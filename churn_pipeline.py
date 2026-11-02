"""
churn_pipeline.py
Shared, importable building blocks for the Telecom Churn Risk Model.

Keeping the custom classes in a real module (instead of the notebook's __main__)
means the saved model (churn_model.joblib) can be loaded in ANY Python session
as long as this file sits next to it.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------
ID_COLS = ["customerID", "CustomerID", "Customer ID"]
TARGET_CANDIDATES = ["Churn", "Churn Label"]
# Columns that only exist AFTER a customer has churned (or are model outputs
# built from the outcome). They would leak the future into the model.
LEAKAGE_PATTERNS = ("churn label", "churn score", "churn reason", "churn category",
                    "churn value", "cltv", "customer status", "satisfaction score")

ADDON_COLS = ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
              "TechSupport", "StreamingTV", "StreamingMovies"]
PROTECT_COLS = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport"]
STREAM_COLS = ["StreamingTV", "StreamingMovies"]

_YES = {"yes", "y", "true", "1"}
_NO = {"no", "n", "false", "0"}
_NO_SERVICE = {"no internet service", "no phone service"}


# --------------------------------------------------------------------------
# Deterministic cleaning (no statistics learned from data -> safe before split)
# --------------------------------------------------------------------------
def _unify_case(s: pd.Series) -> pd.Series:
    """Collapse spelling/case/whitespace variants of the same label."""
    key = s.str.lower().str.replace(r"\s+", " ", regex=True)
    canon = s.groupby(key).agg(lambda x: x.value_counts().index[0])
    return key.map(canon)


def clean_telco(df_raw: pd.DataFrame):
    """Return (X, y, ids). y is None when the target column is absent (scoring)."""
    df = df_raw.copy()
    df.columns = [str(c).strip() for c in df.columns]

    # 1) whitespace -> NaN for every text column
    for c in df.columns:
        if not pd.api.types.is_numeric_dtype(df[c]):
            s = df[c].astype(str).str.strip()
            df[c] = s.mask(s.isin(["", "nan", "NaN", "None", "NA", "N/A"]), np.nan)

    # 2) identifiers / target / leakage columns
    id_col = next((c for c in ID_COLS if c in df.columns), None)
    ids = df[id_col].copy() if id_col else pd.Series(np.arange(len(df)), index=df.index)
    target_col = next((c for c in TARGET_CANDIDATES if c in df.columns), None)
    y = None
    if target_col is not None:
        y = df[target_col].str.lower().isin(_YES).astype(int)
    drop = set([id_col] if id_col else [])
    if target_col:
        drop.add(target_col)
    for c in df.columns:
        low = c.lower().replace("_", " ")
        if c != target_col and any(p in low for p in LEAKAGE_PATTERNS):
            drop.add(c)
    df = df.drop(columns=[c for c in drop if c in df.columns])

    # 3) numeric columns
    for c in ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # 4) inconsistent category labels
    for c in [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]:
        low = df[c].str.lower()
        vals = set(low.dropna().unique())
        if vals and vals <= (_YES | _NO | _NO_SERVICE):
            # "No internet service" / "No phone service" are just another way
            # of saying "No" (the real information lives in InternetService /
            # PhoneService), so they are merged into "No".
            df[c] = low.map(lambda v: np.nan if pd.isna(v) else ("Yes" if v in _YES else "No"))
        else:
            df[c] = _unify_case(df[c])

    # 5) blank TotalCharges belong to brand-new customers (tenure == 0): nothing
    #    has been billed yet, so the true value is 0 (business rule, not a guess)
    if {"TotalCharges", "tenure"} <= set(df.columns):
        df.loc[df["TotalCharges"].isna() & (df["tenure"] == 0), "TotalCharges"] = 0.0
    return df, y, ids


# --------------------------------------------------------------------------
# Feature engineering (stateless transformer with switchable groups)
# --------------------------------------------------------------------------
FEATURE_GROUPS = {
    "tenure_lifecycle": ["tenure_band", "is_new_customer"],
    "spend_pattern": ["avg_monthly_spend", "charge_delta", "charge_per_service"],
    "service_bundle": ["addon_count", "protect_count", "stream_count"],
    "contract_payment": ["is_month_to_month", "is_electronic_check", "is_autopay"],
    "risk_combos": ["fiber_no_support", "family_score"],
}
ALL_GROUPS = tuple(FEATURE_GROUPS)
N_ENGINEERED = sum(len(v) for v in FEATURE_GROUPS.values())


def _yes(df, col):
    return (df[col] == "Yes").astype(int) if col in df.columns else 0


class ChurnFeatureEngineer(BaseEstimator, TransformerMixin):
    """Adds engineered columns. `groups=None` -> all groups, `groups=()` -> none."""

    def __init__(self, groups=None):
        self.groups = groups

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        groups = ALL_GROUPS if self.groups is None else tuple(self.groups)
        tenure = X["tenure"].astype(float)
        mc = X["MonthlyCharges"].astype(float)
        tc = pd.to_numeric(X["TotalCharges"], errors="coerce")
        avg_spend = (tc / tenure.where(tenure > 0)).fillna(mc)          # history-based price
        addon = sum(_yes(X, c) for c in ADDON_COLS)
        n_services = addon + _yes(X, "PhoneService") + (X["InternetService"] != "No").astype(int)

        if "tenure_lifecycle" in groups:
            X["tenure_band"] = pd.cut(tenure, [-1, 6, 12, 24, 48, np.inf],
                                      labels=["0-6m", "7-12m", "13-24m", "25-48m", "49m+"]).astype(str)
            X["is_new_customer"] = (tenure <= 6).astype(int)
        if "spend_pattern" in groups:
            X["avg_monthly_spend"] = avg_spend
            X["charge_delta"] = mc - avg_spend                          # >0 : paying more than historically
            X["charge_per_service"] = mc / (1 + n_services)
        if "service_bundle" in groups:
            X["addon_count"] = addon
            X["protect_count"] = sum(_yes(X, c) for c in PROTECT_COLS)
            X["stream_count"] = sum(_yes(X, c) for c in STREAM_COLS)
        if "contract_payment" in groups:
            X["is_month_to_month"] = (X["Contract"] == "Month-to-month").astype(int)
            X["is_electronic_check"] = (X["PaymentMethod"] == "Electronic check").astype(int)
            X["is_autopay"] = X["PaymentMethod"].str.contains("automatic", case=False, na=False).astype(int)
        if "risk_combos" in groups:
            X["fiber_no_support"] = ((X["InternetService"] == "Fiber optic")
                                     & (X["TechSupport"] == "No")
                                     & (X["OnlineSecurity"] == "No")).astype(int)
            X["family_score"] = _yes(X, "Partner") + _yes(X, "Dependents")
        return X


# --------------------------------------------------------------------------
# Pre-processing (ColumnTransformer) and model pipeline builder
# --------------------------------------------------------------------------
def numeric_columns(df):
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]


def categorical_columns(df):
    return [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]


def make_preprocessor(scale=True):
    num = Pipeline([("imp", SimpleImputer(strategy="median")),
                    ("sc", StandardScaler() if scale else "passthrough")])
    cat = Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                    ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    return ColumnTransformer([("num", num, numeric_columns),
                              ("cat", cat, categorical_columns)],
                             verbose_feature_names_out=False)


def build_pipeline(model, groups=None, scale=True):
    return Pipeline([("fe", ChurnFeatureEngineer(groups=groups)),
                     ("prep", make_preprocessor(scale)),
                     ("clf", model)])


# --------------------------------------------------------------------------
# Gradient boosting with class_weight support (sklearn's GBM only takes sample_weight)
# --------------------------------------------------------------------------
class BalancedGB(GradientBoostingClassifier):
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3,
                 min_samples_leaf=1, subsample=1.0, max_features=None,
                 class_weight=None, random_state=None):
        super().__init__(n_estimators=n_estimators, learning_rate=learning_rate,
                         max_depth=max_depth, min_samples_leaf=min_samples_leaf,
                         subsample=subsample, max_features=max_features,
                         random_state=random_state)
        self.class_weight = class_weight

    def fit(self, X, y, sample_weight=None):
        if self.class_weight is not None:
            cw = compute_sample_weight(self.class_weight, y)
            sample_weight = cw if sample_weight is None else cw * sample_weight
        return super().fit(X, y, sample_weight=sample_weight)
