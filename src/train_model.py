"""Train and export the BizLens customer churn model pipeline.

Run from the project root:
    python src/train_model.py --data data/customer_churn_data.csv
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import joblib
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "customer_churn_data.csv"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "bizlens_churn_pipeline.joblib"


@dataclass(frozen=True)
class ProjectConfig:
    """Training configuration shared by the export script and model pipeline."""

    random_state: int = 42
    test_size: float = 0.2
    target_col: str = "Churn"
    cat_features: Tuple[str, ...] = (
        "gender",
        "SeniorCitizen",
        "Partner",
        "Dependents",
        "PhoneService",
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "Contract",
        "PaperlessBilling",
        "PaymentMethod",
    )
    num_features: Tuple[str, ...] = ("tenure", "MonthlyCharges", "TotalCharges")

    @property
    def feature_columns(self) -> Tuple[str, ...]:
        return self.num_features + self.cat_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the BizLens churn pipeline and save bizlens_churn_pipeline.joblib."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="Path to customer_churn_data.csv. Defaults to data/customer_churn_data.csv.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Output path for the trained joblib pipeline. Defaults to the project root.",
    )
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_dataset(data_path: Path, config: ProjectConfig) -> pd.DataFrame:
    resolved_path = resolve_path(data_path)
    if not resolved_path.exists():
        raise FileNotFoundError(
            "Dataset not found. Place the Telco churn CSV at "
            f"{DEFAULT_DATA_PATH.relative_to(PROJECT_ROOT)} or pass a path with "
            "--data path/to/customer_churn_data.csv."
        )

    df = pd.read_csv(resolved_path)
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    required_columns = set(config.feature_columns + (config.target_col,))
    missing_columns = sorted(required_columns - set(df.columns))
    if missing_columns:
        raise ValueError(
            "Dataset is missing required column(s): " + ", ".join(missing_columns)
        )

    print(f"Loaded dataset: {resolved_path}")
    print(f"Dataset shape: {df.shape}")
    return df


def build_preprocessor(config: ProjectConfig) -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, list(config.num_features)),
            ("cat", categorical_pipeline, list(config.cat_features)),
        ],
        verbose_feature_names_out=False,
    )


def build_pipeline(config: ProjectConfig) -> ImbPipeline:
    return ImbPipeline(
        steps=[
            ("preprocessor", build_preprocessor(config)),
            ("smote", SMOTE(random_state=config.random_state)),
            (
                "classifier",
                XGBClassifier(
                    eval_metric="logloss",
                    random_state=config.random_state,
                    n_estimators=200,
                    learning_rate=0.05,
                    max_depth=5,
                ),
            ),
        ]
    )


def prepare_training_data(
    df: pd.DataFrame, config: ProjectConfig
) -> tuple[pd.DataFrame, pd.Series]:
    features = df.drop(columns=[config.target_col, "customerID"], errors="ignore")
    features = features.loc[:, list(config.feature_columns)]

    target = df[config.target_col]
    if target.dtype == "object":
        target = target.str.strip().map({"Yes": 1, "No": 0})

    if target.isna().any():
        raise ValueError("Target column Churn must contain only Yes/No or 1/0 values.")

    return features, target.astype(int)


def train_and_export(data_path: Path, output_path: Path, config: ProjectConfig) -> float:
    df = load_dataset(data_path, config)
    x, y = prepare_training_data(df, config)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=y,
    )
    print(f"Training shape: X={x_train.shape}, y={y_train.shape}")
    print(f"Test shape: X={x_test.shape}, y={y_test.shape}")

    pipeline = build_pipeline(config)
    pipeline.fit(x_train, y_train)

    y_prob = pipeline.predict_proba(x_test)[:, 1]
    roc_auc = roc_auc_score(y_test, y_prob)
    print(f"ROC AUC score: {roc_auc:.4f}")

    resolved_output = resolve_path(output_path)
    resolved_output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, resolved_output)
    print(f"Saved trained pipeline to: {resolved_output}")
    return roc_auc


def main() -> None:
    args = parse_args()
    config = ProjectConfig()
    try:
        train_and_export(args.data, args.output, config)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
