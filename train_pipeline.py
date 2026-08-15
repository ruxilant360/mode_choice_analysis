import os
from pathlib import Path
import argparse

import mlflow

from src.data import (
    load_dataset,
    process_data_xgboost,
    process_data_NN,
    get_sample_weights,
    get_class_weights,
    map_label,
)
from src.model import custom_Net
from src.training import train_xgboost, train_NN

PROJECT_ROOT = Path(__file__).resolve().parent
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
TARGET_COL = "Transport_mode"


def train_xgboost_stage(df):
    X_train, X_test, y_train, y_test = process_data_xgboost(df, TARGET_COL)

    xgmodels, fold_thresholds = train_xgboost(
        X_train,
        y_train,
        max_depth=10,
        n_estimators=2000,
        lr=0.01,
        random_state=42,
        n_splits=10,
        weight_method="balanced",
        weight_function=get_sample_weights,
        experiment_name="xgb_experiment",
    )
    return xgmodels, fold_thresholds


def train_nn_stage(df):
    target_map = map_label(df[TARGET_COL])
    X_proc, y_train, X_test, y_test, preprocessor = process_data_NN(
        df, TARGET_COL, target_map
    )

    model_fold_dict, fold_thresholds = train_NN(
        X_proc,
        y_train,
        n_splits=5,
        epochs=50,
        batch_size=64,
        learning_rate=1e-3,
        class_weight_method="balanced",
        experiment_name="nn_experiment",
        random_state=42,
        netfunction=custom_Net,
        get_class_weights=get_class_weights,
    )
    return model_fold_dict, fold_thresholds


def main(args):
    mlflow.set_tracking_uri(f"sqlite:///{(NOTEBOOKS_DIR / 'mlflow.db').as_posix()}")

    #os.chdir(NOTEBOOKS_DIR)

    df = load_dataset(args.input, True)

    xgmodels, xgb_fold_thresholds = train_xgboost_stage(df)
    nn_models, nn_fold_thresholds = train_nn_stage(df)

    return {
        "xgboost": {"models": xgmodels, "fold_thresholds": xgb_fold_thresholds},
        "nn": {"models": nn_models, "fold_thresholds": nn_fold_thresholds},
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="My script")
    parser.add_argument("input", help='path to your dataset file')

    args = parser.parse_args()
    main(args)
