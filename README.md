# mode_choice

Predicting the **transport mode** of a trip from trip, person and household
attributes in the 2017 U.S. National Household Travel Survey (NHTS). The
project covers data extraction, exploratory analysis, multi-class model
training (XGBoost and a Keras neural network) tracked with MLflow, and an
in-progress LangGraph agent that can query a Postgres database and call the
trained model through natural language.

## Project layout

```
mode_choice/
├── train_pipeline.py          # CLI entry point: load data -> train models -> save folds
├── src/
│   ├── data/data_preprocess.py   # loading, encoding, train/test split, class/sample weights
│   ├── model/model.py            # custom_Net: Keras functional MLP classifier
│   └── training/train.py         # train_xgboost / train_NN with stratified K-fold + MLflow
├── agents/
│   ├── agent.py               # LangGraph ReAct agent (Groq LLM + Postgres SQL tools + XGBoost tool)
│   └── requirements.txt       # agent dependencies
├── notebooks/
│   ├── extract_data.ipynb     # joins raw NHTS trip/person/household tables -> trip_data.csv
│   ├── data_analysis.ipynb    # EDA, cleaning, label mapping -> final_data.csv
│   ├── (test) trian_pipeline.ipynb
│   └── xgboost_models/        # 10 saved XGBoost fold models + per-fold class thresholds
├── artifacts/
│   └── xgb_lbl_enc.pkl        # LabelEncoder fitted on the target classes
├── dataset/                   # raw and processed CSVs (git-ignored)
├── server.py / client.py      # FastAPI service + CLI client (see "Status" below)
└── (test) test_pipeline.ipynb
```

## Data

Source: the 2017 NHTS public-use files (`trippub.csv`, `perpub.csv`,
`hhpub.csv`, `vehpub.csv`). Data files are git-ignored, so place them under
`dataset/` (or `notebooks/raw_dataset/`, which is the path the extraction
notebook reads from).

1. **`extract_data.ipynb`** replaces NHTS missing-value codes
   (`-1, -7, -8, -9, -88, -99`) with NaN, keeps a subset of columns, and joins
   trips → persons (`HOUSEID`, `PERSONID`) → households (`HOUSEID`), writing
   `trip_data.csv`.
2. **`data_analysis.ipynb`** drops sparse columns, removes nulls, maps coded
   values to readable labels (trip purpose, travel day, etc.), groups modes
   under 1% of trips into `other_mode`, and writes `final_data.csv`.

Final dataset: about 818k trips × 16 columns.

| Feature group | Columns |
|---|---|
| Trip | `TRVLCMIN` (minutes), `TRPMILES`, `WHYTO` (purpose), `TRAVDAY`, `URBRUR` |
| Person | `R_AGE`, `SEX`, `WORKER`, `DRIVER` |
| Household | `HHFAMINC (AVERAGE)`, `HHVEHCNT`, `HHSIZE`, `HOMEOWN` |
| IDs | `HOUSEID`, `PERSONID` |
| **Target** | `Transport_mode` |

Class distribution of `Transport_mode` (imbalanced):

| Class | Trips |
|---|---|
| suv | 363,086 |
| pickup-truck | 349,712 |
| car | 71,569 |
| other_mode | 33,614 |

## Modeling

Both models use a 70/30 train/test split, then **10-fold stratified CV** on
the training portion. Class imbalance is handled with `balanced` sample or class
weights (an `effective`-number-of-samples option is also available in
`get_class_weights`).

**XGBoost** (`train_xgboost`): native categorical support
(`enable_categorical=True`, `tree_method="hist"`), `max_depth=10`,
`n_estimators=2000`, `learning_rate=0.01`, early stopping after 50 rounds.

**Neural network** (`train_NN`, `custom_Net`): one-hot encoded categoricals plus
standard-scaled numerics, fed to Dense 256 → 128 → 64 with BatchNorm and
Dropout(0.5), softmax output, Adam, and sparse categorical cross-entropy. It
uses early stopping (patience 8) and ReduceLROnPlateau, and picks
Mirrored/OneDevice strategies automatically when GPUs are available.

For each fold, both trainers:
- log per-class PR-AUC, macro PR-AUC, accuracy, and macro/weighted F1 to MLflow
  (fold runs are nested under a `cv_parent` run)
- compute a per-class decision threshold that maximizes F1 on the
  precision-recall curve (returned as `fold_thresholds`)
- log the fitted model to MLflow

At the end of the run, the CV mean and standard deviation of macro-F1 are logged.

## Getting started

Requires Python 3.12.

```bash
python -m venv venv
venv\Scripts\activate          # Windows  (source venv/bin/activate on Linux/macOS)
pip install pandas numpy scikit-learn xgboost tensorflow mlflow joblib matplotlib seaborn
```

### Train

```bash
python train_pipeline.py --input dataset/final_data.csv
```

- `--input` is the path to `final_data.csv`. It defaults to an S3 URI
  (`s3://personal-ds-data-bucket/final_data.csv`), which also needs `s3fs`.
- `--output` is the output folder (currently unused, since outputs go to `artifacts/`).

At the moment the pipeline runs only the **NN stage**. The XGBoost stage is
commented out in `main()`. The fold models are saved to
`artifacts/nn_models/nn_fold_{k}.keras`.

### View experiments

```bash
mlflow ui                      # experiments: xgb_experiment, nn_experiment
```

`mlruns/` and the `*.db` files are git-ignored. To use a SQLite backend,
uncomment the `mlflow.set_tracking_uri(...)` line in `train_pipeline.py`.

## Agent (`agents/agent.py`)

This is a LangGraph ReAct loop built by hand (`agent → should_continue →
action → agent …`) with in-memory checkpointing per thread. The LLM is served by
Groq and has two kinds of tools:

- **Postgres SQL toolkit**: list tables, inspect schema, check queries, run queries
- **`use_xgboost_model`**: takes a JSON object of feature values, builds a
  one-row DataFrame, and returns the predicted class and probabilities

```bash
pip install -r agents/requirements.txt
python agents/agent.py         # interactive REPL
```

Configure it with environment variables or a `.env` file:

| Variable | Purpose |
|---|---|
| `GROQ_API_KEY` | Groq API key |
| `GROQ_MODEL_NAME` | e.g. `llama-3.3-70b-versatile` |
| `POSTGRES_URI` | SQLAlchemy URI, default `postgresql+psycopg2://user:password@localhost:5432/mode_choice` |
| `XGBOOST_MODEL_PATH` | trained model (`.json`/`.ubj` Booster or joblib), default `artifacts/xgb_model.json` |

The tool's input contract is still a placeholder, so the expected feature
names have not been enforced yet.

