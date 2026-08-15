import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight, compute_class_weight
from sklearn.compose import ColumnTransformer
import joblib
import mlflow
from pathlib import Path

def load_dataset(memory):
    return pd.read_csv('final_data.csv', low_memory=memory)

def process_data_xgboost(df: pd.DataFrame, target : str ):

    le = LabelEncoder()

    X = df.drop(columns=[target])
    y = df[target] 

    y = le.fit_transform(y)

    project_root = Path(__file__).resolve().parents[2]
    save_path = project_root / "notebooks" / "xgb_lbl_enc.pkl"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(le, save_path)

    mlflow.log_artifact(str(save_path))


    cat_features = X.select_dtypes(include=["object", "category", "bool"]).columns
    X[cat_features] = X[cat_features].astype('category')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    
    
    return X_train, X_test, y_train, y_test



def get_sample_weights(y: np.ndarray, method:str) -> np.ndarray:
    sample_weights = compute_sample_weight(method, y)
    return sample_weights 

def map_label(target):
    unique_labels = target.unique()
   # map_dict = {1:'car', 2: 'pickup-truck' , 3: 'rec-car',4: 'pickup-truck' , 6: 'rec-car' , 7: 'motorcycle' , 8: 'public-bus' , 9: 'school-bus' ,10: 'street-car' , 11: 'subway' , 12: 'comm-rail' ,
    #13: 'amtrak' , 14: 'airplane' , 15: 'taxi' , 16: 'ride-share' , 17: 'paratransit' , 18: 'bicycle' , 
    #19:'e-scooter', 20:'walked', 21: 'noneabove' }

    #sort_keys = sorted(map_dict.keys())
    #raw_to_tf = {raw_key : tf_index for tf_index, raw_key in enumerate(sort_keys)}

    label_map = {label: tf_index  for tf_index, label in enumerate(unique_labels)}

    return label_map



def get_class_weights(y: np.ndarray, method: str = "balanced", beta: float = 0.999) -> dict:
  
    classes = np.unique(y)
 
    if method == "balanced":
        weights = compute_class_weight(class_weight="balanced", classes=classes, y=y)
        return {int(c): float(w) for c, w in zip(classes, weights)}

    if method == "effective":
        counts = np.array([np.sum(y == c) for c in classes], dtype=np.float64)
        effective_num = 1.0 - np.power(beta, counts)
        weights = (1.0 - beta) / effective_num
        weights = weights / weights.sum() * len(classes)  # normalize, mean ~1
        return {int(c): float(w) for c, w in zip(classes, weights)}
 
    raise ValueError("method must be 'balanced' or 'effective'")
 




def process_data_NN(
    df: pd.DataFrame, target_col: str , target_map: dict, process_object: ColumnTransformer | None = None):
    """
    preprocesses my data and returns it in addition to the preprocessor for testing
    """

    df = df.copy()
    df[target_col] = df[target_col].map(target_map)
    y = df[target_col].values
    X = df.drop(columns=[target_col])

    X_training, X_testing, y_training, y_testing = train_test_split( X, y, test_size=0.3)
 
    categorical_features = X.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()
    
    numerical_features = X.select_dtypes(
        include=["int64", "int32", "float64", "float32"]
    ).columns.tolist()

    if process_object is None:
        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "cat",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                    categorical_features,
                ),
                ("num", StandardScaler(), numerical_features),
            ],
            remainder="drop",
        )
        
        X_proc = preprocessor.fit_transform(X_training)

    else:
        passed_processor = process_object
        X_proc = passed_processor.fit_transform(X_training)

    print(f"Training data shape after processing = {X_proc.shape} and label shape = {y_training.shape}")
    print(f"Testing data shape before processing = {X_testing.shape} and label shape = {y_testing.shape}")
        

    return X_proc, y_training, X_testing, y_testing, preprocessor
    