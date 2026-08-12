import tensorflow as tf
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report, precision_recall_curve, auc
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
import mlflow
import time
import mlflow.tensorflow


def train_xgboost(X, y , max_depth: int, n_estimators: int, lr: float, random_state: int, n_splits: int, weight_method: str, weight_function, experiment_name:str) :

    skf = StratifiedKFold(n_splits = n_splits, shuffle=True, random_state= random_state)

    params = {
    "n_splits" : n_splits,
    "sample_weight": weight_method,
     }

    xgb_params = {
        "n_estimators": n_estimators,
        "max_depth" : max_depth,
        "learning_rate" : lr,
        "random_state": random_state,
        "enable_categorical" :True
    }
        

    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name="cv_parent") as parent_run:
        mlflow.log_params(params)
        xgmodels = {}

        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]

            sample_weights = weight_function(y_tr, method="balanced")

            with mlflow.start_run(run_name=f"fold_{fold_idx}", nested=True):
                pr_auc_scores = []
                mlflow.log_param('fold', fold_idx)
                mlflow.log_dict(sample_weights, f"class_weights_fold_{fold_idx}.json")

                veh_clf = XGBClassifier(**xgb_params)

                mlflow.log_params(xgb_params)

                veh_clf.fit(X_tr, y_tr, sample_weight=sample_weights)
                xgmodels[fold_idx] = veh_clf

                y_pred = veh_clf.predict(X_val)
                y_probs = veh_clf.predict_proba(X_val)
                num_classes = y_probs.shape[1]

                T = np.zeros(num_classes)
                
                for i in range(num_classes):
                  y_true_b = (y_val == i).astype(int)
                  precisions, recalls, thresholds = precision_recall_curve(y_true_b, y_probs[:,i])
                  score = auc(recalls, precisions)
                  pr_auc_scores.append(score)

                  mlflow.log_metric(f"Fold {fold_idx} pr_auc_class{i}", score)

                  f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-9)

                  best_f1_id = np.argmax(f1_scores)

                  if best_f1_id < len(thresholds):
                      T[i] = thresholds[best_f1_id]
                  else:
                      T[i] = 0.5

                macro_pr_auc = np.mean(pr_auc_scores)
                mlflow.log_metric(f"macro_pr_auc_{fold_idx}", macro_pr_auc)
                #mlflow.log_metric(f"Thresholds_for_fold{fold_idx}", T)      

                adjusted_proba = y_probs / np.clip(T, 1e-6, None)
                y_pred_tuned = np.argmax(adjusted_proba, axis=1)


                report = classification_report(y_val, y_pred_tuned, output_dict=True)
                
                mlflow.log_metric(f"Fold {fold_idx}accuracy", report["accuracy"])
                mlflow.log_metric(f"Fold {fold_idx} macro_f1", report["macro avg"]["f1-score"])
                mlflow.log_metric(f"Fold {fold_idx} weighted_f1", report["weighted avg"]["f1-score"])

                mlflow.xgboost.log_model(
                    xgb_model = veh_clf,
                    artifact_path = f"Fold_{fold_idx}_xgb_model",
                    input_example = X_tr[:5],
                )

    return xgmodels            





def train_NN(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int ,
    epochs: int ,
    batch_size: int ,
    learning_rate: float,
    class_weight_method: str,
    experiment_name: str,
    random_state: int ,
    netfunction ,
    get_class_weights
):

    """
    Train my tf model logs params with mlflow
    """
    
    y = np.asarray(y)
    num_classes = len(np.unique(y))
    input_dim = X.shape[1]
 
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
 
    mlflow.set_experiment(experiment_name)
    mlflow.tensorflow.autolog(log_models=False)  
    model_fold_dict = {}
 
    with mlflow.start_run(run_name="cv_parent") as parent_run:
        mlflow.log_params(
            {
                "n_splits": n_splits,
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "class_weight_method": class_weight_method,
                "num_classes": num_classes,
                "input_dim": input_dim,
            }
        )
 
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]

            pr_auc_scores = []
 
            class_weights = get_class_weights(y_tr, method=class_weight_method)
 
            with mlflow.start_run(run_name=f"fold_{fold_idx}", nested=True):
      
                class MLflowEpochLogger(tf.keras.callbacks.Callback):
                  def on_epoch_end(self, epoch, logs=None):
                      mlflow.log_metric("train_loss", logs["loss"], step=epoch)
                      mlflow.log_metric("val_loss", logs["val_loss"], step=epoch) 

                mlflow.log_param("fold", fold_idx)
                mlflow.log_dict(class_weights, f"class_weights_fold_{fold_idx}.json")
 
                model = netfunction(input_dim, num_classes, learning_rate)
 
                callbacks = [
                    tf.keras.callbacks.EarlyStopping(
                        monitor="val_loss", patience=8, restore_best_weights=True
                    ),
                    tf.keras.callbacks.ReduceLROnPlateau(
                        monitor="val_loss", factor=0.5, patience=4
                    ),
                    MLflowEpochLogger()
                ]
 
                history = model.fit(
                    X_tr,
                    y_tr,
                    validation_data=(X_val, y_val),
                    epochs=epochs,
                    batch_size=batch_size,
                    class_weight=class_weights,
                    callbacks=callbacks,
                    verbose=0,
                )

                best_val_loss = min(history.history["val_loss"])
                mlflow.log_metric("best_val_loss", best_val_loss)
                mlflow.tensorflow.log_model(model, artifcat_path=f"model_Fold{fold_idx}")

                model_fold_dict[fold_idx] = model

                y_probs = model.predict(X_val)
                y_preds = np.argmax(y_probs, axis=1)
                num_classes = y_probs.shape[1]
                T = np.zeros(num_classes)

                for i in range(num_classes):
                  y_true_b = (y_val == i).astype(int)
                  precisions, recalls, thresholds = precision_recall_curve(y_true_b, y_probs[:,i])
                  score = auc(recalls, precisions)
                  pr_auc_scores.append(score)

                  mlflow.log_metric(f"Fold {fold_idx} pr_auc_class{i}", score)

                  f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-9)

                  best_f1_id = np.argmax(f1_scores)

                  if best_f1_id < len(thresholds):
                      T[i] = thresholds[best_f1_id]
                  else:
                      T[i] = 0.5

                macro_pr_auc = np.mean(pr_auc_scores)
                mlflow.log_metric(f"macro_pr_auc_{fold_idx}", macro_pr_auc)      

                adjusted_proba = y_probs / np.clip(T, 1e-6, None)
                y_pred_tuned = np.argmax(adjusted_proba, axis=1)


                report = classification_report(y_val, y_pred_tuned, output_dict=True)
                
                mlflow.log_metric(f"Fold {fold_idx}accuracy", report["accuracy"])
                mlflow.log_metric(f"Fold {fold_idx} macro_f1", report["macro avg"]["f1-score"])
                mlflow.log_metric(f"Fold {fold_idx} weighted_f1", report["weighted avg"]["f1-score"])


               
        
                    
                                

                
            