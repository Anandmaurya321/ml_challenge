
import pandas as pd
import xgboost as xgb

### Training on your model 

# Create XGBoost's special data format
dtrain = xgb.DMatrix(train_features, label=train_labels)
dval = xgb.DMatrix(val_features, label=val_labels)

# Hyperparameters
params = {
    "max_depth": 5,
    "eta": 0.2,
    "gamma": 4,
    "min_child_weight": 6,
    "subsample": 0.8,
    "objective": "binary:logistic",
    "eval_metric": "logloss"
}

print("Training locally... (no separate machine needed)")
model = xgb.train(
    params, dtrain, num_boost_round=100,
    evals=[(dtrain, "train"), (dval, "validation")],
    verbose_eval=10
)
print("\nTraining complete!")






### Important : Here we used to save our model to make the prediction 
# Save the trained model locally
model.save_model("xgboost_churn_model.json")
print("Model saved! You can reload it anytime with:")
print("  loaded_model = xgb.Booster()")
print("  loaded_model.load_model('xgboost_churn_model.json')")




