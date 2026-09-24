import pandas as pd
import xgboost as xgb

# ==========================================
# 1. LOAD TRAINED MODEL
# ==========================================

model = xgb.XGBClassifier()
model.load_model("xgb_model.json")


# ==========================================
# 2. LOAD TEST DATA
# ==========================================

test_data_no_target = pd.read_csv("test_data_no_target.csv")


# ==========================================
# 3. MAKE PREDICTIONS
# ==========================================

predictions = model.predict_proba(test_data_no_target)[:, 1]


# ==========================================
# 4. CONVERT PROBABILITY TO CLASS
# ==========================================

predicted_class = (predictions > 0.5).astype(int)


# ==========================================
# 5. CREATE PREDICTION DATAFRAME
# ==========================================

prediction_df = pd.DataFrame({
    "churn_probability": predictions, ### if we also required to show probability of class.
    "prediction": predicted_class
})


# ==========================================
# 6. SAVE PREDICTIONS
# ==========================================

prediction_df.to_csv(
    "predictions.csv",
    index=False
)


# ==========================================
# 7. DISPLAY SAMPLE
# ==========================================

print("Predictions saved successfully!")
print(prediction_df.head(10))


## Now we just requried to upload it >>>>>>>>>
