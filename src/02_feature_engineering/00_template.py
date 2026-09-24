

# Drop non-predictive columns
df = df.drop("Phone", axis=1)

# Area Code is a code, not a number
df["Area Code"] = df["Area Code"].astype(object)

# Remove redundant charge columns (charges = minutes x rate)
df = df.drop(["Day Charge", "Eve Charge", "Night Charge", "Intl Charge"], axis=1)

# One-hot encode categorical features (Yes/No -> 1/0)
model_data = pd.get_dummies(df)

# Move target to first column (XGBoost convention)
model_data = pd.concat(
    [model_data["Churn?_True."],
     model_data.drop(["Churn?_False.", "Churn?_True."], axis=1)],
    axis=1
)
model_data = model_data.astype(float)



# ==========================================
# ENCODE CATEGORICAL VARIABLES
# ==========================================

# One-Hot Encoding
df = pd.get_dummies(
    df,
    columns=cat_cols,
    drop_first=True
)

print(f"Processed: {model_data.shape}")
model_data.head()




