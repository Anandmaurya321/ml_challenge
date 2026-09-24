
### Here we tries to understand 
- the problem,  
- dataset, 
- requirements, 
- matrix

### and used to write our own understanding regarding that and will talk about the baseline of the model 


# Basic setup code 

# Step1: Install libraries

```python
!pip install xgboost scikit-learn -q
```


# Step2: Setup segamaker session
```python
import sagemaker
import boto3
import pandas as pd
import numpy as np
import time

session = sagemaker.Session()
role = sagemaker.get_execution_role()
region = session.boto_region_name
bucket = session.default_bucket()

print(f"Region: {region}")
print(f"Role: {role}")
print(f"Bucket: {bucket}")
```





# step3: Load dataset
```python
data = pd.read_csv(
    f"s3://sagemaker-example-files-prod-{region}/datasets/tabular/synthetic/churn.txt"
)
df = data.copy()

print(f"Dataset shape: {df.shape}")
df.head()
```





# step4: Feature Engineering

```python
from sklearn.model_selection import train_test_split

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

print(f"Processed: {model_data.shape}")
model_data.head()
```

# step 5: Split the data
```python
# Split into train (67%), validation (22%), test (11%)
train_data, validation_data = train_test_split(model_data, test_size=0.33, random_state=42)
validation_data, test_data = train_test_split(validation_data, test_size=0.33, random_state=42)

# Separate test labels (we'll use these to check accuracy later)
test_target = test_data['Churn?_True.']
test_data_no_target = test_data.drop(['Churn?_True.'], axis=1)

# Separate features and labels for train/validation
train_features = train_data.iloc[:, 1:]
train_labels = train_data.iloc[:, 0]
val_features = validation_data.iloc[:, 1:]
val_labels = validation_data.iloc[:, 0]

print(f"Training:   {train_data.shape[0]} rows")
print(f"Validation: {validation_data.shape[0]} rows")
print(f"Test:       {test_data.shape[0]} rows")
```






# Step 6: Train with your model 

```python
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
```



## Everything that is purely deterministic and doesn't learn from the dataset can generally be done before splitting.

Before Split                    After Split
─────────────────               ─────────────────
Remove duplicates               Imputation
Fix data types                  Encoding
Clean strings                   Scaling
Create new features             Vectorization
Handle obvious invalid values   PCA
Feature engineering             Feature Selection




