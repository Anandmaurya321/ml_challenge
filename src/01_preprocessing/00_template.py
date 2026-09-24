


# ==========================================
# 1. IMPORT LIBRARIES
# ==========================================

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer





df = pd.read_csv("data.csv")





# ==========================================
# 4. REMOVE DUPLICATES
# ==========================================

df = df.drop_duplicates()





# ==========================================
# 5. HANDLE MISSING VALUES
# ==========================================

# Numerical columns
num_cols = df.select_dtypes(include=np.number).columns

# Categorical columns
cat_cols = df.select_dtypes(include="object").columns

# Fill numerical missing values
for col in num_cols:
    df[col] = df[col].fillna(df[col].median())

# Fill categorical missing values
for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])





# ==========================================
# 6. HANDLE OUTLIERS (OPTIONAL)
# ==========================================

# Example using IQR

for col in num_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)

    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    df[col] = df[col].clip(lower, upper)






