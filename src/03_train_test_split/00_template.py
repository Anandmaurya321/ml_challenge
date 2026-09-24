

from sklearn.model_selection import train_test_split




# ============================================================
# TRAIN / VALIDATION / TEST SPLIT
# ============================================================

train_data, validation_data = train_test_split(
    model_data,
    test_size=0.33,
    random_state=42
)

validation_data, test_data = train_test_split(
    validation_data,
    test_size=0.33,
    random_state=42
)





# ============================================================
# SEPARATE FEATURES AND TARGET
# ============================================================

X_train = train_data.drop(columns=["target"])
y_train = train_data["target"]

X_val = validation_data.drop(columns=["target"])
y_val = validation_data["target"]

X_test = test_data.drop(columns=["target"])
y_test = test_data["target"]





# ============================================================


# Before Split                    After Split
# ─────────────────               ─────────────────
# Remove duplicates               Imputation
# Fix data types                  Encoding
# Clean strings                   Scaling
# Create new features             Vectorization
# Handle obvious invalid values   PCA
# Feature engineering             Feature Selection




# ============================================================








# OPERATIONS THAT MUST BE FIT ONLY ON TRAIN DATA


# 1. IMPUTATION
# imputer.fit(X_train)
# X_train = imputer.transform(X_train)
# X_val   = imputer.transform(X_val)
# X_test  = imputer.transform(X_test)


# 2. ENCODING
# encoder.fit(X_train)
# X_train = encoder.transform(X_train)
# X_val   = encoder.transform(X_val)
# X_test  = encoder.transform(X_test)


# 3. FEATURE SCALING / NORMALIZATION
# scaler.fit(X_train)
# X_train = scaler.transform(X_train)
# X_val   = scaler.transform(X_val)
# X_test  = scaler.transform(X_test)


# 4. TEXT VECTORIZATION
# vectorizer.fit(X_train)
# X_train = vectorizer.transform(X_train)
# X_val   = vectorizer.transform(X_val)
# X_test  = vectorizer.transform(X_test)


# 5. PCA / DIMENSIONALITY REDUCTION
# pca.fit(X_train)
# X_train = pca.transform(X_train)
# X_val   = pca.transform(X_val)
# X_test  = pca.transform(X_test)


# 6. FEATURE SELECTION
# selector.fit(X_train, y_train)
# X_train = selector.transform(X_train)
# X_val   = selector.transform(X_val)
# X_test  = selector.transform(X_test)



