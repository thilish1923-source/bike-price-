import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score
import xgboost as xgb
import joblib

# Load dataset
df = pd.read_csv("Used_Bikes.csv")

# Features & Target
X = df.drop(columns=["price", "bike_name"])
y = df["price"]

# Categorical and Numerical columns
categorical_cols = ["city", "owner", "brand"]
numerical_cols = ["kms_driven", "age", "power"]

# Preprocessing pipeline
preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
    ("num", "passthrough", numerical_cols)
])

# 🚀 XGBoost with better hyperparameters
xgb_model = xgb.XGBRegressor(
    n_estimators=300,        # more trees
    learning_rate=0.05,      # slower learning, better results
    max_depth=8,             # deeper trees
    subsample=0.8,           # use 80% rows to prevent overfitting
    colsample_bytree=0.8,    # use 80% columns per tree
    gamma=0,                 # minimum loss reduction to make a split
    reg_alpha=0.5,           # L1 regularization (reduce overfitting)
    reg_lambda=1,            # L2 regularization
    random_state=42
)

# Build full pipeline
model_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", xgb_model)
])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train
model_pipeline.fit(X_train, y_train)

# Evaluate
y_pred = model_pipeline.predict(X_test)
print(f"MAE: ₹{mean_absolute_error(y_test, y_pred):.2f}")
print(f"R² Score: {r2_score(y_test, y_pred) * 100:.2f}%")

# Save the model
joblib.dump(model_pipeline, "bike_price_model.pkl")
