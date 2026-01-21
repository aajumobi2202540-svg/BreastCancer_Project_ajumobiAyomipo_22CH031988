import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Load Dataset
data = pd.read_csv("train.csv")

# Feature Selection

selected_features = [
    "OverallQual",
    "GrLivArea",
    "TotalBsmtSF",
    "GarageCars",
    "YearBuilt",
    "Neighborhood",
    "SalePrice"
]

data = data[selected_features]



# Handle Missing Values

data["TotalBsmtSF"].fillna(data["TotalBsmtSF"].median(), inplace=True)
data["GarageCars"].fillna(data["GarageCars"].median(), inplace=True)

# Categorical feature
data["Neighborhood"].fillna(data["Neighborhood"].mode()[0], inplace=True)


# Encode Categorical Variable
label_encoder = LabelEncoder()
data["Neighborhood"] = label_encoder.fit_transform(data["Neighborhood"])



# Split Features and Target
X = data.drop("SalePrice", axis=1)
y = data["SalePrice"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# Train Model
model = LinearRegression()
model.fit(X_train, y_train)


# Evaluate Model

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("Model Evaluation Metrics")
print("------------------------")
print(f"MAE  : {mae:.2f}")
print(f"MSE  : {mse:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"R²   : {r2:.4f}")



# Save Model and Encoder
with open("house_price_model.pkl", "wb") as model_file:
    pickle.dump(model, model_file)

with open("label_encoder.pkl", "wb") as encoder_file:
    pickle.dump(label_encoder, encoder_file)

print("\nModel and Label Encoder saved successfully!")
print("Files saved:")
print("- house_price_model.pkl")
print("- label_encoder.pkl")
