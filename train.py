# ----------------------------------------------------
# IMPORTS
# ----------------------------------------------------
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import pickle

# ----------------------------------------------------
# LOAD DATA
# ----------------------------------------------------
df = pd.read_csv("C:\\Users\\sahil\\Desktop\\diabetes_project\\diabetes.csv")


# ----------------------------------------------------
# HANDLE INVALID ZEROS
# ----------------------------------------------------
invalid_zero_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

for col in invalid_zero_cols:
    median_value = df[col].median()
    df[col] = df[col].replace(0, median_value)

# ----------------------------------------------------
# OUTLIER CLIPPING USING IQR
# ----------------------------------------------------
cols = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
        "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]

for col in cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    df[col] = np.where(df[col] < lower, lower,
                       np.where(df[col] > upper, upper, df[col]))

# ----------------------------------------------------
# TRAIN-TEST SPLIT
# ----------------------------------------------------
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ----------------------------------------------------
# SCALING
# ----------------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_raw)
X_test_scaled = scaler.transform(X_test_raw)

# ----------------------------------------------------
# LOGISTIC REGRESSION
# ----------------------------------------------------
log_model = LogisticRegression()
log_model.fit(X_train_scaled, y_train)
y_pred_log = log_model.predict(X_test_scaled)

print("\n--- Logistic Regression ---")
print("Accuracy:", accuracy_score(y_test, y_pred_log))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_log))

# ----------------------------------------------------
# SAVE MODEL + SCALER
# ----------------------------------------------------
pickle.dump(log_model, open("model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))

print("\nModel and scaler saved successfully as model.pkl and scaler.pkl")
