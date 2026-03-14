import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# Load dataset
# -----------------------------
dataset = pd.read_excel("HousePricePrediction.xlsx")

print(dataset.head())
print("Dataset shape:", dataset.shape)

# -----------------------------
# Identify column types
# -----------------------------
categorical_cols = dataset.select_dtypes(include=['object','string']).columns
print("Categorical variables:", len(categorical_cols))

integer_cols = dataset.select_dtypes(include=['int64']).columns
print("Integer variables:", len(integer_cols))

float_cols = dataset.select_dtypes(include=['float64']).columns
print("Float variables:", len(float_cols))

# -----------------------------
# Correlation Heatmap
# -----------------------------
numerical_dataset = dataset.select_dtypes(include=['int64','float64'])

plt.figure(figsize=(12,6))
sns.heatmap(numerical_dataset.corr(),
            cmap='BrBG',
            annot=True,
            fmt='.2f',
            linewidths=2)

plt.title("Correlation Heatmap of Numerical Features")
plt.tight_layout()
plt.savefig("correlation_heatmap.png")

print("Heatmap saved as correlation_heatmap.png")

# -----------------------------
# Categorical feature analysis
# -----------------------------
unique_values = []

for col in categorical_cols:
    unique_values.append(dataset[col].nunique())

plt.figure(figsize=(10,6))
plt.title("No. Unique values of Categorical Features")
plt.xticks(rotation=90)

sns.barplot(x=categorical_cols, y=unique_values)

plt.tight_layout()
plt.savefig("categorical_features.png")

# -----------------------------
# Data cleaning
# -----------------------------
dataset.drop(['Id'], axis=1, inplace=True)

dataset['SalePrice'] = dataset['SalePrice'].fillna(dataset['SalePrice'].mean())

# Drop rows with remaining missing values
dataset = dataset.dropna()

# -----------------------------
# Encode categorical variables
# -----------------------------
dataset_encoded = pd.get_dummies(dataset, drop_first=True)

# -----------------------------
# Split features and target
# -----------------------------
from sklearn.model_selection import train_test_split

X = dataset_encoded.drop(['SalePrice'], axis=1)
y = dataset_encoded['SalePrice']

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, train_size=0.8, random_state=42
)

# -----------------------------
# Train Models
# -----------------------------
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

# SVR
model_SVR = SVR()
model_SVR.fit(X_train, y_train)

y_pred = model_SVR.predict(X_valid)
print("SVR Error:", mean_absolute_percentage_error(y_valid, y_pred))

# Random Forest
model_RF = RandomForestRegressor(n_estimators=100, random_state=42)
model_RF.fit(X_train, y_train)

y_pred = model_RF.predict(X_valid)
print("Random Forest Error:", mean_absolute_percentage_error(y_valid, y_pred))

# Linear Regression
model_LR = LinearRegression()
model_LR.fit(X_train, y_train)

y_pred = model_LR.predict(X_valid)
print("Linear Regression Error:", mean_absolute_percentage_error(y_valid, y_pred))