import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# Step 1: Load dataset
data = pd.read_csv('C:\Users\dell\OneDrive\Desktop\Linear-Regression-Sales-Prediction\data\advertising.csv')

print("First 5 rows of dataset:")
print(data.head())

# Step 2: Define features and target
X = data[['TV', 'Radio', 'Newspaper']]
y = data['Sales']

# Step 3: Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 4: Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 5: Predict
y_pred = model.predict(X_test)

# Step 6: Evaluate
print("\nModel Evaluation:")
print("MAE:", mean_absolute_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# Step 7: Visualization
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs Predicted Sales")

plt.savefig('C:\Users\dell\OneDrive\Desktop\Linear-Regression-Sales-Prediction\outputs/actual_vs_predicted.png')
plt.show()