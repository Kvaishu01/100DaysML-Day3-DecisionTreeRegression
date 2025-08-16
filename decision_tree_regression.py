# Day 3 - Decision Tree Regression
# Predict Salary based on Years of Experience

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor

# --------------------------
# Step 1: Create Sample Dataset
# --------------------------
data = {
    'YearsExperience': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Salary': [45000, 50000, 60000, 65000, 70000,
               85000, 95000, 105000, 125000, 150000]
}
df = pd.DataFrame(data)

X = df[['YearsExperience']]   # Feature
y = df['Salary']              # Target

# --------------------------
# Step 2: Train Model
# --------------------------
regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(X, y)

# --------------------------
# Step 3: Make Prediction
# --------------------------
predicted_salary = regressor.predict([[6.5]])
print(f"Predicted salary for 6.5 years experience: ₹{predicted_salary[0]:,.2f}")

# --------------------------
# Step 4: Visualization
# --------------------------
# High-resolution curve for smooth plot
X_grid = np.arange(min(X.values), max(X.values), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))

plt.scatter(X, y, color='red', label="Actual Data")
plt.plot(X_grid, regressor.predict(X_grid), color='blue', label="Model Prediction")
plt.title("Decision Tree Regression (Day 3)")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.legend()
plt.show()
