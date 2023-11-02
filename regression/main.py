# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Step 1: Data Preparation
# Load the dataset
data = pd.read_csv("regression/houseprice_data.csv")

# Check for missing values and handle them if necessary
data.dropna(inplace=True)  # Remove rows with missing values

# Encode categorical variables if needed
data = pd.get_dummies(data)

# Step 2: Data Exploration
# Explore the data
print(data.head())  # Display the first few rows of the dataset

# Visualize the relationships between features and target variable
# Example: Scatter plot of the number of bedrooms vs. house price
plt.scatter(data['bedrooms'], data['price'])
plt.xlabel('Number of Bedrooms')
plt.ylabel('House Price')
plt.title('Number of Bedrooms vs. House Price')
plt.show()

# Step 3: Feature Selection (Assuming all features are used for simplicity)
X = data.drop('price', axis=1)
y = data['price']

# Step 4: Model Building
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Linear Regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Step 5: Model Evaluation
# Make predictions on the test data
y_pred = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error: {mae}")
print(f"Mean Squared Error: {mse}")
print(f"Root Mean Squared Error: {rmse}")
print(f"R-squared: {r2}")

# Step 6: Visualization
# Visualize the model's predictions against actual house prices
plt.scatter(y_test, y_pred)
plt.xlabel('Actual House Prices')
plt.ylabel('Predicted House Prices')
plt.title('Actual vs. Predicted House Prices')
plt.show()

# Step 7: Improvements
# You can experiment with feature engineering and other regression techniques here.

# Step 8: Conclusion
# Summarize the findings and discuss the model's effectiveness and potential improvements.
