import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Create sample weather data (you can replace this with real data)
np.random.seed(42)
n_samples = 100  # Reduced sample size for clarity

# Generate simple weather data
data = pd.DataFrame({
    'temperature': np.random.normal(25, 5, n_samples),  # Temperature in Celsius
    'humidity': np.random.normal(60, 10, n_samples),    # Humidity in %
    'wind_speed': np.random.normal(10, 3, n_samples),   # Wind speed in km/h
    'rainfall': np.random.normal(5, 2, n_samples)       # Rainfall in mm
})

# Split features (X) and target (y)
X = data[['temperature', 'humidity', 'wind_speed']]
y = data['rainfall']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = RandomForestRegressor(n_estimators=50, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate accuracy metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print results
print("\nModel Performance:")
print(f"Mean Squared Error: {mse:.2f}")
print(f"R² Score: {r2:.2f}")

# Feature importance
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
})
print("\nFeature Importance:")
print(feature_importance.sort_values(by='Importance', ascending=False))

# Create visualizations
plt.figure(figsize=(12, 4))

# Plot 1: Actual vs Predicted
plt.subplot(121)
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel("Actual Rainfall (mm)")
plt.ylabel("Predicted Rainfall (mm)")
plt.title("Actual vs Predicted Rainfall")

# Plot 2: Feature Importance
plt.subplot(122)
feature_importance.plot(x='Feature', y='Importance', kind='bar', ax=plt.gca())
plt.title("Feature Importance")
plt.xlabel("Weather Features")
plt.ylabel("Importance Score")
plt.tight_layout()

# Example: Make a new prediction
new_weather = pd.DataFrame({
    'temperature': [24],
    'humidity': [65],
    'wind_speed': [12]
})

prediction = model.predict(new_weather)
print(f"\nExample Prediction:")
print(f"Weather Conditions:")
print(f"Temperature: 24°C")
print(f"Humidity: 65%")
print(f"Wind Speed: 12 km/h")
print(f"Predicted Rainfall: {prediction[0]:.2f} mm")

plt.show()
