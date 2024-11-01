# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt

# Load the data
file_path = 'rockville,-maryland-air-quality.csv'  # Update this path to your actual file path
air_quality_data = pd.read_csv(file_path)

# Data cleaning
# Remove any leading/trailing whitespace in column names
air_quality_data.columns = air_quality_data.columns.str.strip()
# Convert date to datetime format, PM2.5 and O3 to numeric
air_quality_data['date'] = pd.to_datetime(air_quality_data['date'], errors='coerce')
air_quality_data['pm25'] = pd.to_numeric(air_quality_data['pm25'], errors='coerce')
air_quality_data['o3'] = pd.to_numeric(air_quality_data['o3'], errors='coerce')
# Drop any rows with NaN values
air_quality_data = air_quality_data.dropna()

# Sort data by date
air_quality_data = air_quality_data.sort_values(by='date')

# Exploratory Analysis - Visualizing Trends
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
plt.plot(air_quality_data['date'], air_quality_data['pm25'], color='orange', label='PM2.5 Levels')
plt.axhline(y=50, color='green', linestyle='--', label='Good AQI Threshold (50)')
plt.axhline(y=100, color='yellow', linestyle='--', label='Moderate AQI Threshold (100)')
plt.axhline(y=150, color='red', linestyle='--', label='Unhealthy Threshold (150)')
plt.title("PM2.5 Levels Over Time")
plt.xlabel("Date")
plt.ylabel("PM2.5 Concentration")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(air_quality_data['date'], air_quality_data['o3'], color='blue', label='Ozone Levels')
plt.axhline(y=50, color='green', linestyle='--', label='Good AQI Threshold (50)')
plt.axhline(y=100, color='yellow', linestyle='--', label='Moderate AQI Threshold (100)')
plt.axhline(y=150, color='red', linestyle='--', label='Unhealthy Threshold (150)')
plt.title("Ozone Levels Over Time")
plt.xlabel("Date")
plt.ylabel("Ozone Concentration")
plt.legend()

plt.tight_layout()
plt.show()

# Data Standardization
scaler = StandardScaler()
scaled_data = scaler.fit_transform(air_quality_data[['pm25', 'o3']])

# Calculating Euclidean Distances
# Compute distances between consecutive time periods to measure change in air quality levels
distances = euclidean_distances(scaled_data[:-1], scaled_data[1:])
distance_df = pd.DataFrame(distances, columns=['PM2.5-O3 Distance'], index=air_quality_data['date'][:-1])

# Output analysis
print("Summary of changes in air quality levels (PM2.5 and O3) over time:")
print(distance_df.describe())

# Plot Euclidean distances to visualize the magnitude of changes over time
plt.figure(figsize=(10, 5))
plt.plot(distance_df.index, distance_df['PM2.5-O3 Distance'], label='Distance Between Time Periods')
plt.title("Change in Air Quality Levels Over Time (Euclidean Distance)")
plt.xlabel("Date")
plt.ylabel("Euclidean Distance")
plt.legend()
plt.show()
