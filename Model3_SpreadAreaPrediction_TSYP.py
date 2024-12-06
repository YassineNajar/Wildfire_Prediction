#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import mean_absolute_error
import geopandas as gpd
from shapely.geometry import Point


# In[2]:


# Load dataset
#change the file path after downloading the dataset (GOAT) available in the drive link provided
file_path = r"C:\Users\najar\Downloads\GOAT DATASET - Data.csv"
fire_data = pd.read_csv(file_path)


# In[3]:


# Convert date columns to datetime
fire_data['fire_start_date'] = pd.to_datetime(fire_data['fire_start_date'], errors='coerce')
fire_data['discovered_date'] = pd.to_datetime(fire_data['discovered_date'], errors='coerce')
fire_data['dispatch_date'] = pd.to_datetime(fire_data['dispatch_date'], errors='coerce')


# In[4]:


# Convert numeric columns to float
numeric_columns = ['current_size', 'wind_speed', 'bh_hectares', 'uc_hectares', 'to_hectares', 'ex_hectares']
for col in numeric_columns:
    fire_data[col] = fire_data[col].replace({',': '.'}, regex=True).astype(float)

# Handle missing values
fire_data[numeric_columns] = fire_data[numeric_columns].fillna(fire_data[numeric_columns].median())


# In[5]:


# For non-numeric columns, fill missing values with a placeholder
non_numeric_columns = fire_data.select_dtypes(exclude=[np.number]).columns
for col in non_numeric_columns:
    fire_data[col] = fire_data[col].fillna("Unknown")


# In[6]:


# Clean the latitude and longitude columns
fire_data['fire_location_latitude'] = fire_data['fire_location_latitude'].str.replace(',', '.').astype(float)
fire_data['fire_location_longitude'] = fire_data['fire_location_longitude'].str.replace(',', '.').astype(float)

# Drop rows with missing latitude or longitude
fire_data = fire_data.dropna(subset=['fire_location_latitude', 'fire_location_longitude'])


# In[7]:


# Create 'geometry' for mapping
fire_data['geometry'] = fire_data.apply(lambda row: Point(row['fire_location_longitude'], row['fire_location_latitude']), axis=1)

# Create GeoDataFrame
gdf = gpd.GeoDataFrame(fire_data, geometry='geometry')


# In[8]:


# Ensure 'fire_spread_rate' is numeric and drop rows with missing target values
fire_data['fire_spread_rate'] = pd.to_numeric(fire_data['fire_spread_rate'], errors='coerce')
fire_data = fire_data.dropna(subset=['fire_spread_rate'])


# In[9]:


# Define features (X) and target variable (y)
X = fire_data[['current_size', 'wind_speed', 'bh_hectares', 'uc_hectares', 'to_hectares', 'ex_hectares']]
y = fire_data['fire_spread_rate']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[10]:


# Hyperparameter Tuning using GridSearchCV

# Define hyperparameter grid
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2']
}

# Instantiate the RandomForestRegressor
rf_model = RandomForestRegressor(random_state=42)

# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)

# Fit GridSearchCV to the training data
grid_search.fit(X_train, y_train)

# Get the best parameters
best_params = grid_search.best_params_
print(f"Best hyperparameters found: {best_params}")


# In[11]:


# Get the best model from GridSearchCV
best_model = grid_search.best_estimator_


import joblib

# Save the best model to a file
joblib_file = "best_rf_model.pkl"
joblib.dump(best_model, joblib_file)

print(f"Model saved to {joblib_file}")


# In[12]:


# Evaluate the model using cross-validation
cv_scores = cross_val_score(best_model, X, y, cv=5, scoring='neg_mean_absolute_error')

# Convert negative MAE to positive
cv_scores = -cv_scores

# Print the cross-validation results
print(f"Cross-Validation Mean MAE: {np.mean(cv_scores):.2f}")
print(f"Cross-Validation Standard Deviation MAE: {np.std(cv_scores):.2f}")


# In[13]:


# Make predictions on the test set
y_pred = best_model.predict(X_test)

# Calculate and print MAE on the test set
test_mae = mean_absolute_error(y_test, y_pred)
print(f"Test Set MAE: {test_mae:.2f}")


# In[14]:


# Plot Feature Importances
feature_importances = best_model.feature_importances_
features = X.columns

plt.figure(figsize=(10, 6))
plt.bar(features, feature_importances)
plt.title("Feature Importance (Random Forest)")
plt.xlabel("Features")
plt.ylabel("Importance")
plt.show()


# In[15]:


# Ensure both columns are in datetime format
fire_data['assessment_datetime'] = pd.to_datetime(fire_data['assessment_datetime'])
fire_data['reported_date'] = pd.to_datetime(fire_data['reported_date'])

# Step 1: Calculate the time difference (in hours) between the assessment time and the reported time
fire_data['time_difference_hours'] = (fire_data['assessment_datetime'] - fire_data['reported_date']).dt.total_seconds() / 3600

# Step 2: Calculate the spread area (spread_rate * time_difference)
fire_data['spread_area'] = fire_data['fire_spread_rate'] * fire_data['time_difference_hours']

# Step 3: Calculate the radius (in km) of the spread, assuming the area forms a circle
fire_data['spread_radius_km'] = np.sqrt(fire_data['spread_area'] / np.pi)

# Load the world map shapefile
world12 = gpd.read_file(r'C:\Users\najar\Downloads\ne_10m_admin_0_countries\ne_10m_admin_0_countries.shp')

# Randomly select 3 fires from the dataset
sampled_fires = fire_data.sample(n=50, random_state=42)

# Plot the world map
fig, ax = plt.subplots(figsize=(15, 10))
world12.plot(ax=ax, color='lightgray')

# Add the sampled fire locations and spread radius circles
for idx, row in sampled_fires.iterrows():
    # Get the fire's center (latitude, longitude)
    latitude = row['fire_location_latitude']
    longitude = row['fire_location_longitude']
    radius_km = row['spread_radius_km']
    
    # Create a circle representing the fire's spread
    circle = plt.Circle((longitude, latitude), radius_km, color='red', alpha=0.5, edgecolor='black')
    
    # Add the circle to the plot
    ax.add_patch(circle)

# Set limits to show only Canada
ax.set_xlim([-141, -52])  # Longitude from -141 (west) to -52 (east)
ax.set_ylim([41, 83])     # Latitude from 41 (south) to 83 (north)

# Add title and labels
plt.title("Fire Spread Areas in Canada")
plt.xlabel("Longitude")
plt.ylabel("Latitude")

# Show the plot
plt.show()


# In[ ]:




