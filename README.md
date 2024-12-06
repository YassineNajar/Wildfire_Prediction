# Wildfire_Prediction

## Fire Spread Prediction and Detection

This repository contains three machine learning models for predicting and detecting fire spread in various regions, with a focus on remote sensing data and geospatial analysis. The models use a combination of deep learning (CNN), classical machine learning (Random Forest), and spatial analytics to achieve this goal.

## Table of Contents:

1. [Project Overview](#project-overview)
2. [Setup Instructions](#setup-instructions)
3. [Usage Instructions](#usage-instructions)
4. [Models Overview](#models-overview)
   - Model 1: Wildfire Occurrence Prediction using Random Forest
   - Model 2: Fire Detection using CNN
   - Model 3: Fire Spread Visualization and Geospatial Analysis
5. [Evaluation and Results](#evaluation-and-results)
   - Model 1 Evaluation
   - Model 2 Evaluation
   - Model 3 Evaluation
6. [Additional Notes](#additional-notes)

---

## 1) Project Overview

This project focuses on predicting fire spread rates, detecting fire locations, and visualizing fire spread using geospatial data and image classification techniques. The models are built using a mix of deep learning and machine learning algorithms. The data comes from a combination of satellite imagery (for fire detection) and geospatial datasets (for fire spread prediction).

- **Model 1** predicts wildfire occurrence using a Random Forest Regressor based on various fire-related features.
- **Model 2** uses a Convolutional Neural Network (CNN) for classifying aerial images of fires.
- **Model 3** visualizes fire spread areas on a map and calculates the radius of fire spread over time using geospatial data.

---

## 2) Setup Instructions

### 1. Install Required Libraries

Make sure you have the following libraries installed:

pip install tensorflow scikit-learn geopandas matplotlib numpy pandas joblib

The following libraries are used in this project:

- **TensorFlow**: For building deep learning models.
- **scikit-learn**: For machine learning models and evaluation metrics.
- **GeoPandas**: For handling geospatial data.
- **Matplotlib**: For plotting graphs and visualizations.
- **Joblib**: For saving models.

## 2. Datasets

You can download the required datasets from the following links:

- **Fire Dataset (GOAT)**: Download from the provided link. Adjust the `file_path` variable in the script to the location where the dataset is saved.
  - [Dataset Link](https://drive.google.com/drive/folders/1OFfNRu5bWIth2HiWtiAxw5_lkfTu1SR5)
- **Geospatial Dataset**: Download a shapefile of world countries for mapping and fire location visualization.

### 3. File Paths

Make sure to update the file paths in the scripts to point to the correct location of the datasets on your system.

## 3. Usage Instructions

### 3.1 Train Model 1 (Random Forest for Fire Spread Prediction)

1. Run the script to train the **Random Forest** model. It uses the geospatial dataset and fire-related features (like wind speed and fire size) to predict the fire spread rate.
2. The model will be saved as `best_rf_model.pkl`.

### 3.2 Train Model 2 (CNN for Fire Detection)

1. Run the script to train the **CNN** model. It processes images in the specified directories (`Fire` and `No_Fire`) and classifies them as fire or no-fire images.
2. The trained model will be saved as `fire_no_fire_mdl`.

### 3.3 Model 3 (Fire Spread Visualization and Geospatial Analysis)

1. Run the script to generate a **geospatial visualization** of fire spread in Canada, displaying the radius of each fire's spread over time.
2. The map will be generated, showing fire spread areas based on the calculated radius.

## 4. Models Overview

### Model 1: Wildfire Prediction using Random Forest

This model predicts the probability of wildfire occurrence using a **Random Forest Regressor**. The model is trained with various features like wind speed, current fire size, and hectares affected.

**Hyperparameter Tuning**:  
GridSearchCV is used for hyperparameter tuning, testing different values for parameters such as `n_estimators`, `max_depth`, and `min_samples_split`.

**Features**:
- Current fire size
- Wind speed
- Affected hectares, etc.

**Evaluation**:  
Model performance is evaluated using **Mean Absolute Error (MAE)** on the test set.

### Model 2: Fire Detection using CNN

This model classifies aerial images into "Fire" and "No Fire" categories using a **Convolutional Neural Network (CNN)**. The architecture is built using TensorFlow and Keras.

**Architecture**:
- **Input**: 256x256 RGB images
- **Layers**:
  - Data Augmentation (for training images)
  - Three **Conv2D** layers (with increasing filters: 32, 64, 128)
  - **MaxPooling** layers after each Conv2D layer
  - Fully connected **Dense** layers (128 neurons)
- **Output**: 2 classes (Fire, No Fire)

**Training**:  
The model is trained with class weights to handle imbalanced data.

### Model 3: Fire Spread Visualization and Geospatial Analysis

This model uses **geospatial data** to calculate fire spread over time and visualize the affected areas on a map.

**Process**:
1. **Data Preprocessing**:
   - Latitude and longitude are cleaned and combined to create a geometry column for mapping.
   - The dataset is cleaned for missing or erroneous values.
   - The fire spread rate and time difference between reported and assessment times are calculated.
   
2. **Geospatial Visualization**:
   - The model visualizes fire spread on a world map (with a focus on Canada) by displaying fire locations and the calculated spread radius as circles.
   - The map shows the spread area based on calculated data.

**Evaluation**:  
This model is evaluated based on its ability to accurately visualize and spatially analyze fire spread. No specific performance metric is used, as it focuses on accurate spatial visualization.

## 5. Evaluation and Results

### Model 1 Evaluation (Random Forest)

- **Predicted vs Actual**:  
  A plot comparing predicted vs actual **BUI** (Burnt Area Index) values, showing the model’s accuracy.

- **Confusion Matrix**:  
  A confusion matrix to evaluate the model’s classification performance (for fire prediction).

### Model 3 Evaluation (Fire Spread Visualization)

- **Fire Spread Radius**:  
  The radius is calculated based on the spread rate and time difference, and visualized with circles on the map.

## 6. Future Improvements

- Explore other **geospatial features** like terrain type, historical weather patterns, and proximity to infrastructure.
- **Additional Datasets**: More data from different regions or longer time periods would enhance the model's accuracy and robustness.
