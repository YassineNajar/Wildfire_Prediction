#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, classification_report, roc_auc_score
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns


# In[19]:


# Load the dataset
#change the file_path with the actual path of the tranformed_file available in the drive link:https://drive.google.com/drive/folders/1OFfNRu5bWIth2HiWtiAxw5_lkfTu1SR5?usp=drive_link
file_path = r"C:\Users\najar\Downloads\transformed_file - transformed_file.csv"  # Update this with your file path
data = pd.read_csv(file_path)

# Analyze the dataset
print("Dataset Shape:", data.shape)
print("Missing Values:\n", data.isnull().sum())
print("Sample Rows:\n", data.head())

# Ensure all numerical features are numeric
features = ['Temperature', 'RH', 'Ws', 'Rain', 'FFMC', 'DMC', 'DC', 'ISI']
for col in features:
    data[col] = pd.to_numeric(data[col], errors='coerce')

# Drop or fill problematic rows
data.dropna(subset=features, inplace=True)


# In[3]:


# Visualize feature distribution and correlation heatmap
plt.figure(figsize=(12, 6))
sns.heatmap(data[features].corr(), annot=True, cmap='coolwarm', fmt='.2f', linewidths=1)
plt.title("Correlation Heatmap")
plt.show()

# Visualize the distribution of features
data[features].hist(figsize=(12, 8), bins=20)
plt.suptitle("Feature Distribution")
plt.show()


# In[4]:


# Separate rows for regression
bui_data = data.dropna(subset=['BUI'])
missing_bui_data = data[data['BUI'].isna()]
fwi_data = data.dropna(subset=['FWI'])
missing_fwi_data = data[data['FWI'].isna()]


# In[5]:


# Function to evaluate models
def evaluate_regression_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    rmse = mean_squared_error(y_test, predictions, squared=False)
    r2 = r2_score(y_test, predictions)
    return mae, rmse, r2

def evaluate_classification_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    class_report = classification_report(y_test, predictions)
    try:
        auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    except:
        auc = None
    return accuracy, class_report, auc


# In[6]:


# Regression: Predict BUI using XGBoost
X_bui = bui_data[features]
y_bui = bui_data['BUI']
X_train_bui, X_test_bui, y_train_bui, y_test_bui = train_test_split(X_bui, y_bui, test_size=0.2, random_state=42)

# Train and evaluate XGBoost for BUI prediction
xgb_model_bui = xgb.XGBRegressor(random_state=42)
print("Evaluating XGBoost for BUI Prediction:")
mae_bui, rmse_bui, r2_bui = evaluate_regression_model(xgb_model_bui, X_train_bui, X_test_bui, y_train_bui, y_test_bui)
print(f"XGBoost BUI MAE: {mae_bui}")
print(f"XGBoost BUI RMSE: {rmse_bui}")
print(f"XGBoost BUI R²: {r2_bui}\n")

# Predict missing BUI values
missing_bui_data['BUI'] = xgb_model_bui.predict(missing_bui_data[features])
data.update(missing_bui_data)


# In[7]:


# Regression: Predict FWI using RandomForest
X_fwi = fwi_data[features]
y_fwi = fwi_data['FWI']
X_train_fwi, X_test_fwi, y_train_fwi, y_test_fwi = train_test_split(X_fwi, y_fwi, test_size=0.2, random_state=42)

# Train and evaluate RandomForest for FWI prediction
rf_model_fwi = RandomForestRegressor(random_state=42)
print("Evaluating RandomForest for FWI Prediction:")
mae_fwi, rmse_fwi, r2_fwi = evaluate_regression_model(rf_model_fwi, X_train_fwi, X_test_fwi, y_train_fwi, y_test_fwi)
print(f"RandomForest FWI MAE: {mae_fwi}")
print(f"RandomForest FWI RMSE: {rmse_fwi}")
print(f"RandomForest FWI R²: {r2_fwi}\n")

# Predict missing FWI values
missing_fwi_data['FWI'] = rf_model_fwi.predict(missing_fwi_data[features])
data.update(missing_fwi_data)


# In[8]:


# Regression: Predict FWI using RandomForest
X_fwi = fwi_data[features]
y_fwi = fwi_data['FWI']
X_train_fwi, X_test_fwi, y_train_fwi, y_test_fwi = train_test_split(X_fwi, y_fwi, test_size=0.2, random_state=42)

# Train and evaluate RandomForest for FWI prediction
rf_model_fwi = RandomForestRegressor(random_state=42)
print("Evaluating RandomForest for FWI Prediction:")
mae_fwi, rmse_fwi, r2_fwi = evaluate_regression_model(rf_model_fwi, X_train_fwi, X_test_fwi, y_train_fwi, y_test_fwi)
print(f"RandomForest FWI MAE: {mae_fwi}")
print(f"RandomForest FWI RMSE: {rmse_fwi}")
print(f"RandomForest FWI R²: {r2_fwi}\n")

# Predict missing FWI values
missing_fwi_data['FWI'] = rf_model_fwi.predict(missing_fwi_data[features])
data.update(missing_fwi_data)


# In[9]:


# Classification: Predict Fire/No Fire using different models
data.dropna(subset=['Classes'], inplace=True)  # Ensure no missing class labels
X_class = data[features + ['BUI', 'FWI']]
y_class = data['Classes'].apply(lambda x: 1 if x == 'fire' else 0)  # Binary encode

X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(X_class, y_class, test_size=0.2, random_state=42)


# In[11]:


from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix

# Define the classification models you want to evaluate
classification_models = {
    'Random Forest': RandomForestClassifier(),
    'Logistic Regression': LogisticRegression(),
    'SVM': SVC(probability=True)
}

# Function to evaluate classification models and track Accuracy
def evaluate_classification_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    
    # Calculate accuracy and classification report
    accuracy = accuracy_score(y_test, predictions)
    class_report = classification_report(y_test, predictions)
    
    # Confusion Matrix to calculate False Negatives (if needed for extended functionality)
    cm = confusion_matrix(y_test, predictions)
    
    if len(set(y_test)) == 2:  # Binary classification
        tn, fp, fn, tp = cm.ravel()
        auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    else:  # Multiclass classification
        auc = None  # AUC for multiclass is not straightforward, could be extended if needed
    
    return accuracy, class_report, auc

# Initialize variables to track the model with the best Accuracy
best_model_name = None
best_model = None
best_accuracy = -float('inf')  # Start with a very low value to ensure we pick the highest

# Loop through all models and evaluate them
for model_name, model in classification_models.items():
    print(f"Evaluating {model_name} for Fire Prediction:")
    accuracy, class_report, auc = evaluate_classification_model(model, X_train_class, X_test_class, y_train_class, y_test_class)
    
    print(f"{model_name} Accuracy: {accuracy:.4f}")
    print(f"{model_name} Classification Report:\n{class_report}")
    if auc is not None:
        print(f"{model_name} AUC: {auc:.4f}")
    else:
        print(f"{model_name} AUC: N/A (Multiclass problem)")

    # Track the model with the highest accuracy
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model_name = model_name
        best_model = model

print(f"\nModel with best Accuracy: {best_model_name} with Accuracy: {best_accuracy:.4f}")


# In[12]:


import joblib

# After the model evaluation code

# Plot Predicted vs Actual for BUI (Regression)
plt.figure(figsize=(8, 6))
plt.scatter(y_test_bui, xgb_model_bui.predict(X_test_bui), color='blue', label='Predicted vs Actual (BUI)', alpha=0.7)
plt.plot([min(y_test_bui), max(y_test_bui)], [min(y_test_bui), max(y_test_bui)], color='red', linestyle='--', label='Perfect Prediction Line')
plt.title('Predicted vs Actual for BUI (XGBoost)')
plt.xlabel('Actual BUI')
plt.ylabel('Predicted BUI')
plt.legend()
plt.show()


# In[ ]:





# In[14]:


# Plot Confusion Matrix for Fire Prediction (Classification)
from sklearn.metrics import confusion_matrix
import seaborn as sns


# In[15]:


# Get confusion matrix for best model (classification)
cm_class = confusion_matrix(y_test_class, best_model.predict(X_test_class))

# Plot confusion matrix using heatmap
plt.figure(figsize=(6, 6))
sns.heatmap(cm_class, annot=True, fmt='d', cmap='Blues', xticklabels=['No Fire', 'Fire'], yticklabels=['No Fire', 'Fire'])
plt.title(f'Confusion Matrix for {best_model_name} Model')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()


# In[16]:


# Check if all FWI predictions are zero
predicted_fwi_values = rf_model_fwi.predict(X_test_fwi)
unique_values = np.unique(predicted_fwi_values)


# In[17]:


# Print the unique values of the predicted FWI values
print(f"Unique predicted FWI values: {unique_values}")


# In[18]:


# Visualize the distribution of predicted FWI values
plt.figure(figsize=(8, 6))
plt.hist(predicted_fwi_values, bins=20, color='purple', edgecolor='black')
plt.title('Distribution of Predicted FWI Values')
plt.xlabel('Predicted FWI')
plt.ylabel('Frequency')
plt.show()


# In[20]:


# Save the best model using joblib
model_filename = f"{best_model_name}_best_model.joblib"
joblib.dump(best_model, model_filename)
print(f"\nBest model saved as {model_filename}")


# In[ ]:




