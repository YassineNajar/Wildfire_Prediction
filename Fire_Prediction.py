#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd

# Use the raw URL for the dataset
url = "https://raw.githubusercontent.com/aravind-selvam/forest-fire-prediction/refs/heads/main/dataset/Algerian_forest_fires_dataset_UPDATE.csv"

# Load the data with proper handling of delimiters
data = pd.read_csv(url, delimiter=',', header=1)  # header=1 skips the first line which contains column names

# Check the first few rows to understand the structure
print(data.head(126))


# In[7]:


# Drop lines 124 and 125 (index 125 and 124)
data = data.drop([122, 123], axis=0)  # Drop rows by index


# In[8]:


# Check the first few rows to understand the structure
print(data.head(126))


# In[9]:


# Combine the day, month, and year columns into a single date column
data['date'] = pd.to_datetime(data[['year', 'month', 'day']])


# In[10]:


import matplotlib.pyplot as plt
# Plot FWI vs Date
plt.figure(figsize=(10, 6))
plt.plot(data['date'], data['FWI'], label='FWI')


# In[11]:


print(data.columns)


# In[12]:


data.isnull().sum()


# In[13]:


# Assuming 'data' is your DataFrame
unique_classes = data['Classes  '].unique()
print(unique_classes)


# In[14]:


# Strip whitespace from column names
data.columns = data.columns.str.strip()
# Mapping 'Classes' column values
data['Classes'] = data['Classes'].apply(lambda x: "not fire" if 'n' in str(x).lower() else "fire")


# In[15]:


# Remove rows with missing values in 'Classes'
data = data.dropna(subset=['Classes'])


# In[16]:


import numpy as np
print(data.isna().sum())  # Check for NaN values in each column


# In[17]:


# Assuming 'data' is your DataFrame
unique_classes = data['Classes'].unique()
print(unique_classes)


# In[18]:


# Mapping 'Classes' column values
data['Classes'] = data['Classes'].apply(lambda x: 0 if 'n' in str(x).lower() else 1)

# Check the result
print(data['Classes'])


# In[19]:


# Count the number of occurrences of 0 and 1 in the 'Classes' column
class_counts = data['Classes'].value_counts()

# Print the counts
print(class_counts)


# In[20]:


data.info()


# In[21]:


# Convert columns to numeric, forcing errors to NaN
columns_to_convert = ['day', 'month', 'year', 'Temperature', 'RH', 'Ws', 'FFMC', 'DMC', 'DC', 'ISI', 'BUI', 'FWI']

# Convert each column to numeric, invalid parsing will be set as NaN
for col in columns_to_convert:
    data[col] = pd.to_numeric(data[col], errors='coerce')

# Check data types after conversion
print(data.info())


# In[19]:





# In[24]:


import numpy as np
print(data.isna().sum())  # Check for NaN values in each column
 


# In[25]:


# Display rows where there are missing values
missing_rows = data[data.isna().any(axis=1)]
print(missing_rows)


# In[28]:


# Drop rows with missing values and update the original DataFrame
data = data.dropna()

print(data)


# In[29]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Step 1: Prepare your feature set (X) and target variable (y)
X = data[['day', 'month', 'year', 'Temperature', 'RH', 'Ws', 'FFMC', 'DMC', 'DC', 'ISI', 'BUI', 'FWI']]
y = data['Classes']  # Target variable


# In[30]:


# Step 2: Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Instantiate and train the model (RandomForestClassifier)
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)


# In[31]:


# Step 4: Make predictions
y_pred = model.predict(X_test)

# Step 5: Evaluate the model
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print("Classification Report:")
print(classification_report(y_test, y_pred))


# In[ ]:





# In[ ]:




