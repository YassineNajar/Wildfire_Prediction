#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import pickle
from tensorflow.keras import layers

# Updated directories for training images
# link for the trainig dataset(download the 7th link: Training_Validation images for Fire_vs_NoFire image classification and unzip it
#    : https://ieee-dataport.org/open-access/flame-dataset-aerial-imagery-pile-burn-detection-using-drones-uavs
dir_fire = r"C:/Users/najar/Downloads/Training/Training/Fire"
dir_no_fire = r"C:/Users/najar/Downloads/Training/Training/No_Fire"



# Get the number of images in each class
fire = len([name for name in os.listdir(dir_fire) if os.path.isfile(os.path.join(dir_fire, name))])
no_fire = len([name for name in os.listdir(dir_no_fire) if os.path.isfile(os.path.join(dir_no_fire, name))])

# Check if the classes have images
if fire == 0:
    raise ValueError(f"No images found in Fire directory: {dir_fire}")
if no_fire == 0:
    raise ValueError(f"No images found in No_Fire directory: {dir_no_fire}")

total = fire + no_fire
weight_for_fire = (1 / fire) * total / 2.0
weight_for_no_fire = (1 / no_fire) * total / 2.0

print(f"Weight for class fire : {weight_for_fire:.2f}")
print(f"Weight for class No_fire : {weight_for_no_fire:.2f}")


# Image Preprocessing and Augmentation
image_size = (256, 256)
batch_size = 26

data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
    ]
)

# Load datasets from the directory and split into training and validation sets
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "C:/Users/najar/Downloads/Training/Training", 
    validation_split=0.2, 
    subset="training", 
    seed=1337, 
    image_size=image_size, 
    batch_size=batch_size, 
    shuffle=True
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "C:/Users/najar/Downloads/Training/Training", 
    validation_split=0.2, 
    subset="validation", 
    seed=1337, 
    image_size=image_size, 
    batch_size=batch_size, 
    shuffle=True
)

# Cache and prefetch datasets for performance
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# Build the model
model = keras.Sequential(
    [
        layers.InputLayer(input_shape=(256, 256, 3)),
        data_augmentation,
        layers.Rescaling(1.0 / 255),
        layers.Conv2D(32, 3, activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(128, 3, activation="relu"),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dense(2, activation="softmax"),
    ]
)

# Compile the model
model.compile(
    optimizer=keras.optimizers.Adam(),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Train the model with class weights
class_weights = {0: weight_for_fire, 1: weight_for_no_fire}
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10,
    class_weight=class_weights
)

# Save the model
model.save("fire_no_fire_mdl")

# Plot training and validation accuracy and loss
def plot_history(history):
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()

plot_history(history)

# Save the model history to a file for future analysis
with open("model_history.pkl", "wb") as f:
    pickle.dump(history.history, f)

print("Training Complete!")


# In[ ]:





# In[ ]:





# In[ ]:




