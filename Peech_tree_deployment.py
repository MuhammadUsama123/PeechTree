#!/usr/bin/env python
# coding: utf-8

# In[16]:


import os
import random
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

from tensorflow.keras.preprocessing.image import ImageDataGenerator

test_directory = "peach_tree_disease_dataset/test"

test_datagen = ImageDataGenerator(rescale=1./255)
# Load and preprocess test dataset
test_generator = test_datagen.flow_from_directory(
    test_directory,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical'
)

def predict_images(image_paths, model, test_generator):
    predictions = []
    
    for image_path in image_paths:
        # Load and preprocess the image
        image = Image.open(image_path)
        image = image.convert("RGB")  # Convert to RGB if necessary
        image = image.resize((150, 150))  # Resize to model input size
        image_array = np.expand_dims(np.array(image), axis=0) / 255.0  # Add batch dimension and normalize
        # Make prediction using the trained model
        prediction = model.predict(image_array)
        # Get the predicted class index
        reverse_class_indices = {v: k for k, v in test_generator.class_indices.items()}
        predicted_class_index = np.argmax(prediction)
        # Get the predicted class label using the reverse mapping
        class_label = reverse_class_indices[predicted_class_index]
        # Append the predicted class label to the list of predictions
        predictions.append(class_label)

    return predictions

# Example usage
test_dir = "peach_tree_disease_dataset/test"
subfolders = [subfolder for subfolder in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, subfolder))]
print(subfolders);
image_paths = []

# Loop 12 times to collect 12 images
for _ in range(12):
    # Randomly select a subfolder
    random_subfolder = random.choice(subfolders)
    subfolder_path = os.path.join(test_dir, random_subfolder)
    
    # Get the list of image files in the selected subfolder
    image_files = [file for file in os.listdir(subfolder_path) if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff'))]
    random.shuffle(image_files)  # Shuffle the list of image files
    
    # Check if there are valid image files in the subfolder
    if image_files:
        # Select one image randomly from the subfolder
        image_path = os.path.join(subfolder_path, image_files[0])
        image_paths.append(image_path)

# Shuffle the list of selected image paths

random.shuffle(image_paths)
# Make predictions for the randomly selected images

model = load_model('CSVD/PeechTree_model.h5')
predicted_classes = predict_images(image_paths, model, test_generator)

for i, image_path in enumerate(image_paths):
    image = Image.open(image_path)
    print(f"Actual: {os.path.basename(os.path.dirname(image_path))}\nPredicted: {predicted_classes[i]}")

