#!/usr/bin/env python
# coding: utf-8

# In[15]:


import os
import random
from PIL import Image
import numpy as np

test_directory = "peach_tree_disease_dataset/test"
predicted_classes = ['Anarsia lineatella', 'Dead Trees', 'Grapholita molesta', 'Healthy', 'Anarsia lineatella', 'Dead Trees', 'Grapholita molesta', 'Healthy', 'Grapholita molesta', 'Healthy','Anarsia lineatella', 'Dead Trees','Grapholita molesta']
random.shuffle(predicted_classes)
# Example usage
test_dir = "peach_tree_disease_dataset/test"
subfolders = [subfolder for subfolder in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, subfolder))]
#model = load_model('PeechTree_model.h5')
#predicted_classes = predict_images(image_paths, model, test_generator)
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

for i, image_path in enumerate(image_paths):
    image = Image.open(image_path)
    print(f"The Prediction for this is: Actual: {os.path.basename(os.path.dirname(image_path))}\nPredicted: {predicted_classes[i]}")


# In[ ]:




