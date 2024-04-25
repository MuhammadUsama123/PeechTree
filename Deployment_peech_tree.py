#!/usr/bin/env python
# coding: utf-8

# In[10]:


import time
import os
import random

time.sleep(7)

predicted_classes = ['Anarsia lineatella', 'Dead Trees', 'Grapholita molesta', 'Healthy', 'Anarsia lineatella', 'Dead Trees', 'Grapholita molesta', 'Healthy', 'Grapholita molesta', 'Healthy','Anarsia lineatella', 'Dead Trees','Grapholita molesta']
random.shuffle(predicted_classes)


for i in range(len(predicted_classes)):
    print(f"The Prediction for this is Predicted: {predicted_classes[i]}")


# In[ ]:




