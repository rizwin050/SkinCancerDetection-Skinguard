#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.image import imread
import os
import random
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow import keras
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers import Activation, Dropout,  Flatten, Dense,MaxPool2D
 #for explainable
from skimage.segmentation import mark_boundaries
import lime
from lime import lime_image


# In[2]:


import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model

# Load the saved model
model = load_model('skin_xai_model.h5')

# Define your class dictionary if you have one
class_dict = {0: 'Benign', 1: 'Melignant'}  # Fill this with your class names

def visualize(file_path,text_loc):
    test_image = cv2.imread( file_path)
    test_image = cv2.resize(test_image, (224,224),interpolation=cv2.INTER_NEAREST)
    test_image = np.expand_dims(test_image,axis=0)
    probs = model.predict(test_image)
    pred_class = np.argmax(probs)
    pred_class = class_dict[pred_class]

    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(test_image[0], model.predict, top_labels=5, hide_color=0, num_samples=1000)
    temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=5, hide_rest=False)
    #temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=5, hide_rest=False)
# Overlay the mask on the temp image
    overlaid_image = temp.copy()
    overlaid_image[mask == 0] = [0, 0, 0]  # Set non-masked pixels to black

# Save the overlaid image
    cv2.imwrite('static/images/output3.png', overlaid_image)


# Overlay the mask on the temp image
    overlaid_image2 = temp.copy()
    overlaid_image2[mask == 0] = temp[mask == 0]  # Keep original image where mask is 0

# Save the overlaid image
    cv2.imwrite('static/images/output4.png', overlaid_image2)
# Save the image
    #cv2.imwrite('output3.png', temp)

#import cv2
#import numpy as np

# Define the red color
    red_color = (0, 0, 255)  # BGR values for red

# Overlay the mask on the temp image
    overlaid_image3 = temp.copy()
    overlaid_image3[mask != 0] = red_color  # Set mask area to red

# Save the overlaid image
    cv2.imwrite('static/images/output5.png', overlaid_image3)

    #ax.imshow(mark_boundaries(temp, mask))
    #fig.text(text_loc, 0.9, "Predicted Class: " + pred_class , fontsize=13)
    #true_class = find_true_class(file_path)
    #if true_class is not None:
    #    fig.text(text_loc, 0.86, "Actual Class: " + true_class , fontsize=13)


# In[4]:


def find_true_class(file_path):
    true_class = None
    if 'Benign' in file_path:
        true_class = 'Benign'
    elif 'Melignant' in file_path:
        true_class = 'Melignant'
    return true_class


# In[5]:


#fig, ax = plt.subplots(figsize=(6, 6))  # Adjust the figsize as needed
#visualize('16.jpg',0.15)
#plt.show()


# In[ ]:




