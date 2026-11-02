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


import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model

def find_true_class(file_path):
    true_class = None
    if 'Benign' in file_path:
        true_class = 'Benign'
    elif 'Melignant' in file_path:
        true_class = 'Melignant'
    return true_class

def visualize(file_path):
    test_image = cv2.imread(file_path)
    test_image = cv2.resize(test_image, (224, 224), interpolation=cv2.INTER_NEAREST)
    test_image = np.expand_dims(test_image, axis=0)
    probs = model.predict(test_image)
    pred_class = np.argmax(probs)
    pred_class = class_dict[pred_class]

    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(test_image[0], model.predict, top_labels=5, hide_color=0, num_samples=1000)
    temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=5, hide_rest=False)

    # Save the result directly without displaying
    fig = plt.figure(figsize=(temp.shape[1] / 100, temp.shape[0] / 100), dpi=100)
    #plt.imshow(mark_boundaries(temp, mask))
    plt.axis('off')  # Turn off axis
    plt.savefig("output2.png", bbox_inches='tight', pad_inches=0)  # Save without extra whitespace
    plt.close(fig)  # Close the figure to release memory
    print("Result saved as output2.png")
# Load the saved model
model = load_model('skin_xai_model.h5')

# Define your class dictionary if you have one
class_dict = {0: 'Benign', 1: 'Melignant'}  # Fill this with your class names
def detector_model(imag_path):
			# Load the test image
			file_path =imag_path
			test_image = cv2.imread(file_path)
			test_image = cv2.resize(test_image, (224, 224), interpolation=cv2.INTER_NEAREST)
			#plt.imshow(cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB))  # Convert colorspace for matplotlib display
			#plt.show()

			# Preprocess the test image
			test_image = np.expand_dims(test_image, axis=0)

			# Make prediction
			probs = model.predict(test_image)
			pred_class = np.argmax(probs)
			pred_class = class_dict[pred_class]  # Convert predicted class index to class name

			print('Prediction: ', pred_class)
			#fig, ax = plt.subplots(figsize=(6, 6))
			#visualize(imag_path)
			#visualize(imag_path,0.15)

			return pred_class



