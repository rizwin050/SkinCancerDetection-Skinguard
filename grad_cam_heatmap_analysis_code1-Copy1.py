#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os, glob
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import Callback,EarlyStopping
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from sklearn.metrics import  classification_report


# In[2]:


file_path = 'dataset'


# In[3]:


name_class = os.listdir(file_path)
name_class


# In[4]:


filepaths = list(glob.glob(file_path+'/**/*.*'))


# In[5]:


filepaths


# In[6]:


labels = list(map(lambda x: os.path.split(os.path.split(x)[0])[1], filepaths))
labels


# In[7]:



filepath = pd.Series(filepaths, name='Filepath').astype(str)
labels = pd.Series(labels, name='Label')
data = pd.concat([filepath, labels], axis=1)
data = data.sample(frac=1).reset_index(drop=True)
data.head(5)


# In[8]:


#balancing


# In[9]:


counts = data.Label.value_counts()
sns.barplot(x=counts.index, y=counts)
plt.xlabel('Type')
plt.xticks(rotation=90);


# In[10]:


#balance by deleting extra images (Augmentor API)
#augmentation 
#GAN


# In[ ]:





# In[ ]:





# In[11]:


train, test = train_test_split(data, test_size=0.25, random_state=42)


# In[12]:


fig, axes = plt.subplots(nrows=5, ncols=3, figsize=(10,8), subplot_kw={'xticks':[],'yticks':[]})
for i, ax in enumerate(axes.flat):
    ax.imshow(plt.imread(data.Filepath[i]))
    ax.set_title(data.Label[i])
plt.tight_layout()
plt.show()


# In[13]:


train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)


# In[14]:


train_gen = train_datagen.flow_from_dataframe(
    dataframe=train,
    x_col='Filepath',
    y_col='Label',
    target_size=(100,100),
    class_mode='categorical',
    batch_size=32,
    shuffle=True,
    seed=42
)
valid_gen = train_datagen.flow_from_dataframe(
    dataframe=test,
    x_col='Filepath',
    y_col='Label',
    target_size=(100,100),
    class_mode='categorical',
    batch_size=32,
    shuffle=False,
    seed=42
)
test_gen = test_datagen.flow_from_dataframe(
    dataframe=test,
    x_col='Filepath',
    y_col='Label',
    target_size=(100,100),
    class_mode='categorical',
    batch_size=32,
    shuffle=False
)


# In[24]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model

# Define the base pre-trained model
pretrained_model = ResNet50(
    input_shape=(100,100, 3),
    include_top=False,
    weights='imagenet',
    pooling='avg'
)

pretrained_model.trainable = False

# Define the additional layers for classification
inputs = Input(shape=(100, 100, 3))
x = pretrained_model(inputs)
x = Dense(128, activation='relu')(x)
x = Dense(128, activation='relu')(x)
outputs = Dense(2, activation='softmax')(x)

# Combine the base model with additional layers
model = Model(inputs=inputs, outputs=outputs)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    train_gen,
    steps_per_epoch=train_gen.samples // train_gen.batch_size,
    epochs=10,
    validation_data=valid_gen,
    validation_steps=valid_gen.samples // valid_gen.batch_size
)


# In[30]:


class ExplainableAttention(Layer):
    def __init__(self, **kwargs):
        super(ExplainableAttention, self).__init__(**kwargs)

    def call(self, inputs):
        # Construct query and value inputs for attention
        query = inputs
        value = inputs

        attention_weights = Attention(use_scale=True)([query, value])
        return attention_weights


# In[31]:


attention_weights = ExplainableAttention()(x)


# In[ ]:





# In[ ]:





# In[35]:


import numpy as np
import cv2
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing import image

def get_gradcam(image_path, model, layer_name):
    img = image.load_img(image_path, target_size=(100, 100))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = tf.keras.applications.resnet50.preprocess_input(x)

    preds = model.predict(x)
    predicted_class = np.argmax(preds[0])
    
    with tf.GradientTape() as tape:
        last_conv_layer = model.get_layer(layer_name)
        iterate = tf.keras.models.Model([model.inputs], [model.output, last_conv_layer.output])
        model_out, last_conv_layer = iterate(x)
        class_out = model_out[:, predicted_class]
        grads = tape.gradient(class_out, last_conv_layer)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
    heatmap = tf.reduce_mean(tf.multiply(pooled_grads, last_conv_layer), axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)

    img = cv2.imread(image_path)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(img, 0.5, heatmap, 0.5, 0)

    return superimposed_img

# Define the layer name from which to get the activation
layer_name = 'conv5_block3_out'

# Get the Grad-CAM visualization for an image
image_path = '16.jpg' # Provide the path to your image
gradcam_result = get_gradcam(image_path, model, layer_name)

# Plot the original image and Grad-CAM visualization
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(gradcam_result, cv2.COLOR_BGR2RGB))
plt.title('Grad-CAM')
plt.axis('off')

plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[15]:


pretrained_model = ResNet50(
    input_shape=(100,100, 3),
    include_top=False,
    weights='imagenet',
    pooling='avg'
)

pretrained_model.trainable = False


# In[16]:


inputs = pretrained_model.input

x = Dense(128, activation='relu')(pretrained_model.output)
x = Dense(128, activation='relu')(x)

outputs = Dense(2, activation='softmax')(x)

model = Model(inputs=inputs, outputs=outputs)


# In[17]:


model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)


# In[18]:


my_callbacks  = [EarlyStopping(monitor='val_accuracy',
                              min_delta=0,
                              patience=2,
                              mode='auto')]


# In[19]:


history = model.fit(
    train_gen,
    validation_data=valid_gen,
    epochs=100
)


# In[22]:


model.save("model_resnet50.h5")  


# In[23]:


pd.DataFrame(history.history)[['accuracy','val_accuracy']].plot()
plt.title("Accuracy")
plt.show()

pd.DataFrame(history.history)[['loss','val_loss']].plot()
plt.title("Loss")
plt.show()


# In[24]:


results = model.evaluate(test_gen, verbose=0)

print("    Test Loss: {:.5f}".format(results[0]))
print("Test Accuracy: {:.2f}%".format(results[1] * 100))


# In[25]:


# Predict the label of the test_gen
pred = model.predict(test_gen)
pred = np.argmax(pred,axis=1)

# Map the label
labels = (train_gen.class_indices)
labels = dict((v,k) for k,v in labels.items())
pred = [labels[k] for k in pred]


# In[26]:


y_test = list(test.Label)
print(classification_report(y_test, pred))


# In[27]:


fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(12, 8),
                        subplot_kw={'xticks': [], 'yticks': []})

for i, ax in enumerate(axes.flat):
    ax.imshow(plt.imread(test.Filepath.iloc[i]))
    ax.set_title(f"True: {test.Label.iloc[i]}\nPredicted: {pred[i]}")
plt.tight_layout()
plt.show()


# In[28]:


import cv2
import numpy as np
from matplotlib.pyplot import imread
from matplotlib.pyplot import imshow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.imagenet_utils import decode_predictions
from tensorflow.keras.applications.imagenet_utils import preprocess_input

img_path = 'images/real/1.jpg'

#img = image.load_img(img_path, target_size=(224, 224))
#x = img.img_to_array(img)

img = cv2.imread(img_path)
img = cv2.resize(img, (100, 100))

x = np.expand_dims(img, axis=0)
x = preprocess_input(x)
result = model.predict(x)
print((result*100).astype('int'))

#print('Input image shape:', x.shape)

#my_image = imread(img_path)
#imshow(my_image)


# In[1]:


from tensorflow.keras.models import load_model
loaded_model_imageNet = load_model("skin_xai_model.h5")
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions


# In[ ]:





# In[2]:


import cv2
import numpy as np
from matplotlib.pyplot import imread
from matplotlib.pyplot import imshow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.imagenet_utils import decode_predictions
from tensorflow.keras.applications.imagenet_utils import preprocess_input

img_path = '16.jpg'

#img = image.load_img(img_path, target_size=(224, 224))
#x = img.img_to_array(img)

img = cv2.imread(img_path)
img = cv2.resize(img, (224,224))

x = np.expand_dims(img, axis=0)
x = preprocess_input(x)
result = loaded_model_imageNet.predict(x)
print((result*100).astype('int'))

#print('Input image shape:', x.shape)

#my_image = imread(img_path)
#imshow(my_image)


# In[3]:


p=list((result*100).astype('int'))
pp=list(p[0])
print(pp)


# In[4]:


print("Largest element is:", max(pp))


# In[5]:


index = pp.index(max(pp))


# In[ ]:





# In[6]:


name_class=['benign', 'Melignant']


# In[7]:


name_class[index]


# In[8]:


import matplotlib.pyplot  as plt


# In[9]:


plt.imread(img_path)


# In[10]:


import matplotlib.pyplot as plt
import matplotlib.image as mpimg
img = mpimg.imread(img_path)
imgplot = plt.imshow(img)
plt.title(name_class[index])
plt.show()


# In[21]:


import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
img_path = '7.jpg'

def get_grad_cam(model, img_path, layer_name):
    # Load and preprocess the image
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)  # Convert image to numpy array
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    
    # Get the image prediction
    preds = model.predict(x)
    predicted_class = np.argmax(preds[0])
    
    # Get the output of the specified layer
    grad_model = Model(
        [model.inputs], 
        [model.get_layer(layer_name).output, model.output]
    )
    
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(x)
        loss = predictions[:, predicted_class]

    # Get the gradients
    grads = tape.gradient(loss, conv_outputs)[0]

    # Pool the gradients across the channels
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Get the activation maps
    conv_output = conv_outputs[0]
    heatmap = tf.reduce_mean(tf.multiply(conv_output, pooled_grads), axis=-1)

    # Normalize the heatmap
    heatmap = np.maximum(heatmap, 0) / np.max(heatmap)

    # Resize heatmap to the original image size
    img_array = image.img_to_array(img)
    heatmap = cv2.resize(heatmap, (img_array.shape[1], img_array.shape[0]))

    # Convert heatmap to RGB
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Superimpose the heatmap on the original image
    superimposed_img = cv2.addWeighted(img_array.astype('uint8'), 0.5, heatmap, 0.5, 0)

    return superimposed_img, predicted_class, preds[0][predicted_class]


layer_name = "conv2d_3"
superimposed_img, predicted_class, confidence = get_grad_cam(loaded_model_imageNet, img_path, layer_name)

# Load and display the original image
original_img = cv2.imread(img_path)
original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
plt.imshow(original_img)
plt.axis('off')
plt.title('Original Image')
plt.show()

# Display the Grad-CAM visualization
plt.imshow(superimposed_img)
plt.axis('off')
plt.title(f'Grad-CAM\nPredicted Class: {predicted_class}, Confidence: {confidence:.2f}')
plt.show()


# In[23]:


from vis.utils import utils
from vis.visualization import visualize_cam

# Define the function to generate Grad-CAM visualization
def generate_grad_cam(model, img_path, layer_name):
    # Load the image
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    # Get the image prediction
    preds = model.predict(x)
    predicted_class = np.argmax(preds[0])

    # Retrieve the desired layer output
    layer_output = utils.find_layer_output(model, layer_name)

    # Generate Grad-CAM heatmap
    grad_cam = visualize_cam(model, layer_idx=utils.find_layer_idx(model, layer_name),
                              filter_indices=predicted_class, seed_input=x[0], backprop_modifier='guided')

    # Overlay the heatmap onto the original image
    heatmap = cv2.applyColorMap(np.uint8(255 * grad_cam), cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(img_array.astype('uint8'), 0.5, heatmap, 0.5, 0)

    return superimposed_img, predicted_class, preds[0][predicted_class]

# Define the layer name
layer_name = "conv2d_3"

# Generate Grad-CAM visualization
superimposed_img, predicted_class, confidence = generate_grad_cam(loaded_model_imageNet, img_path, layer_name)

# Display the original image
original_img = cv2.imread(img_path)
original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
plt.imshow(original_img)
plt.axis('off')
plt.title('Original Image')
plt.show()

# Display the Grad-CAM visualization
plt.imshow(superimposed_img)
plt.axis('off')
plt.title(f'Grad-CAM\nPredicted Class: {predicted_class}, Confidence: {confidence:.2f}')
plt.show()


# In[24]:


import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image

def get_grad_cam(model, img_path, layer_name):
    # Load and preprocess the image
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)  # Convert image to numpy array
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    
    # Get the image prediction
    preds = model.predict(x)
    predicted_class = np.argmax(preds[0])
    
    # Get the output of the specified layer
    grad_model = Model(
        [model.inputs], 
        [model.get_layer(layer_name).output, model.output]
    )
    
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(x)
        loss = predictions[:, predicted_class]

    # Get the gradients
    grads = tape.gradient(loss, conv_outputs)[0]

    # Pool the gradients across the channels
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Get the activation maps
    conv_output = conv_outputs[0]
    heatmap = tf.reduce_mean(tf.multiply(conv_output, pooled_grads), axis=-1)

    # Normalize the heatmap
    heatmap = np.maximum(heatmap, 0) / np.max(heatmap)

    # Resize heatmap to the original image size
    img_array = image.img_to_array(img)
    heatmap = cv2.resize(heatmap, (img_array.shape[1], img_array.shape[0]))

    # Convert heatmap to RGB
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Superimpose the heatmap on the original image
    superimposed_img = cv2.addWeighted(img_array.astype('uint8'), 0.5, heatmap, 0.5, 0)

    return superimposed_img, predicted_class, preds[0][predicted_class]


# Define the layer name
layer_name = "conv2d_3"

# Call the function to generate Grad-CAM
superimposed_img, predicted_class, confidence = get_grad_cam(loaded_model_imageNet, img_path, layer_name)

# Load and display the original image
original_img = cv2.imread(img_path)
original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
plt.imshow(original_img)
plt.axis('off')
plt.title('Original Image')
plt.show()

# Display the Grad-CAM visualization
plt.imshow(superimposed_img)
plt.axis('off')
plt.title(f'Grad-CAM\nPredicted Class: {predicted_class}, Confidence: {confidence:.2f}')
plt.show()


# In[17]:


# Print the names of all layers in the model
for layer in loaded_model_imageNet.layers:
    print(layer.name)


# In[32]:


import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # Create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = Model(
        model.inputs, [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Compute the gradient of the top predicted class for the input image
    # for the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # Calculate the gradients
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Calculate the heatmap
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purposes, normalize the heatmap
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)

    return heatmap.numpy()


def save_and_display_gradcam(model, img_path, layer_name, cam_path="save_cam_image.jpg", alpha=0.4):
    # Load and preprocess the image
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    
    # Get the Grad-CAM heatmap
    heatmap = make_gradcam_heatmap(img_array, model, layer_name)
    
    # Resize heatmap to the original image size
    heatmap = cv2.resize(heatmap, (img_array.shape[2], img_array.shape[1]))

    # Convert heatmap to RGB
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Superimpose the heatmap on the original image
    superimposed_img = cv2.addWeighted(img_array[0].astype('uint8'), alpha, heatmap, 1-alpha, 0)

    # Save the superimposed image
    cv2.imwrite(cam_path, superimposed_img)

    # Display the Grad-CAM visualization
    plt.imshow(cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title(f'Grad-CAM\nLayer: {layer_name}')
    plt.show()


# Define the layer name
layer_name = "conv2d_3"

# Specify the image path
img_path = '7.jpg'

# Call the function to generate Grad-CAM and display
save_and_display_gradcam(loaded_model_imageNet, img_path, layer_name)


# In[ ]:




