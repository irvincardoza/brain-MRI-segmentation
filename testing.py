
import os
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("ggplot")


import cv2
from tqdm import tqdm_notebook, tnrange
from glob import glob
from itertools import chain
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras.models import Model, load_model, save_model
from tensorflow.keras.layers import Input, Activation, BatchNormalization, Dropout, Lambda, Conv2D, Conv2DTranspose, MaxPooling2D, concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
class Config:#resize images
    IMAGE_WIDTH = 256  
    IMAGE_HEIGHT = 256 
    CHANNELS = 3  


im_width = Config.IMAGE_WIDTH  
im_height = Config.IMAGE_HEIGHT 


image_filenames_train = []
mask_files = glob('kaggle_3m/*/*_mask*')

for i in mask_files:
    image_filenames_train.append(i.replace('_mask', ''))

print(image_filenames_train[:10])
len(image_filenames_train) 

df = pd.DataFrame(data={'image_filenames_train': image_filenames_train, 'mask': mask_files })

df_train, df_test = train_test_split(df, test_size=0.1)

# Further split this val and train
df_train, df_val = train_test_split(df_train, test_size=0.2)

print(df_train.shape)
print(df_test.shape)
print(df_val.shape)
#helper functions
def plot_from_img_path(rows, columns, list_img_path, list_mask_path):
    fig = plt.figure(figsize=(12, 12))
    for i in range(1, rows * columns + 1):
        fig.add_subplot(rows, columns, i)
        img_path = list_img_path[i]
        mask_path = list_mask_path[i]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path)
        plt.imshow(image)
        plt.imshow(mask, alpha=0.4)
    plt.show()

def dice_coefficients(y_true, y_pred, smooth=100):
    y_true_flatten = tf.keras.backend.flatten(y_true)
    y_pred_flatten = tf.keras.backend.flatten(y_pred)

    intersection = tf.keras.backend.sum(y_true_flatten * y_pred_flatten)
    union = tf.keras.backend.sum(y_true_flatten) + tf.keras.backend.sum(y_pred_flatten)
    return (2 * intersection + smooth) / (union + smooth)
def dice_coefficients_loss(y_true, y_pred, smooth=100):
    return -dice_coefficients(y_true, y_pred, smooth)

def iou(y_true, y_pred, smooth=100):
    intersection = tf.keras.backend.sum(y_true * y_pred)
    sum = tf.keras.backend.sum(y_true + y_pred)
    iou = (intersection + smooth) / (sum - intersection + smooth)
    return iou

# print(visualize_image_mask(3, 3 , image_filenames_train, mask_files ))
model = load_model('unet.hdf5', custom_objects={'dice_coefficients_loss': dice_coefficients_loss, 'iou': iou, 'dice_coefficients': dice_coefficients  } )
# for i in range(20):
    
#     index = np.random.randint(1, len(df_test.index))
#     img = cv2.imread(df_test["image_filenames_train"].iloc[index])
#     img = cv2.resize(img, (im_height, im_width))
#     img = img/255
#     img = img[np.newaxis, :, :, :]
    
#     predicted_img = model.predict(img)
    
#     plt.figure(figsize=(12, 12))
#     plt.subplot(1, 3, 1)
#     plt.imshow(np.squeeze(img))
#     plt.title("Original Image")
    
#     plt.subplot(1, 3, 2)
#     mask_path = df_test['mask'].iloc[index]
#     mask_image = cv2.imread(mask_path)
#     plt.imshow(np.squeeze(mask_image))
#     plt.title("Original mask")
    
#     plt.subplot(1, 3, 3)
#     plt.imshow(np.squeeze(predicted_img) > 0.5)
#     plt.title("Prediction")
#     plt.show()

img = cv2.imread('kaggle_3m/TCGA_DU_7019_19940908/TCGA_DU_7019_19940908_21.tif')

# Resize the image to match the model input size
img = cv2.resize(img, (im_height, im_width))

# Normalize the image (convert pixel values to [0, 1])
img = img / 255.0

# Add batch dimension (required for model input)
img_batch = np.expand_dims(img, axis=0)

# Predict using the model
predicted_img = model.predict(img_batch)

# Visualization
plt.figure(figsize=(12, 12))

# Original Image
plt.subplot(1, 3, 1)
plt.imshow(np.squeeze(img))  # Remove batch dimension for visualization
plt.title("Original Image")

# Predicted Mask
plt.subplot(1, 3, 3)
plt.imshow(np.squeeze(predicted_img) > 0.5, cmap="gray")  # Threshold and remove batch dimension
plt.title("Prediction")

plt.show()
