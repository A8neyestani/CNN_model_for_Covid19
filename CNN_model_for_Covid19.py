# Import necessary libraries
import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
from glob import glob
import shutil
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import optimizers

# Set environment for Keras
os.environ['KERAS_BACKEND'] = 'theano'

# Define directories
DATA_ROOT = '/kaggle/input/covidct/'
PATH_POSITIVE_CASES = os.path.join('CT_COVID/')
PATH_NEGATIVE_CASES = os.path.join('CT_NonCOVID/')

# Gather positive and negative cases
positive_images = glob(os.path.join(PATH_POSITIVE_CASES, "*.png"))
negative_images = glob(os.path.join(PATH_NEGATIVE_CASES, "*.png")) + glob(os.path.join(PATH_NEGATIVE_CASES, "*.jpg"))

# Print dataset statistics
print(f"Total Positive Cases Covid19 images: {len(positive_images)}")
print(f"Total Negative Cases Covid19 images: {len(negative_images)}")

# Sample images to visualize
image_positive = cv2.imread(positive_images[51])
image_negative = cv2.imread(negative_images[22])

# Display sample images
plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.imshow(image_negative)
plt.subplot(1, 2, 2)
plt.imshow(image_positive)
plt.show()

# Create Train-Test directories
for subdir in ['train/', 'test/']:
    for labeldir in ['CT_COVID', 'CT_NonCOVID']:
        os.makedirs(subdir + labeldir, exist_ok=True)

# Split dataset into train and test
test_ratio = 0.1
datasets = [{'class': 'CT_COVID', 'images': positive_images},
            {'class': 'CT_NonCOVID', 'images': negative_images}]

for dataset in datasets:
    test_samples = random.sample(dataset['images'], int(test_ratio * len(dataset['images'])))
    for img in test_samples:
        shutil.copy2(img, 'test/' + dataset['class'])
    train_samples = [img for img in dataset['images'] if img not in test_samples]
    for img in train_samples:
        shutil.copy2(img, 'train/' + dataset['class'])

# Print dataset split statistics
print(f"Train set COVID: {len(os.listdir('train/CT_COVID'))}")
print(f"Train set Non-COVID: {len(os.listdir('train/CT_NonCOVID'))}")
print(f"Test set COVID: {len(os.listdir('test/CT_COVID'))}")
print(f"Test set Non-COVID: {len(os.listdir('test/CT_NonCOVID'))}")

# Constants for training
BATCH_SIZE = 32
EPOCHS = 50
IMG_HEIGHT, IMG_WIDTH = 224, 224

# Data generators
train_image_generator = ImageDataGenerator(rescale=1./255, horizontal_flip=True)
test_image_generator = ImageDataGenerator(rescale=1./255, horizontal_flip=True)

# Load pre-trained VGG16 model + higher level layers
base_model = VGG16(input_shape=[IMG_HEIGHT, IMG_WIDTH, 3], weights='imagenet', include_top=False)

# Freeze base model layers
for layer in base_model.layers:
    layer.trainable = False

# Add custom layers
x = Flatten()(base_model.output)
x = Dense(4096, activation='relu')(x)
x = Dense(4096, activation='relu')(x)
output = Dense(len(glob(DATA_ROOT + '/*/')), activation='softmax')(x)

# Define and compile the final model
model = Model(inputs=base_model.input, outputs=output)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Prepare data generators
train_data_gen = train_image_generator.flow_from_directory(directory='train', batch_size=BATCH_SIZE, target_size=(IMG_HEIGHT, IMG_WIDTH), class_mode='categorical')
test_data_gen = test_image_generator.flow_from_directory(directory='test', batch_size=BATCH_SIZE, target_size=(IMG_HEIGHT, IMG_WIDTH), class_mode='categorical')

# Train the model
history = model.fit_generator(train_data_gen, validation_data=test_data_gen, epochs=EPOCHS, steps_per_epoch=len(train_data_gen), validation_steps=len(test_data_gen))

# Plot training results
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='train accuracy')
plt.plot(history.history['val_accuracy'], label='val accuracy')
plt.legend()
plt.show()
