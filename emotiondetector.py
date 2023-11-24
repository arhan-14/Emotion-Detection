import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import MobileNet
from keras.layers import GlobalAveragePooling2D, Dense
from keras.models import Model
import os

#Set random seeds for reproducibility
seed_value = 1234
tf.random.set_seed(seed_value)

#Directory path containing the training data
directory_path = '/Users/arhan/Downloads/emotion-detector-data/train'

#Image parameters
img_height = 180
img_width = 180
batch_size = 32
num_classes = 7

#Data augmentation
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True, validation_split=0.2)
train_ds = train_datagen.flow_from_directory(directory_path, target_size=(img_height, img_width), batch_size=batch_size, class_mode='categorical', subset='training', seed=seed_value)
val_ds = train_datagen.flow_from_directory(directory_path, target_size=(img_height, img_width), batch_size=batch_size, class_mode='categorical', subset='validation', seed=seed_value)

#Load MobileNet model with pre-trained ImageNet weights but exclude the top layers
base_model = MobileNet(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))

#Add custom top layers for emotion detection on top of the base model
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
predictions = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

#Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#Train the model
epochs = 10
history = model.fit(train_ds, validation_data=val_ds, epochs=epochs)

#Save the trained model
model.save('/Users/arhan/Emotion Detection/emotion_detection_model_mobilenet.h5')
