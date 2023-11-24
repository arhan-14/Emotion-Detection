import numpy as np
import os
import cv2
import emotiondetector

model = emotiondetector.model

test_directory_path = '/Users/arhan/positive-samples'

test_images = []

for test_file_name in os.listdir(test_directory_path):
    test_file_path = os.path.join(test_directory_path, test_file_name)
    test_image = cv2.imread(test_file_path)
    
    if test_image is not None:
        test_images.append(test_image)
    else:
        print(f"Unable to read test image file: {test_file_path}")

test_images = np.array(test_images, dtype = object)
test_images = test_images / 255.0  #Normalize the pixel values

predictions = model.predict(test_images)
