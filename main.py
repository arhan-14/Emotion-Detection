import cv2
import numpy as np
import tensorflow as tf

#Load the cascade classifiers
face_cascade = cv2.CascadeClassifier('/Users/arhan/Downloads/haarcascade_frontalface_default.xml')

#Load the emotion detector model
model = tf.keras.models.load_model('/Users/arhan/Emotion Detection/emotion_detection_model_mobilenet.h5')
class_names = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

#Read the video stream
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print('--(!)Error opening video capture')
    exit(0)

while True:
    ret, frame = cap.read()
    if frame is None:
        print('--(!) No captured frame -- Break!')
        break

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.equalizeHist(frame_gray)

    #Detect faces
    faces = face_cascade.detectMultiScale(frame_gray)
    for (x, y, w, h) in faces:
        center = (x + w // 2, y + h // 2)
        frame = cv2.ellipse(frame, center, (w // 2, h // 2), 0, 0, 360, (255, 0, 255), 4)

        faceROI = frame_gray[y:y + h, x:x + w]

        #Preprocess the face image for emotion detection
        faceROI_resized = cv2.resize(faceROI, (180, 180))
        faceROI_resized = cv2.cvtColor(faceROI_resized, cv2.COLOR_GRAY2RGB)
        faceROI_resized = np.expand_dims(faceROI_resized, axis=0)
        faceROI_resized = faceROI_resized / 255.0

        #Convert the face image to a TensorFlow tensor
        face_tensor = tf.convert_to_tensor(faceROI_resized, dtype=tf.float32)

        #Perform emotion prediction using the emotion detector model
        emotion_prediction = model.predict(face_tensor)
        emotion_label = class_names[np.argmax(emotion_prediction)]

        #Draw the emotion label on the frame
        cv2.putText(frame, emotion_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    cv2.imshow('Capture - Face detection', frame)

    if cv2.waitKey(10) == 27:
        break

cap.release()
cv2.destroyAllWindows()
