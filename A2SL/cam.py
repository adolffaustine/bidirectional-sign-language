# views.py

import cv2
import numpy as np
import pyttsx3
from django.shortcuts import render
from keras.models import load_model

# Load the sign language recognition model
model = load_model('./ml_model/BidirectionsignLanguage.h5')  # Replace 'action.h5' with the path to your model file

# Function to preprocess the frame
def preprocess_frame(frame):
    # Preprocess the frame (resize, normalize, etc.)
    resized_frame = cv2.resize(frame, (224, 224))  # Resize to match the input size expected by the model
    normalized_frame = resized_frame / 255.0  # Normalize pixel values to range [0, 1]
    return normalized_frame

# Function to recognize sign language from a frame
def recognize_sign(frame):
    preprocessed_frame = preprocess_frame(frame)
    preprocessed_frame = np.expand_dims(preprocessed_frame, axis=0)  # Add batch dimension
    predictions = model.predict(preprocessed_frame)
    predicted_class = np.argmax(predictions)  # Get the index of the class with highest probability
    return str(predicted_class)  # Convert the index to the recognized sign (e.g., '0' to '0')

# Function to speak the recognized sign
def speak(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

# Django view for camera input and sign language recognition
def camera_view(request):
    # Initialize the camera
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()  # Capture a frame from the camera

        # Perform sign language recognition
        sign = recognize_sign(frame)

        # Speak the recognized sign
        speak(sign)

        # Render the frame with recognized sign in the template
        return render(request, 'camera.html', {'frame': frame, 'sign': sign})

    cap.release()  # Release the camera when done
