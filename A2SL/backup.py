from django.http import HttpResponse
from django.shortcuts import render, redirect
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from django.contrib.auth import login, logout
from django.contrib.auth.decorators import login_required
import threading
import cv2
import logging
import mediapipe as mp
import numpy as np
import pyttsx3
import time
import torch
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from django.contrib.staticfiles import finders
from django.views.decorators.csrf import csrf_exempt
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.corpus import wordnet as wn
wn.ensure_loaded()
from django.core.management import call_command
from django.core.wsgi import get_wsgi_application
from django.conf import settings
from transformers import pipeline


# Function to run Django server

def run_server():
    # Configure Django settings
    from django.conf import settings
    settings.configure()
    # Get the WSGI application
    django_application = get_wsgi_application()
    # Start the server
    call_command('runserver')

# Start Django server in a separate thread
django_thread = threading.Thread(target=run_server)
django_thread.daemon = True
django_thread.start()

# Load model directly
from transformers import AutoImageProcessor, AutoModelForImageClassification
import requests


def run_camera_feed():
    processor = AutoImageProcessor.from_pretrained("RavenOnur/Sign-Language", do_rescale=False)
    model = AutoModelForImageClassification.from_pretrained("RavenOnur/Sign-Language")
    labels_dict = { 
        0: 'A', 
        1: 'B', 
        2: 'C', 
        3: 'where', 
        4: 'E',
        5: 'F',
        6: 'what',
        7: 'H',
        8: 'I',
        9: 'what',
        10: 'Big up',
        11: 'L',
        12: 'M',
        13: 'N',
        14: 'O',
        15: 'P',
        16: 'Q',
        17: 'R',
        18: 'S',
        19: 'we are', 
        20: 'U', 
        21: 'hello',
        22: 'W',
        23: 'X',
        24: 'Y',
        25: 'Z'
    }
    color_dict = (0, 255, 0)
    img_size = 224  # Update to match the input size expected by the Hugging Face model
    source = cv2.VideoCapture(0)
    string = " "

    # Initialize TTS engine
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)  # Set the speed of speech

    while True:
        ret, img = source.read()
        if not ret:
            break
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.rectangle(img, (24, 24), (250, 250), color_dict, 2)
        crop_img = gray[24:250, 24:250]
        
        resized = cv2.resize(crop_img, (img_size, img_size))
        normalized = resized / 255.0
        reshaped = np.stack([normalized] * 3, axis=-1)  # Stack to create a 3-channel image

        inputs = processor(images=reshaped, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=1)
        max_prob, label = torch.max(probabilities, dim=1)

        label = label.item()
        max_prob = max_prob.item()
        confidence_threshold = 0.5  # Adjust the threshold as needed for testing

        print(f"Predicted Label: {labels_dict[label]}, Confidence: {max_prob}")

        if max_prob >= confidence_threshold:
            predicted_label = labels_dict[label]
            if label == 0:
                string += " "
            else:
                string += predicted_label

            # Speak the predicted letter
            engine.say(predicted_label)
            engine.runAndWait()

            cv2.putText(img, predicted_label, (24, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        cv2.putText(img, string, (275, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
        cv2.imshow("Gray", resized)
        cv2.imshow('LIVE', img)
        key = cv2.waitKey(1)

        if key == 27:  # press Esc to exit
            break

    print(string)
    cv2.destroyAllWindows()
    source.release()
# Call the function to start the camera feed and sign language prediction



# Start camera feed in a separate thread
camera_thread = threading.Thread(target=run_camera_feed)
camera_thread.daemon = True
camera_thread.start()

# Django views
def home_view(request):
    return render(request, 'home.html')

def about_view(request):
    return render(request, 'about.html')

def contact_view(request):
    return render(request, 'contact.html')




@csrf_exempt
def animation_view(request):
    global camera_thread, microphone_thread

    if request.method == 'POST':
        text = request.POST.get('sen', '').strip()


        # Tokenizing the sentence
        text = text.lower()
        words = word_tokenize(text)
        tagged = nltk.pos_tag(words)

        tense = {
            "future": len([word for word in tagged if word[1] == "MD"]),
            "present": len([word for word in tagged if word[1] in ["VBP", "VBZ", "VBG"]]),
            "past": len([word for word in tagged if word[1] in ["VBD", "VBN"]]),
            "present_continuous": len([word for word in tagged if word[1] == "VBG"]),
        }

        stop_words = set([
            "mightn't", 're', 'wasn', 'wouldn', 'be', 'has', 'that', 'does', 'shouldn',
            'do', "you've", 'off', 'for', "didn't", 'm', 'ain', 'haven', "weren't", 'are',
            "she's", "wasn't", 'its', "haven't", "wouldn't", 'don', 'weren', 's', "you'd",
            "don't", 'doesn', "hadn't", 'is', 'was', "that'll", "should've", 'a', 'then',
            'the', 'mustn', 'i', 'nor', 'as', "it's", "needn't", 'd', 'am', 'have', 'hasn',
            'o', "aren't", "you'll", "couldn't", "you're", "mustn't", 'didn', "doesn't",
            'll', 'an', 'hadn', 'whom', 'y', "hasn't", 'itself', 'couldn', 'needn',
            "shan't", 'isn', 'been', 'such', 'shan', "shouldn't", 'aren', 'being', 'were',
            'did', 'ma', 't', 'having', 'mightn', 've', "isn't", "won't"
        ])

        lr = WordNetLemmatizer()
        filtered_text = []
        for w, p in zip(words, tagged):
            if w not in stop_words:
                if p[1] in ['VBG', 'VBD', 'VBZ', 'VBN', 'NN']:
                    filtered_text.append(lr.lemmatize(w, pos='v'))
                elif p[1] in ['JJ', 'JJR', 'JJS', 'RBR', 'RBS']:
                    filtered_text.append(lr.lemmatize(w, pos='a'))
                else:
                    filtered_text.append(lr.lemmatize(w))

        words = ['Me' if w == 'I' else w for w in filtered_text]

        probable_tense = max(tense, key=tense.get)
        if probable_tense == "past" and tense["past"] >= 1:
            words.insert(0, "Before")
        elif probable_tense == "future" and tense["future"] >= 1:
            if "Will" not in words:
                words.insert(0, "Will")
        elif probable_tense == "present" and tense["present_continuous"] >= 1:
            words.insert(0, "Now")

        filtered_text = []
        for w in words:
            path = w + ".mp4"
            f = finders.find(path)
            if not f:
                filtered_text.extend(w)
            else:
                filtered_text.append(w)
        words = filtered_text

        return render(request, 'animation.html', {'words': words, 'text': text})
    else:
        return render(request, 'animation.html')

def signup_view(request):
    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            return redirect('animation')
    else:
        form = UserCreationForm()
    return render(request, 'signup.html', {'form': form})

def login_view(request):
    if request.method == 'POST':
        form = AuthenticationForm(data=request.POST)
        if form.is_valid():
            user = form.get_user()
            login(request, user)
            return redirect(request.POST.get('next', 'animation'))
    else:
        form = AuthenticationForm()
    return render(request, 'login.html', {'form': form})

def logout_view(request):
    logout(request)
    return redirect("home")