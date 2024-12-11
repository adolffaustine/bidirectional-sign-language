import cv2
import numpy as np
import pyttsx3
import time
from keras.models import load_model

 
def run_camera_feed():
    model = load_model("./ml_model/model.h5")
    labels_dict = { 
        0: 'A', 
        1: 'B', 
        2: 'C', 
        3: 'D', 
        4: 'E',
        5: 'F',
        6: 'G',
        7: 'H',
        8: 'I',
        9: 'J',
        10: 'K',
        11: 'L',
        12: 'M',
        13: 'N',
        14: 'O',
        15: 'P',
        16: 'Q',
        17: 'R',
        18: 'S',
        19: 'T', 
        20: 'U', 
        21: 'V',
        22: 'W',
        23: 'X',
        24: 'Y',
        25: 'Z'
    }
    color_dict = (0, 255, 0)
    img_size = 28
    minValue = 70
    source = cv2.VideoCapture(0)
    current_string = ""
    spoken_string = ""
    prev = ""
    prev_time = time.time()
    idle_time = 0

    # Initialize TTS engine
    engine = pyttsx3.init()

    while True:
        ret, img = source.read()

        cv2.namedWindow('LIVE', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('LIVE', 800, 600)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        x_start, y_start, width, height = 100, 100, 400, 400
        cv2.rectangle(img, (x_start, y_start), (x_start + width, y_start + height), color_dict, 2)

        crop_img = gray[y_start:y_start + height, x_start:x_start + width]

        blur = cv2.GaussianBlur(crop_img, (5, 5), 2)
        th3 = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        ret, res = cv2.threshold(th3, minValue, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        resized = cv2.resize(res, (img_size, img_size))
        normalized = resized / 255.0
        reshaped = np.reshape(normalized, (1, img_size, img_size, 1))

        result = model.predict(reshaped)
        prediction_accuracy = np.max(result)
        label = np.argmax(result, axis=1)[0]

        if prediction_accuracy < 0.90:
            current_sign = "No sign recognized"
        else:
            current_sign = labels_dict[label]

        if current_sign != "No sign recognized":
            if current_sign != prev:
                prev_time = time.time()
                prev = current_sign
                idle_time = 0
            else:
                idle_time = time.time() - prev_time
                if idle_time >= 1:
                    if current_sign == " ":
                        current_string += " "
                    else:
                        current_string += current_sign
                    prev_time = time.time()

        # If idle for more than 1.5 seconds and current_string is not empty, speak the current_string
        if idle_time >= 1.5:
            if current_string:
                engine.say(current_string)
                engine.runAndWait()
                spoken_string += current_string + " "
                current_string = ""
            idle_time = 0

        cv2.putText(img, current_sign, (x_start, y_start - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(img, spoken_string, (x_start, y_start + height + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)

        cv2.imshow("Gray", res)
        cv2.imshow('LIVE', img)

        key = cv2.waitKey(1)
        if key == 27:  # press Esc to exit
            break

    if current_string:
        engine.say(current_string)
        engine.runAndWait()

    cv2.destroyAllWindows()
    source.release()

