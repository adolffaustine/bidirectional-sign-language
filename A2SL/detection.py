import cv2
import mediapipe as mp
import pyttsx3

# Initialize pyttsx3
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # Speed of speech



previous_word = ""

def speak(word):
    global previous_word
    if word != previous_word:
        engine.say(word)
        engine.runAndWait()
        previous_word = word


def run_camera_feed():
  # Initialize MediaPipe Hands and pyttsx3
  mp_hands = mp.solutions.hands
  hands = mp_hands.Hands()
  mp_draw = mp.solutions.drawing_utils
  cap = cv2.VideoCapture(0)
  finger_tips = [8, 12, 16, 20]
  thumb_tip = 4
       
  while True:
      ret, img = cap.read()
      if not ret:
          break

      img = cv2.flip(img, 1)
      h, w, c = img.shape
      results = hands.process(img)

      if results.multi_hand_landmarks:
          for hand_landmark in results.multi_hand_landmarks:
              lm_list = []
              for id, lm in enumerate(hand_landmark.landmark):
                  lm_list.append(lm)
              finger_fold_status = []
              for tip in finger_tips:
                  x, y = int(lm_list[tip].x * w), int(lm_list[tip].y * h)

                  if lm_list[tip].x < lm_list[tip - 2].x:
                      finger_fold_status.append(True)
                  else:
                      finger_fold_status.append(False)

              x, y = int(lm_list[8].x * w), int(lm_list[8].y * h)

              word = ""

            # Stop
              if lm_list[4].y < lm_list[2].y and lm_list[8].y < lm_list[6].y and lm_list[12].y < lm_list[10].y and \
                      lm_list[16].y < lm_list[14].y and lm_list[20].y < lm_list[18].y and lm_list[17].x < lm_list[0].x < lm_list[5].x:
                  word = "STOP"
            
            # Forward
              elif lm_list[3].x > lm_list[4].x and lm_list[8].y < lm_list[6].y and lm_list[12].y > lm_list[10].y and \
                      lm_list[16].y > lm_list[14].y and lm_list[20].y > lm_list[18].y:
                  word = "FORWARD"

            # Backward
              elif lm_list[3].x > lm_list[4].x and lm_list[3].y < lm_list[4].y and lm_list[8].y > lm_list[6].y and \
                      lm_list[12].y < lm_list[10].y and lm_list[16].y < lm_list[14].y and lm_list[20].y < lm_list[18].y:
                  word = "BACKWARD"
            
            # Left
              elif lm_list[4].y < lm_list[2].y and lm_list[8].x < lm_list[6].x and lm_list[12].x > lm_list[10].x and \
                      lm_list[16].x > lm_list[14].x and lm_list[20].x > lm_list[18].x and lm_list[5].x < lm_list[0].x:
                  word = "LEFT"

            # Right
              elif lm_list[4].y < lm_list[2].y and lm_list[8].x > lm_list[6].x and lm_list[12].x < lm_list[10].x and \
                      lm_list[16].x < lm_list[14].x and lm_list[20].x < lm_list[18].x:
                  word = "RIGHT"
            
            # Other gestures
              elif all(finger_fold_status):
                  if lm_list[thumb_tip].y < lm_list[thumb_tip - 1].y < lm_list[thumb_tip - 2].y and lm_list[0].x < lm_list[3].y:
                      word = "LIKE"
                  elif lm_list[thumb_tip].y > lm_list[thumb_tip - 1].y > lm_list[thumb_tip - 2].y and lm_list[0].x < lm_list[3].y:
                      word = "DISLIKE"

            # Numbers
              if lm_list[3].x > lm_list[4].x and lm_list[8].y < lm_list[6].y and lm_list[12].y > lm_list[10].y and \
                      lm_list[16].y > lm_list[14].y and lm_list[20].y > lm_list[18].y and lm_list[4].y < lm_list[12].y:
                  word = "ONE"
              elif lm_list[3].x > lm_list[4].x and lm_list[8].y < lm_list[6].y and lm_list[12].y < lm_list[10].y and \
                      lm_list[16].y > lm_list[14].y and lm_list[20].y > lm_list[18].y:
                  word = "TWO"
              elif lm_list[2].x < lm_list[4].x and lm_list[8].y < lm_list[6].y and lm_list[12].y < lm_list[10].y and \
                      lm_list[16].y > lm_list[14].y and lm_list[20].y > lm_list[18].y:
                  word = "THREE"
              elif lm_list[2].x > lm_list[4].x and lm_list[8].y < lm_list[6].y and lm_list[12].y < lm_list[10].y and \
                      lm_list[16].y < lm_list[14].y and lm_list[20].y < lm_list[18].y and lm_list[2].x < lm_list[8].x:
                  word = "FOUR"
              elif lm_list[2].x < lm_list[4].x and lm_list[8].y < lm_list[6].y and lm_list[12].y < lm_list[10].y and \
                      lm_list[16].y < lm_list[14].y and lm_list[20].y < lm_list[18].y and lm_list[17].x < lm_list[0].x < lm_list[5].x:
                  word = "FIVE"
              elif lm_list[2].x > lm_list[4].x and lm_list[8].y < lm_list[6].y and lm_list[12].y < lm_list[10].y and \
                      lm_list[16].y < lm_list[14].y and lm_list[20].y > lm_list[18].y and lm_list[17].x < lm_list[0].x < lm_list[5].x:
                  word = "SIX"
              elif lm_list[2].x > lm_list[4].x and lm_list[8].y < lm_list[6].y and lm_list[12].y < lm_list[10].y and \
                      lm_list[16].y > lm_list[14].y and lm_list[20].y < lm_list[18].y and lm_list[17].x < lm_list[0].x < lm_list[5].x:
                  word = "SEVEN"
              elif lm_list[2].x > lm_list[4].x and lm_list[8].y < lm_list[6].y and lm_list[12].y > lm_list[10].y and \
                      lm_list[16].y < lm_list[14].y and lm_list[20].y < lm_list[18].y and lm_list[17].x < lm_list[0].x < lm_list[5].x:
                  word = "EIGHT"
              elif lm_list[2].x > lm_list[4].x and lm_list[8].y > lm_list[6].y and lm_list[12].y < lm_list[10].y and \
                      lm_list[16].y < lm_list[14].y and lm_list[20].y < lm_list[18].y and lm_list[17].x < lm_list[0].x < lm_list[5].x:
                  word = "NINE"

            # Speak the word if it's different from the previous one
              if word:
                  cv2.putText(img, word, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                  speak(word)

              mp_draw.draw_landmarks(img, hand_landmark,
                                     mp_hands.HAND_CONNECTIONS,
                                     mp_draw.DrawingSpec((0, 0, 255), 6, 3),
                                     mp_draw.DrawingSpec((0, 255, 0), 4, 2)
                                   )

      cv2.imshow("Hand Sign Detection", img)
      if cv2.waitKey(1) & 0xFF == ord('q'):
          break

  cap.release()
  cv2.destroyAllWindows()

