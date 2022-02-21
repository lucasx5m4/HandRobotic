import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)  #criando obejto de video e chamando a webcam

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    #print(results.multi_hand_landmarks)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:                            #extraindo somente uma das mãos da imagem
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)       #marcando os pontos de referencia e ligando-os

    cv2.imshow("Image", img)                              #abrindo a webcam
    cv2.waitKey(1)
