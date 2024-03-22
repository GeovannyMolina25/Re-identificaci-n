from ultralytics import YOLO
import cv2
import math 
import imutils
import numpy as np
import os
import time
import torch
# start webcam
personName = 'Jhon'
dataPath = 'Data_TH' 
personPath = dataPath + '/' + personName
if not os.path.exists(personPath):
    print('Carpeta creada: ',personPath)
    os.makedirs(personPath)
cap = cv2.VideoCapture("pruebas/Jhon1_Cam1.mp4")
model = YOLO("yolov8n.pt")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
count = 0
fps = 0
frame_count = 0
start_time = time.time()
frame_skip = 5

while True:
    frame_count += 1
    success, frame = cap.read()
    if not success:
        break
    if frame_count % frame_skip != 0:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue
    frame = cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_CUBIC)
    results = model(frame,verbose=False,conf=0.40)
    # coordinates
    for r in results:
        boxes = r.boxes
        #auxFrame = frame.copy()
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values
            body = frame[y1:y2,x1:x2]
            body = cv2.resize(body, (64, 128), interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(personPath + '/body_{}.jpg'.format(count),body)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 1)
            count = count + 1
    cv2.imshow('Webcam', frame)
    k=cv2.waitKey(1)
    if k==27 or count>=1000:
        break
cap.release()
cv2.destroyAllWindows()