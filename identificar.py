from ultralytics import YOLO
import cv2
import math 
import imutils
import numpy as np
import os
import time
import joblib
import torch
from skimage.feature import hog
from skimage.feature import local_binary_pattern
from sklearn.metrics import average_precision_score, auc,confusion_matrix, roc_curve, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import warnings
from sklearn.exceptions import UndefinedMetricWarning
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
dataPath = 'Data_Body' # Nombre del dataset a listar
imagePaths = os.listdir(dataPath)
svm_model = joblib.load('THC.pkl')  # Lectura del modelo entrenado
def Color_Gabor_Hog():
    clase = 2
    print(imagePaths)
    cap = cv2.VideoCapture("pruebas/Diego2_Cam2.mp4")
    fps = 0
    frame_count = 0
    start_time = time.time()
    model = YOLO("yolov8n.pt")
    classNames = model.names
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    numPoints=24
    radius=3
    eps=1e-7
    true_labels = []
    predicted_labels = []
    predictions=[]
    y_mAP=[]
    y_score=[]
    while cap.isOpened () :
        success, frame = cap.read()
        if not success:
            break
        frame = cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_CUBIC)
        results = model(frame, conf=0.25, verbose=False)[0]
        gray_textura = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_color = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls = int(box.cls[0])
                if classNames[cls] != "person":
                    continue
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                body_texture = gray_textura[y1:y2, x1:x2]
                body_color = frame_color[y1:y2, x1:x2]
                if not body_texture.size: 
                    continue
                body_color = cv2.resize(body_color, (64, 128), interpolation=cv2.INTER_CUBIC) 
                body_texture = cv2.resize(body_texture, (64, 128), interpolation=cv2.INTER_CUBIC)
                lbp_feature = local_binary_pattern(body_texture, numPoints, radius, method='uniform')
                
                (hist, _) = np.histogram(lbp_feature.ravel(),
                                    bins=np.arange(0, numPoints + 3),
                                    range=(0, numPoints + 2))
                hist = hist.astype("float")
                hist /= (hist.sum() + eps)
                hog_feature,visualizacion = hog(body_texture, orientations=9,
                            pixels_per_cell=(8, 8),
                            cells_per_block=(2, 2),
                            block_norm='L2-Hys',
                            visualize=True,
                            transform_sqrt=True)
                
                # Calcular el histograma de color en HSV
                hist_hsv = cv2.calcHist([body_color], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
                hist_hsv = cv2.normalize(hist_hsv, hist_hsv).flatten()
            
                combined_features = np.concatenate((hog_feature, hist,hist_hsv))
                prediction = svm_model.predict([combined_features])
                confidence = svm_model.decision_function([combined_features])
                mean_confidence = np.mean(confidence)
                numero_decimal = float(prediction[0])
                numero_entero = int(prediction[0])
                max_confidence = np.max(confidence)
                predictions.append([round(max_confidence ,4)])
                if prediction[0]==clase:
                    true_labels.append(1)
                    y_mAP.append([1])
                else:
                    true_labels.append(0)
                    y_mAP.append([0])
                if prediction[0] == -1:
                    predicted_labels.append(0)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 3)
                    cv2.putText(frame, 'Desconocido', (x1, y1 - 20), 1, 0.8, (0, 0, 255), 1, cv2.LINE_AA)
                else:
                    salida_color = cv2.resize(body_color, (170, 250), interpolation=cv2.INTER_CUBIC)
                    cv2.imshow("imagne de color", salida_color)
                    salida_hog = cv2.resize(visualizacion, (170, 250), interpolation=cv2.INTER_CUBIC)
                    cv2.imshow("imagne de hog", salida_hog)
                    salida_lbp = cv2.resize(lbp_feature, (170, 250), interpolation=cv2.INTER_CUBIC)
                    cv2.imshow("imagne de lbp", salida_lbp)
                    predicted_labels.append(1)
                    cv2.rectangle(frame,  (x1, y1), (x2, y2), (0, 255, 0), 1)
                    cv2.putText(frame, "{:.2f}".format(mean_confidence-1.6) + " " + '{}'.format(imagePaths[prediction[0]]),
                                (x1, y1 - 25), 1, 1.5, (255, 0, 0), 2, cv2.LINE_AA)
                #cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 1)
                
        #frame = cv2.resize(frame, (800, 500), interpolation=cv2.INTER_CUBIC)
        cv2.imshow('Webcam', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        ret, frame = cap.read()
        if time.time() - start_time >= 1:
            fps = frame_count
            frame_count = 0
            start_time = time.time()
    cap.release()
    cv2.destroyAllWindows()
    confusion = confusion_matrix(true_labels, predicted_labels)
    print("Matriz de confusión")
    print(confusion)
    # Obtener los valores de la matriz de confusión
    vn = confusion[0, 0]  # Verdadero Negativo
    fn = confusion[1, 0]  # Falso Negativo
    fp = confusion[0, 1]  # Falso Positivo
    vp = confusion[1, 1]  # Verdadero Positivo

    # Imprimir los valores
    # Imprimir los valores

    print("Verdadero Negativo:", vn)
    print("Falso Negativo:", fn)
    print("Falso Positivo:", fp)
    print("Verdadero Positivo:", vp)
    accuracy = (vn + vp) / (vn + fn + fp + vp)
    precision = vp / (vp + fp)
    recall = vp / (vp + fn)
    f1_score = 2 * (precision * recall) / (precision + recall)
    # Cálculo de métricas
    accuracy = (vn + vp) / (vn + fn + fp + vp)
    precision = vp / (vp + fp)
    recall = vp / (vp + fn)
    f1_score = 2 * (precision * recall) / (precision + recall)
    # Imprimir métricas
    print("Accuracy:", round(accuracy,4))
    print("Precision:", round(precision,4))
    print("Recall:", round(recall,4))
    print("F1 Score:", round(f1_score,4))
    # Calcular la curva mAP
    true_labels = np.array(y_mAP)
    predictions = np.array(predictions)
    #print("label1",true_labels)
    #print("label2",predictions)
    num_classes = true_labels.shape[1]
    ap_scores = []
    for i in range(num_classes):
        ap_scores.append(average_precision_score(true_labels[:, i], predictions[:, i]))
    # Calcular el mean Average Precision (mAP)
    mAP = np.mean(ap_scores)

    print("mAP:", round(mAP,4))