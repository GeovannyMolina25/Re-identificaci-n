import cv2
import os
import numpy as np
from sklearn import svm
from skimage.feature import local_binary_pattern
import joblib
from skimage.feature import hog
dataPath = 'Data_Body' 
peopleList = os.listdir(dataPath)
print('Lista de personas:', peopleList)
labels = []
features = []
label = 0
for nameDir in peopleList:
    personPath = dataPath + '/' + nameDir
    print('Leyendo las imágenes')
    for fileName in os.listdir(personPath):
        print('Textura:', nameDir + '/' + fileName)
        labels.append(label)
        imgPath = personPath + '/' + fileName
        img = cv2.imread(imgPath)
        numPoints=24
        radius=3
        eps=1e-7
        imagen_gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        lbp_feature = local_binary_pattern(imagen_gris, numPoints, radius, method='uniform').flatten()
        (hist, _) = np.histogram(lbp_feature.ravel(),
                                 bins=np.arange(0, numPoints + 3),
                                 range=(0, numPoints + 2))
        hist = hist.astype("float")
        hist /= (hist.sum() + eps)
        hog_feature = hog(imagen_gris, orientations=9,
                          pixels_per_cell=(8, 8),
                          cells_per_block=(2, 2),
                          block_norm='L2-Hys',
                          visualize=False,
                          transform_sqrt=True)
        # Combina el descriptor LBPu con el histograma de color en HSV
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # Calcular el histograma de color en HSV
        hist_hsv = cv2.calcHist([img_hsv], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
        hist_hsv = cv2.normalize(hist_hsv, hist_hsv).flatten()
        combined_features = np.concatenate((hog_feature, hist,hist_hsv))
        features.append(combined_features)
    label = label + 1
unknownPath = 'Data_Desconocido' 
#print('Leyendo imágenes de personas desconocidas')
for feature in features:
    print(len(feature))
    
for fileName in os.listdir(unknownPath):
    #print('Textura:', 'Desconocido/' + fileName)
    labels.append(-1)  # Etiqueta de desconocido (-1)
    imgPath = unknownPath + '/' + fileName
    img = cv2.imread(imgPath)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    numPoints=24
    radius=3
    eps=1e-7
    imagen_gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lbp_feature = local_binary_pattern(imagen_gris, numPoints, radius, method='uniform').flatten()
    (hist, _) = np.histogram(lbp_feature.ravel(),
                                bins=np.arange(0, numPoints + 3),
                                range=(0, numPoints + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + eps)
    hog_feature = hog(imagen_gris, orientations=9,
                        pixels_per_cell=(8, 8),
                        cells_per_block=(2, 2),
                        block_norm='L2-Hys',
                        visualize=False,
                        transform_sqrt=True)
    # Calcular el histograma de color en HSV
    hist_hsv = cv2.calcHist([img_hsv], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
    hist_hsv = cv2.normalize(hist_hsv, hist_hsv).flatten()
    combined_features = np.concatenate((hog_feature, hist,hist_hsv))
    features.append(combined_features)
# Convierte las listas en arrays numpy
features = np.array(features)
labels = np.array(labels)
# Crea un clasificador SVM
svm_classifier = svm.SVC()
# Entrena el clasificador SVM
print('Entrenando el clasificador SVM...')
svm_classifier.fit(features, labels)
# Guarda el modelo entrenado
joblib.dump(svm_classifier, 'THC.pkl')
print('Entrenamiento completo.')
