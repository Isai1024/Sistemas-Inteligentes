import cv2
import os
import numpy as np

#Url donde se encuetra las imagenes de los usuarios
dataPath = '/home/alan/Documentos/Sistemas-Inteligentes/Face detection/Datos'

listaPersonas = os.listdir(dataPath)
print('Personas: ', listaPersonas)

labels = []
faceData = []
label = 0

for nameDir in listaPersonas:
    personaPath = dataPath + '/' + nameDir
    print("Leyendo imagenes")

    for fileName in os.listdir(personaPath):
        print('Rostro: ', nameDir + '/' + fileName)
        labels.append(label)
        faceData.append(cv2.imread(personaPath + '/' + fileName, 0))
        image = cv2.imread(personaPath + '/' + fileName, 0)
    
    label = label + 1

cv2.destroyAllWindows

face_recognizer = cv2.face.LBPHFaceRecognizer_create()

print('Entrenando')
face_recognizer.train(faceData, np.array(labels))

face_recognizer.write('Modelo_Rostros_2024.xml')
print('Modelo entrenado')

