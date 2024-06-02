import os
import sys
import time

import numpy as np
import cv2
import imutils

#=============================================================================
# Variables 
n_max           = 200   # Cantidad de imagenes a tomar de un rostro
max_length      = 110   # Largo maximo de los mensajes a mostrar

#=============================================================================
# Direcciones a considerar
path_db = 'C:/Users/rodri/Desktop/ReconocimientoFacial_Grupo3/Proyecto/User_DataBase'
usr_name = input("Ingrese su nombre y apellido: ")
path_usr = path_db + '/' + usr_name

# Se crea la carpeta del usuario en caso de no existir
if not os.path.exists(path_usr):
	os.makedirs(path_usr)

# Captura de video
cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)

# Clasificador a utilizar
faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
n = 0

m1 = f"\r¡Mueve la cabeza lentamente de lado a lado y haz varias expresiones faciales para capturar mejor tu rostro!"
sys.stdout.write(m1.ljust(max_length) + '\r')
sys.stdout.flush()
time.sleep(1)

# Captura de imagenes
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Redimensionamiento y cambio de color
    frame = imutils.resize(frame, width=640)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_rgb = frame.copy()

    # Detección de rostros usando clasificador en cascada
    faces = faceClassif.detectMultiScale(gray, 1.3, 5)

    # Recorte y almacenamiento del rostro
    for (x, y, w, h) in faces:
        rostro = frame_rgb[y:y+h, x:x+w]
        rostro = cv2.resize(rostro, (150, 150), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(path_usr + '/rostro_{}.jpg'.format(n), rostro)
        n += 1

    # Se termina al tener 300 imágenes
    if n >= n_max:
        break

    # Actualiza la consola sin agregar nuevas líneas
    m2 = f"\r{int(((n+1)*100)/n_max)}% completado"
    sys.stdout.write(m2.ljust(max_length) + '\r')
    sys.stdout.flush()
cap.release()

m3 = f"\r¡Captura completada! Entrenando modelo...\n"
sys.stdout.write(m3.ljust(max_length) + '\r')
sys.stdout.flush()
#=============================================================================
people_list = os.listdir(path_db)

# Arreglos a considerar para entrenamiento
labels = []
facesData = []
label = 0

# Se lee la informacion de todas las personas presentes en la base de datos
for person in people_list:
    # Direccion de cada persona
	person_path = path_db + '/' + person
    # Se pasa por cada imagen
	for picture in os.listdir(person_path):
		labels.append(label)
		facesData.append(cv2.imread(person_path + '/' + picture, 0))
    # Label para proxima persona
	label = label + 1

# Objeto de reconocedor de rostros y entrenamiento
face_recognizer = cv2.face.EigenFaceRecognizer_create()
face_recognizer.train(facesData, np.array(labels))

# Almacenamiento del modelo
face_recognizer.write('modelo_rf.xml')
print("¡Modelo entrenado y almacenado!\n")
