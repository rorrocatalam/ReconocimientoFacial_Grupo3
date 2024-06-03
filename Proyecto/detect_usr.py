import cv2
import os
import time

# Variables
rostro          = 0     # Indicador si se detecta un rostro que mira al frente
persona         = 0     # Indicador si hay una persona (con movimiento)
parpadeo        = 0     # Indicador si hay parpadeos
contador        = 0     # Contador de parpadeos
step            = 0     # Estapa del sistema

start           = 0     # Indicador para comenzar temporizador
start_time      = 0     # Instante en que se inicia el temporizador
max_time        = 5    # Tiempo maximo a esperar para detectar a una persona
contador_a      = 0     # Contador de frames en que se detecta a una persona
max_frames      = 20    # Cantidad de frames para confirmar deteccion de una persona
usr_a           = None  # Usuario detectado por confirmar

# Objeto reconocedor de rostros
face_recognizer = cv2.face.EigenFaceRecognizer_create()
face_recognizer.read('modelo_rf.xml')

# Objeto clasificador de rostros frontales
face_clasiff = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')

# Lista de usuarios registrados
path_db = 'C:/Users/rodri/Desktop/ReconocimientoFacial_Grupo3/Proyecto/User_DataBase'
usr_list = os.listdir(path_db)

# Captura de video (configuración inicial)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(3, 1280)
cap.set(4, 720)

def reset():
    global start, start_time, contador_a, usr_a
    start = 0
    start_time = 0
    contador_a = 0
    usr_a = None

def detect_usr():
    global start, start_time, max_time,  contador_a, max_frames, usr_a, step, cap
    
    # Si hay captura entonces se procede
    if cap is not None:
        ret, frame = cap.read()
        if not ret: 
            return
        # Cambio de color
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_copy = frame_gray.copy()

        # Detección de rostros
        faces = face_clasiff.detectMultiScale(frame_gray, 1.3, 5)

        # Se ve el primer rostro
        for (x, y, w, h) in faces:
            rostro = frame_copy[y:y+h, x:x+w]
            rostro = cv2.resize(rostro, (150, 150), interpolation=cv2.INTER_CUBIC)
            result = face_recognizer.predict(rostro)

            # Si es alguien de la base de datos se confirma su presencia con varias detecciones
            if result[1] < 5700:
                usr = usr_list[result[0]]
                # En la primera deteccion se inicia el temporizador
                if start == 0:
                    start_time = time.time()
                    # Se guarda la persona detectada
                    usr_a = usr
                    # Se cambia start
                    start = 1

                # Si se reconoce al usuario actual se aumenta el contador
                if usr == usr_a:
                    contador_a += 1
                    # Si se reconoce suficientes veces se da el acceso
                    if contador_a >= max_frames:
                        print(f"¡Bienvenido {usr}! Pase nomas mi rey ^_^")
                        # Finalizacion de la funcion
                        cap.release()
                        # Reseteo de variables globales
                        reset()
                        return
            
			# Solo el primer rostro
            break

        k = cv2.waitKey(1)
        if k == 27:
            cap.release()
            return

    # En todas las iteraciones se revisa el contador
    if start == 1:
        if time.time()-start_time >= max_time:
            print(f"No hubo reconocimiento en {max_time} segundos :(")
            cap.release()
            # Reseteo de variables globales
            reset()
            return


while True:
    detect_usr()
