import cv2
# import face_recognition as fr
import numpy as np
import mediapipe as mp
import os

from tkinter import *
from PIL import Image, ImageTk
import imutils
import random

#-----------------------------------------------------------------------------
# Variables
parpadeo = 0
contador = 0
muestra = 0
step = 0

# Parametros para deteccion de rostro
offset_x = 20
offset_y = 40
face_threshold = 0.5 # Umbral detección de rostro

# Herramienta de dibujo para la malla facial
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

# Objeto de malla facial
mp_face_mesh = mp.solutions.face_mesh
mp_drawing_styles = mp.solutions.drawing_styles
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1,
                                 refine_landmarks=True,
                                 min_detection_confidence=0.5,
                                 min_tracking_confidence=0.5)

# Objeto detector de rostros
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5, model_selection=1)

#-----------------------------------------------------------------------------
# Funcion para nuevos usuarios
def log_rf():
    """
    Variables globales a considerar:
        screen =    la pantalla a mostrar el video con tkinter
        parpadeo =  indica si la persona parpadea o no
        contador =  la cantidad de parpadeos de la persona
        step =      indica el paso del proceso en que se encuentra
        cap =       la captura del video
    """
    global screen, parpadeo, contador, step, cap

    # Si hay captura entonces se procede
    if cap is not None:
        ret, frame = cap.read()
        # Copia en caso de tener que guardar parte de la imagen
        frame_save = frame
        # Redimensionamiento y cambio de color
        frame = imutils.resize(frame,width=1280)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Copia sobre la que se identificaran objetos
        frame_rgb = frame

        # Dimensiones de la imagen
        height, width, c = frame.shape
        
        # Se busca primero una malla facial para detectar solamente un rostro
        mesh = face_mesh.process(frame_rgb)
        # Lista para guardar resultados
        px = []
        py = []
        lista = []

        # Se procede en caso de tener detecciones
        if mesh.multi_face_landmarks:
            for rostros in mesh.multi_face_landmarks:
                # Se extraen puntos importantes para identificar orientacion del rostro
                for id, puntos in enumerate(rostros.landmark):
                    x, y = int(puntos.x*width), int(puntos.y*height)
                    px.append(x)
                    py.append(y)
                    lista.append([id, x, y])

                    # Una vez se tienen los 468 que caracterizan un rostro se procede
                    if len(lista) == 468:
                        # Ojo derecho
                        x1, y1 = lista[145][1:]
                        x2, y2 = lista[159][1:]
                        l1 = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                        
                        # Ojo izquierdo
                        x3, y3 = lista[374][1:]
                        x4, y4 = lista[386][1:]
                        l2 = np.sqrt((x4-x3)**2 + (y4-y3)**2)

                        # Parietal derecho
                        x5, y5 = lista[139][1:]
                        # Parietal izquierdo
                        x6, y6 = lista[368][1:]

                        # Ceja derecha
                        x7, y7 = lista[70][1:]
                        # Ceja izquierda
                        x8, y8 = lista[300][1:]

                        # Se realiza la deteccion del rostro
                        faces = face_detection.process(frame_rgb)
                        if faces.detections is not None:
                            for face in faces.detections:
                                # Se obtiene su probabilidad de ser un rostro
                                score = face.score[0]
                                
                                # Se procede en caso de superar el umbral
                                if score > face_threshold:
                                    cv2.putText(frame,"Hay un rostro!", (1070,375), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255,255,255), 1)
                                    
                                    # Si se esta en step = 0, se deben contar parpadeos
                                    if step == 0:
                                        # Se mira de frente
                                        if x7 > x5 and x8 < x6: 
                                            cv2.putText(frame,"Parpadea!", (1070,400), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255,255,255), 1)
                                            # Contador de parpadeos
                                            if l1 <= 10 and l2 <= 10 and parpadeo == 0:
                                                contador +=1
                                                parpadeo = 1
                                            elif l1 > 10 and l2 > 10 and parpadeo == 1:
                                                parpadeo = 0
                                            cv2.putText(frame,f"Parpadeos: {contador}", (1070,425), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255,255,255), 1)
                                            
                                            # Se cumple cantidad de parpadeos y se guarda usuario
                                            if contador >= 3:
                                                # Se obtiene imagen del usuario cuando se abran los ojos
                                                if l1 > 10 and l2 > 10:
                                                    # Cuadrado que encierra el rostro
                                                    bbox = face.location_data.relative_bounding_box
                                                    # Se calculan dimensiones que tendra la imagen
                                                    xb = int(bbox.xmin*width)
                                                    yb = int(bbox.ymin*height)
                                                    wb = int(bbox.width*width)
                                                    hb = int(bbox.height*height)

                                                    # Se le da olgura con un offset
                                                    offset_w = (offset_x*wb)/100
                                                    xb = int(xb-offset_w/2)
                                                    wb = int(wb + offset_w)

                                                    offset_h = (offset_y*hb)/100
                                                    yb = int(yb-offset_h)
                                                    hb = int(hb + offset_h)

                                                    # Casos de error
                                                    if xb < 0:
                                                        xb = 0
                                                    if yb < 0:
                                                        yb = 0
                                                    if wb < 0:
                                                        wb = 0
                                                    if hb < 0:
                                                        hb = 0

                                                    # Recuadro a recortar del frame
                                                    xf = xb + wb
                                                    yf = yb + hb
                                                    cut = frame_save[yb:yf, xb:xf]

                                                    # Se da un usuario aleatorio
                                                    n = random.randint(10000, 99999)
                                                    cv2.imwrite(f"{pathFaces}/{n}.png", cut)
                                                    print(f"Usuario {n} registrado!")

                                                    # Se pasa a la siguiente etapa
                                                    step = 1
                    
                                        # No se mira de frente
                                        else:
                                            cv2.putText(frame,"Mira hacia el frente!", (1070,400), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255,255,255), 1)
                                            # Se reinicia el contador
                                            contador = 0

        # Convertir el video
        im = Image.fromarray(frame)
        img = ImageTk.PhotoImage(image=im)

        # Mostrar el video
        labelVideo.configure(image=img)
        labelVideo.image = img
        labelVideo.after(10, log_rf)
    
    else:
        cap.release()

def log_rf_test():
    """
    Variables globales a considerar:
        screen =    la pantalla a mostrar el video con tkinter
        parpadeo =  indica si la persona parpadea o no
        contador =  la cantidad de parpadeos de la persona
        step =      indica el paso del proceso en que se encuentra
        cap =       la captura del video
    """
    global screen, parpadeo, contador, step, cap

    # Si hay captura entonces se procede
    if cap is not None:
        ret, frame = cap.read()
        # Copia en caso de tener que guardar parte de la imagen
        frame_save = frame
        # Redimensionamiento y cambio de color
        frame = imutils.resize(frame,width=1280)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Copia sobre la que se identificaran objetos
        frame_rgb = frame

        # Dimensiones de la imagen
        height, width, c = frame.shape
        
        # Se busca primero una malla facial para detectar solamente un rostro
        mesh = face_mesh.process(frame_rgb)
        # Lista para guardar resultados
        px = []
        py = []
        lista = []

        # Se procede en caso de tener detecciones
        if mesh.multi_face_landmarks:
            for rostros in mesh.multi_face_landmarks:
                # Se dibujan
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=rostros,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())
                # Se extraen puntos importantes para identificar orientacion del rostro
                for id, puntos in enumerate(rostros.landmark):
                    x, y = int(puntos.x*width), int(puntos.y*height)
                    px.append(x)
                    py.append(y)
                    lista.append([id, x, y])

                    # Una vez se tienen los 468 que caracterizan un rostro se procede
                    if len(lista) == 468:
                        # Ojo derecho
                        x1, y1 = lista[145][1:]
                        x2, y2 = lista[159][1:]
                        l1 = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                        # Visualizar informacion
                        cv2.circle(frame,(x1,y1),2,(255,0,0),cv2.FILLED)
                        cv2.circle(frame,(x2,y2),2,(255,0,0),cv2.FILLED)
                        
                        # Ojo izquierdo
                        x3, y3 = lista[374][1:]
                        x4, y4 = lista[386][1:]
                        l2 = np.sqrt((x4-x3)**2 + (y4-y3)**2)
                        # Visualizar informacion
                        cv2.circle(frame,(x3,y3),2,(0,255,0),cv2.FILLED)
                        cv2.circle(frame,(x4,y4),2,(0,255,0),cv2.FILLED)

                        # Parietal derecho
                        x5, y5 = lista[139][1:]
                        # Parietal izquierdo
                        x6, y6 = lista[368][1:]
                        # Visualizar informacion
                        cv2.circle(frame,(x5,y5),2,(255,0,0),cv2.FILLED)
                        cv2.circle(frame,(x6,y6),2,(0,255,0),cv2.FILLED)

                        # Ceja derecha
                        x7, y7 = lista[70][1:]
                        # Ceja izquierda
                        x8, y8 = lista[300][1:]
                        # Visualizar informacion
                        cv2.circle(frame,(x7,y7),2,(255,0,0),cv2.FILLED)
                        cv2.circle(frame,(x8,y8),2,(0,255,0),cv2.FILLED)

                        # Se realiza la deteccion del rostro
                        faces = face_detection.process(frame_rgb)
                        if faces.detections is not None:
                            for face in faces.detections:
                                # Se obtiene su probabilidad de ser un rostro y la bbox que lo encierra
                                score = face.score[0]
                                # Cuadrado que encierra el rostro
                                bbox = face.location_data.relative_bounding_box
                                # Se calculan dimensiones que tendra la bbox a mostrar
                                xb = int(bbox.xmin*width)
                                yb = int(bbox.ymin*height)
                                wb = int(bbox.width*width)
                                hb = int(bbox.height*height)

                                # Se le da olgura con un offset
                                offset_w = (offset_x*wb)/100
                                xb = int(xb-offset_w/2)
                                wb = int(wb + offset_w)

                                offset_h = (offset_y*hb)/100
                                yb = int(yb-offset_h)
                                hb = int(hb + offset_h)

                                # Se procede en caso de superar el umbral
                                if score > face_threshold:
                                    cv2.putText(frame,"Hay un rostro!", (1070,375), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255,255,255), 1)
                                    
                                    # Si se esta en step = 0, se deben contar parpadeos
                                    if step == 0:
                                        cv2.rectangle(frame, (xb, yb, wb, hb), (0,0,255),2)
                                        # Se mira de frente
                                        if x7 > x5 and x8 < x6: 
                                            cv2.putText(frame,"Parpadea!", (1070,400), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255,255,255), 1)
                                            # Contador de parpadeos
                                            if l1 <= 10 and l2 <= 10 and parpadeo == 0:
                                                contador +=1
                                                parpadeo = 1
                                            elif l1 > 10 and l2 > 10 and parpadeo == 1:
                                                parpadeo = 0
                                            cv2.putText(frame,f"Parpadeos: {contador}", (1070,425), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255,255,255), 1)
                                            
                                            # Se cumple cantidad de parpadeos y se guarda usuario
                                            if contador >= 3:
                                                # Se obtiene imagen del usuario cuando se abran los ojos
                                                if l1 > 10 and l2 > 10:
                                                    # Recuadro a recortar del frame
                                                    xf = xb + wb
                                                    yf = yb + hb
                                                    cut = frame_save[yb:yf, xb:xf]

                                                    # Se da un usuario aleatorio
                                                    n = random.randint(10000, 99999)
                                                    cv2.imwrite(f"{pathFaces}/{n}.png", cut)
                                                    print(f"Usuario {n} registrado!")

                                                    # Se pasa a la siguiente etapa
                                                    step = 1
                    
                                        # No se mira de frente
                                        else:
                                            cv2.putText(frame,"Mira hacia el frente!", (1070,400), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255,255,255), 1)
                                            # Se reinicia el contador
                                            contador = 0

        # Convertir el video
        im = Image.fromarray(frame)
        img = ImageTk.PhotoImage(image=im)

        # Mostrar el video
        labelVideo.configure(image=img)
        labelVideo.image = img
        labelVideo.after(10, log_rf_test)
    
    else:
        cap.release()

#-----------------------------------------------------------------------------
# Direcciones de carpetas (Modificable)
pathFaces = "C:/Users/rodri/Desktop/ReconocimientoFacial_Grupo3/Proyecto/DataBase/Faces"
pathUsers = "C:/Users/rodri/Desktop/ReconocimientoFacial_Grupo3/Proyecto/DataBase/Users"

# Informacion
info = []

# Ventana principal
screen = Tk()
screen.title("Reconocimiento Facial")
screen.geometry("1280x720")

# Ubicacion del video en la ventana
labelVideo = Label(screen)
labelVideo.place(x=0, y=0)

# Captura de video
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(3, 1280)
cap.set(4, 720)
log_rf_test()

screen.mainloop()