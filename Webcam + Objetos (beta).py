# import cv2
# import numpy as np
# from keras.applications.vgg16 import VGG16
# from keras.applications.vgg16 import preprocess_input
# from keras.preprocessing import image
# from keras.applications.vgg16 import decode_predictions
# from keras.models import Model
# import logging as log
# import datetime as dt

# # Carregando o modelo VGG16 pré-treinado
# base_model = VGG16(weights='vgg16_weights_tf_dim_ordering_tf_kernels.h5', include_top=True)

# # Criando uma camada de saída personalizada para classificar objetos
# model = Model(inputs=base_model.input, outputs=base_model.get_layer('predictions').output)

# casc_path = "haarcascade_frontalface_default.xml"
# face_cascade = cv2.CascadeClassifier(casc_path)
# log.basicConfig(filename='webcam.log',level=log.INFO)

# video_capture = cv2.VideoCapture(0)

# if not video_capture.isOpened():
#     print('Não foi possível carregar a câmera.')
#     exit()

# anterior = 0

# while True:
#     ret, frame = video_capture.read()
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
#     for (x, y, w, h) in faces:
#         cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
#         # Criar uma subimagem a partir do rosto detectado
#         face = frame[y:y+h, x:x+w]
        
#         # Redimensionar a subimagem para as dimensões esperadas pelo modelo VGG16
#         face = cv2.resize(face, (224, 224))
        
#         # Fazer o pré-processamento da imagem de entrada
#         face = preprocess_input(face)
        
#         # Passar a imagem através do modelo para obter uma predição
#         pred = model.predict(np.expand_dims(face, axis=0))
        
#         # Decodificar a predição para obter a classificação mais provável
#         pred_class = decode_predictions(pred, top=1)[0][0][1]
        
#         # Escrever a classificação na tela
#         cv2.putText(frame, pred_class, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
#     if anterior != len(faces):
#         anterior = len(faces)
#         log.info("faces: "+str(len(faces))+" at "+str(dt.datetime.now()))

#     cv2.imshow('Video', frame)

#     key = cv2.waitKey(1)

#     if key == ord('s'):
#         filename = "saved_img.jpg"
#         cv2.imwrite(filename, frame)
#         print("Imagem salva como", filename)
#         break
#     elif key == ord('q'):
#         print("Programa encerrado.")
#         break

# video_capture.release()
# cv2.destroyAllWindows()

#<--------------------------------------------------------------------------------------------> Codigo acima para ajuste (beta)

import cv2
import numpy as np
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from keras.applications.vgg16 import decode_predictions
from keras.models import Model
import logging as log
import datetime as dt
import dlib

# Carregando o modelo VGG16 pré-treinado
base_model = VGG16(weights='vgg16_weights_tf_dim_ordering_tf_kernels.h5', include_top=True)

# Criando uma camada de saída personalizada para classificar objetos
model = Model(inputs=base_model.input, outputs=base_model.get_layer('predictions').output)

# Carregando o detector de faces e marcas faciais
face_cascade = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('path/to/shape_predictor_68_face_landmarks.dat')

log.basicConfig(filename='webcam.log',level=log.INFO)

webcam = cv2.VideoCapture(0)

if not webcam.isOpened():
    print('Não foi possível carregar a câmera.')
    exit()

anterior = 0

while True:
    ret, frame = webcam.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detectando os rostos na imagem usando o dlib
    faces = face_cascade(gray, 1)
    
    for face in faces:
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Criar uma subimagem a partir do rosto detectado
        face_img = frame[y:y+h, x:x+w]
        
        # Redimensionar a subimagem para as dimensões esperadas pelo modelo VGG16
        face_img = cv2.resize(face_img, (224, 224))
        
        # Fazer o pré-processamento da imagem de entrada
        face_img = preprocess_input(face_img)
        
        # Passar a imagem através do modelo para obter uma predição
        pred = model.predict(np.expand_dims(face_img, axis=0))
        
        # Decodificar a predição para obter a classificação mais provável
        pred_class = decode_predictions(pred, top=1)[0][0][1]
        
        # Escrever a classificação na tela
        cv2.putText(frame, pred_class, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    if anterior != len(faces):
        anterior = len(faces)
        log.info("faces: "+str(len(faces))+" at "+str(dt.datetime.now()))

    cv2.imshow('Video', frame)

    key = cv2.waitKey(1)

    if key == ord('s'):
        filename = "saved_img.jpg"
        cv2.imwrite(filename, frame)
        print("Imagem salva como", filename)
        break
    elif key == ord('q'):
        print("Programa encerrado.")
        break

webcam.release()
cv2.destroyAllWindows()