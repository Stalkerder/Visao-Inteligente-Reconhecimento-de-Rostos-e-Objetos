import cv2
import keras
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
import numpy as np
from keras.preprocessing import image
from keras.applications.vgg16 import decode_predictions

# Carregando o modelo VGG16 pré-treinado
base_model = VGG16(weights='vgg16_weights_tf_dim_ordering_tf_kernels.h5', include_top=True)

# Criando uma camada de saída personalizada para classificar objetos
model = Model(inputs=base_model.input, outputs=base_model.get_layer('predictions').output)

# Inicializando a webcam
webcam = cv2.VideoCapture(0)

while True:
    # Capturando uma imagem da webcam
    ret, frame = webcam.read()

    # Convertendo a imagem para o formato de entrada do modelo
    img = cv2.resize(frame, (224, 224))
    img = img.astype('float32')
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)

    # Classificando a imagem
    preds = model.predict(img)

    # Convertendo a saída do modelo para uma string legível
    pred_class = keras.applications.vgg16.decode_predictions(preds, top=1)
    pred_class_name = pred_class[0][0][1]

    # Exibindo a imagem e a classificação na tela
    cv2.putText(frame, pred_class_name, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.imshow('Webcam', frame)

    key = cv2.waitKey(1)

    # Finalizando o loop com a tecla 'q'
    if key == ord('s'):
        filename = "saved_img.jpg"
        cv2.imwrite(filename, frame)
        print("Imagem salva como", filename)
        break
    elif key == ord('q'):
        print("Programa encerrado.")
        break

# Finalizando a webcam e fechando as janelas
webcam.release()
cv2.destroyAllWindows()
