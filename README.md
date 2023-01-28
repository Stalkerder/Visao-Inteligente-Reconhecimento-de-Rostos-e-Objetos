LEIA-ME:

{Stalker Red}:

**Webcam**

Este programa utiliza a biblioteca OpenCV para detectar rostos em imagens capturadas pela webcam. Para usá-lo, você precisará instalar a seguinte biblioteca:

OpenCV: Versão 4.7.0 ou superior
Para instalar o OpenCV no Windows, você pode baixar o arquivo de instalação em https://pypi.org/project/opencv-python/ ou utilizar o comando "pip install opencv-python".

Uma vez instalado o OpenCV, você pode executar o arquivo webcam.py. Certifique-se de que a câmera esteja conectada e habilitada

O programa abrirá uma janela com a imagem capturada pela webcam, marcando os rostos detectados com um retângulo azul.
O usuário também tem a opção de pressionar a tecla "s" para salvar a imagem atual ou pressionar "q" para encerrar o programa.


**Objetos**

Este programa utiliza o modelo VGG16 pré-treinado para classificar objetos em imagens capturadas pela webcam. Ele utiliza a biblioteca OpenCV para capturar imagens da webcam e a biblioteca Keras para carregar e utilizar o modelo VGG16.

Para executar o programa, é necessário ter o OpenCV e o Keras instalados, utilize "pip install cv2" "pip install keras" "pip install numpy" além dos pesos do modelo VGG16 (arquivo "vgg16_weights_tf_dim_ordering_tf_kernels.h5"). É necessário também uma webcam conectada ao computador.

O programa inicializa a webcam e, em um loop, captura imagens e as classifica usando o modelo VGG16. A classificação é exibida na imagem capturada e na tela. 
O usuário também tem a opção de pressionar a tecla "s" para salvar a imagem atual ou pressionar "q" para encerrar o programa.


**Webcam + Objetos (em beta)**

Este programa utiliza a biblioteca OpenCV para capturar imagens de uma webcam e a biblioteca Keras para fazer classificações de objetos em imagens. Ele faz uso do modelo VGG16 pré-treinado para identificar objetos em imagens capturadas pela webcam.

O programa inicialmente carrega o modelo VGG16 e cria uma camada de saída personalizada para classificar objetos. Em seguida, ele carrega um detector de faces e marcas faciais usando a biblioteca dlib.

A webcam é inicializada e as imagens capturadas são convertidas para tons de cinza. As faces são detectadas na imagem usando o detector de faces do dlib e são marcadas com um retângulo verde. Uma subimagem é criada a partir da face detectada, redimensionada para as dimensões esperadas pelo modelo VGG16 e pré-processada. A imagem é então passada através do modelo para obter uma predição e a classificação mais provável é escrita na tela.

Para executar o programa você deve instalar as biblioteca necessária com o comando "pip install opencv-python numpy keras keras_applications keras_preprocessing dlib logging" Além disso, é necessário ter o arquivo "vgg16_weights_tf_dim_ordering_tf_kernels.h5" e "shape_predictor_68_face_landmarks.dat" na máquina para carregar os pesos pré-treinados e o detector de faces e marcas faciais respectivamente.

O usuário também tem a opção de pressionar a tecla "s" para salvar a imagem atual ou pressionar "q" para encerrar o programa. O número de faces detectadas também é registrado em um arquivo de log.



**Exemplos**

EXEMPLO_img.jpg  ------> Programa Webcam, identifica vários rosto

EXEMPLO2_img.jpg -------> Programa Objetos, identificou o microfone


**NOTAS**

Os códigos estão ainda em fase de testes

É necessário aumentar a precisão do modelo VGG16, deve ser feito um treinamento com uma base de dados de imagens de rostos e objetos específicos. Além disso, temos que ajustar os parâmetros do detector de faces (como scaleFactor e minNeighbors) para detectar com mais precisão os rostos na imagem. No momento desse programa em "Webcam + Objetos (beta)" é utilizado o dlib que atualmente é mais preciso que o Cascade usado nos outros 2 programas. Lembrando que são modelos pré-treinados, para aumentar a precisão deve também aumentar o numero de dados de treinamento, e usar transfer learning.

