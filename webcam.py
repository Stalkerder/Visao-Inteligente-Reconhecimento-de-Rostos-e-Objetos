import cv2
from keras.applications import mobilenet_v2

model = mobilenet_v2.MobileNetV2(weights='imagenet')
face_detection = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

webcam = cv2.VideoCapture(0)

while True:
    ret, frame = webcam.read()
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detection.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)        
    cv2.imshow("Detecção de Face por Webcam com Captura de Imagem", frame)

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
