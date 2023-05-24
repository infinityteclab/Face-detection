import cv2

arqcasc1 = 'haarcascade_frontalface_default.xml'
arqcasc2 = 'haarcascade_eye.xml'
arqcasc3 = 'haarcascade_eye_tree_eyeglasses.xml'

faceCascade1 = cv2.CascadeClassifier(arqcasc1)  # classificador para o rosto
faceCascade2 = cv2.CascadeClassifier(arqcasc2)  # classificador para os olhos
faceCascade3 = cv2.CascadeClassifier(arqcasc3)  # classificador para óculos

webcam = cv2.VideoCapture(0)  # instancia o uso da webcam

while True:
    s, imagem = webcam.read()  # pega efeticamente a imagem da webcam
    imagem = cv2.flip(imagem, 180)  # espelha a imagem

    faces = faceCascade1.detectMultiScale(
        imagem,
        minNeighbors=20,
        minSize=(30, 30),
        maxSize=(300, 300)
    )

    olhos = faceCascade3.detectMultiScale(
        imagem,
        minNeighbors=20,
        minSize=(10, 10),
        maxSize=(90, 90)
    )

    # Desenha um retângulo nas faces e olhos detectados
    for (x, y, w, h) in faces:
        cv2.rectangle(imagem, (x, y), (x + w, y + h), (0, 255, 0), 4)

    for (x, y, w, h) in olhos:
        cv2.rectangle(imagem, (x, y), (x + w, y + h), (255, 0, 0), 2)

    cv2.imshow('Video', imagem)  # mostra a imagem capturada na janela

    # o trecho seguinte é apenas para parar o código e fechar a janela
    if cv2.waitKey(1) == 27:  # 27 é o código ASCII para a tecla "Esc"
        break

webcam.release()  # dispensa o uso da webcam
cv2.destroyAllWindows()  # fecha todas as janelas abertas
