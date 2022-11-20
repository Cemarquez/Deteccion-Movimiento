import cv2
import time
from tkinter import *
from tkinter.filedialog import askopenfilename

base = Tk()
modoUso = 0


base.title("Detector de movimiento")
base.geometry("398x281")
base.resizable(width=False, height=False)


def cam():
    global modoUso
    modoUso = 0
    base.destroy()


def file():
    global modoUso
    root = Tk()
    root.withdraw()
    root.update()
    pathString = askopenfilename(filetypes=[("video files", "*.mp4")])
    modoUso = pathString
    root.destroy()
    base.destroy()


Label = Label(base, text="Elija un modo de uso")
Label.place(x=60, y=10, height=119, width=266)

CamButton = Button(base, font=('Arial', 12, 'bold'),
                   text="Cámara", bg="#dfdfdf", activebackground="#3e3e3e", fg="#ffffff", command=cam)
CamButton.place(x=210, y=190, height=39, width=89)

FileButton = Button(base, font=('Arial', 12, 'bold'),
                    text="Vídeo", bg="#dfdfdf", activebackground="#3e3e3e", fg="#ffffff", command=file)
FileButton.place(x=90, y=190, height=39, width=89)


base.mainloop()


cap = cv2.VideoCapture(modoUso)
fondo = None
fgbg = cv2.createBackgroundSubtractorKNN(history=500, dist2Threshold=400, detectShadows=False)

# Deshabilitamos OpenCL, si no hacemos esto no funciona
cv2.ocl.setUseOpenCL(False)

while True:
    ret, frame = cap.read()
# Si hemos llegado al final del vídeo salimos
    if not ret:
        break

    # Aplicamos el algoritmo
    fgmask = fgbg.apply(frame)

    # Copiamos el umbral para detectar los contornos
    contornosimg = fgmask.copy()

     # Buscamos contorno en la imagen
    contornos, hierarchy = cv2.findContours(
          contornosimg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

       # Recorremos todos los contornos encontrados
    for c in contornos:
            # Eliminamos los contornos más pequeños
            if cv2.contourArea(c) < 500:
                continue

            # Obtenemos el bounds del contorno, el rectángulo mayor que engloba al contorno
            (x, y, w, h) = cv2.boundingRect(c)
            # Dibujamos el rectángulo del bounds
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Mostramos las capturas
    cv2.imshow('Camara', frame)
    cv2.imshow('Umbral', fgmask)
    cv2.imshow('Contornos', contornosimg)
    time.sleep(0.015)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
