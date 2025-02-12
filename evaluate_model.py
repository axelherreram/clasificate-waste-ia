import cv2
import numpy as np
from tensorflow.keras.models import load_model
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import os

# Obtener la ruta absoluta del directorio actual
current_dir = os.path.dirname(os.path.abspath(__file__))

# Cargar el modelo de clasificación de residuos
classification_model = load_model('./models/waste_classification_final_model.h5')  # Cambia al otro modelo si es necesario

# Etiquetas de las clases
class_labels = {0: 'No Reciclable', 1: 'Orgánico', 2: 'Reciclable'}

# Cargar imágenes de los contenedores
container_images = {
    'No Reciclable': cv2.imread(os.path.join(current_dir, 'contenedores/black_container.png')),
    'Orgánico': cv2.imread(os.path.join(current_dir, 'contenedores/green_container.png')),
    'Reciclable': cv2.imread(os.path.join(current_dir, 'contenedores/brown_container.png'))
}

# Rutas absolutas a los archivos YOLO
yolov3_weights_path = os.path.join(current_dir, "./YOLO/yolov3.weights")
yolov3_cfg_path = os.path.join(current_dir, "./YOLO/yolov3.cfg")
coco_names_path = os.path.join(current_dir, "./YOLO/coco.names")

# Cargar YOLOv3
net = cv2.dnn.readNet(yolov3_weights_path, yolov3_cfg_path)

# Cargar las clases COCO
with open(coco_names_path, "r", encoding="utf-8") as f:
    classes = f.read().strip().split("\n")

# Obtener las capas de salida de YOLO
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Tamaño fijo para las imágenes
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480

# Función para preprocesar la imagen para el modelo de clasificación
def preprocess_image(image):
    """
    Redimensiona y normaliza la imagen para que coincida con la entrada del modelo.
    """
    resized_image = cv2.resize(image, (150, 150))
    normalized_image = resized_image / 255.0
    input_image = np.expand_dims(normalized_image, axis=0)
    return input_image

# Función para clasificar la imagen
def classify_image(image, model):
    """
    Clasifica la imagen utilizando el modelo cargado.
    """
    input_image = preprocess_image(image)
    predictions = model.predict(input_image)
    predicted_class = np.argmax(predictions, axis=1)[0]
    return predicted_class

# Función para detectar objetos usando YOLO
def detect_objects(image):
    """
    Detecta objetos en la imagen usando YOLO.
    """
    height, width, channels = image.shape

    # Preprocesar la imagen para YOLO
    blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Información de detección
    class_ids = []
    confidences = []
    boxes = []

    # Procesar las detecciones
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:  # Umbral de confianza
                # Coordenadas del objeto detectado
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectángulo del objeto
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Aplicar supresión de no máximos para eliminar detecciones redundantes
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Devolver las detecciones
    detected_objects = []
    for i in indices:
        detected_objects.append({
            "class": classes[class_ids[i]],
            "confidence": confidences[i],
            "box": boxes[i]
        })
    return detected_objects

# Función para procesar un frame
def process_frame(frame):
    """
    Procesa un frame: detecta objetos y clasifica los residuos.
    """
    # Detectar objetos
    detected_objects = detect_objects(frame)

    # Clasificar cada objeto detectado
    for obj in detected_objects:
        x, y, w, h = obj["box"]

        # Verificar que las dimensiones del objeto sean válidas
        if w > 0 and h > 0:
            object_image = frame[y:y+h, x:x+w]

            # Verificar que el objeto recortado no esté vacío
            if object_image.size > 0:
                # Clasificar el objeto
                predicted_class = classify_image(object_image, classification_model)
                label = class_labels[predicted_class]

                # Obtener la imagen del contenedor correspondiente
                container_image = container_images[label]
                container_image = cv2.resize(container_image, (50, 50))  # Redimensionar la imagen del contenedor

                # Verificar que la región donde se superpondrá la imagen esté dentro de los límites del frame
                if y + 50 <= frame.shape[0] and x + 50 <= frame.shape[1]:
                    # Superponer la imagen del contenedor en el frame
                    frame[y:y+50, x:x+50] = container_image
                else:
                    # Si la región es más pequeña, redimensionar la imagen del contenedor para que coincida
                    region_height = min(50, frame.shape[0] - y)
                    region_width = min(50, frame.shape[1] - x)
                    resized_container = cv2.resize(container_image, (region_width, region_height))
                    frame[y:y+region_height, x:x+region_width] = resized_container

                # Dibujar el rectángulo y la etiqueta en el frame
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            else:
                print("Advertencia: El objeto recortado está vacío.")
        else:
            print("Advertencia: Dimensiones inválidas para el objeto detectado.")

    return frame

# Interfaz gráfica con Tkinter
class WasteClassificationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Clasificación de Residuos")

        # Definir el tamaño de la ventana
        self.root.geometry("800x600")  # Ancho x Alto

        # Marco para los botones
        button_frame = tk.Frame(root)
        button_frame.pack(side=tk.TOP, pady=10)

        # Botón para abrir una imagen
        self.open_button = tk.Button(button_frame, text="Abrir Imagen", command=self.open_image)
        self.open_button.pack(side=tk.LEFT, padx=10)

        # Botón para iniciar la cámara
        self.camera_button = tk.Button(button_frame, text="Iniciar Cámara", command=self.start_camera)
        self.camera_button.pack(side=tk.LEFT, padx=10)

        # Etiqueta para mostrar la imagen
        self.image_label = tk.Label(root)
        self.image_label.pack(side=tk.TOP, pady=10)

        # Variable para controlar la cámara
        self.camera_active = False
        self.cap = None

    def open_image(self):
        """
        Abre una imagen y la procesa.
        """
        file_path = filedialog.askopenfilename()
        if file_path:
            image = cv2.imread(file_path)
            if image is None:
                print("Error: No se pudo cargar la imagen.")
                return
            processed_image = process_frame(image)
            self.show_image(processed_image)

    def start_camera(self):
        """
        Inicia la cámara y procesa los frames en tiempo real.
        """
        if not self.camera_active:
            self.camera_active = True
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                print("Error: No se pudo abrir la cámara.")
                self.camera_active = False
                return
            self.camera_button.config(text="Detener Cámara")
            self.update_camera()
        else:
            self.camera_active = False
            self.cap.release()
            self.camera_button.config(text="Iniciar Cámara")

    def update_camera(self):
        """
        Actualiza el frame de la cámara en la interfaz.
        """
        if self.camera_active:
            ret, frame = self.cap.read()
            if ret:
                processed_frame = process_frame(frame)
                self.show_image(processed_frame)
            self.root.after(10, self.update_camera)

    def show_image(self, image):
        """
        Muestra una imagen en la interfaz.
        """
        # Redimensionar la imagen al tamaño fijo
        resized_image = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT))
        resized_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(resized_image)
        image_tk = ImageTk.PhotoImage(image_pil)
        self.image_label.config(image=image_tk)
        self.image_label.image = image_tk

# Iniciar la aplicación
if __name__ == "__main__":
    root = tk.Tk()
    app = WasteClassificationApp(root)
    root.mainloop()