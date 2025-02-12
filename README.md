# Clasificación de Residuos con Redes Neuronales Convolucionales (CNN)

Este proyecto tiene como objetivo clasificar imágenes de residuos en tres categorías: **No Reciclable**, **Orgánico** y **Reciclable**. Utiliza una red neuronal convolucional (CNN) entrenada con TensorFlow y Keras para realizar la clasificación.

## Tabla de Contenidos

- [Descripción del Proyecto](#descripción-del-proyecto)
- [Instalación](#instalación)
- [Uso](#uso)
- [Estructura del Proyecto](#estructura-del-proyecto)
- [Contribución](#contribución)

## Descripción del Proyecto

El proyecto utiliza un modelo de red neuronal convolucional (CNN) para clasificar imágenes de residuos en tres categorías principales:

1. **No Reciclable**: Residuos que no pueden ser reciclados.
2. **Orgánico**: Residuos biodegradables como restos de comida.
3. **Reciclable**: Residuos que pueden ser reciclados, como plástico, vidrio y papel.

El modelo se entrena con un conjunto de datos preprocesado y se guarda para su uso en una aplicación de clasificación en tiempo real.

## Instalación

Sigue estos pasos para configurar el proyecto en tu máquina local:
1. **Versión de Python recomendada**:

   Asegúrate de tener Python 3.12.8. 
   Puedes verificar tu versión de Python con el siguiente comando:
   ```bash
   python --version

3. **Clona el repositorio**:
   ```bash
   git clone https://github.com/axelherreram/clasificate-waste-ia.git
   cd clasificacion-residuos
   ```
4. **Crea y activa un entorno virtual**:
   ```bash
   python -m venv waste_clasificate
   ```
5. **Instala las dependencias**:
   ```bash
   pip install -r requirements.txt
   ```

### 4. Descarga y organiza los archivos necesarios para YOLO

Sigue estos pasos para obtener y ubicar los archivos esenciales de YOLO:

1. **Descarga los siguientes archivos:**

   - [`yolov3.weights`](https://github.com/patrick013/Object-Detection---Yolov3/blob/master/model/yolov3.weights)
   - [`yolov3.cfg`](https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg)
   - [`coco.names`](https://github.com/pjreddie/darknet/blob/master/data/coco.names)

2. **Ubicación de los archivos:**
   - Coloca los archivos descargados en la carpeta `YOLO`, ubicada en la raíz del proyecto.

## Uso

Para ejecutar la aplicación de clasificación de residuos, utiliza el siguiente comando:

```bash
python evaluate_model.py
```

Esto abrirá una interfaz gráfica donde podrás cargar imágenes o utilizar la cámara para clasificar residuos en tiempo real.

## Estructura del Proyecto

.gitignore

```
└── 📁contenedores
    └── black_container.png
    └── brown_container.png
    └── green_container.png
└── 📁model
   └──modelo_entrenado.h5
└── 📁YOLO
    └── coco.names
    └── yolov3.cfg
    └── yolov3.weights
└── .gitignore
└── evaluate_model.py
└── README.md
└── requirements.txt
```

## Contribución

Si deseas contribuir a este proyecto, por favor sigue estos pasos:

1. Haz un fork del repositorio.
2. Crea una nueva rama
   ```bash
   git checkout -b feature/nueva-funcionalidad
   ```
3. Realiza tus cambios y haz commit
   ```bash
    git commit -am 'Añadir nueva funcionalidad'
   ```
4. Sube tus cambios a tu rama

```bash
git push origin feature/nueva-funcionalidad
```

5. Abre un Pull Request.
