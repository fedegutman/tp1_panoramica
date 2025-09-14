# 📌 TP1 – Panorámicas con Homografía y Blending

Este trabajo práctico consiste en implementar un pipeline de visión artificial para construir imágenes panorámicas a partir de tripletes de fotografías. El objetivo es alinear correctamente las imágenes laterales sobre la central mediante homografías y combinarlas con técnicas de blending para obtener un resultado visual continuo.  

Se trabajó con tres conjuntos de datos:  
- **Udesa**: fotografías del campus universitario.  
- **Cuadro**: fotografías de una pintura con mesa y pared.  
- **Árboles**: conjunto propio tomado en la vía pública.  

---

## 📂 Estructura del Proyecto
- `tp1_pano.ipynb`: Notebook principal con el desarrollo del trabajo.
- `utils.py`: Funciones auxiliares.
- `visualization.py`: Funciones de visualización.
- `methods.py`: Implementación de los métodos principales.
- `requirements.txt`: Dependencias necesarias.
- `img/`: Carpeta con las imágenes de los datasets.
- `script.py`: Código para seleccionar puntos manualmente.

---

## 📊 Contenido del Trabajo

### 🔹 Detección y selección de puntos
Se detectaron keypoints con **SIFT** y se aplicó **ANMS** para obtener una distribución más uniforme.  

### 🔹 Emparejamiento
Se usó la regla de Lowe con verificación cruzada para descartar correspondencias falsas y quedarse con un conjunto confiable.  

### 🔹 Homografía
- **Manual (DLT)**: sobre 4 puntos seleccionados a mano.  
- **Automática (RANSAC)**: sobre matches filtrados, eliminando outliers.  

### 🔹 Warping y canvas óptimo
Se calcularon los límites mínimos del lienzo para evitar bordes innecesarios y recortes en las imágenes transformadas.  

### 🔹 Blending
Se aplicó un blending con **distance transform** para suavizar las uniones y mantener continuidad de color en el panorama final.  

### 🔹 Resultados
- En **Udesa** se obtuvo una panorámica continua, con buena alineación en techo, columnas y vegetación.  
- En **Cuadro**, pese a la dificultad de objetos 3D como la mesa, el cuadro plano se reconstruyó correctamente.  
- En **Árboles**, los keypoints se concentraron en ramas y hojas, generando una panorámica estable y natural.  

---

## ⚙️ Requisitos

Las principales dependencias se encuentran en `requirements.txt`:
- `opencv-python`
- `numpy`
- `matplotlib`

Instalación rápida:
```bash
pip install -r requirements.txt
