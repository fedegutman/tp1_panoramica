import matplotlib.pyplot as plt
import cv2

# Cargar la imagen
img = cv2.imread("img/arbol_2.jpeg")
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Lista para guardar los puntos
points = []

def onclick(event):
    if event.xdata is not None and event.ydata is not None:
        x, y = event.xdata, event.ydata
        points.append((x, y))
        idx = len(points)  # número del punto
        print(f"Punto {idx}: ({x:.2f}, {y:.2f})")
        # Dibujar un punto rojo
        ax.scatter(x, y, c='r', s=50)
        # Dibujar el número al lado del punto
        ax.text(x+5, y-5, str(idx), color='yellow', fontsize=12, weight='bold')
        fig.canvas.draw()

# Crear figura y conectar el evento del mouse
fig, ax = plt.subplots()
ax.imshow(img_rgb)
cid = fig.canvas.mpl_connect('button_press_event', onclick)

plt.title("Click en 4 puntos de interés")
plt.show()

# Después de cerrar la ventana, los puntos estarán en la lista
print("Puntos seleccionados:", points)