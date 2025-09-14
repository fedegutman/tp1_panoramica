import matplotlib.pyplot as plt
import cv2

def onclick(event):
    if event.xdata is not None and event.ydata is not None:
        x, y = event.xdata, event.ydata
        points.append((x, y))
        index = len(points)
        print(f'Punto {index}: ({x:.2f}, {y:.2f})')
        ax.scatter(x, y, c='r', s=50) # punto rojo
        ax.text(x+5, y-5, str(index), color='yellow', fontsize=12, weight='bold') # numero en el punto
        fig.canvas.draw()

img = cv2.imread("img/arbol_1.jpeg")
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

points = [] # esto para ir guardanddo 

fig, ax = plt.subplots()
ax.imshow(img_rgb)
cid = fig.canvas.mpl_connect('button_press_event', onclick)

plt.show()