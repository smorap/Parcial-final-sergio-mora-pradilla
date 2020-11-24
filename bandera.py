#* bandera.py******************************************************************
#*                                                                            *
#*                         caracterizacion de banderas                        *
#*                                                                            *
#******************************************************************************

#******************************************************************************
#*                                                                            *
#*   Registro de Revisiones:                                                  *
#*                                                                            *
#*   FECHA           RESPONSABLE         REVISION                             *
#*   -------------------------------------------------------------------------*
#*   Noviemrbe 24/20    S. Mora Pradilla    Implementación inicial.           *
#******************************************************************************

#Librerias de opencv
import cv2
import numpy as np
import collections
from sklearn.cluster import KMeans
from sklearn.utils import shuffle

#Definicion de clase "colorImage"
class bandera:
    #atributos de la case colorimage
    def __init__(self, path):
        self.path = path                    #Direccion de ubicación de la imágen en el directorio
        self.img = cv2.imread(self.path)    #atributo donde yace la imágen

    def colores(self):
        image = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        image = np.array(image, dtype=np.float64) / 255
        rows, cols, ch = image.shape
        assert ch == 3
        image_array = np.reshape(image, (rows * cols, ch))
        image_array_sample = shuffle(image_array, random_state=0)[:10000]
        self.model = KMeans(n_clusters=4, random_state=0).fit(image_array_sample)
        self.labels = self.model.predict(image_array)
        self.centers = self.model.cluster_centers_
        colors=(max(self.labels)-min(self.labels))+1
        return colors

    def porcentajes(self):
        percent=[0,0,0,0]
        labels= self.model.labels_
        contador=collections.Counter(labels)
        tam=len (labels)
        for i in range(len(contador)):
            percent[i]=int((contador[i]/tam)*100)
        return percent

    def orientacion(self):
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        n=1
        SCALE = 1
        DELTA = 0
        DDEPTH = cv2.CV_16S

        grad_x = cv2.Sobel(gray, DDEPTH, 1, 0, ksize=3, scale=SCALE, delta=DELTA)
        grad_y = cv2.Sobel(gray, DDEPTH, 0, 1, ksize=3, scale=SCALE, delta=DELTA)

        grad_x = np.float32(grad_x)
        grad_x = grad_x * (1 / 512)
        grad_y = np.float32(grad_y)
        grad_y = grad_y * (1 / 512)

        # Gradient and smoothing
        grad_x2 = cv2.multiply(grad_x, grad_x)
        grad_y2 = cv2.multiply(grad_y, grad_y)
        grad_xy = cv2.multiply(grad_x, grad_y)
        g_x2 = cv2.blur(grad_x2, (n, n))
        g_y2 = cv2.blur(grad_y2, (n, n))
        g_xy = cv2.blur(grad_xy, (n, n))

        # Magnitude of the gradient
        Mag = np.sqrt(grad_x2 + grad_y2)
        M = cv2.blur(Mag, (n, n))

        # Gradient local aggregation
        vx = 2 * g_xy
        vy = g_x2 - g_y2
        fi = cv2.divide(vx, vy + np.finfo(float).eps)

        case1 = vy >= 0
        case2 = np.logical_and(vy < 0, vx >= 0)
        values1 = 0.5 * np.arctan(fi)
        values2 = 0.5 * (np.arctan(fi) + np.pi)
        values3 = 0.5 * (np.arctan(fi) - np.pi)
        theta = np.copy(values3)
        theta[case1] = values1[case1]
        theta[case2] = values2[case2]

        return theta
