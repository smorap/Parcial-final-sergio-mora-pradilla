#* main.py *****************************************************************
#*                                                                         *
#*                               Parcial 1                                  *
#*                    Proc. de Imágenes y Visión-PUJ                       *
#***************************************************************************

#***************************************************************************
#*                                                                         *
#*   FECHA           RESPONSABLE           OBSERVACION                     *
#*   ----------------------------------------------------------------------*
#*   Noviembre 24/20    S. Mora Pradilla      Implementación inicial.      *
#***************************************************************************
import cv2
from bandera import *         #llamado a la clase bandera

#imprime solicitud de dirección de archivo
path=input("Inserte la dirección de ubicación de la imágen con el nombre y su extensiónal final")
#D:\Datos\sergio\UNIVERSIDAD\2020\Proc_ Imagens\Final\Imagenes\flag1.png

#Constructor
Banderita = bandera(path)        #contructor de la imágen
image=cv2.imread(Banderita.path)
#punto1
color=Banderita.colores()
print('La cantidad de colores en la imagen es: ',color)
#punto2
porcentajes=Banderita.porcentajes()
print('El porcentaje de cada uno de esos colores es: ',porcentajes)
#punto3
