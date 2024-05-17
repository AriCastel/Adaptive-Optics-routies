#Autofoco SPIM 2
#Autorxs: Cristhian Perdomo, Luis Bastidas, Alex Artemis Castelblanco
#cd.perdomo10@uniandes.edu.co
#l.bastidash@uniandes.edu.co
#Versión: 2024/02/29

#Aqui importar las funciones necesitadas de FuncionesSPIM


import matplotlib.pyplot as plt
from skimage import io
import numpy as np
import pytic
from time import sleep

from random import randint
from pycromanager import Bridge
from pycromanager import Acquisition, multi_d_acquisition_events
import numpy as np
import time
#import napari

import sys

from datetime import datetime
import os

import cv2 as cv

from PIL import Image, ImageSequence

'''
You may notice that the function names are have changed slightly from the example above to the onces listed here. 
Specifically, "snapImage" was called as "snap_image". This is because Pycro-Manager automatically converts from the 
Java convention of "functionsNamesLikeThis()" to the Python convention of "functions_named_like_this()". It is possible to 
change this behavior when creating the bridge with Bridge(convert_camel_case=False)
'''

bridge = Bridge()
core = bridge.get_core() #No borrar!!

#'''
properties = core.get_device_property_names("HamamatsuHam_DCAM")
for i in range (properties.size() ): 
    prop = properties.get(i)
    val = core.get_property("HamamatsuHam_DCAM", prop)
    print ("Name: " + prop + ", value: " + val)
#'''

'''
COM5
COM1
HamamatsuHam_DCAM
Sutter MPC
Sutter MPC Z stage
Sutter MPC XY stage
Oxxius LaserBoxx LBX or LMX or LCX
GenericSLM
Core
'''
#FinalAlignementTest

import numpy as np
import matplotlib.pyplot as plt
import aotools
from UtilitySPIM2 import matriarch
import slmpy
from slmAberrationCorrection import make_now
from slmAberrationCorrection import iterate

import logging
logging.basicConfig(level=logging.DEBUG)


# Y,X coordinates
slmShape = (1154,1920)
fouriershape = (900,900)
centerpoint = (660,860)
stretch = 1.01
degree, g_0, epsilon, totalIterations = 21, 0.1, 0.5, 200


#CorrectionPhasemask, iterations, metrics = iterate(slmShape, fouriershape, centerpoint, stretch, degree, g_0, epsilon, totalIterations, preview=True)


ogMask = make_now.generate_corkscrew_optimized(int(fouriershape[0]/2))


angledMask = matriarch.stretch_image(ogMask, stretch)

display = np.zeros(slmShape)

DoubleHelixphaseMask = matriarch.frame_image(display, angledMask, centerpoint)

#completeMask = DoubleHelixphaseMask + CorrectionPhasemask

#completeMask = completeMask - completeMask.min()

#completeMask = completeMask%255


slm = slmpy.SLMdisplay(monitor = 1)
slm.updateArray(DoubleHelixphaseMask.astype('uint8'))

plt.imshow(DoubleHelixphaseMask)
plt.show()

core.set_property("Sutter MPC", 'Step Size', '1')

def InitPos (): #lee la posición del micromanipulador y la devuelve

    x_pos = core.get_property('Sutter MPC', 'Current X')
    y_pos = core.get_property('Sutter MPC', 'Current Y')
    z_pos = core.get_property('Sutter MPC', 'Current Z')
    return float(x_pos), float(y_pos),float(z_pos)

#TODO poner comentario no  cutre 
def GoToX (Posicion: float): #Mueve la posicion de la microcrontrolacion 
    core.set_property('Sutter MPC', 'Current X', str(Posicion))


def DepthOfField(fileName: str, folderPath: str):
    #Posición inicial de la muestra
    xI, yI, zI =   InitPos() 

    stepSize =  1 #Cambio para el micromanipulador, 1micra
    #Posicion final
    xF=10 
    #Array que tendra las posiciones predeterminadas para las metricas
  
    x1= np.arange(xI-xF, xI, stepSize)
    x2=  np.arange(xI, xI+xF+stepSize, stepSize)
    x= np.concatenate((x1,x2))
    
    GoToX(xI-20)
    sleep(1)
    #Recorrido de las posiciones
    for i in range(x.size):
        
        #Mueve el micromanipulador y el objetivo de iluminacion
        GoToX(x[i])
    
        #Se hace la toma de datos y se recibe el nombre del archivo de la imagen
        folderPath.replace("\\" , '/') 

        core.snap_image()
        tagged_image = core.get_tagged_image()
        #If using micro-manager multi-camera adapter, use core.getTaggedImage(i), where i is the camera index
        #pixels by default come out as a 1D array. We can reshape them into an image
        pixels = Image.fromarray( np.reshape(tagged_image.pix,newshape=[tagged_image.tags['Height'], tagged_image.tags['Width']]) , mode='I;16'  )
        pixels.save((folderPath + '/'+fileName+str(i)+'.tiff'), "TIFF")  
            
        print(i)
        sleep(0.3)
        #Una vez acaba el recorrido, volver a la posicion inicial
        
filename= input("Coloca el nombre del archivo: ")
folderPath= input("Coloca la ubicación para el archivo: ")
DepthOfField(filename, folderPath)