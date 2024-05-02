"""
Corrects aberrations using Miguels algorithm 
"""
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import aotools
from UtilitySPIM2 import matriarch
from adaptOptFunct import adaptiveOpt
from ephestus import make_now
from tqdm import tqdm
import time
from pycromanager import Bridge
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def countdown(seconds):
    for i in range(seconds, 0, -1):
        print(i)
        time.sleep(1)
    print("Begginning")

logging.info("Initializing")
#SLM Calibration Variables 
slmShape = (1154,1900)
fouriershape = (900,900)
centerpoint = (700,800)
stretch = 1.01

#Zernike Constants
degree = 12
Zernikes= aotools.zernike.zernikeArray(degree,fouriershape[0])
N = len(Zernikes)

#Iteration Constants
array =  (int(N)*[0])
g_0 = 0.3
epsilon = 0.01 #Medida heurística de perturbación en la máscara de fase
totalIterations = 500

#Microscope Connection
logging.info("Accessing Microscope")
slm = slmpy.SLMdisplay(monitor = 1)
bridge = Bridge()
core = bridge.get_core()

#Iteration Generator
logging.info("Preadquisition")

#initializes the optimization variables
startimage = adaptiveOpt.get_guidestar(core)
M0 = adaptiveOpt.metric_r_power_integral(startimage)
metric_t = M0
a_dash_t = np.array(array)  

#saves some data
metrics = []
polynomialSeries = []

countdown(5)

#main Iteration Cycle
logging.info("Iteration Begins")
iteration = 1 

for i in tqdm(range(totalIterations), desc="Optimizing", unit='iteration'): 

    #Disturbs the polynomial series Terms
    C_t= (make_now.random_signs(len(a_dash_t)))
    D_t = epsilon*C_t
    a_plus = a_dash_t + D_t
    a_minus = a_dash_t - D_t

    #Creates phase masks 
    phaseMask_p = make_now.zernike_phase_mask(a_plus, fouriershape[0], stretch, slmShape, centerpoint)
    phaseMask_m = make_now.zernike_phase_mask(a_minus, fouriershape[0], stretch, slmShape, centerpoint)

    #takes phase masked images
    slm.updateArray(phaseMask_p.astype('uint8'))
    guideStar_p = adaptiveOpt.get_guidestar(core)

    slm.updateArray(phaseMask_m.astype('uint8'))
    guideStar_m = adaptiveOpt.get_guidestar(core)
    
    #Evaluates the metric for each phase mask image
    metric_p = adaptiveOpt.metric_r_power_integral(guideStar_p)
    metric_m = adaptiveOpt.metric_r_power_integral(guideStar_m)
    
    diff = metric_p - metric_m
    logging.debug(f"Metric Difference ={diff}")
    
    #Creates the iteration's Phase mask
    phaseMask_t = make_now.zernike_phase_mask(a_dash_t, fouriershape[0], stretch, slmShape, centerpoint)
    slm.updateArray(phaseMask_t.astype('uint8'))
    guideStar_t = adaptiveOpt.get_guidestar(core)
    metric_t = adaptiveOpt.metric_r_power_integral(guideStar_t)
    
    metrics.append(metric_t)
    polynomialSeries.append(a_dash_t)
    
    #calculates the terms for the next iteration
    g_t = g_0*M0/metric_t
    a_dash_t  = a_dash_t - g_t*diff*D_t
    
    iteration +=1
    
    logging.debug(f"Iteration {i}")
    
logging.info("Iterations Finished :3")

#Plot the metric progression    
logging.info("Plotting Results")
plt.figure(figsize=(10, 5))

plt.plot(np.arange(0,totalIterations,len(metrics)), metrics, color='#240046', linewidth=2, linestyle='-')
plt.scatter(np.arange(0,totalIterations,len(metrics)), metrics, marker='d', color='#B40424')
plt.title('Metric progression',fontsize=20)
plt.grid(True, color='#E1E2EF')  # Add grid lines
plt.xlabel('Iteration', fontsize=18)
plt.ylabel('Metric Value', fontsize=18)
