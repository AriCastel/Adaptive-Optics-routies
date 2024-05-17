#Optimal Epsilon
import numpy as np
import matplotlib.pyplot as plt
import aotools
from UtilitySPIM2 import matriarch
import slmpy
from slmAberrationCorrection import make_now
from slmAberrationCorrection import iterate
from pycromanager import Bridge
import logging
import msvcrt
from tqdm import tqdm
logging.basicConfig(level=logging.DEBUG)
from slmAberrationCorrection import adaptiveOpt

distroSamples = 100
metricTolerance = 3
degree = 21

# Y,X coordinates
slmShape = (1154,1920)
fouriershape = (1000,1000)
centerpoint = (660,860)
stretch = 1.01

#Connect to the SLM
logging.info("Connecting to Microscope")
bridge = Bridge()
core = bridge.get_core()
slm = slmpy.SLMdisplay(monitor = 1)
display = np.zeros(slmShape)
slm.updateArray(display.astype('uint8'))

print("Press Any key to confirm Depth")
msvcrt.getch()
print("Depth Confirmed")

logging.info("Taking Images")
metrics = [] 
#Distribution Sampling Loop
for i in tqdm(range(distroSamples), desc="Sampling", unit='sample'):
    image_i = adaptiveOpt.get_guidestar(core,graph=False)
    m_i = adaptiveOpt.metric_better_r(image_i)
    metrics.append()

#STD Calculation    
logging.info("Calculating Distribution") 
Metrics = np.array(metrics)
sigma = np.std(Metrics)

print(f"Standard Deviation is sigma={sigma}")

#Optimal Epsilon Iteration 
diff = 0

Zernikes= aotools.zernike.zernikeArray(degree,fouriershape[0])
N = len(Zernikes)
array =  (int(N)*[0])
a_t = np.array(array)
C_t= (make_now.random_signs(len(a_t)))
epsilon = 0.1

criteria = sigma*metricTolerance
logging.info("Entering the Iteration Cycle") 
with tqdm(total=criteria) as pbar:
    while diff < criteria:
        
        D_t = epsilon*C_t
        a_plus = a_t + D_t
        a_minus = a_t - D_t
        
        #Creates phase masks 
        phaseMask_p = make_now.zernike_optimized(Zernikes, a_plus, stretch, slmShape, centerpoint)
        phaseMask_m = make_now.zernike_optimized(Zernikes, a_minus, stretch, slmShape, centerpoint)

        #takes phase masked images
        slm.updateArray(phaseMask_p.astype('uint8'))
        guideStar_p = adaptiveOpt.get_guidestar(core)
        slm.updateArray(phaseMask_m.astype('uint8'))
        guideStar_m = adaptiveOpt.get_guidestar(core)
            
        #Evaluates the metric for each phase mask image
        metric_p = adaptiveOpt.metric_better_r(guideStar_p)
        metric_m = adaptiveOpt.metric_better_r(guideStar_m)
        diff = metric_p - metric_m
        logging.debug(f"Metric Difference ={diff}")
        
        epsilon = epsilon + 0.1

logging.info("Iterations finished") 
print(f"optimal Epsilon value Found E={epsilon}")

    