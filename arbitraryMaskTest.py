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
fouriershape = (1000,1000)
centerpoint = (660,860)
stretch = 1.01
degree, g_0, epsilon, totalIterations = 21, 0.05, 0.1, 50

#HelixPhaseMask
ogMask = make_now.generate_corkscrew_optimized(int(fouriershape[0]/2))
angledMask = matriarch.stretch_image(ogMask, stretch)
display = np.zeros(slmShape)
DoubleHelixphaseMask = matriarch.frame_image(display, angledMask, centerpoint)

#Adaptive Optics
logging.info("Connecting to SLM")
slm = slmpy.SLMdisplay(monitor = 1)

logging.info("Preparing Phase Mask correction")
CorrectionPhasemask, iterations, metrics = iterate(slmShape, fouriershape, centerpoint, stretch, degree, g_0, epsilon, totalIterations, slm, preview=True)

#Adaptive Optics and Double Helix
logging.info("Preparing Phase Mask correction")
completeMask = DoubleHelixphaseMask + CorrectionPhasemask
completeMask = completeMask - completeMask.min()
completeMask = completeMask%255
slm.updateArray(completeMask.astype('uint8'))

logging.info("Displaying phase mask")
plt.imshow(DoubleHelixphaseMask)
plt.show()

