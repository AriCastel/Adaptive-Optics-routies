#FinalAlignementTest

import numpy as np
import matplotlib.pyplot as plt
import aotools
from UtilitySPIM2 import matriarch

# Y,X coordinates
slmShape = (1154,1900)
fouriershape = (900,900)
centerpoint = (700,800)
stretch = 1.01




ogMask = np.full(fouriershape, 128) - aotools.circle(450,900)*128

angledMask = matriarch.stretch_image(ogMask, stretch)

display = np.full(slmShape,128)

phaseMask = matriarch.frame_image(display, angledMask, centerpoint)

plt.imshow(phaseMask)
plt.show()