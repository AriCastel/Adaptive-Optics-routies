"""
PhaseMask from Zernike Polynomials
By Artemis the Lynx
Ver 0.0.20240429
"""
import numpy as np
import matplotlib.pyplot as plt
import logging
import aotools
import random
from UtilitySPIM2 import matriarch
#Logging Config
logging.basicConfig(level=logging.INFO)

class make_now:
    
    def random_signs(length):
        # Initialize an empty list
        array = np.zeros(length, dtype=int)  # Create an array of zeros with the specified length
        for i in range(length):
            array[i] = random.choice([-1, 1])  # Set each element to either -1 or 1 randomly
        return array
    
    def zernike_optimized (Z_v, V_coef, strt, S_sz, C_cntr, norma=True, preview = False, maxAdmisible = 255, safeguard = 'truncating'):
        for i in range(len(Z_v)):
            Z_v[i]=Z_v[i]*V_coef[i]

        Z_i = np.zeros((Z_v.shape[1],Z_v.shape[2]))
        for i in range(len(Z_v)):
            Z_i = Z_i + Z_v[i]      
             
        
        Z_strt = matriarch.stretch_image(Z_i, strt)
        C_cnvas = np.zeros(S_sz)
        Z_result = matriarch.frame_image(C_cnvas, Z_strt, C_cntr )

        if norma: 
            Z_result = Z_result - Z_result.min()
        
        if Z_result.max() > maxAdmisible:
            logging.warning(f"Maximum value {Z_result.max()} stretches over the admissible limit {maxAdmisible}")
            if safeguard == 'resize': 
                logging.warning(f"Rezising the Matrix")
                Z_result = Z_result*(maxAdmisible/Z_result.max())
            elif safeguard == 'fresnel': 
                logging.warning(f"Fresnel Lens yet to be implemented, Matrix left as it is")
            else: 
                logging.warning(f"Truncating the Matrix to {maxAdmisible}")
                Z_result = matriarch.truncate(Z_result, maxAdmisible)
        
        if preview: 
            plt.imshow(Z_result)
            plt.show()
        
        return Z_result
    
    def zernike_phase_mask (V_coef, dmtr, strt, S_sz, C_cntr, norma=True, preview = False, maxAdmisible = 255, safeguard = 'truncating'):
        Z_i = aotools.functions.zernike.phaseFromZernikes(V_coef, dmtr)
        Z_strt = matriarch.stretch_image(Z_i, strt)
        C_cnvas = np.zeros(S_sz)
        Z_result = matriarch.frame_image(C_cnvas, Z_strt, C_cntr )
        
        if norma: 
            Z_result = Z_result - Z_result.min()
        
        if Z_result.max() > maxAdmisible:
            logging.warning(f"Maximum value {Z_result.max()} stretches over the admissible limit {maxAdmisible}")
            if safeguard == 'resize': 
                logging.warning(f"Rezising the Matrix")
                Z_result = Z_result*(maxAdmisible/Z_result.max())
            elif safeguard == 'fresnel': 
                logging.warning(f"Fresnel Lens yet to be implemented, Matrix left as it is")
            else: 
                logging.warning(f"Truncating the Matrix to {maxAdmisible}")
                Z_result = matriarch.truncate(Z_result, maxAdmisible)
        
        if preview: 
            plt.imshow(Z_result)
            plt.show()
        
        return Z_result
        
    pass