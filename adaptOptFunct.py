"""_summary_
Aberration Correction Functions
"""
import numpy as np
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
import aotools
import matplotlib.pyplot as plt
import scipy
import math

class adaptiveOpt:
    
    #power Integral Metric
    def metric_r_power_integral(img, integration_radius=20, power=2):
        """Metric of PSF quality based on integration of image(r) x r^2 over a circle of defined radius. 
        From Vorontsov, Shmalgausen, 1985 book. For best accuracy, img dimensions should be odd, with peak at the center.
        Parameters:
            img (array):  a 2D image with PSF peak at the center
            integration_radius (int) = for the circle of integration, default 20.
            power (int) = the power radious
        returns: 
            float
        """
        img = (img-img.min())/(img.max()-img.min())
        h, w = img.shape[0], img.shape[1]
        if np.min(img.shape) < 2 * integration_radius:
            raise ValueError("Radius too large for image size")
        else:
            cmass = scipy.ndimage.measurements.center_of_mass(img)
            x_center, y_center = cmass[1], cmass[0]
            y, x = np.ogrid[:h, :w]
            dist_from_center = np.sqrt((x - x_center) ** 2 + (y - y_center) ** 2)
            mask = (dist_from_center <= integration_radius).astype(int)
            metric = np.sum(img * mask * (dist_from_center ** power)) 
        return metric
    
    def metric_better_r(img, integration_radius=30):
        cmass = scipy.ndimage.measurements.center_of_mass(img)
        h, w = img.shape[0], img.shape[1]
        x_center, y_center = cmass[1], cmass[0]
        y, x = np.ogrid[:h, :w]
        radious = np.sqrt((x - x_center) ** 2 + (y - y_center) ** 2)
        mask = (radious <= integration_radius).astype(int)
        metric = np.sum(img*mask*radious)/np.sum(img)

        return metric
    
    def extract_centered_matrix(matrix, center_coord, size):
        """
        Extract a smaller matrix centered around a specific coordinate from a larger matrix.

        Parameters:
            matrix (numpy.ndarray): Input matrix.
            center_coord (tuple): Coordinates (row, column) of the center.
            size (tuple): Size of the extracted matrix (rows, columns).

        Returns:
            numpy.ndarray: Extracted smaller matrix.
        """
        # Unpack center coordinates
        center_row, center_col = center_coord
        
        # Unpack size of the extracted matrix
        rows, cols = size
        
        # Calculate starting and ending indices for the extraction
        start_row = max(0, center_row - rows // 2)
        end_row = min(matrix.shape[0], center_row + (rows - rows // 2))
        start_col = max(0, center_col - cols // 2)
        end_col = min(matrix.shape[1], center_col + (cols - cols // 2))
        
        # Extract the smaller matrix
        smaller_matrix = matrix[start_row:end_row, start_col:end_col]
        
        return smaller_matrix

    def threshold_matrix(matrix, percentile=90):
        """
        Thresholds the values in a matrix below a certain percentile to 0.

        Parameters:
            matrix (numpy.ndarray): Input matrix.
            percentile (float): Percentile value below which to threshold.

        Returns:
            numpy.ndarray: Thresholded matrix.
        """
        # Calculate the threshold value based on the percentile
        threshold_value = np.percentile(matrix, percentile)
        
        # Threshold the matrix
        thresholded_matrix = np.where(matrix < threshold_value, 0, matrix)
        result = (thresholded_matrix - thresholded_matrix.min)*(100/thresholded_matrix.max)
        
        return result
    
    def center_of_mass(image):
        """
        Calculate the center of mass of an image matrix.

        Parameters:
            image (numpy.ndarray): Input image matrix.

        Returns:
            tuple: Coordinates (row, column) of the center of mass.
        """
        # Create grid of coordinates
        rows, cols = np.indices(image.shape)
        
        # Calculate total intensity
        total_intensity = np.sum(image)
        
        # Calculate center of mass coordinates
        center_row = np.sum(rows * image) / total_intensity
        center_col = np.sum(cols * image) / total_intensity
        
        return int(center_row), int(center_col)
    
    def get_guidestar(core, size=(101,101), graph=False):
        """Takes a photo of the sample in the microscope and returns a small matrix containing
        the guide star. Function only usable when no more than 1 fluorescent sample is visible in the microscope
        Args:
            core (pycromanagerCore): the pycromanager microscope's core object
            size (tuple, optional): size of the wanted gudiestar image. Defaults to (101,101).
            graph (bool, optional): wheter or not to show a preview of the guidestar image. Defaults to False.

        Returns:
            Numpy Matrix: Matrix image of the guidestar
        """
        
        core.snap_image()
        tagged_image = core.get_tagged_image()
        imageH = tagged_image.tags['Height']
        imageW = tagged_image.tags['Width']
        image = tagged_image.pix.reshape((imageH,imageW))

        threshStar = adaptiveOpt.threshold_matrix(image)
        centerOfMass = scipy.ndimage.measurements.center_of_mass(threshStar)
        
        guideStar = adaptiveOpt.extract_centered_matrix(threshStar,centerOfMass,size)
        if graph:
            plt.imshow(guideStar) 
            plt.colorbar
            plt.show()
        
        return guideStar 
        
            
    