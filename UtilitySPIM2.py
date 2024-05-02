########
"""
Utility Functions for SPIM2 related code 
Code that doesn't explicitly relate to the SLM or .CORE 
functions should go here. 
Artemis Castelblanco
Versión: 1.3.20240421
"""
#######
import logging
import numpy as np
import os
import cv2
import aotools
import matplotlib.pyplot as plt
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

######
"""
Matrix Related Functions
"""
######    
class matriarch():
    
    def truncate(matrix, threshold):
        """
        Truncates matrix values after a certain threshold.

        Args:
        - matrix (list of lists): The input matrix
        - threshold (float): The threshold value

        Returns:
        - truncated_matrix (list of lists): The matrix with values truncated after the threshold
        """
        truncated_matrix = [[min(value, threshold) for value in row] for row in matrix]
        return truncated_matrix
    
    
    
    def stretch_image(image, stretch_factor, axis='x'):
        """
        Stretch an image array along a given axis by a stretch factor.
        
        Parameters:
            image (numpy.ndarray): Input image array.
            stretch_factor (float): Factor by which to stretch the image.
            axis (str): Axis along which to stretch the image ('x' or 'y').
        
        Returns:
            numpy.ndarray: Stretched image array.
        """
        if axis == 'x':
            new_width = int(image.shape[1] * stretch_factor)
            stretched_image = cv2.resize(image, (new_width, image.shape[0]))
        elif axis == 'y':
            new_height = int(image.shape[0] * stretch_factor)
            stretched_image = cv2.resize(image, (image.shape[1], new_height))
        else:
            raise ValueError("Invalid axis. Please use 'x' or 'y'.")

        return stretched_image
    
    def frame_image (frame, image, center_point):
        """
        places the values of the image matrix inside the frame matrix with 
        center_point as the center point.
        Stable only where frame can contain image

        Parameters:
            frame (numpy.ndarray): Larger matrix to be modified.
            image (numpy.ndarray): Smaller matrix whose values will be placed into the larger matrix.
            center_point (tuple): Coordinates (row, column) specifying the center point.

        Returns:
            numpy.ndarray: Modified frame matrix.
        """
        smaller_rows, smaller_cols = image.shape
        center_row, center_col = center_point
        start_row = max(center_row - smaller_rows // 2, 0)
        end_row = min(start_row + smaller_rows, frame.shape[0])
        start_col = max(center_col - smaller_cols // 2, 0)
        end_col = min(start_col + smaller_cols, frame.shape[1])

        frame[start_row:end_row, start_col:end_col] = image[:end_row-start_row, :end_col-start_col]

        return frame

    def generate_torus(shape,innerRad, outerRad):
        """
        Generates a Toroidal shape
        Parameters:
            shape (tuple): size of the array that will contain the torus 
            innerRad (int): inner radious of the torus 
            outerRad (int): outer radious of the torus
            graph (bool): Whether or not to plot a graph of the torus matrix
            
        Returns:
            numpy.ndarray: matrix with the torus.
        """
        matrix = np.zeros(shape) + aotools.circle(outerRad,shape[0]) - aotools.circle(innerRad,shape[0])
        
        return matrix        
        
    def generate_curtain(shape, split_column, ):
        matrix = np.zeros(shape)  # Create a matrix of zeros
        matrix[:, :split_column] = 1     # Set columns before split_column to 1
        
        maxmat =matrix.max()
        if maxmat > 0: 
            alpha = (np.pi)/maxmat
        else: 
            alpha = np.pi
        matrixinPi = matrix*alpha
        
    
        return matrix

######
"""
File management related Functions
"""
######  
class librarian: 
    def save_data():
        #TODO Make a function that saves the files in the standard format
        pass
    
    
    
    
    def save_graph(plot, filename):
        """
        Save a matplotlib plot as a PDF file in the parent folder of the Python file.
        
        Parameters:
            plot (matplotlib.pyplot plot): The plot to save.
            filename (str): The filename (without extension) to save the plot as.
        """
        parent_folder = os.path.dirname(os.path.abspath(__file__))
        
        
        pass
    