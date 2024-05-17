########
"""
Utility Functions for SPIM2 related code 
Code that doesn't explicitly relate to the SLM or .CORE functions should go here. 
Artemis Castelblanco
Version: 3.0.20240517
"""
#######
import logging
import numpy as np
import os
import cv2
import aotools
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd
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

    def generate_torus(shape,innerRad, outerRad, factor=127):
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
        shape=(int(outerRad*2), int(outerRad*2))

        matrix = (np.zeros(shape) + aotools.circle(outerRad,shape[0]) - aotools.circle(innerRad,shape[0]))*factor
        
        return matrix        
        
    def generate_curtain(shape, split_column, value=1 ):
        """Generates a Curtain phase mask

        Args:
            shape (tuple): Shape of the curtain phase mask
            split_column (int): column where the curtain stops 

        Returns:
            ndarray: phase mask containing curtain
        """
        
        matrix = np.zeros(shape)  # Create a matrix of zeros
        matrix[:, :split_column] = value     # Set columns before split_column to 1
    
        return matrix

######
"""
File management related Functions
"""
######  
class librarian: 
    def save_data(file, name, foldername, extension='csv'):
        pandaValued = pd.DataFrame(file)
        current_datetime = datetime.now()
        folder_name = f"{current_datetime.strftime('%Y-%m-%d')}_{foldername}"
        os.makedirs(folder_name, exist_ok=True)
        filename = f"{name}.{extension}"
        file_path = os.path.join(folder_name, filename)
        pandaValued.to_csv(file_path, index = False)
        pass
    
    def save_graph(file, name, foldername, extension='pdf'):
        """
        Save a matplotlib plot as a PDF file in the parent folder of the Python file.
        
        Parameters:
            plot (matplotlib.pyplot plot): The plot to save.
            filename (str): The filename (without extension) to save the plot as.
        """
        current_datetime = datetime.now()
        folder_name = f"{current_datetime.strftime('%Y-%m-%d')}_{foldername}"
        os.makedirs(folder_name, exist_ok=True)
        filename = f"{name}.{extension}"
        file_path = os.path.join(folder_name, filename)
        file.save(file_path)
        pass
    