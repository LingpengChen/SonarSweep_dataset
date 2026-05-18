#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np
import argparse
from typing import Tuple
import math
    
def padding_sonar_image(input_image, top_padding_pixels=20):
    """
    Add black padding to the top of a rectangular sonar image.

    Args:
        input_image: Input OpenCV image as a NumPy array.
        top_padding_pixels: Number of black rows to add at the top.

    Returns:
        Padded OpenCV image as a NumPy array.
    """
    top_padding = top_padding_pixels
    bottom_padding = 0
    left_padding = 0
    right_padding = 0
    padding_value = 0

    padded_image = cv2.copyMakeBorder(
        input_image, 
        top_padding, 
        bottom_padding, 
        left_padding, 
        right_padding, 
        cv2.BORDER_CONSTANT, 
        value=padding_value
    )
    
    return padded_image


def rect_to_sonar_map(rect_image: np.ndarray,
                      range_pixels: int = 1000, 
                      azimuth_pixels: int = 150,
                      azimuth_bounds: Tuple[float, float] = (-0.5236, 0.5236)) -> np.ndarray:
    """
    Map a rectangular sonar image into a Cartesian fan-shaped view.
    
    Args:
        rect_image: Input rectangular sonar image.
        range_pixels: Number of range samples.
        azimuth_pixels: Number of azimuth samples.
        azimuth_bounds: Azimuth range as (min_angle, max_angle), in radians.
        
    Returns:
        Cartesian sonar image.
    """
    azimuth_pixels = rect_image.shape[1]
    
    
    minus_width = math.floor(range_pixels * math.sin(azimuth_bounds[0]))
    plus_width = math.ceil(range_pixels * math.sin(azimuth_bounds[1]))
    width = plus_width - minus_width
    

    map_x = np.zeros((range_pixels, width), dtype=np.float32)
    map_y = np.zeros((range_pixels, width), dtype=np.float32)
    
    db = (azimuth_bounds[1] - azimuth_bounds[0]) / azimuth_pixels
    origin_x = abs(minus_width)
    
    for x in range(width):
        for y in range(range_pixels):
            dx = x - origin_x
            dy = y
            
            range_val = math.sqrt(dx * dx + dy * dy)
            azimuth = math.atan2(dx, dy)
            
            map_x[y, x] = (azimuth - azimuth_bounds[0]) / db
            map_y[y, x] = range_val
    
    sonar_image = cv2.remap(rect_image, map_x, map_y, cv2.INTER_CUBIC,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        
    return sonar_image


def process_images_in_folder(input_dir, output_dir):
    """
    Apply sonar padding to every image in a folder and save the results.

    Args:
        input_dir: Folder containing source images.
        output_dir: Folder for processed images.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    try:
        image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    except FileNotFoundError:
        print(f"Error: input directory '{input_dir}' does not exist.")
        return

    if not image_files:
        print(f"No image files found in '{input_dir}'.")
        return
        
    print(f"Found {len(image_files)} images in '{input_dir}'. Processing...")
    
    for i, filename in enumerate(sorted(image_files)):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        
        image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
        
        if image is None:
            print(f"\nWarning: could not read image {filename}. Skipping.")
            continue
            
        cv2.imshow("image original", image)
        padded_image = padding_sonar_image(image, top_padding_pixels=20)
        cv2.imshow("padded original", padded_image)
        sonar_image = rect_to_sonar_map(padded_image, 
                                        range_pixels= 1000, 
                                        azimuth_pixels = 150,
                                        azimuth_bounds = (-0.5236, 0.5236))
        cv2.imshow("sonar image", sonar_image)
        
        cv2.waitKey(0)

        cv2.imwrite(output_path, padded_image)
        
        print(f"\rProcessed: {i + 1}/{len(image_files)} - {filename}", end="")

    print(f"\n\nProcessing complete. Saved all images to '{output_dir}'.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Pad rectangular sonar images.")
    parser.add_argument("-input_dir", default="cp", help="Input image directory, e.g. tmp.")
    parser.add_argument("-o", "--output_dir", default="cp_padded", help="Output directory. Default: cp_padded.")

    args = parser.parse_args()

    process_images_in_folder(args.input_dir, args.output_dir)
