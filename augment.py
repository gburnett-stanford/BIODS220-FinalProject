# This script converts image data from the HANDS dataset Raw Data folder
# into a format useable by TensorFlow 

import os
import glob
import shutil
import argparse
import numpy as np
import pandas as pd 
from PIL import Image, ImageFilter

def main(input_folder, output_folder):

    for image_name in os.listdir(input_folder): 
        
        # Load the original image 
        image_basename = image_name.split('_')[0]
        input_image_path = os.path.join(input_folder, image_name)
        original_image = Image.open(input_image_path)

        # Randomly blur the image once for each participant 
        for i in range(1, 12): 

            # Blur the image 
            blur_size = int(np.random.random() * 10)
            blurred = original_image.filter(ImageFilter.BoxBlur(blur_size))

            # Save the new image 
            output_image_path = os.path.join(output_folder, image_basename+'_'+str(i)+'.jpg')
            blurred.save(output_image_path)
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--input_folder', type=str, help='input folder of images', required=True)
    parser.add_argument('--output_folder', type=str, help='output folder of images', required=True)

    args = parser.parse_args()

    main(args.input_folder, args.output_folder)