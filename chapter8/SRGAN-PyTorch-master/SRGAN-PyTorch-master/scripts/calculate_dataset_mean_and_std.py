
import os
import cv2
import argparse
import logging
import numpy as np

## adding argument parser - command line data directory check
parser = argparse.ArgumentParser()
parser.add_argument("data", metavar="DIR", help="dataset dir")
args = parser.parse_args()
logger = logging.getLogger(__name__)
logging.basicConfig(format="[ %(levelname)s ] %(message)s", level=logging.DEBUG)

def main():
    
    data_path = os.path.abspath(args.data)
    # Initialize all 3 dimensions - input image
    mean = [0., 0., 0.]
    std = [0., 0., 0.]
    image_count = 0

    for data_dir in os.listdir(data_path):
        
        image_dir = os.path.join(data_path, data_dir)

        for file in os.listdir(image_dir):
            ## going through each file instances

            file_dir = os.path.join(image_dir, file)
            image = cv2.imread(file_dir)
            image = image.astype(np.float32)
            image =  image / 255.

            for i in range(3):
                mean[i] += image[:, :, i].mean()
                std[i] += image[:, :, i].std()

            image_count += 1


    mean.reverse()
    std.reverse()
    mean = np.asarray(mean) / image_count
    std = np.asarray(std) / image_count

if __name__ == "__main__":
    logger.info("Script:")
    logger.info("\t version .......... 0.1.0")
    logger.info("\tBuild Success")

    main()
