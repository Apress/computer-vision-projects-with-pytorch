
import os
import logging
import random
import shutil


logger = logging.getLogger(__name__)
logging.basicConfig(format="[ %(levelname)s ] %(message)s", level=logging.INFO)

def main():
    r""" Train and test """
    image_list = os.listdir(os.path.join("train", "input"))

    test_img_list = random.sample(image_list,
                                     int(len(image_list) / 10))

    ## Iterating through the test files

    for test_img_file in test_img_list:
        filename = os.path.join("train", "input", test_img_file)
        logger.info(f"Process: `{filename}`.")

        shutil.move(os.path.join("train", "input", test_img_file),
                    os.path.join("test", "input", test_img_file))
        shutil.move(os.path.join("train", "target", test_img_file),
                    os.path.join("test", "target", test_img_file))


if __name__ == "__main__":
    logger.info("ScriptEngine:")
    logger.info("\tversion ..... 0.1.0")
    logger.info("\tBuild Success")

    main()
