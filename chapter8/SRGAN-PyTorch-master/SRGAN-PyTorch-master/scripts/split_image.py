import os
## pillow lib
from PIL import Image

def main():
    img_sizex0  = 128
    img_sizex2  = 64
    img_sizex4  = 32

    file_names = os.listdir("HR")
    for file_name in sorted(file_names):
        print(f"Process: `{file_name}`.")

        imgx0 = Image.open(os.path.join("HR",          file_name) )
        imgx2 = Image.open(os.path.join("LRunknownx2", file_name) )
        imgx4 = Image.open(os.path.join("LRunknownx4", file_name) )

        crop_imgx0 = crop_image(imgx0, img_sizex0)
        crop_imgx2 = crop_image(imgx2, img_sizex2)
        crop_imgx4 = crop_image(imgx4, img_sizex4)
        save_images(os.path.join("HR",          file_name), crop_imgx0)
        save_images(os.path.join("LRunknownx2", file_name), crop_imgx2)
        save_images(os.path.join("LRunknownx4", file_name), crop_imgx4)
        os.remove(os.path.join("HR",          file_name) )
        os.remove(os.path.join("LRunknownx2", file_name) )
        os.remove(os.path.join("LRunknownx4", file_name) )

## crop image - Augmentation
def crop_image(img, crop_sizes: int):
    assert img.size[0] == img.size[1]
    crop_num = img.size[0] // crop_sizes

    box_list = []
    for width_index in range(0, crop_num):
        for height_index in range(0, crop_num):
            box_info = ( (height_index + 0)*crop_sizes,(width_index + 0) * crop_sizes,
                   (height_index + 1)*crop_sizes,(width_index + 1) * crop_sizes)
            box_list.append(box_info)


    cropped_images = [img.crop(box_info) for box_info in box_list]
    return cropped_images


def save_images(raw_file_name, image_list):
    index = 1
    for image in image_list:
        image.save(raw_file_name.split(".")[0] + f"_{index:08d}.bmp")
        index += 1


if __name__ == "__main__":
    main()
