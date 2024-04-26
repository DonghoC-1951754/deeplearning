import mnist
import cv2
import os
import zipfile
import numpy as np
from enum import Enum

class ToolType(Enum):
    COMBWRENCH = 0
    HAMMER = 1
    SCREWDRIVER = 2
    WRENCH = 3


def iterate_folders_in_zip(zip_file_path):
    folders = []
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        for file_name in zip_ref.namelist():
            folder_name = os.path.dirname(file_name)
            if folder_name and folder_name not in folders:
                folders.append(folder_name)
    folders.reverse()
    return folders


def show_image(image):
    cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Image', 500, 500)
    cv2.imshow('Image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def load_data():
    dataset_dir = './dataset.zip'
    all_images = []
    all_labels = []
    folders = iterate_folders_in_zip(dataset_dir)
    for folder in folders:
        images = []
        with zipfile.ZipFile(dataset_dir, 'r') as zip_ref:
            for file_name in sorted(zip_ref.namelist()):
                if os.path.dirname(file_name) == folder:
                    with zip_ref.open(file_name) as file:
                        image_data = file.read()
                        image_array = np.frombuffer(image_data, np.uint8)
                        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
                        image = scale_image(image)
                        images.append(image)
                        all_labels.append(ToolType[folder.upper()].value)
        all_images.append(images)
    return all_images, all_labels


    # with zipfile.ZipFile(dataset_dir, 'r') as zip_ref:
    #     for file_name in sorted(zip_ref.namelist()):  # Sort the file names to maintain order
    #         with zip_ref.open(file_name) as file:
    #             image_data = file.read()
    #                 # image_data = zip_ref.read(file_info.filename)
    #             image_np = np.frombuffer(image_data, np.uint8)
    #             image_cv2 = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
    #             image_cv2 = scale_image(image_cv2)
    #             images.append(image_cv2)
    #             cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
    #             cv2.resizeWindow('Image', 500, 500)
    #             cv2.imshow('Image', image_cv2)
    #             cv2.waitKey(0)
    #             cv2.destroyAllWindows()

    print(f"Number of images loaded: {len(images)}")

def scale_image(image):
    # Define the new size (width, height)
    new_size = (500, 500)  # Adjust the size as needed
    # Downscale the image
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)

def turn_180(image):
    return cv2.rotate(image, cv2.ROTATE_180)

def flip_image_horizontal(image):
    return cv2.flip(image, 1)

def flip_image_vertical(image):
    return cv2.flip(image, 0)