import mnist
import cv2
import os
import zipfile
import numpy as np
import random
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

# def load_data():
#     dataset_dir = './dataset.zip'
#     images = []
#     with zipfile.ZipFile(dataset_dir, 'r') as zip_ref:
#         for file_name in sorted(zip_ref.namelist()):  # Sort the file names to maintain order
#             with zip_ref.open(file_name) as file:
#                 image_data = file.read()
#                 # image_data = zip_ref.read(file_info.filename)
#                 image_np = np.frombuffer(image_data, np.uint8)
#                 image_cv2 = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
#                 image_cv2 = scale_image(image_cv2)
#                 images.append(image_cv2)
#                 cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
#                 cv2.resizeWindow('Image', 1000, 1000)
#                 #cv2.imshow('Image', image_cv2)
#                 cv2.imshow('Image', change_saturation(image_cv2))
#                 cv2.waitKey(0)
#                 cv2.destroyAllWindows()
#
#     print(f"Number of images loaded: {len(images)}")
def load_data(test_percent = 0.2, validate_percent = 0):
    dataset_dir = './dataset.zip'
    # list to store images
    test_images = []
    train_images = []
    validate_images = []
    # list to store labels for each image
    test_labels = []
    train_labels = []
    validate_labels = []

    folders = iterate_folders_in_zip(dataset_dir)
    for folder in folders:
        folder_images = []
        with zipfile.ZipFile(dataset_dir, 'r') as zip_ref:
            for file_name in sorted(zip_ref.namelist()):
                if os.path.dirname(file_name) == folder:
                    with zip_ref.open(file_name) as file:
                        image_data = file.read()
                        image_array = np.frombuffer(image_data, np.uint8)
                        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
                        image = scale_image(image)

                        folder_images.append(image)
                        # all_labels.append(ToolType[folder.upper()].value)

        random.shuffle(folder_images)
        test_index = int(len(folder_images) * test_percent)
        validate_index = int(len(folder_images) * (test_percent + validate_percent))

        test_images.extend(folder_images[:test_index])
        test_labels.extend([ToolType[folder.upper()].value] * test_index)

        validate_images.extend(folder_images[test_index:validate_index])
        validate_labels.extend([ToolType[folder.upper()].value] * (validate_index - test_index))

        train_images.extend(folder_images[validate_index:])
        train_labels.extend([ToolType[folder.upper()].value] * (len(folder_images) - validate_index))

    return train_images, test_images, validate_images, train_labels, test_labels, validate_labels

def add_augmented_images(images, labels):
    # print(f"Number of images loaded: {len(images)}")
    # print(f"labels pre: {labels}")
    original_length = len(images)
    for i in range(original_length):
        image = images[i]
        augmented_images = [
            turn_180(image),
            flip_image_horizontal(image),
            flip_image_vertical(image),
            shear_transform(image),
            blur_image(image),
            add_noise(image),
            change_color(image),
            change_brightness(image),
            change_contrast(image),
            change_saturation(image)
        ]
        images.extend(augmented_images)
        labels += [labels[i]] * len(augmented_images)
    return images, labels
    # print(f"labels post: {labels}")
    # print(f"Number of images loaded: {len(images)}")
    # show_image(images[0])
    # for i in range(203,213,1):
    #     show_image(images[i])

def load_data_augmented(test_percent=0.2, validate_percent=0):
    train_images, test_images, validate_images, train_labels, test_labels, validate_labels = load_data(test_percent, validate_percent)
    train_images, train_labels = add_augmented_images(train_images, train_labels)
    test_images, test_labels = add_augmented_images(test_images, test_labels)
    validate_images, validate_labels = add_augmented_images(validate_images, validate_labels)
    return train_images, test_images, validate_images, train_labels, test_labels, validate_labels

def scale_image(image):
    # Define the new size (width, height)
    new_size = (100, 100)  # Adjust the size as needed
    # Downscale the image
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)

def turn_180(image):
    return cv2.rotate(image, cv2.ROTATE_180)

def flip_image_horizontal(image):
    return cv2.flip(image, 1)

def flip_image_vertical(image):
    return cv2.flip(image, 0)

def shear_transform(image, factor = 0.2):
    # Define the shear matrix
    shear_matrix = np.array([[1, factor, 0], [0, 1, 0]])
    # Apply the shear matrix
    return cv2.warpAffine(image, shear_matrix, (image.shape[1], image.shape[0]))

def blur_image(image, kernel_size=5):
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

def add_noise(image, noise_factor=0.2):
    # Generate random noise
    noise = np.random.normal(0, 1, image.shape).astype(np.uint8)
    # Add the noise to the image
    return cv2.addWeighted(image, 1 - noise_factor, noise, noise_factor, 0)

def change_color(image, hue_shift=0.5, saturation_scale=2, value_scale=3):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 0] += hue_shift
    hsv[..., 1] *= saturation_scale
    hsv[..., 2] *= value_scale
    return cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)

def change_brightness(image, scale=1.5):
    return cv2.convertScaleAbs(image, alpha=scale, beta=0)

def change_contrast(image, scale=1.5):
    mean_intensity = np.mean(image)
    return cv2.convertScaleAbs(image, alpha=scale, beta=mean_intensity * (1 - scale))

def change_saturation(image, scale=2.5):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] *= scale
    return cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)

def get_one_hammer_augmentation_list(image):
    return [
            turn_180(image),
            flip_image_horizontal(image),
            flip_image_vertical(image),
            shear_transform(image),
            blur_image(image),
            add_noise(image),
            change_color(image),
            change_brightness(image),
            change_contrast(image),
            change_saturation(image)
        ]