import mnist
import cv2
import os
import zipfile
import random
import numpy as np
import itertools
from enum import Enum


class ToolType(Enum):
    COMBWRENCH = 0
    HAMMER = 1
    SCREWDRIVER = 2
    WRENCH = 3


def load_single_image():
    # Load image from file
    image = cv2.imread("./IMG_0655.JPEG")

    if image is None:
        print("Error: Unable to load image at", image_path)
        return None

    # Convert image from BGR to RGB (OpenCV loads images in BGR format by default)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image

def get_one_hammer_augmentation_list(image):
    return [
            turn_180(image),
            flip_image_horizontal(image),
            flip_image_vertical(image),
            blur_image(image),
            add_noise(image),
            change_color(image),
            change_brightness(image, 0.3),
            change_brightness(image, 2),
            change_contrast(image),
            change_saturation(image)
        ]

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
    cv2.resizeWindow('Image', 100, 100)
    cv2.imshow('Image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def scale_image(image):
    # Define the new size (width, height)
    new_size = (100, 100)  # Adjust the size as needed
    # Downscale the image
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)


# -----------------Original Images-----------------#
# Load data from the dataset.zip file, this only gives the standard images
def load_data(test_percent=0.2, validate_percent=0):
    dataset_dir = './dataset.zip'

    # Lists to store images and labels for each set
    test_images, train_images, validate_images = [], [], []
    test_labels, train_labels, validate_labels = [], [], []

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
                        image = scale_image(image)  # You need to define the scale_image function

                        folder_images.append(image)

        random.shuffle(folder_images)
        total_images = len(folder_images)

        # Calculate indexes for splitting into train, test, and validate sets
        test_index = int(total_images * test_percent)
        validate_index = int(total_images * (test_percent + validate_percent))

        # Append images and labels to respective lists
        test_images.extend(folder_images[:test_index])
        test_labels.extend([ToolType[folder.upper()].value] * test_index)

        validate_images.extend(folder_images[test_index:validate_index])
        validate_labels.extend([ToolType[folder.upper()].value] * (validate_index - test_index))

        train_images.extend(folder_images[validate_index:])
        train_labels.extend([ToolType[folder.upper()].value] * (total_images - validate_index))

    return train_images, test_images, validate_images, train_labels, test_labels, validate_labels


# -----------------Augmented Images-----------------#

def add_augmented_images(images, labels):
    # print(f"Number of images loaded: {len(images)}")
    # print(f"labels pre: {labels}")
    original_length = len(images)
    for i in range(original_length):
        augmented_images = get_seperate_augmented_images(images[i])
        images.extend(augmented_images)
        labels += [labels[i]] * len(augmented_images)
    return images, labels
    # print(f"labels post: {labels}")
    # print(f"Number of images loaded: {len(images)}")
    # show_image(images[0])
    # for i in range(203,213,1):
    #     show_image(images[i])


def get_seperate_augmented_images(image):
    augmented_images = [
        turn_180(image),
        flip_image_horizontal(image),
        flip_image_vertical(image),
        blur_image(image),
        add_noise(image),
        change_color(image),
        change_brightness(image, 0.3),
        change_brightness(image, 2),
        change_contrast(image),
        change_saturation(image)
    ]
    return augmented_images

#Load data with augmented images where each image has 10 augmented versions so in total there is 11 images of each image to work with
def load_data_augmented(test_percent=0.2, validate_percent=0):
    train_images, test_images, validate_images, train_labels, test_labels, validate_labels = load_data(test_percent,
                                                                                                       validate_percent)
    train_images, train_labels = add_augmented_images(train_images, train_labels)
    test_images, test_labels = add_augmented_images(test_images, test_labels)
    validate_images, validate_labels = add_augmented_images(validate_images, validate_labels)
    return train_images, test_images, validate_images, train_labels, test_labels, validate_labels


# -----------------Augmented Permutations-----------------#
def add_permuted_augmented_images(images, labels):
    # print(f"Number of images loaded: {len(images)}")
    # print(f"labels pre: {labels}")
    original_length = len(images)
    for i in range(original_length):
        augmented_images = all_permutations_of_image(images[i])
        images.extend(augmented_images)
        labels += [labels[i]] * len(augmented_images)
    return images, labels
    # print(f"labels post: {labels}")
    # print(f"Number of images loaded: {len(images)}")
    # show_image(images[0])
    # for i in range(203,213,1):
    #     show_image(images[i])


def all_permutations_of_image(image):
    # also use functions with eachother
    flipping_func = [
        turn_180,
        flip_image_horizontal,
        flip_image_vertical
    ]

    brightness_func = [
        lambda x: change_brightness(x, 0.45),
        lambda x: change_brightness(x, 1.6)
    ]

    changing_func = [
        change_color,
        change_contrast,
        change_saturation
    ]

    obscurity_func = [
        blur_image,
        add_noise
    ]

    permutations_changing = list(itertools.permutations(changing_func))

    # Generate all permutations of functions
    all_images = []
    for flip_func in flipping_func:
        for bright_func in brightness_func:
            for perm_changing in permutations_changing:
                result = image
                result = flip_func(result)
                result = bright_func(result)
                skip = random.randint(0, 2)
                for i in range(0,2):
                    if i == skip:
                        continue
                    else:
                        result = perm_changing[i](result)

                for obs_func in obscurity_func:
                    result = obs_func(result)
                    all_images.append(result)

    return all_images

#Load data with augmented images where each image has different permutations of the image of using multiple augmentations and different orders of augmentations at once
def load_data_augmented_permutations(test_percent=0.2, validate_percent=0):
    train_images, test_images, validate_images, train_labels, test_labels, validate_labels = load_data(test_percent,
                                                                                                       validate_percent)
    train_images, train_labels = add_permuted_augmented_images(train_images, train_labels)
    test_images, test_labels = add_permuted_augmented_images(test_images, test_labels)
    validate_images, validate_labels = add_permuted_augmented_images(validate_images, validate_labels)
    return train_images, test_images, validate_images, train_labels, test_labels, validate_labels


# -----------------Augmentation Functions-----------------
def turn_180(image):
    return cv2.rotate(image, cv2.ROTATE_180)


def flip_image_horizontal(image):
    return cv2.flip(image, 1)


def flip_image_vertical(image):
    return cv2.flip(image, 0)


def invert_image(image):
    return cv2.bitwise_not(image)


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


# -----------------Alternative Implementation to loading-----------------

# def get_folder_names(zip_file_path):
#     folders = set()
#     with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
#         for file_name in zip_ref.namelist():
#             folder_name = os.path.dirname(file_name)
#             if folder_name:
#                 folders.add(folder_name)
#     return list(folders)
#
# def read_image(zip_ref, file_name):
#     with zip_ref.open(file_name) as file:
#         image_data = file.read()
#         image_array = np.frombuffer(image_data, np.uint8)
#         return cv2.imdecode(image_array, cv2.IMREAD_COLOR)
#
# def split_data(folder_images, test_percent, validate_percent):
#     random.shuffle(folder_images)
#     total_images = len(folder_images)
#     test_index = int(total_images * test_percent)
#     validate_index = int(total_images * (test_percent + validate_percent))
#     return folder_images[:test_index], folder_images[test_index:validate_index], folder_images[validate_index:]
#
# def load_data(test_percent=0.2, validate_percent=0.1, dataset_dir='./dataset.zip'):
#     test_images, train_images, validate_images = [], [], []
#     test_labels, train_labels, validate_labels = [], [], []
#
#     folders = get_folder_names(dataset_dir)
#
#     with zipfile.ZipFile(dataset_dir, 'r') as zip_ref:
#         for folder in folders:
#             folder_images = []
#             for file_name in sorted(zip_ref.namelist()):
#                 if os.path.dirname(file_name) == folder:
#                     image = read_image(zip_ref, file_name)
#                     image = scale_image(image)  # You need to define the scale_image function
#                     folder_images.append(image)
#
#             test, validate, train = split_data(folder_images, test_percent, validate_percent)
#
#             test_images.extend(test)
#             test_labels.extend([ToolType[folder.upper()].value] * len(test))
#
#             validate_images.extend(validate)
#             validate_labels.extend([ToolType[folder.upper()].value] * len(validate))
#
#             train_images.extend(train)
#             train_labels.extend([ToolType[folder.upper()].value] * len(train))
#
#     return train_images, test_images, validate_images, train_labels, test_labels, validate_labels
