import itertools
import os
import random
import zipfile
from enum import Enum

import cv2
import numpy as np
from tqdm import tqdm


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


def scale_image(image, width=256, height=256):
    # Define the new size (width, height)
    new_size = (width, height)  # Adjust the size as needed
    # Downscale the image
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)


# -----------------Original Images-----------------#
# Load data from the dataset.zip file, this only gives the standard images
def load_data(dataset_dir='./dataset.zip', test_percent=0.2, validate_percent=0, scale: tuple = (100, 100)):
    # Lists to store images and labels for each set
    test_images, train_images, validate_images = [], [], []
    test_labels, train_labels, validate_labels = [], [], []

    folders = iterate_folders_in_zip(dataset_dir)

    for folder in folders:
        folder_images = []

        with zipfile.ZipFile(dataset_dir, 'r') as zip_ref:
            sorted_names = sorted(zip_ref.namelist())
            for file_name in tqdm(sorted_names, "loading " + folder):
                if os.path.dirname(file_name) == folder:
                    with zip_ref.open(file_name) as file:
                        image_data = file.read()
                        image_array = np.frombuffer(image_data, np.uint8)
                        try:
                            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
                            image = scale_image(image, scale[0], scale[1])  # You need to define the scale_image function

                            folder_images.append(image)
                        except:
                            print("Error: Unable to load image at", file_name)

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
                for i in range(0, 2):
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

# -----------------function to augment a file of images and to save it-----------------

def augment_images_in_folder(folder_path):
    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)
        img = cv2.imread(img_path)
        if img is not None:
            augmented_images = all_permutations_of_image(scale_image(img))
            for i, augmented_img in enumerate(augmented_images):
                output_filename = f"{os.path.splitext(filename)[0]}_aug_{i + 1}{os.path.splitext(filename)[1]}"
                output_path = os.path.join(folder_path, output_filename)
                cv2.imwrite(output_path, augmented_img)

def rescale_images_in_folder(folder_path):
    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)
        img = cv2.imread(img_path)
        if img is not None:
            rescaled_img = scale_image(img)
            cv2.imwrite(img_path, rescaled_img)

def augment_rescale_images_in_main_directory(main_directory):
    # Loop over each subdirectory in the main directory
    for subdir in os.listdir(main_directory):
        subfolder_path = os.path.join(main_directory, subdir)
        # Check if the item in the main directory is indeed a directory
        if os.path.isdir(subfolder_path):
            rescale_images_in_folder(subfolder_path)
            augment_images_in_folder(subfolder_path)

def check_corrupted_images_in_folder(folder_path):
    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Corrupted image found: {img_path}")
            os.remove(img_path)
            print(f"{img_path} deleted.")

def check_corrupted_images_in_main_directory(main_directory):
    # Loop over each subdirectory in the main directory
    for subdir in os.listdir(main_directory):
        subfolder_path = os.path.join(main_directory, subdir)
        # Check if the item in the main directory is indeed a directory
        if os.path.isdir(subfolder_path):
            check_corrupted_images_in_folder(subfolder_path)