# import tensorflow as tf
import dataset
import cnn
import mnist
import matplotlib.pyplot as plt
# from tensorflow.keras import datasets, layers, models

def main():
    # (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
    train_images, test_images, validate_images, train_labels, test_labels, validate_labels = dataset.load_data_augmented()
    # cnn.train_model(train_images, train_labels, test_images, test_labels)
    print("test")
    i=0
    for item in test_labels:
        if item == 1:
            break
        i+=1
    hammer_list = dataset.get_one_hammer_augmentation_list(test_images[i])
    cnn.predict_image(hammer_list)


if __name__ == '__main__':
    main()

