# import tensorflow as tf
import dataset
import cnn
# import mnist
import matplotlib.pyplot as plt
# from tensorflow.keras import datasets, layers, models

def main():
    # (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
    train_images, test_images, validate_images, train_labels, test_labels, validate_labels = dataset.load_data_augmented_permutations()


    #print amount of images in each dataset
    print("Train images: ", len(train_images))
    print("Test images: ", len(test_images))
    print("Validate images: ", len(validate_images))

    #print amount of labels in each dataset
    print("Train labels: ", len(train_labels))
    print("Test labels: ", len(test_labels))
    print("Validate labels: ", len(validate_labels))

    cnn.train_model(train_images, train_labels, test_images, test_labels)
    print("test")
    # i=0
    # for item in test_labels:
    #     if item == 1:
    #         break
    #     i+=1
    # hammer_list = dataset.get_one_hammer_augmentation_list(test_images[i])
    # cnn.predict_image(hammer_list)


if __name__ == '__main__':
    main()

