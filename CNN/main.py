import cnn
import dataset
import tensorflow as tf


# from CNN import dataset


def main():
    # ---Run this code below if you need to scale and augment a new dataset---
    # dataset.augment_rescale_images_in_main_directory("../real only")
    # ---Run this code below if you need to check for corrupted images in a directory---
    # dataset.check_corrupted_images_in_main_directory("../Dataset real")

    # cnn.train_model_load('../Dataset real', batch_size=64, epochs=40)
    cnn.train_model_load('../Dataset synth', '../real only', batch_size=64, epochs=40)

    # cnn.train_model_load('../data-medium', batch_size=64, epochs=40, #conv2d_filters=[64, 128, 256, 512],
    #                      )
    # (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
    # train_images, test_images, validate_images, train_labels, test_labels, validate_labels = dataset.load_data_augmented_permutations()
    # train_images, test_images, validate_images, train_labels, test_labels, validate_labels = dataset.load_data('./dataset_real.zip',
    #     drop_percent=.00, scale=(100, 100))

    #print amount of images in each dataset
    # print("Train images: ", len(train_images))
    # print("Test images: ", len(test_images))
    # print("Validate images: ", len(validate_images))

    #print amount of labels in each dataset
    # print("Train labels: ", len(train_labels))
    # print("Test labels: ", len(test_labels))
    # print("Validate labels: ", len(validate_labels))
    #
    # print("Image shape: ", np.array(train_images[0]).shape)
    # cnn.train_model(train_images, train_labels, test_images, test_labels, epochs=10)
    # print("test")
    # i=0
    # for item in test_labels:
    #     if item == 1:
    #         break
    #     i+=1
    # hammer_list = dataset.get_one_hammer_augmentation_list(test_images[i])
    # cnn.predict_image(hammer_list)


if __name__ == '__main__':
    main()
