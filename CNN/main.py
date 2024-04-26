# import tensorflow as tf
from dataset import load_data
# import mnist
import matplotlib.pyplot as plt

def main():
    images, labels = load_data()
    print(images.shape)


if __name__ == '__main__':
    main()

