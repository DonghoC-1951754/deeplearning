from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.models import load_model
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def train_model(train_images, train_labels, test_images, test_labels, batch_size=128, epochs=20, conv2d_filters=[32, 64, 128], dense_units=128):
    train_images = np.array(train_images)
    test_images = np.array(test_images)
    train_labels = np.array(train_labels)
    train_labels = train_labels.reshape(-1, 1)
    test_labels = np.array(test_labels)
    test_labels = test_labels.reshape(-1, 1)
    train_images, test_images = train_images / 255.0, test_images / 255.0

    model = models.Sequential()
    model.add(layers.Conv2D(conv2d_filters[0], (3, 3), activation='relu', input_shape=(100, 100, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(conv2d_filters[1], (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(conv2d_filters[2], (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.summary()

    model.add(layers.Flatten())
    model.add(layers.Dense(dense_units, activation='relu'))
    model.add(layers.Dense(4))
    model.summary()

    model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

    history = model.fit(train_images, train_labels, epochs=epochs, batch_size=batch_size, validation_data=(test_images, test_labels))
    model.save('my_model.keras')
    plot_model(history, model, test_images, test_labels)


def predict_image(images):
    loaded_model = load_model('my_model.keras')
    images = np.array(images)
    predictions = loaded_model.predict(images)
    for prediction in predictions:
        print(np.argmax(prediction))


def plot_model(history, model, test_images, test_labels):
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.0, 1])
    plt.legend(loc='lower right')
    plt.show()
    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
    print("Test accuracy: ", test_acc)