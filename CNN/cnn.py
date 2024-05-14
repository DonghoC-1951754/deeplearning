import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import datasets, layers, models, preprocessing
from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.keras.callbacks import ModelCheckpoint


def train_model(train_images, train_labels, test_images, test_labels, batch_size=128, epochs=20,
                conv2d_filters=[32, 64, 128], dense_units=128):
    train_images = np.array(train_images)
    test_images = np.array(test_images)
    train_labels = np.array(train_labels)
    train_labels = train_labels.reshape(-1, 1)
    test_labels = np.array(test_labels)
    test_labels = test_labels.reshape(-1, 1)

    model = models.Sequential()
    model.add(layers.Rescaling(1. / 255))
    model.add(layers.Conv2D(conv2d_filters[0], (3, 3), activation='relu', input_shape=train_images[0].shape))
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

    model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    history = model.fit(train_images, train_labels, epochs=epochs, batch_size=batch_size,
                        validation_data=(test_images, test_labels))
    model.save('my_model.keras')
    plot_model(history, model, test_images, test_labels)


class SaveCB(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()

    def on_train_begin(self, logs=None):
        # Initialize the best as infinity.
        self.best = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get("val_loss")
        if np.less(current, self.best):
            self.best = current
            self.model.save('best_model.keras')


class EarlyStoppingAtMinLoss(tf.keras.callbacks.Callback):
    """Stop training when the loss is at its min, i.e. the loss stops decreasing.

    Arguments:
        patience: Number of epochs to wait after min has been hit. After this
        number of no improvement, training stops.
    """

    def __init__(self, patience=0):
        super().__init__()
        self.patience = patience
        # best_weights to store the weights at which the minimum loss occurs.
        self.best_weights = None

    def on_train_begin(self, logs=None):
        # The number of epoch it has waited when loss is no longer minimum.
        self.wait = 0
        # The epoch the training stops at.
        self.stopped_epoch = 0
        # Initialize the best as infinity.
        self.best = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get("val_loss")
        if np.less(current, self.best):
            self.best = current
            self.wait = 0
            # Record the best weights if current results is better (less).
            self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                print("Restoring model weights from the end of the best epoch.")
                self.model.set_weights(self.best_weights)

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            print("Epoch %05d: early stopping" % (self.stopped_epoch + 1))


def train_model_load(dir1, dir2='', batch_size=128, epochs=20, conv2d_filters=[32, 64, 128], dense_units=128,
                     shape: tuple = (256, 256, 3)):
    colormodes = {1: 'grayscale',
                  3: 'rgb'}
    training_data, val_data = preprocessing.image_dataset_from_directory(dir1, validation_split=0.2, subset='both',
                                                                         batch_size=batch_size, seed=1337,
                                                                         image_size=(shape[0], shape[1]),
                                                                         color_mode=colormodes[shape[2]])

    if dir2:
        training_data = tf.data.Dataset.sample_from_datasets(
            [training_data, preprocessing.image_dataset_from_directory(dir2, validation_split=0.2, subset='training',
                                                                       batch_size=batch_size, seed=1337,
                                                                       image_size=(shape[0], shape[1]),
                                                                       color_mode=colormodes[shape[2]])],
            weights=[0.9, 0.1], stop_on_empty_dataset=True, rerandomize_each_iteration=True)
        # val_data = tf.data.Dataset.sample_from_datasets(
        #     [val_data, preprocessing.image_dataset_from_directory(dir2, validation_split=0.2, subset='validation',
        #                                                           batch_size=batch_size, seed=1337,
        #                                                           image_size=(shape[0], shape[1]),
        #                                                           color_mode=colormodes[shape[2]])],
        #     weights=[0.9, 0.1], stop_on_empty_dataset=True, rerandomize_each_iteration=True)

    model = models.Sequential([
        layers.InputLayer(shape),
        layers.Rescaling(1. / 255),
        layers.Conv2D(conv2d_filters[0], 3, activation='relu'),
        layers.MaxPooling2D((2, 2)),
        # layers.Dropout(0.1),
        layers.Conv2D(conv2d_filters[1], 3, activation='relu'),
        layers.MaxPooling2D((2, 2)),
        # layers.Dropout(0.1),
        layers.Conv2D(conv2d_filters[2], 3, activation='relu'),
        layers.MaxPooling2D((2, 2)),
        # layers.Conv2D(conv2d_filters[3], 3, activation='relu'),
        # layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.1),
        layers.Flatten(),
        layers.Dense(dense_units, activation='relu'),
        # layers.Dropout(0.1),
        layers.Dense(4),
    ])
    model.summary()

    model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    # add this to autosave current best, for when you need to interrupt mid-training
    checkpoint = SaveCB()

    stop = EarlyStoppingAtMinLoss(4)
    history = model.fit(training_data, epochs=epochs, batch_size=batch_size, validation_data=val_data,
                        callbacks=[stop])
    model.save('my_model.keras')
    plot_model(history, model, val_data)


def predict_image(images):
    loaded_model = load_model('my_model.keras')
    images = np.array(images)
    predictions = loaded_model.predict(images)
    for prediction in predictions:
        print(np.argmax(prediction))


# def plot_model(history, model, test_images, test_labels):
#     plt.plot(history.history['accuracy'], label='accuracy')
#     plt.plot(history.history['val_accuracy'], label='val_accuracy')
#     plt.xlabel('Epoch')
#     plt.ylabel('Accuracy')
#     plt.ylim([0.0, 1])
#     plt.legend(loc='lower right')
#     plt.show()
#     test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
#     print("Test accuracy: ", test_acc)

def plot_model(history, model, dataset):
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.0, 1])
    plt.legend(loc='lower right')
    plt.show()
    test_loss, test_acc = model.evaluate(dataset, verbose=2)
    print("Test accuracy: ", test_acc)
