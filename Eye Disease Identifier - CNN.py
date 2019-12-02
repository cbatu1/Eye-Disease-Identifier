import numpy as np
import os
import cv2
import random
import pickle
import time
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D
from keras.callbacks import TensorBoard
from keras.models import load_model

DATA_DIRECTORY = "/Users/chris/Desktop/School/Project Files/Eye Disease Identification/Eye_Images"
CATEGORIES = ["Healthy_Eyes", "Bulging_Eyes", "Cataracts", "Crossed_Eyes", "Glaucoma", "Uveitis"]

training_data = []
IMAGE_SIZE = 50


def create_training_data():
    for category in CATEGORIES:
        path = os.path.join(DATA_DIRECTORY,
                            category)  # Join each category in CATEGORIES with its respective data set in DATA_DIRECTORY
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (IMAGE_SIZE, IMAGE_SIZE))
                training_data.append([new_array, class_num])
            except Exception as e:
                pass


create_training_data()

random.shuffle(training_data)

X = []  # Uppercase X for features set
y = []  # Lowercase y for labels set

for features, label in training_data:
    X.append(features)
    y.append(label)

X = np.array(X).reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 1)

pickle_out = open("features.pickle", "wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("labels.pickle", "wb")
pickle.dump(y, pickle_out)
pickle_out.close()

X = pickle.load(open("features.pickle", "rb"))
y = pickle.load(open("labels.pickle", "rb"))

X = X / 255.0  # Scaling data to normalize it. Dividing by 255 because that is the max value for pixel data.

dense_layers = [0]
layer_sizes = [64]
conv_layers = [1]

for dense_layer in dense_layers:
    for layer_size in layer_sizes:
        for conv_layer in conv_layers:
            NAME = "{}-conv-{}-nodes-{}-dense-{}".format(conv_layer, layer_size, dense_layer, int(time.time()))
            tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME))

            # Building the model
            model = Sequential()

            model.add(Conv2D(layer_size, (3, 3), input_shape=X.shape[1:]))
            model.add(Activation("relu"))
            model.add(MaxPooling2D(pool_size=(2, 2)))

            for l in range(conv_layer - 1):
                model.add(Conv2D(layer_size, (3, 3)))
                model.add(Activation("relu"))
                model.add(MaxPooling2D(pool_size=(2, 2)))

            model.add(Flatten())
            for l in range(dense_layer):
                model.add(Dense(dense_layer))
                model.add(Activation("relu"))

            model.add(Dense(7))
            model.add(Activation("sigmoid"))

            # Training the model
            model.compile(loss="sparse_categorical_crossentropy",
                          optimizer="adam",
                          metrics=["accuracy"])

            model.fit(X, y, batch_size=20, epochs=5, validation_split=0.3, callbacks=[tensorboard])

            model.save("64x0-CNN.model")
