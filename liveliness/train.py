from tensorflow.python.keras.preprocessing.image import DataFrameIterator
from liveliness import LivenessNet
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from imutils import paths

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

import matplotlib.pyplot as plt
import numpy as np
import pickle
import cv2
import os


def read_images(path):
    """
    Reads all images from inside the path, resize these images to 32 x 32 and normalizes

    Returns two lists:
        data - numpy array of images
        label - list of labels
    """

    # loop over all image paths
    imagePaths = list(paths.list_images(path))
    print(imagePaths)
    data = []
    labels = []
    for imagePath in imagePaths:
            # extract the class label from the filename, load the image and
            # resize it to be a fixed 32x32 pixels, ignoring aspect ratio
            label = imagePath.split(os.path.sep)[-2]
            # print(label)
            # break
            image = cv2.imread(imagePath)
            image = cv2.resize(image, (32, 32))
            # update the data and labels lists, respectively
            data.append(image)
            labels.append(label)
    data = np.array(data, dtype="float") / 255.0
    return data,labels

def main():
    # preprocessing
    path = './images/'
    
    data,labels = read_images(path)

    le = LabelEncoder()
    labels = le.fit_transform(labels)
    labels = to_categorical(labels, 2)
    # partition the data into training and testing splits using 75% of
    # the data for training and the remaining 25% for testing
    (trainX, testX, trainY, testY) = train_test_split(data, labels,	test_size=0.25, random_state=88)
    
    aug = ImageDataGenerator(rotation_range=20, zoom_range=0.15, 
                        width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15,
                         horizontal_flip=True, fill_mode="nearest")
    INIT_LR = 1e-4
    BS = 8
    EPOCHS = 100

    opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
    model = LivenessNet.build(width=32, height=32, depth=3, classes=len(le.classes_))
    model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

    H = model.fit(x=aug.flow(trainX, trainY, batch_size=BS),validation_data=(testX, testY), steps_per_epoch=len(trainX) // BS,epochs=EPOCHS)

    model_path = os.path.join('.','model')
    predictions = model.predict(x=testX, batch_size=BS)
    print(classification_report(testY.argmax(axis=1),predictions.argmax(axis=1), target_names=le.classes_))

    model.save(os.path.join(model_path,'liveliness.h5'), save_format="h5")
    # save the label encoder to disk
    f = open(os.path.join(model_path,'labels/labels'), "wb")
    f.write(pickle.dumps(le))
    f.close()

if __name__ == '__main__':
    main()
