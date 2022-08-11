import os 
import cv2
import sys
import keras
import constants 
from resnet50 import ResNet50 
import numpy as np
import pandas as pd
from keras import optimizers
from keras import applications
from keras import backend as K
from os import listdir, makedirs
from keras.utils.data_utils import Sequence
from os.path import join, exists, expanduser
from keras.models import Sequential, Model, load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from keras.layers import Conv2D, MaxPooling2D, Dense, GlobalAveragePooling2D, Activation, Dropout, Flatten, MaxPool2D
from keras.utils.vis_utils import plot_model
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
import matplotlib 
import matplotlib.pyplot as plt 
# import tensorflow as tf
# gpus = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(gpus[0], True)

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)
val_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    constants.TRAIN_DATA,
    target_size=(256, 256),
    batch_size=16,
    class_mode='categorical',
    shuffle = True)

test_generator = test_datagen.flow_from_directory(
    constants.TEST_DATA,
    target_size=(256, 256),
    batch_size=16,
    class_mode='categorical')

validation_generator = val_datagen.flow_from_directory(
    constants.VAL_DATA,
    target_size=(256, 256),
    batch_size=16,
    class_mode='categorical')

def train_resnet50(train_generator, validation_generator):
    model = Sequential()
    model.add(ResNet50(include_top = False, pooling = 'avg', weights = 'imagenet'))
    model.add(Dense(constants.NUM_CLASSES, activation = 'sigmoid'))
    model.layers[0].trainable = False
    model.compile(loss='categorical_crossentropy',
                  optimizer=constants.ADAM, 
                  metrics=['CategoricalAccuracy'])
    # plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
    print(model.summary())
    model.fit_generator(train_generator,
                        steps_per_epoch=constants.NB_TRAIN_SAMPLES // constants.BATCH_SIZE,
                        epochs=constants.EPOCHS, validation_data=validation_generator,
                        validation_steps=constants.NB_VALIDATION_SAMPLES // constants.BATCH_SIZE,callbacks=[constants.CHECKPOINTER])
    print(" Saving Model ")
    # model.save_weights(constants.WEIGHTS_PATH1)
    model.save(constants.MODELPATH)

# def build_resnet_model():
#     model = Sequential()
#     model.add(ResNet50(include_top = False, pooling = 'avg', weights = 'imagenet'))
#     model.add(Dense(constants.NUM_CLASSES, activation = 'softmax'))
#     model.layers[0].trainable = False
#     return model

# def test_resnet_model(test_image):
#     model = build_resnet_model()
#     model.load_weights(constants.CHECKPOINT_PATH)
#     im = cv2.imread(test_image)
#     im = im.resize([ 256, 256, 3])
#     print(model.predict(im))
#     # print(model.predict_proba(im))


def test_resnet():
    model = Sequential()
    model.add(ResNet50(include_top = False, pooling = 'avg', weights = 'imagenet'))
    model.add(Dense(constants.NUM_CLASSES, activation = 'sigmoid'))
    model.layers[0].trainable = False
    model.compile(loss='categorical_crossentropy',
                  optimizer=constants.SGD, 
                  metrics=['accuracy'])
    model.load_weights('models/cp.ckpt.data-00000-of-00001')

    print("loaded model") 
    pred=model.predict_generator(test_generator, steps=len(test_generator), verbose=1)
    y_true = test_generator.classes
    print("y_true:", y_true.shape)
    print("pred: ", pred.shape) 

    y_pred = np.argmax(pred, axis = 1)
    print("y_pred: ", y_pred.shape) 

    font = {
            'family': 'Times New Roman',
            'size': 12
            }
    matplotlib.rc('font', **font)
    mat = confusion_matrix(y_true, y_pred)
    # plot_confusion_matrix(conf_mat=mat, figsize=(8, 8), show_normed=False)
    # plt.show()
    print(mat)
    # cl = np.round(pred)
    # filenames=test_generator.filenames
    # results=pd.DataFrame({"file":filenames,"pr":pred[:,0], "class":cl[:,0]})
    # results.to_csv("resnetresults.csv")

# def build_vgg_model():
#     model = Sequential()
#     model.add(VGG16(include_top = False, pooling = 'avg', weights = 'imagenet'))
#     model.add(Dense(constants.NUM_CLASSES, activation = 'softmax'))
#     model.layers[0].trainable = False
#     return model

# def test_batch(img_folder):
#     model = build_model()
#     model.load_weights(weights_path)

#     # batch_test_datagen = ImageDataGenerator(rescale=1. / 255)

#     # batch_test_generator = batch_test_datagen.flow_from_directory(
#     #     img_folder,
#     #     target_size=(img_width, img_height),
#     #     batch_size=batch_size, class_mode='binary')

#     for img in os.listdir(img_folder):
#         im = cv2.imread(img_folder + img)
#         im = im.reshape([-1, 256, 256, 3])
#         print(img_folder + img, model.predict_proba(im))

def sequence_model():
    model= Sequential()
    model.add(Conv2D(kernel_size=(3,3), filters=32, activation='tanh', input_shape=(200,200,3,)))
    model.add(Conv2D(filters=30,kernel_size = (3,3),activation='tanh'))
    model.add(MaxPool2D(2,2))
    model.add(Conv2D(filters=30,kernel_size = (3,3),activation='tanh'))
    model.add(MaxPool2D(2,2))
    model.add(Conv2D(filters=30,kernel_size = (3,3),activation='tanh'))

    model.add(Flatten())

    model.add(Dense(20,activation='relu'))
    model.add(Dense(15,activation='relu'))
    model.add(Dense(11,activation = 'softmax'))
        
    model.compile(
                loss='categorical_crossentropy', 
                metrics=['acc'],
                optimizer='adam'
                )

    return model 

def train_seq(train_generator, validation_generator):
    model = sequence_model()
    history = model.fit_generator(train_generator,
                        steps_per_epoch=constants.NB_TRAIN_SAMPLES // constants.BATCH_SIZE,
                        epochs=constants.EPOCHS, validation_data=validation_generator,
                        validation_steps=constants.NB_VALIDATION_SAMPLES // constants.BATCH_SIZE)
    print(" Saving Model ")
    # model.save_weights(constants.WEIGHTS_PATH1)
    model.save(constants.MODELPATH)

    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

if __name__ == "__main__":
    if sys.argv[1] == 'train_seq':
        train_seq(train_generator, validation_generator)
    if sys.argv[1] == 'test_resnet50':
        test_resnet() 