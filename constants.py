import keras
from keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from keras.optimizers import gradient_descent_v2, adam_v2

IMG_WIDTH, IMG_HEIGHT = 256, 256
NB_TRAIN_SAMPLES = 2062
NB_VALIDATION_SAMPLES = 334
NB_TEST_SAMPLES = 143
EPOCHS = 20
BATCH_SIZE = 64 
NUM_CLASSES = 11

TENSORBOARD_DIR = './logs'
CHECKPOINT_PATH = 'models/resnetmodel1.h5'
WEIGHTS_PATH1 = 'models/resnetweights.h5'
MODELPATH = 'models/resnetmodel.h5'
DATA_PATH = 'dataset/' 
TRAIN_DATA = DATA_PATH+ 'train/'
TEST_DATA = DATA_PATH+ 'test/'
VAL_DATA = DATA_PATH+ 'val/' 

CHECKPOINTER = ModelCheckpoint(filepath=CHECKPOINT_PATH, verbose=1, save_best_only=True, save_weights_only=True)

TENSORBOARD = TensorBoard(log_dir=TENSORBOARD_DIR, histogram_freq=0,
                          write_graph=True, write_images=False)


SGD = gradient_descent_v2.SGD(learning_rate=0.1, decay=5e-6, momentum=0.87, nesterov=True)
ADAM = adam_v2.Adam()


LR_REDUCER = ReduceLROnPlateau(patience = 5, monitor = 'loss', factor = 0.95, verbose = 1)