import numpy as np
import pandas as pd
from tensorflow.keras.applications.efficientnet import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, BatchNormalization, Flatten, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

#0. 변수
batch = 100000
seed = 42


#1. 데이터
train_gen = ImageDataGenerator(
    validation_split=0.2,
    rescale = 1/255.
)

test_gen = ImageDataGenerator(
    rescale = 1/255.
)

# Found 39000 images belonging to 1000 classes.
train_data = train_gen.flow_from_directory(
    'C:/workspace/lotte/train/',
    target_size = (128, 128),
    class_mode = 'categorical',
    batch_size = batch,
    seed = seed,
    subset = 'training'
)

# Found 9000 images belonging to 1000 classes.
val_data = train_gen.flow_from_directory(
    'C:/workspace/lotte/train/',
    target_size = (128, 128),
    class_mode = 'categorical',
    batch_size = batch,
    seed = seed,
    subset = 'validation'
)

# Found 72000 images belonging to 1 classes.
test_data = test_gen.flow_from_directory(
    'C:/workspace/lotte/test/',
    target_size = (128, 128),
    class_mode = None,
    batch_size = batch,
    seed = seed,
    shuffle = False
)
np.save('C:/workspace/lotte/npy/x_val.npy',arr=val_data[0][0])
np.save('C:/workspace/lotte/npy/y_val.npy',arr=val_data[0][1])
np.save('C:/workspace/lotte/npy/x_test.npy',arr=test_data[0][0])
np.save('C:/workspace/lotte/npy/y_test.npy',arr=test_data[0][1])