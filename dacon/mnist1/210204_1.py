import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import string
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from scipy import stats
from tensorflow.keras.layers.experimental.preprocessing import Resizing
import warnings
warnings.filterwarnings('ignore')

#데이터 끌어오기
train = pd.read_csv('C:/data/dacon/mnist1/data/train.csv')
test = pd.read_csv('C:/data/dacon/mnist1/data/test.csv')
sub = pd.read_csv('C:/data/dacon/mnist1/data/submission.csv')

#선언해주기
skf = StratifiedKFold(n_splits=15, random_state=33, shuffle=True)
# datagen = ImageDataGenerator(width_shift_range= 0.2, height_shift_range= 0.2)
# datagen2 = ImageDataGenerator()
es = EarlyStopping(monitor = 'val_loss', patience = 30)
lr = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.25, patience = 10)

#RESIZE , RESHAPE
x1_train = train.loc[:, '0':].to_numpy().reshape(-1,28,28,1)
x1_train = Resizing(128,128)(x1_train)/255.
x1_train = np.array(x1_train)
y1_train = train.loc[:, 'digit'].to_numpy()
x_test = test.loc[:, '0':].to_numpy().reshape(-1,28,28,1)
x_test = Resizing(128,128)(x_test)/255.
x_test = np.array(x_test)


cp = ModelCheckpoint(monitor= 'val_loss', filepath=f'C:/data/dacon/mnist1/modelcheck/0204_1.h5', save_best_only= True)


for train_index, valid_index in skf.split(x1_train, y1_train) :
    x_train = x1_train[train_index]
    x_val = x1_train[valid_index]
    y_train = y1_train[train_index]
    y_val = y1_train[valid_index]


# cutout => 0.25 ~ 0.5 랜덤한 비율로 이미지 일부를 삭제 하여 학습을 합니다

def cutout(images, cut_length):
    """
    Perform cutout augmentation from images.
    :param images: np.ndarray, shape: (N, H, W, C).
    :param cut_length: int, the length of cut(box).
    :return: np.ndarray, shape: (N, h, w, C).
    """

    H, W, C = images.shape[1:4]
    augmented_images = []
    for image in images:    # image.shape: (H, W, C)
        image_mean = image.mean(keepdims=True)
        image -= image_mean

        mask = np.ones((H, W, C), np.float32)

        y = np.random.randint(H)
        x = np.random.randint(W)
        length = cut_length

        y1 = np.clip(y - (length // 2), 0, H)
        y2 = np.clip(y + (length // 2), 0, H)
        x1 = np.clip(x - (length // 2), 0, W)
        x2 = np.clip(x + (length // 2), 0, W)

        mask[y1: y2, x1: x2] = 0.
        image = image * mask

        image += image_mean
        augmented_images.append(image)

    return np.stack(augmented_images)    # shape: (N, h, w, C)
#MODEL
def mymodel():
    model = Sequential()

    model.add(Conv2D(16,(3,3),activation='relu',input_shape=(28,28,1),padding='same'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    
    model.add(Conv2D(32,(3,3),activation='relu',padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(32,(5,5),activation='relu',padding='same')) 
    model.add(BatchNormalization())
    model.add(Conv2D(32,(5,5),activation='relu',padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(32,(5,5),activation='relu',padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((3,3)))
    model.add(Dropout(0.3))

    model.add(Conv2D(64,(3,3),activation='relu',padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(64,(5,5),activation='relu',padding='same')) 
    model.add(BatchNormalization())
    model.add(MaxPooling2D((3,3)))
    model.add(Dropout(0.3))

    model.add(Flatten())

    model.add(Dense(128,activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(64,activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    model.add(Dense(10,activation='softmax'))
    return model

model = mymodel()
model.compile(loss = 'sparse_categorical_crossentropy', optimizer = 'adam', metrics = ['acc'])
model.fit(x_train, epochs = 1, validation_data= (x_val, y_val), callbacks = [es,cp,lr])
