import numpy as np
import pandas as pd
from tensorflow.keras.applications.efficientnet import EfficientNetB4
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, BatchNormalization, Flatten, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.applications import VGG19, MobileNet

#0. 변수
batch = 16
seed = 42
dropout = 0.3
epochs = 1000
model_path = 'C:/workspace/lotte/h5/lotte_0318_{epoch:02d}-{val_loss:,4f}.hdf5'
sub = pd.read_csv('C:/workspace/lotte/sample.csv', header = 0)
es = EarlyStopping(patience = 5)
lr = ReduceLROnPlateau(factor = 0.25, patience = 3, verbose = 1)
cp = ModelCheckpoint(model_path, save_best_only= True)

#1. 데이터
train_gen = ImageDataGenerator(
    validation_split = 0.2,
    width_shift_range= 0.1,
    height_shift_range= 0.1,
    preprocessing_function= preprocess_input
)

test_gen = ImageDataGenerator(
    preprocessing_function= preprocess_input
)

# Found 39000 images belonging to 1000 classes.
train_data = train_gen.flow_from_directory(
    'C:/workspace/lotte/train',
    target_size = (128, 128),
    class_mode = 'sparse',
    batch_size = batch,
    seed = seed,
    subset = 'training'
)

# Found 9000 images belonging to 1000 classes.
val_data = train_gen.flow_from_directory(
    'C:/workspace/lotte/train',
    target_size = (128, 128),
    class_mode = 'sparse',
    batch_size = batch,
    seed = seed,
    subset = 'validation'
)

# Found 72000 images belonging to 1 classes.
test_data = test_gen.flow_from_directory(
    'C:/workspace/lotte/test',
    target_size = (128, 128),
    class_mode = None,
    batch_size = batch,
    seed = seed,
    shuffle = False
)

#2. 모델
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Flatten, BatchNormalization, Dense, Activation, DepthwiseConv2D
from tensorflow.keras.applications import VGG19, MobileNet

mobile_net = MobileNet(weights="imagenet", include_top=False, input_shape=(128, 128, 3),)

top_model = mobile_net.output
top_model = Flatten()(top_model)
top_model = Dense(1228, activation="relu")(top_model)
top_model = Dense(1000, activation="softmax")(top_model)

model = Model(inputs=mobile_net.input, outputs = top_model)

#3. 컴파일 훈련
model.compile(loss = 'sparse_categorical_crossentropy', optimizer = 'adam', metrics = ['acc'])
model.fit(train_data, steps_per_epoch = np.ceil(39000/batch), validation_data= val_data, validation_steps= np.ceil(9000/batch),\
    epochs = epochs, callbacks = [es, cp, lr])

model = load_model(model_path)

#4. 평가 예측
pred = model.predict(test_data)
pred = np.argmax(pred, 1)
print(pred)
sub.loc[:,'prediction'] = pred
sub.to_csv('C:/workspace/lotte/submit/sample_001.csv', index = False)