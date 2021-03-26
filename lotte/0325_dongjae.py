import numpy as np
import pandas as pd
import os
from tensorflow.keras.applications.efficientnet import EfficientNetB4
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, BatchNormalization, Flatten, Dropout, GlobalAveragePooling2D, Input, GaussianDropout
from tensorflow.keras.activations import swish
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from scipy import stats
from tensorflow.keras.applications import MobileNet
filenum = 16
direcotry = 'C:/workspace/lotte/train_new2'
batch_size = 3
img_height = 100
img_width = 100
validation_split = 0.3
seed = 42
epochs = 1000
model_path = 'C:/workspace/lotte/h5/lpd_{0:03}.hdf5'.format(filenum)
save_folder = 'C:/workspace/lotte/submit/submit_{0:03}'.format(filenum)
sub = pd.read_csv('C:/workspace/lotte/sample.csv', header = 0)
es = EarlyStopping(patience = 11)
lr = ReduceLROnPlateau(factor = 0.5, patience = 3, verbose = 1)
cp = ModelCheckpoint(model_path, save_best_only= True)
class MixupImageDataGenerator():
    def __init__(self, generator, directory, batch_size, img_height, img_width, alpha=0.2, subset=None):
        """Constructor for mixup image data generator.

        Arguments:
            generator {object} -- An instance of Keras ImageDataGenerator.
            directory {str} -- Image directory.
            batch_size {int} -- Batch size.
            img_height {int} -- Image height in pixels.
            img_width {int} -- Image width in pixels.

        Keyword Arguments:
            alpha {float} -- Mixup beta distribution alpha parameter. (default: {0.2})
            subset {str} -- 'training' or 'validation' if validation_split is specified in
            `generator` (ImageDataGenerator).(default: {None})
        """

        self.batch_index = 0
        self.batch_size = batch_size
        self.alpha = alpha

        # First iterator yielding tuples of (x, y)
        self.generator1 = generator.flow_from_directory(directory,
                                                        target_size=(
                                                            img_height, img_width),
                                                        class_mode="sparse",
                                                        batch_size=batch_size,
                                                        shuffle=True,
                                                        subset=subset)

        # Second iterator yielding tuples of (x, y)
        self.generator2 = generator.flow_from_directory(directory,
                                                        target_size=(
                                                            img_height, img_width),
                                                        class_mode="sparse",
                                                        batch_size=batch_size,
                                                        shuffle=True,
                                                        subset=subset)

        # Number of images across all classes in image directory.
        self.n = self.generator1.samples

    def reset_index(self):
        """Reset the generator indexes array.
        """

        self.generator1._set_index_array()
        self.generator2._set_index_array()

    def on_epoch_end(self):
        self.reset_index()

    def reset(self):
        self.batch_index = 0

    def __len__(self):
        # round up
        return (self.n + self.batch_size - 1) // self.batch_size

    def get_steps_per_epoch(self):
        """Get number of steps per epoch based on batch size and
        number of images.

        Returns:
            int -- steps per epoch.
        """

        return self.n // self.batch_size

    def __next__(self):
        """Get next batch input/output pair.

        Returns:
            tuple -- batch of input/output pair, (inputs, outputs).
        """

        if self.batch_index == 0:
            self.reset_index()

        current_index = (self.batch_index * self.batch_size) % self.n
        if self.n > current_index + self.batch_size:
            self.batch_index += 1
        else:
            self.batch_index = 0

        # random sample the lambda value from beta distribution.
        l = np.random.beta(self.alpha, self.alpha, self.batch_size)

        X_l = l.reshape(self.batch_size, 1, 1, 1)
        y_l = l.reshape(self.batch_size, 1)

        # Get a pair of inputs and outputs from two iterators.
        X1, y1 = self.generator1.next()
        X2, y2 = self.generator2.next()

        # Perform the mixup.
        X = X1 * X_l + X2 * (1 - X_l)
        y = y1 * y_l + y2 * (1 - y_l)
        return X, y

    def __iter__(self):
        while True:
            yield next(self)
#0. 변수

test_dir = 'C:/workspace/lotte/test_new'


if not os.path.exists(save_folder):
    os.mkdir(save_folder)

#1. 데이터
input_imgen = ImageDataGenerator(
    rotation_range=5,
    width_shift_range=0.05,
    height_shift_range=0,
    shear_range=0.05,
    zoom_range=0,
    brightness_range=(1, 1.3),
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=validation_split
    )

test_gen = ImageDataGenerator(
    preprocessing_function= preprocess_input,
    width_shift_range= 0.03,
    height_shift_range= 0.03
)

train_gen = MixupImageDataGenerator(generator=input_imgen,
                                          directory=direcotry,
                                          batch_size=batch_size,
                                          img_height=img_height,
                                          img_width=img_width,
                                          subset='training')

validation_gen = input_imgen.flow_from_directory(direcotry,
                                                       target_size=(
                                                           img_height, img_width),
                                                       class_mode="categorical",
                                                       batch_size=batch_size,
                                                       shuffle=True,
                                                       subset='validation')
'''
# Found 58000 images belonging to 1000 classes.
train_data = train_gen.flow_from_directory(
    train_dir,
    target_size = (img_size, img_size),
    class_mode = 'sparse',
    batch_size = batch_size,
    seed = seed,
    subset = 'training'
)

# Found 14000 images belonging to 1000 classes.
val_data = validation_gen.flow_from_directory(
    train_dir,
    target_size = (img_size, img_size),
    class_mode = 'sparse',
    batch_size = batch_size,
    seed = seed,
    subset = 'validation'
)
'''
# Found 72000 images belonging to 1 classes.
test_data = test_gen.flow_from_directory(
    test_dir,
    target_size = (img_height, img_width),
    class_mode = None,
    batch_size = batch_size,
    shuffle = False
)

#2. 모델
eff = MobileNet(include_top = False, input_shape=(img_height, img_width, 3))
eff.trainable = False

a = eff.output
a = Dense(1000, activation= 'swish') (a)
a = Dropout(0.5) (a)
a = GlobalAveragePooling2D() (eff.output)
a = Dense(1000, activation= 'softmax') (a)

model = Model(inputs = eff.input, outputs = a)

#3. 컴파일 훈련
model.compile(loss = 'sparse_categorical_crossentropy', optimizer = 'adam', metrics = ['sparse_categorical_accuracy'])
history = model.fit_generator(train_gen, steps_per_epoch = len(train_gen), validation_data= validation_gen, validation_steps= len(validation_gen),\
    epochs = epochs, callbacks = [es, cp, lr])

model = load_model(model_path)


# history = model.fit_generator(
#     train_generator,
#     steps_per_epoch=train_generator.get_steps_per_epoch(),
#     validation_data=validation_generator,
#     validation_steps=validation_generator.samples // batch_size,
#     epochs=epochs)
#4. 평가 예측
cumsum = np.zeros([72000, 1000])
count_result = []
for tta in range(50):
    print(f'{tta+1} 번째 TTA 진행중 - TTA')
    pred = model.predict(test_data, steps = len(test_data), verbose = True) # (72000, 1000)
    pred = np.array(pred)
    cumsum = np.add(cumsum, pred)
    temp = cumsum / (tta+1)
    temp_sub = np.argmax(temp, 1)
    temp_percent = np.max(temp, 1)

    count = 0
    i = 0
    for percent in temp_percent:
        if percent < 0.3:
            print(f'{i} 번째 테스트 이미지는 {percent}% 의 정확도를 가짐')
            count += 1
        i += 1
    print(f'TTA {tta+1} : {count} 개가 불확실!')
    count_result.append(count)
    print(f'기록 : {count_result}')
    sub.loc[:, 'prediction'] = temp_sub
    sub.to_csv(save_folder + '/sample_{0:03}_{1:02}.csv'.format(filenum, (tta+1)), index = False)

# Function call stack:
# train_function

# 2021-03-26 09:49:00.647669: W tensorflow/core/kernels/data/generator_dataset_op.cc:103] Error occurred when finalizing GeneratorDataset iterator: Failed precondition: Python interpreter state is not initialized. The process may be terminated.
#          [[{{node PyFunc}}]]
