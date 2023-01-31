# Copyright (c) 2023 Intel Corporation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
# SPDX-License-Identifier: MIT

from __future__ import division, print_function
import keras
from keras_applications.imagenet_utils import _obtain_input_shape
from imutils import paths
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import get_source_inputs
import os
import warnings
import numpy as np
from keras import backend as K
from keras.layers import Input
from keras.layers.core import Dense, Flatten
from keras.layers.pooling import MaxPooling2D
from keras.models import Model
from keras.layers import Conv2D
from keras.regularizers import l2
from keras.layers.core import Dropout
from keras.layers import GlobalAveragePooling2D
from keras.layers import GlobalMaxPooling2D
from keras.utils.data_utils import get_file
from keras.utils import layer_utils
from tensorflow.keras.optimizers import SGD
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import argparse
import pandas as pd
from cnvrg import Experiment
import shutil

parser = argparse.ArgumentParser(description="""Creator""")
parser.add_argument(
    "-f",
    "--img_dir",
    action="store",
    dest="img_dir",
    default="/data/img_dir/dataset/",
    required=True,
    help="""fine-tuning dataset """,
)
parser.add_argument(
    "--epochs",
    action="store",
    dest="epochs",
    default="50",
    required=True,
    help="""number of iterations that the model training will take to fit the model """,
)
parser.add_argument(
    "--dimensions",
    action="store",
    dest="dimensions",
    default="224,224",
    required=True,
    help="""dimensions for flow from directory""",
)
parser.add_argument(
    "--target_size",
    action="store",
    dest="target_size",
    default="",
    required=True,
    help="""dimension to which the images will be resized""",
)
parser.add_argument(
    "--lr",
    action="store",
    dest="lr",
    default="1e-4",
    required=True,
    help="""learning rate of the fine-tuning""",
)
parser.add_argument(
    "--momentum",
    action="store",
    dest="momentum",
    default="0.9",
    required=True,
    help="""dimension to which the images will be resized""",
)
parser.add_argument(
    "--training_args",
    action="store",
    dest="training_args",
    default="30,0.15,0.2,0.2,0.15",
    required=True,
    help="""training image preprocessing arguments""",
)
parser.add_argument(
    "--loss",
    action="store",
    dest="loss",
    default="categorical_crossentropy",
    required=True,
    help="""loss function while fitting the model on training data """,
)
parser.add_argument(
    "--metrics",
    action="store",
    dest="metrics",
    default="accuracy",
    required=True,
    help="""metric on which the optimization will work while fitting the model """,
)
## extracting values from parameters
args = parser.parse_args()
classes = []
img_dir = args.img_dir
epochs = int(args.epochs)
dimensions = args.dimensions.split(",")
d1 = float(dimensions[0])
d2 = float(dimensions[1])
d3 = float(dimensions[2])
target_size = args.target_size.split(",")
t1 = int(float(target_size[0]))
t2 = int(float(target_size[1]))
lr = float(args.lr)
momentum = float(args.momentum)
training_args = args.training_args.split(",")
rotation_range = float(training_args[0])
zoom_range = float(training_args[1])
w_s_r = float(training_args[2])
h_s_r = float(training_args[3])
shear_range = float(training_args[4])
metrics = args.metrics
loss = args.loss
cnt_classes = 0

#### list each folder in the given directory do os.listdir ####
classes = [f for f in os.listdir(img_dir) if ".cnvrgignore" not in f and '.cnvrg' not in f and 'idx.' not in f]

### create two folders train and test ###
os.mkdir(img_dir+"/train")
os.mkdir(img_dir+"/test")

### in train and test create folders with the same name as the folders in the given dir
for clss in classes:
    os.mkdir(img_dir+"/train/"+clss)
    os.mkdir(img_dir+"/test/"+clss)
    images = os.listdir(img_dir+"/"+clss)
    imagestrain,imagestest = train_test_split(images,test_size=0.2)
    ### for each image in train move it to train/folder_name dir and repeat step for test to test/folder_name_dir
    for trainimg in imagestrain:
        shutil.move(os.path.join(img_dir,clss,trainimg),os.path.join(img_dir,"train",clss))
    for testimg in imagestest:
        shutil.move(os.path.join(img_dir,clss,testimg),os.path.join(img_dir,"test",clss))


TRAIN = img_dir+"/train/"
TEST = img_dir+"/test/"
## defining weight file locations for the default pretrained model
cnt_classes = len(classes)

WEIGHTS_PATH_NO_TOP = 'https://libhub-readme.s3.us-west-2.amazonaws.com/model_files/scene_detection/vgg16/vgg16-places365_weights_tf_dim_ordering_tf_kernels_notop.h5'
WEIGHTS_PATH = 'https://libhub-readme.s3.us-west-2.amazonaws.com/model_files/scene_detection/vgg16/vgg16-places365_weights_tf_dim_ordering_tf_kernels.h5'


BATCH_SIZE = 32
LE_PATH = os.path.sep.join(["output", "le.cpickle"])
BASE_CSV_PATH = "/cnvrg/output"
MODEL_PATH = os.path.sep.join(["output", "scene1.model"])
UNFROZEN_PLOT_PATH = os.path.sep.join(["output", "unfrozen.png"])
WARMUP_PLOT_PATH = os.path.sep.join(["output", "warmup.png"])

## VGG16 365 pretrained model initialization code
def VGG16_Places365(include_top=True, weights='places',
                    input_tensor=None, input_shape=None,
                    pooling=None,
                    classes=365):
    if not (weights in {'places', None} or os.path.exists(weights)):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization), `places` '
                         '(pre-training on Places), '
                         'or the path to the weights file to be loaded.')

    if weights == 'places' and include_top and classes != 365:
        raise ValueError('If using `weights` as places with `include_top`'
                         ' as true, `classes` should be 365')
    input_shape = _obtain_input_shape(input_shape,
                                      default_size=224,
                                      min_size=48,
                                      data_format=K.image_data_format(),
                                      require_flatten=include_top)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    # Block 1
    x = Conv2D(filters=64, kernel_size=3, strides=(1, 1), padding='same',
               kernel_regularizer=l2(0.0002),
               activation='relu', name='block1_conv1')(img_input)

    x = Conv2D(filters=64, kernel_size=3, strides=(1, 1), padding='same',
               kernel_regularizer=l2(0.0002),
               activation='relu', name='block1_conv2')(x)

    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name="block1_pool", padding='valid')(x)

    # Block 2
    x = Conv2D(filters=128, kernel_size=3, strides=(1, 1), padding='same',
               kernel_regularizer=l2(0.0002),
               activation='relu', name='block2_conv1')(x)

    x = Conv2D(filters=128, kernel_size=3, strides=(1, 1), padding='same',
               kernel_regularizer=l2(0.0002),
               activation='relu', name='block2_conv2')(x)

    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name="block2_pool", padding='valid')(x)

    # Block 3
    x = Conv2D(filters=256, kernel_size=3, strides=(1, 1), padding='same',
               kernel_regularizer=l2(0.0002),
               activation='relu', name='block3_conv1')(x)

    x = Conv2D(filters=256, kernel_size=3, strides=(1, 1), padding='same',
               kernel_regularizer=l2(0.0002),
               activation='relu', name='block3_conv2')(x)

    x = Conv2D(filters=256, kernel_size=3, strides=(1, 1), padding='same',
               kernel_regularizer=l2(0.0002),
               activation='relu', name='block3_conv3')(x)

    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name="block3_pool", padding='valid')(x)

    # Block 4
    x = Conv2D(filters=512, kernel_size=3, strides=(1, 1), padding='same',
               kernel_regularizer=l2(0.0002),
               activation='relu', name='block4_conv1')(x)

    x = Conv2D(filters=512, kernel_size=3, strides=(1, 1), padding='same',
               kernel_regularizer=l2(0.0002),
               activation='relu', name='block4_conv2')(x)

    x = Conv2D(filters=512, kernel_size=3, strides=(1, 1), padding='same',
               kernel_regularizer=l2(0.0002),
               activation='relu', name='block4_conv3')(x)

    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name="block4_pool", padding='valid')(x)

    # Block 5
    x = Conv2D(filters=512, kernel_size=3, strides=(1, 1), padding='same',
               kernel_regularizer=l2(0.0002),
               activation='relu', name='block5_conv1')(x)

    x = Conv2D(filters=512, kernel_size=3, strides=(1, 1), padding='same',
               kernel_regularizer=l2(0.0002),
               activation='relu', name='block5_conv2')(x)

    x = Conv2D(filters=512, kernel_size=3, strides=(1, 1), padding='same',
               kernel_regularizer=l2(0.0002),
               activation='relu', name='block5_conv3')(x)

    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name="block5_pool", padding='valid')(x)

    if include_top:
        # Classification block
        x = Flatten(name='flatten')(x)
        x = Dense(4096, activation='relu', name='fc1')(x)
        x = Dropout(0.5, name='drop_fc1')(x)

        x = Dense(4096, activation='relu', name='fc2')(x)
        x = Dropout(0.5, name='drop_fc2')(x)

        x = Dense(365, activation='softmax', name="predictions")(x)

    else:
        if pooling == 'avg':
            x = GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = GlobalMaxPooling2D()(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input

    # Create model.
    model = Model(inputs, x, name='vgg16-places365')

    # load weights
    if weights == 'places':
        if include_top:
            weights_path = get_file('vgg16-places365_weights_tf_dim_ordering_tf_kernels.h5',
                                    WEIGHTS_PATH,
                                    cache_subdir='models')
        else:
            weights_path = get_file('vgg16-places365_weights_tf_dim_ordering_tf_kernels_notop.h5',
                                    WEIGHTS_PATH_NO_TOP,
                                    cache_subdir='models')

        model.load_weights(weights_path)

        if K.backend() == 'theano':
            layer_utils.convert_all_kernels_in_model(model)

        if K.image_data_format() == 'channels_first':
            if include_top:
                maxpool = model.get_layer(name='block5_pool')
                shape = maxpool.output_shape[1:]
                dense = model.get_layer(name='fc1')
                layer_utils.convert_dense_weights_data_format(dense, shape, 'channels_first')

            if K.backend() == 'tensorflow':
                warnings.warn('You are using the TensorFlow backend, yet you '
                              'are using the Theano '
                              'image data format convention '
                              '(`image_data_format="channels_first"`). '
                              'For best performance, set '
                              '`image_data_format="channels_last"` in '
                              'your Keras config '
                              'at ~/.keras/keras.json.')

    elif weights is not None:
        model.load_weights(weights)
    return model


trainPath = TRAIN
testPath = TEST
totalTrain = len(list(paths.list_images(trainPath)))
totalTest = len(list(paths.list_images(testPath)))
## image extractor function from directories
trainAug = ImageDataGenerator(
    rotation_range=rotation_range,
    zoom_range=zoom_range,
    width_shift_range=w_s_r,
    height_shift_range=h_s_r,
    shear_range=shear_range,
    horizontal_flip=True,
    fill_mode="nearest")


valAug = ImageDataGenerator()
mean = np.array([d1, d2, d3], dtype="float32")
trainAug.mean = mean
valAug.mean = mean

trainGen = trainAug.flow_from_directory(
    trainPath,
    class_mode="categorical",
    target_size=(t1, t2),
    color_mode="rgb",
    shuffle=True,
    batch_size=BATCH_SIZE)


testGen = valAug.flow_from_directory(
    testPath,
    class_mode="categorical",
    target_size=(t1, t2),
    color_mode="rgb",
    shuffle=False,
    batch_size=BATCH_SIZE)

## Model fine tuning code
base_model = VGG16_Places365(weights='places', include_top=False, input_tensor=Input(shape=(t1, t2, 3)))
base_model.trainable = False
for layer in base_model.layers:
    layer.trainable = False

headModel = base_model.output
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(512, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(cnt_classes, activation="softmax")(headModel)
model_new = Model(inputs=base_model.input, outputs=headModel)

opt = SGD(lr=lr, momentum=momentum)
model_new.compile(loss=loss, optimizer=opt, metrics=[metrics])

result = model_new.fit(
    x=trainGen,
    steps_per_epoch=totalTrain // BATCH_SIZE,
    epochs=epochs)

cnvrg_workdir = os.environ.get("CNVRG_WORKDIR", "/cnvrg")
## saving weights and predicting output
model_new.save_weights(cnvrg_workdir+'/my_model_weights.h5')
testGen.reset()
keys1 = pd.DataFrame(columns=['keys', 'label'])
keys1['keys'] = list(testGen.class_indices.keys())
keys1.to_csv(cnvrg_workdir+'/class_names.csv', index=False)
predIdxs = model_new.predict(x=testGen, steps=(totalTest // BATCH_SIZE) + 1)
y_classes = predIdxs.argmax(axis=-1)
predIdxs = np.argmax(predIdxs, axis=1)
print(classification_report(testGen.classes, predIdxs, target_names=testGen.class_indices.keys()))

e = Experiment()
## logging metrics
eval_metrics = (
    pd.DataFrame(classification_report(testGen.classes, predIdxs, target_names=testGen.class_indices.keys(),output_dict=True)).transpose().reset_index())
print("lee")
eval_metrics.to_csv(cnvrg_workdir+"/eval_metrics.csv")
e.log_param(
    "accuracy",
    eval_metrics.loc[eval_metrics["index"] == "accuracy"]["precision"].item(),
)
e.log_param(
    "weighted_precision",
    eval_metrics.loc[eval_metrics["index"] == "weighted avg"]["precision"].item(),
)
e.log_param(
    "weighted_recall",
    eval_metrics.loc[eval_metrics["index"] == "weighted avg"]["recall"].item(),
)
e.log_param(
    "weighted_f1",
    eval_metrics.loc[eval_metrics["index"] == "weighted avg"]["f1-score"].item(),
)
e.log_param(
    "avg_precision",
    eval_metrics.loc[eval_metrics["index"] == "macro avg"]["precision"].item(),
)
e.log_param(
    "avg_recall",
    eval_metrics.loc[eval_metrics["index"] == "macro avg"]["recall"].item(),
)
e.log_param(
    "avg_f1", eval_metrics.loc[eval_metrics["index"] == "macro avg"]["f1-score"].item()
)

for nm in range(len(eval_metrics) - 3):
    e.log_param(eval_metrics["index"][nm] + "_precision", eval_metrics["precision"][nm])
    e.log_param(eval_metrics["index"][nm] + "_recall", eval_metrics["recall"][nm])
    e.log_param(eval_metrics["index"][nm] + "_f1-score", eval_metrics["f1-score"][nm])
    
