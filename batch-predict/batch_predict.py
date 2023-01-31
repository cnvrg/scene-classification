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
import pkg_resources
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
import argparse
import pandas as pd
import pathlib

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
    "--model_dir",
    action="store",
    dest="model_dir",
    default="/data/scene_model/",
    required=True,
    help="""model_files """,
)
parser.add_argument(
    "--using_default",
    action="store",
    dest="using_default",
    default="Yes",
    required=True,
    help="""Whether user wants to use the default model or the pretrained one""",
)
parser.add_argument(
    "--top_pred",
    action="store",
    dest="top_pred",
    default="5",
    required=False,
    help="""How many predictions to return""",
)
parser.add_argument(
    "--dimensions",
    action="store",
    dest="dimensions",
    default="123.68,116.779,103.939",
    required=False,
    help="""dimensions for flow from directory""",
)
parser.add_argument(
    "--target_size",
    action="store",
    dest="target_size",
    default="224,224",
    required=False,
    help="""dimension to which the images will be resized""",
)
parser.add_argument(
    "--class_names",
    action="store",
    dest="class_names",
    default="/data/scene_model_2/class_names.csv",
    required=False,
    help="""dataset with the class names  """,
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
    help="""optimizer argument""",
)
args = parser.parse_args()
testPath = args.img_dir
model_dir = args.model_dir
using_default = args.using_default
top_pred = int(args.top_pred)
dimensions = args.dimensions.split(",")
d1 = float(dimensions[0])
d2 = float(dimensions[1])
d3 = float(dimensions[2])
target_size = args.target_size.split(",")
t1 = int(float(target_size[0]))
t2 = int(float(target_size[1]))
class_names_0 = args.class_names
metrics = args.metrics
loss = args.loss
momentum = float(args.momentum)
lr = float(args.lr)

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

BATCH_SIZE = 32

WEIGHTS_PATH_NO_TOP = 'https://libhub-readme.s3.us-west-2.amazonaws.com/model_files/scene_detection/vgg16/vgg16-places365_weights_tf_dim_ordering_tf_kernels_notop.h5'
WEIGHTS_PATH = 'https://libhub-readme.s3.us-west-2.amazonaws.com/model_files/scene_detection/vgg16/vgg16-places365_weights_tf_dim_ordering_tf_kernels.h5'
cnvrg_workdir = os.environ.get("CNVRG_WORKDIR", "/cnvrg")

if using_default.lower().startswith('y') == True:

    valAug = ImageDataGenerator()
    mean = np.array([d1, d2, d3], dtype="float32")
    valAug.mean = mean
    testGen = valAug.flow_from_directory(
        testPath,
        class_mode="categorical",
        target_size=(t1, t2),
        color_mode="rgb",
        shuffle=False)


    script_dir = pathlib.Path(__file__).parent.resolve()
    cap = pd.read_csv(os.path.join(script_dir, "category_places.csv"))

    model = VGG16_Places365(weights='places')
    predictions_to_return = top_pred
    preds = model.predict(x=testGen)

    df = pd.DataFrame(columns=['Name'] + ['Label_{}'.format(x) for x in range(0, predictions_to_return)])
    for i in range(preds.shape[0]):
        df.at[i, 'Name'] = testGen.filenames[i]
        for j in range(predictions_to_return):
            df.at[i, 'Label_' + str(j)] = cap.loc[cap['label']==np.argsort(preds[i])[::-1][0:predictions_to_return][j]]['category'].item()
            df.at[i, 'Confidence_' + str(j)] = round(float(preds[i][np.argsort(preds[i])[::-1][0:predictions_to_return][j]]),4)

    df.to_csv(cnvrg_workdir+'/output.csv')

else:

    class_names = pd.read_csv(class_names_0)
    valAug = ImageDataGenerator()
    mean = np.array([d1, d2, d3], dtype="float32")
    valAug.mean = mean

    testGen = valAug.flow_from_directory(
        testPath,
        class_mode="categorical",
        target_size=(t1, t2),
        color_mode="rgb",
        shuffle=False)

    base_model = VGG16_Places365(weights='places', include_top=False, input_tensor=Input(shape=(t1, t2, 3)))
    base_model.trainable = False
    for layer in base_model.layers:
        layer.trainable = False

    headModel = base_model.output
    headModel = Flatten(name="flatten")(headModel)
    headModel = Dense(512, activation="relu")(headModel)
    headModel = Dropout(0.5)(headModel)
    headModel = Dense(class_names.shape[0], activation="softmax")(headModel)
    model_new = Model(inputs=base_model.input, outputs=headModel)
    predictions_to_return = top_pred
    df = pd.DataFrame(columns=['Name', 'Label', 'Confidence'])

    opt = SGD(lr=lr, momentum=momentum)
    model_new.compile(loss=loss, optimizer=opt, metrics=[metrics])
    model_new.load_weights(model_dir)
    predIdxs = model_new.predict(x=testGen)
    percentage_pred = predIdxs
    predIdxs = np.argmax(predIdxs, axis=1)
    class_names['label'] = range(len(class_names['keys'].unique()))

    for i in range(predIdxs.shape[0]):

        df.at[i, 'Name'] = testGen.filenames[i]
        df.at[i, 'Label'] = class_names.loc[class_names['label'] == predIdxs[i]]['keys'].item()
        df.at[i, 'Confidence'] = round(float(percentage_pred[i][predIdxs[i]]),4)
    df.to_csv(cnvrg_workdir+'/output.csv')
