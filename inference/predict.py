from __future__ import division, print_function
import keras
from keras_applications.imagenet_utils import _obtain_input_shape
from tensorflow.keras.utils import get_source_inputs
import os
import warnings
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
import numpy as np
from cv2 import resize
import pandas as pd
import base64 as b6
import cv2
import magic
import pathlib
import requests
# defining the VGG16 initialization function

FILES = ['category_places.csv']

BASE_FOLDER_URL = "https://libhub-readme.s3.us-west-2.amazonaws.com/model_files/scene_detection/"

def download_model_files():
    """
    Downloads the model files if they are not already present or pulled as artifacts from a previous train task
    """
    current_dir = str(pathlib.Path(__file__).parent.resolve())
    for f in FILES:
        if not os.path.exists(current_dir + f'/{f}') and not os.path.exists('/input/Train/' + f):
            print(f'Downloading file: {f}')
            response = requests.get(BASE_FOLDER_URL + f)
            f1 = os.path.join(current_dir, f)
            with open(f1, "wb") as fb:
                fb.write(response.content)

download_model_files()


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

    x = MaxPooling2D(pool_size=(2, 2), strides=(
        2, 2), name="block1_pool", padding='valid')(x)

    # Block 2
    x = Conv2D(filters=128, kernel_size=3, strides=(1, 1), padding='same',
               kernel_regularizer=l2(0.0002),
               activation='relu', name='block2_conv1')(x)

    x = Conv2D(filters=128, kernel_size=3, strides=(1, 1), padding='same',
               kernel_regularizer=l2(0.0002),
               activation='relu', name='block2_conv2')(x)

    x = MaxPooling2D(pool_size=(2, 2), strides=(
        2, 2), name="block2_pool", padding='valid')(x)

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

    x = MaxPooling2D(pool_size=(2, 2), strides=(
        2, 2), name="block3_pool", padding='valid')(x)

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

    x = MaxPooling2D(pool_size=(2, 2), strides=(
        2, 2), name="block4_pool", padding='valid')(x)

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

    x = MaxPooling2D(pool_size=(2, 2), strides=(
        2, 2), name="block5_pool", padding='valid')(x)

    if include_top:

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
                layer_utils.convert_dense_weights_data_format(
                    dense, shape, 'channels_first')

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


# giving the paths to the original weights of the pretrained vgg16 model

WEIGHTS_PATH_NO_TOP = 'https://libhub-readme.s3.us-west-2.amazonaws.com/model_files/scene_detection/vgg16/vgg16-places365_weights_tf_dim_ordering_tf_kernels_notop.h5'
WEIGHTS_PATH = 'https://libhub-readme.s3.us-west-2.amazonaws.com/model_files/scene_detection/vgg16/vgg16-places365_weights_tf_dim_ordering_tf_kernels.h5'

model_updated = None
model = None


def predict(data):
    t1 = 224
    t2 = 224
    predicted_response = {}
    cnt = 0
    for i in data['img']:
        # decoding image data and initialization of variables
        predicted_response[cnt] = []
        decoded = b6.b64decode(i)
        extension = magic.from_buffer(decoded, mime=True).split('/')[-1]
        nparr = np.fromstring(decoded, np.uint8)
        img_dec = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
        savepath = f"img.{extension}"
        cv2.imwrite(savepath, img_dec)
        img = cv2.imread(savepath)
        image = resize(img, (224, 224))
        image = np.expand_dims(image, 0)
        if os.path.exists("/input/train/my_model_weights.h5"):
            # fine-tuning the model
            model_dir = "/input/train/my_model_weights.h5"
            class_names = "/input/train/class_names.csv"
            class_names = pd.read_csv(class_names)
            base_model = VGG16_Places365(
                weights='places', include_top=False, input_tensor=Input(shape=(t1, t2, 3)))
            base_model.trainable = False
            for layer in base_model.layers:
                layer.trainable = False
            headModel = base_model.output
            headModel = Flatten(name="flatten")(headModel)
            headModel = Dense(512, activation="relu")(headModel)
            headModel = Dropout(0.5)(headModel)
            headModel = Dense(
                class_names.shape[0], activation="softmax")(headModel)
            model_updated = Model(inputs=base_model.input, outputs=headModel)
            opt = SGD(lr=1e-4, momentum=0.9)
            model_updated.compile(loss="categorical_crossentropy",
                            optimizer=opt, metrics=["accuracy"])
            model_updated.load_weights(model_dir)
            predIdxs_proba = model_updated.predict(image)
            predIdxs = np.argmax(predIdxs_proba, axis=1)
            
            class_names['label'] = range(len(class_names['keys'].unique()))
            response = {}
            response["name"] = str(cnt)+'.' + extension
            response["labels"] = class_names.loc[class_names['label']== predIdxs[0]]['keys'].item()
            response["confidence"] = round(float(predIdxs_proba[0][predIdxs[0]]),4)
            response['model'] = model_dir
        else:
            # original model code
            script_dir = pathlib.Path(__file__).parent.resolve()
            category_path = pd.read_csv(os.path.join(
                script_dir, 'category_places.csv'))
            model = VGG16_Places365(weights='places')
            preds = model.predict(image)[0]
            response = {}
            response["name"] = str(cnt)+'.'+extension
            response["labels"] = [
                category_path.loc[category_path['label'] == np.argsort(
                    preds)[::-1][0:5][0]]['category'].item(),
                category_path.loc[category_path['label'] == np.argsort(
                    preds)[::-1][0:5][1]]['category'].item(),
                category_path.loc[category_path['label'] == np.argsort(
                    preds)[::-1][0:5][2]]['category'].item(),
                category_path.loc[category_path['label'] == np.argsort(
                    preds)[::-1][0:5][3]]['category'].item(),
                category_path.loc[category_path['label'] == np.argsort(
                    preds)[::-1][0:5][4]]['category'].item()
            ]
            response["confidence"] = [
                round(float(preds[np.argsort(preds)[::-1][0:5][0]]),4),
                round(float(preds[np.argsort(preds)[::-1][0:5][1]]),4),
                round(float(preds[np.argsort(preds)[::-1][0:5][2]]),4),
                round(float(preds[np.argsort(preds)[::-1][0:5][3]]),4),
                round(float(preds[np.argsort(preds)[::-1][0:5][4]]),4)
            ]
        predicted_response[cnt].append(response)
        cnt = cnt+1

    #return {'prediction' : str(predicted_response)}
    return predicted_response
