# Scene Classification (Train)
Scene classification, aiming at classifying a scene image to one of the predefined scene categories by comprehending the entire image, is a longstanding, fundamental and challenging problem in computer vision. The rise of large-scale datasets, which constitute the corresponding dense sampling of diverse real-world scenes, and the renaissance of deep learning techniques, which learn powerful feature representations directly from big raw data, have been bringing remarkable progress in the field of scene representation and classification. 

The Training/Fine-tuning is done on the pretrained VGG16 model (trained on places dataset) with the data file that the user can upload in a dataset.
### Features
- Upload the training data images in a particular folder structure (described below) and fill the hyper paramters according to the need
- User can get the custom-trained model weights and the class names file from this module.

# Input Arguments
- `--img_dir` refers to the name of the path of the directory where images which need to be classified are stored.
- `--epochs` number of iterations for which the model will be trained.
- `--dimensions` mean value of the image data generator arguments (to which the images are resized to) while being given as an input
- `--training_args` set of image preprocessing parameters which will be used to preprocess images while given for training.
- `--target_size` (float, default = 0.5) - the resolution to which the image will be resized to, before being implemented to the model
- `--class_names` refers to the categories, into which the custom-trained model will output the results.
- `--loss` (default : 'categorical_crossentropy') refers to the loss function on which the compiling takes place. 
- `--metrics` (default : 'accuracy') refers to the metrics on which the compiling takes place
- `--lr` (default : 1e-4) learning rate schedules can help to converge the optimization process.
- `--momentum` (default : 0.9) Momentum can accelerate training and is used while compiling. 
..

# Model Artifacts
- `--my_model_weights.h5` refers to the name of the file which contains the model weights that need to be incorporated into the compiled model (ion batch_predict module)
- `--class_names.csv` the file that contains the mapping of the categories used for custom training and their indexes
- | keys      |
  | ----------- |
  | ampitheater      |
  | lava   |
  | tunnel   |
## How to run
```
python3 batch_predict/batch_predict_1.py --img_dir /data/scene_2/ --model_dir /data/scene_model_2/my_model_weights.h5 --dimensions "123.68,116.779,103.939" --top_pred 5 --using_default No --target_size "224,224" --class_names /data/scene_model_2/class_names.csv
```

# About VGG-16 Places-365
CNNs trained on Places365 database (latest subset of Places2 Database) could be directly used for scene recognition, while the deep scene features from the higher level layer of CNN could be used as generic features for visual recognition.
The Keras models has been obtained by directly converting the Caffe models provived by the authors
All architectures are compatible with both TensorFlow and Theano, and upon instantiation the models will be built according to the image dimension ordering set in your Keras configuration file at ~/.keras/keras.json. For instance, if you have set image_dim_ordering=tf, then any model loaded from this repository will get built according to the TensorFlow dimension ordering convention, "Width-Height-Depth".
Pre-trained weights can be automatically loaded upon instantiation (weights='places' argument in model constructor for all image models). Weights are automatically downloaded.
[VGG-16_Places-365](https://github.com/GKalliatakis/Keras-VGG16-places365)
[Places-365](http://places2.csail.mit.edu/download.html)
# Fine-tuning Keras Models
[Fine-Tuning Keras Model](https://pyimagesearch.com/2019/06/03/fine-tuning-with-keras-and-deep-learning/)
