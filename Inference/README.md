# Scene Classification (Endpoint)
Scene classification, aiming at classifying a scene image to one of the predefined scene categories by comprehending the entire image, is a longstanding, fundamental and challenging problem in computer vision. The rise of large-scale datasets, which constitute the corresponding dense sampling of diverse real-world scenes, and the renaissance of deep learning techniques, which learn powerful feature representations directly from big raw data, have been bringing remarkable progress in the field of scene representation and classification. 

It uses a pretrained model VGG16-365 that is trained on a Places dataset but is flexible enough to use a custom-trained model in case needed.
### Features
- Upload the image of the natural scenery and get the name of the place as output
- User can either use this endpoint directly with the base, pretrained model or use it alongside the training module (to use the custom model weights) and custom class_names file.

# Model Artifacts
- JSON Response
```
{"prediction":[{"labels":["volcano","mountain","butte","canyon","cliff"],"name":"3.jpg"},{"labels":["volcano","fishpond","pond","lake-natural","mountain"],"name":"55.jpg"}]}
```
## How to run
```
import http.client
with open('/cnvrg/test_images/test/56.jpg', 'rb') as f:
    content = f.read()
    encoded2 = base64.b64encode(content).decode("utf-8")
with open('/cnvrg/test_images/test/55.jpg', 'rb') as f:
    content = f.read()
    encoded = base64.b64encode(content).decode("utf-8")
request_dict = {'img': [{'name': '3.jpg', 'base_64_content': encoded2},{'name': '55.jpg', 'base_64_content': encoded}]}
conn = http.client.HTTPConnection("predict4-3-1.aks-cicd-Grazitti-8766.cnvrg.io", 80)
payload = "{\"input_params\":" + json.dumps(request_dict) + "}"
headers = {
    'Cnvrg-Api-Key': "YrqzFXr4HQRCvzd8LabZrAD4",
    'Content-Type': "application/json"
    }
conn.request("POST", "/api/v1/endpoints/gjtymxxhqbcbclxmrxxg", payload, headers)
res = conn.getresponse()
data = res.read()
print(data.decode("utf-8"))
```

# About VGG-16 Places-365
CNNs trained on Places365 database (latest subset of Places2 Database) could be directly used for scene recognition, while the deep scene features from the higher level layer of CNN could be used as generic features for visual recognition.
The Keras models has been obtained by directly converting the Caffe models provived by the authors
All architectures are compatible with both TensorFlow and Theano, and upon instantiation the models will be built according to the image dimension ordering set in your Keras configuration file at ~/.keras/keras.json. For instance, if you have set image_dim_ordering=tf, then any model loaded from this repository will get built according to the TensorFlow dimension ordering convention, "Width-Height-Depth".
Pre-trained weights can be automatically loaded upon instantiation (weights='places' argument in model constructor for all image models). Weights are automatically downloaded.
[VGG-16_Places-365](https://github.com/GKalliatakis/Keras-VGG16-places365)
[Places-365](http://places2.csail.mit.edu/download.html)