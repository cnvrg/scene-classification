Use this blueprint to immediately detect natural scenery in images. To use this pretrained scene-classification model, create a ready-to-use API endpoint that can be integrated with your data and application.

This inference blueprint uses a pretrained Vgg16_365 model, which is a VGG16 model trained on the Places-365 dataset. To use custom scene data according to your specific requirements, run this counterpart’s [training blueprint](https://metacloud.cloud.cnvrg.io/marketplace/blueprints/scene-classification-training), which trains the model and establishes an endpoint based on the newly trained model.

Complete the following steps to deploy this scene-classifier endpoint:
1. Click the **Use Blueprint** button.
2. In the dialog, select the relevant compute to deploy the API endpoint and click the **Start** button.
3. The cnvrg software redirects to your endpoint. Complete one or both of the following options:
   - Use the Try it Live section with any natural scene image to check the model.
   - Use the bottom integration panel to integrate your API with your code by copying in your code snippet.

An API endpoint that classifies natural scenes in images has now been deployed.

Click [here](link) for detailed instructions to run this blueprint. To learn how this blueprint was created, click [here](https://github.com/cnvrg/scene-classification).