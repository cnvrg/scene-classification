Use this blueprint to train a custom model on your own set of natural scene categories. The training and fine-tuning are performed on the pretrained VGG16-365 model (VGG16 model trained on the Places-365 dataset) with the data file the user uploads as a dataset. This blueprint also establishes an endpoint that can be used to classify scenes in images based on the newly trained model.

Users can use either the pretrained VGG16-365 model or a custom-trained model, the latter’s weights of which can be downloaded after the blueprint run. To train this model with your data, provide in the S3 Connector an ` img-dir` dataset directory with multiple subdirectories containing the different classes of images, organized like the following:
* -class1 – first category of natural sceneries
* -class2 – second category of natural sceneries
* -class3 – third category of natural sceneries

Complete the following steps to train the scene-classifier model:
1. Click the **Use Blueprint** button. The cnvrg Blueprint Flow page displays.
2. In the flow, click the **S3 Connector** task to display its dialog.
   * Within the **Parameters** tab, provide the following Key-Value pair information:
     - Key: `bucketname` - Value: enter the data bucket name
     - Key: `prefix` - Value: provide the main path to the images folder
   * Click the **Advanced** tab to change resources to run the blueprint, as required.
3. Return to the flow and click the **Train** task to display its dialog.
   * Within the **Parameters** tab, provide the following Key-Value pair information:
     - Key: `img_dir` – Value: provide the path to the images folder including the S3 prefix
     - `/input/s3_connector/<prefix>/scene_detection` − ensure the path adheres this format
   NOTE: You can use prebuilt data examples paths already provided.
   * Click the **Advanced** tab to change resources to run the blueprint, as required.
4.	Click the **Run** button. The cnvrg software launches the training blueprint as set of experiments, generating a trained scene-classifier model and deploying it as a new API endpoint.
5. Track the blueprint's real-time progress in its experiments page, which displays artifacts such as logs, metrics, hyperparameters, and algorithms.
6. Click the **Serving** tab in the project, locate your endpoint, and complete one or both of the following options:
   * Use the Try it Live section with any natural scene image to check the model.
   * Use the bottom integration panel to integrate your API with your code by copying in your code snippet.

A custom model and an API endpoint which can classify an image’s scenes have now been trained and deployed.

Click [here](link) for detailed instructions on this blueprint's run. To learn how this blueprint was created, click [here](https://github.com/cnvrg/scene-classification).
