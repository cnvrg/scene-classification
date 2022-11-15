Use this blueprint to classify natural scenes in a batch of images. You can use either the pretrained Vgg16_365 model or a custom-trained model (the latter's weights of which can be downloaded after running the [train blueprint](../scene-classification/train_readme.md)).

To train this model with your data, provide in the S3 Connector an ` img-dir` dataset directory with multiple subdirectories containing the different classes of images, organized like the following:
* -class1 – first category of natural sceneries
* -class2 – second category of natural sceneries
* -class3 – third category of natural sceneries

Complete the following steps to run the scene-classifier model in batch mode:
1. Click the **Use Blueprint** button. The cnvrg Blueprint Flow page displays.
2. Click the **S3 Connector** task to display its dialog. Refer to [Train step 2](../scene-classification/train_readme.md) for instructions.
3. Click the **Batch-Predict** task to display its dialog.
   - Within the **Parameters** tab, provide the following Key-Value pair information:
     - Key: `batch_input` − Value: provide the path to the directory including the S3 prefix
     - `/input/s3_connector/<prefix>/scene_detection` − ensure the path adheres to this format
     NOTE: You can use prebuilt data example paths provided.
   - Click the **Advanced** tab to change resources to run the blueprint, as required.
4. Click the **Run** button. The cnvrg software deploys a scene-classification model that predicts scenes in a batch of images and outputs a CSV file with the information about the scenery.
5. Go to the Experiments > Artifacts section and locate the output CSV file.
6. Click the **output.csv** File Name to view the output CSV file.

A custom model that classifies natural scenery in images has now been deployed in batch mode. To learn how this blueprint was created, click [here](https://github.com/cnvrg/scene-classification).
