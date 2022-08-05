You can use this blueprint to recognize the scene in your images. You can either use the pretrained Vgg16 Model for this or the custom trained model (whose weights you can download after running the train blueprint).

In order to train this model with your data, you would need to provide a directory with multiple sub-directories, containing the different classes of images (whose scene you want to predict) all located in s3:
- img-dir
    The files in that folder should be organized like this:
        - Dataset
            -class1 : category of natural sceneries
            -class2 : category of natural sceneries
            -class3 : category of natural sceneries

1. Click on `Use Blueprint` button
2. You will be redirected to your blueprint flow page
3. In the flow, edit the following tasks to provide your data:

   In the `S3 Connector` task:
    * Under the `bucketname` parameter provide the bucket name of the data
    * Under the `prefix` parameter provide the main path to where the images are located
   In the `Train` task:
    *  Under the `img_dir` parameter provide the path to the directory including the prefix you provided in the `S3 Connector`, it should look like:
       `/input/s3_connector/<prefix>/scene_detection`

**NOTE**: You can use prebuilt data examples paths that are already provided

4. Click on the 'Run Flow' button
5. In a few minutes you will deploy a scene classification model and predict the scene in a batch of images and download the CSV file with the information about the scenery. Go to output artifacts and check for the output csv file. 

Congrats! You have deployed a custom model that classifies natural scenery in images!

[See here how we created this blueprint](https://github.com/cnvrg/scene-classification)