You can use this blueprint to train a custom model on your own set of scene categories. You can either use the pretrained Vgg16 Model for this or the custom trained model (whose weights you can download after running the train blueprint), to get the predictions.
In order to train this model with your data, you would need to provide a directory with multiple sub-directories, containing the different classes of images all located in s3:
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
5. In a few minutes you will train a new scene classification model and deploy as a new API endpoint
6. Go to the 'Serving' tab in the project and look for your endpoint
7. You can use the `Try it Live` section with any image containing natural scenery to check your model
8. You can also integrate your API with your code using the integration panel at the bottom of the page

Congrats! You have trained and deployed a custom model that detects fire elements in images!

[See here how we created this blueprint](https://github.com/cnvrg/scene-classification)