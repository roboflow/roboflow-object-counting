# roboflow-object-counting
Interface with the Roboflow API and Python package for object counting in your computer vision models. https://docs.roboflow.com/python#object-counting

![Roboflow Logo](https://camo.githubusercontent.com/b9468c9d506b644007e50189fd2aa5d5f158b992bb21569222fe3967e608c467/68747470733a2f2f692e696d6775722e636f6d2f6c58436f5674352e706e67)

# Tell us about what you built, ask us questions, or provide feedback for improvements! opensource@roboflow.com and our [Community Forum](https://discuss.roboflow.com):
## [Website](https://docs.roboflow.com/python) • [Docs](https://docs.roboflow.com) • [Blog](https://blog.roboflow.com) • [Twitter](https://twitter.com/roboflow) • [Linkedin](https://www.linkedin.com/company/roboflow-ai) • [Roboflow Universe](https://universe.roboflow.com) • [Knowledge Base](https://help.roboflow.com)

# What is Roboflow?
## **Roboflow** makes managing, preprocessing, augmenting, and versioning datasets for computer vision seamless. This repo utilizes
## the official [Roboflow python package](https://docs.roboflow.com/python) that interfaces with the [Roboflow API](https://docs.roboflow.com/inference/hosted-api). Key features of Roboflow:

- Import and Export image datasets into any supported [formats](https://roboflow.com/formats)
- [Preprocess](https://docs.roboflow.com/image-transformations/image-preprocessing)
  and [augment](https://docs.roboflow.com/image-transformations/image-augmentation) data using Roboflow's dataset
  management tools
- Train computer vision models using [Roboflow Train](https://docs.roboflow.com/train) and deploy
  to [production](https://docs.roboflow.com/inference)
- Use [community curated projects](https://universe.roboflow.com/) to start building your own vision-powered products

## Installation (Dependencies):

To install the Python package, please use `Python 3.6` or higher. We provide three different ways to install the Roboflow
package to use within your own projects.

Install from PyPi (Recommended):

```
pip install roboflow
```

Install from Source:

```
git clone https://github.com/roboflow-ai/roboflow-python.git
cd roboflow-python
python3 -m venv env
source env/bin/activate 
pip3 install -r requirements.txt
```

### Setting up the logic for object counts on a single target class:
- [Obtaining Your API Key](https://docs.roboflow.com/rest-api#obtaining-your-api-key)

Colab Tutorial Here:
<a href="https://colab.research.google.com/drive/1oK9DBe43uImIgH4GKq_rGtDVBSuEXyTb?usp=sharing"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>
- Select "File" in the Google Colab menu, and "Save a Copy in Drive" prior to running the notebook
```
from roboflow import Roboflow
import os, sys, re, glob

# obtaining your API key: https://docs.roboflow.com/rest-api#obtaining-your-api-key
rf = Roboflow(api_key="INSERT_PRIVATE_API_KEY")
workspace = rf.workspace("INSERT_WORKSPACE_ID")
project = rf.workspace("INSERT_WORKSPACE_ID").project("INSERT_MODEL/PROJECT_ID")
# replace REPLACE_WITH_MODEL_VERSION_NUM with your model version number
version = project.version(REPLACE_WITH_MODEL_VERSION_NUM)
model = version.model

def count_object_occurances(predictions, target_class):
  """
    Helper method to count the number of objects in an image for a given class
    :param predictions: predictions returned from calling the predict method
    :param target_class: str, target class for object count
    :return: dictionary with target class and total count of occurrences in image
  """
  object_counts = {target_class : 0}
  for prediction in predictions:
    if prediction['class'] in target_class:
      object_counts[prediction['class']] += 1
  return object_counts
```

### Run model inference and object counting on a single image file:
```
# perform inference on the selected image
predictions = model.predict("YOUR_IMAGE.jpg") # or
## uncomment the following line to run inference on a hosted image
# prediction_hosted = model.predict("https://www.yourimageurl.com", hosted=True)

## replace target_class with name of target_class
## example, target class is 'face': count_object_occurances(predictions, 'face')
class_counts = count_object_occurances(predictions, target_class)
print(predictions, class_counts)
print('\n')
```
### Run model inference and object counting on a folder (directory) of image files:
```
raw_data_location = "INSERT_PATH_TO_IMG_DIRECTORY"
raw_data_extension = ".jpg" # e.g jpg, jpeg, png

globbed_files = glob.glob(raw_data_location + '/*' + raw_data_extension)
## replace target_class with name of target_class
## example, target class is 'face': count_object_occurances(predictions, 'face')
for img_file in globbed_files:
    predictions = model.predict(img_file)
    class_counts = count_object_occurances(predictions, target_class)
    print(predictions, class_counts)
    print('\n')
```