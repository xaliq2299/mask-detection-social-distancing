# **DATA DESCRIPTION**

We used the dataset from the **Kaggle: Face Mask Detection** challenge in order to train our mask detector. In order to use it, please download and extract it from the official webpage ([Kaggle: Face Mask Detection](https://www.kaggle.com/andrewmvd/face-mask-detection)), in the current folder (named "data") and rename it "FaceMaskDetection_Raw".

Each .png image has its own .xml annotation file containing the location of the faces and their labels among:
* without_mask;
* mask_weared_incorrecly;
* with_mask.

Here are a few statistics about the raw data:
- number of images: 853
- number of faces "without_mask":  717;
- number of faecs with "mask_weared_incorrectly": 123;
- number of faces "with_mask": 3232.

Since the data isn't equally distributed, training our models on raw data may lead to unvalid predictions. To solve this issue we proceed to data augmentation during the data preprocessing.