# CV-Assignment1
# Task-1 : Human Classification / Detection
Human Detection is a branch of Object Detection. Object Detection is the task of identifying the presence of predefined types of objects in an image. This task involves both identification of the presence of the objects and identification of the rectangular boundary surrounding each object (i.e. Object Localisation). An object detection system which can detect the class “Human” can work as a Human Detection System.
Human detection is an essential and significant task in any intelligent video surveillance system, as it provides the fundamental information for semantic understanding of the video footages. It has an obvious extension to automotive applications due to the potential for improving safety systems. Many car manufacturers (e.g. Volvo, Ford, GM, Nissan) offer this as an ADAS option in 2017. Some of the applications be given as;
o Self-driving cars: Identifying pedestrians on a road scene
o Security: Restrict access for certain people to certain places
o Retail: Analysing visitors behaviour within a supermarket
o Fashion: Identify specific brands and persons who wear them
Histogram of Oriented Gradients
HOG is an acronym for Histogram of Oriented Gradients. It's an algorithm called a feature descriptor which helps with object detection in computer vision and image processing models. HOG is a kind of feature descriptor that counts occurrences of gradient orientation in localized portions of an image.
Task:


Compute HoG features and train two classifiers i.e SVM and Random Forest. You can use sklearn library’s function feature.hog to compute HOG features. Experiment with different parameters and report the results obtained after training of both the classifiers i.e. Linear SVM and Random Forest Classifier. The classifiers are also available in sklearn library.
After successful training save the model using sklearns’ joblib package .


## Bag of Visual Words

Bag of visual words (BOVW) is commonly used in image classification. Its concept is adapted from information retrieval and NLP’s bag of words (BOW).

The general idea of bag of visual words (BOVW) is to represent an image as a set of features. Features consists of keypoints and descriptors. Keypoints are the “stand out” points in an image, so no matter the image is rotated, shrink, or expand, its keypoints will always be the same. And descriptor is the description of the keypoint. We use the keypoints and descriptors to construct vocabularies and represent each image as a frequency histogram of features that are in the image. From the frequency histogram, later, we can find another similar images or predict the category of the image.

In this assignment,extracted from the image into a bag. Features vector is nothing but a unique pattern that we can find in an image. To put it simply, Bag of Visual Word is nothing but representing an image as a collection of unordered image patches, as shown in the below illustration.

Bag of Visual Words (reference- http://people.csail.mit.edu/torralba/shortCourseRLOC/) What is the Feature? Basically, the feature of the image consists of keypoints and descriptors. Keypoints are the unique points in an image, and even if the image is rotated, shrink, or expand, its keypoints will always be the same. And descriptor is nothing but the description of the keypoint. The main task of a keypoint descriptor is to describe an interesting patch(keypoint)in an image.

Source – http://people.csail.mit.edu/torralba/shortCourseRLOC/ Image classification with Bag of Visual Words This Image classification with Bag of Visual Words technique has three steps: 1. Feature Extraction – Determination of Image features of a given label. 
2. Codebook Construction – Construction of visual vocabulary by clustering, followed by frequency analysis. 
3. 3. Classification – Classification of images based on vocabulary generated using SVM.
Task:
You need to perform image classification on two datasets, which can be downloaded from here## Data Set
INRIA Person Dataset Samples from here (https://drive.google.com/file/d/1pDr3138jwk8WuHF2CFGVlKbKvt27hMIH/view?usp=sharing ).

## Implementation

# Note: The assignement task were decomposed into small tasks containing code segment for that specific functionality
    # Task1 and Task2 are the main function performing the required functionality of assignment.

  # Task1() function containing the code for executing task 1 of assignment
    
  # load_model_plot_results will be used to Load the trained model and display the HOG of image with the predicted and actual lable value


   # load_model_display_matrices will be used to load the model and display the performance matix of the model on the test data

 # TASK2() will be used to perform the task 2 of assignment
  There are two sub functions in TASK2() which are proforming the functionality on two different data set  

   # Task2_a() will be used for performing the BOVW against object detection dataset
   
   # Task2_b() will be used for performing the BOVW against Flower detection dataset

    



## How to Run?

In order to run image classifier, uncomment the relevent Task function and execute the code after giving the  dataset path in the relevent functions.
