
import cv2
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage.feature import hog
from sklearn import svm
from sklearn.metrics import classification_report,accuracy_score,confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import joblib
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split


def resize_Images(list_Images):
    list_resized_img=[]
    for x in list_Images:
        resized_image = cv2.resize(x, (96, 160))
        list_resized_img.append(resized_image)
    return list_resized_img

def calculate_Hog(listImages,label):
    if(label==1):
        print('Calculating hog for pos images')
    else:
        print('Calculating hog for neg images')
    list_fd = []
    list_hogimg = []
    list_label = []
    for img in listImages:
        fd, hog_image = hog(img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)
        list_fd.append(fd)
        list_label.append(label)
        list_hogimg.append(hog_image)
    return list_fd , list_hogimg,list_label

def pil_load_images_from_folder(folder):
    images_gray = []
    images_rgb = []
    images_name = []
    for filename in os.listdir(folder):
        img_gray = Image.open(fp=os.path.join(folder, filename),mode='r').convert('LA')
        newsize = (96, 160)
        img_gray = img_gray.resize(newsize)
        img_rgb = Image.open(fp=os.path.join(folder, filename),mode='r')
        images_name.append(filename)
        if img_rgb is not None:
            images_rgb.append(img_rgb)
        if img_gray is not None:
            images_gray.append(img_gray)
    return images_rgb,images_gray,images_name
    # This method will show image in any image viewer
    #im.show()
def load_images_from_folder(folder):
    images_gray = []
    images_rgb = []
    Images_name=[]
    for filename in os.listdir(folder):
        img_gray = cv2.imread(os.path.join(folder, filename), 0)
        img_rgb = cv2.imread(os.path.join(folder, filename), 1)
        Images_name.append(filename)
        if img_gray is not None:
            images_gray.append(img_gray)
        if img_rgb is not None:
            images_rgb.append(img_rgb)

    return images_gray, images_rgb,Images_name
def read_images_from_array (array_img_path):
    train_labels = np.array([])
    train_label_name = []
    images_gray = []
    images_rgb = []
    for img in array_img_path:
        print("Reading image from "+str(img))

        img_gray = cv2.imread(img, 0)
        img_rgb = cv2.imread(img, 1)

        newsize = (150, 150)
        img_gray = cv2.resize(img_gray, newsize)
        img_rgb= cv2.resize(img_rgb, newsize)
        images_gray.append(img_gray)
        images_rgb.append(img_rgb)
        if ("Soccer_Ball" in img):
            class_index = 0
            train_label_name.append("Soccer_Ball")
        elif ("motorbike" in img):
            class_index = 1
            train_label_name.append("motorbike")
        elif ("dollar_bill" in img):
            class_index = 2
            train_label_name.append("dollar_bill")
        elif ("accordion" in img):
            class_index = 3
            train_label_name.append("accordion")

        train_labels = np.append(train_labels, class_index)
    return images_gray,images_rgb,train_labels,train_label_name
def read_images_from_array2 (array_img_path):
    train_labels = np.array([])
    train_label_name = []
    images_gray = []
    #images_rgb = []
    for img in array_img_path:
        print("Reading image from "+str(img))

        img_gray = cv2.imread(img, 0)
        #img_rgb = cv2.imread(img, 1)

        newsize = (150, 150)
        img_gray = cv2.resize(img_gray, newsize)
        #img_rgb= cv2.resize(img_rgb, newsize)
        images_gray.append(img_gray)
        #images_rgb.append(img_rgb)
        if ("daisy" in img):
            class_index = 0
            train_label_name.append("daisy")
        elif ("dandelion" in img):
            class_index = 1
            train_label_name.append("dandelion")
        elif ("roses" in img):
            class_index = 2
            train_label_name.append("roses")
        elif ("sunflowers" in img):
            class_index = 3
            train_label_name.append("sunflowers")
        elif ("tulips" in img):
            class_index = 4
            train_label_name.append("tulips")

        train_labels = np.append(train_labels, class_index)
    return images_gray,train_labels


def getFiles(train, path):
    images = []
    count = 0
    for folder in os.listdir(path):
        for file in os.listdir(os.path.join(path, folder)):
            images.append(str(os.path.join(path, os.path.join(folder, file))))

    if (train is True):
        np.random.shuffle(images)

    return images
def get_lables_Objects_Dataset_Images(arr_image_path):
    train_labels = np.array([])
    train_label_name=[]
    label_count = 4
    for img_path in arr_image_path:
        if ("Soccer_Ball" in img_path):
            class_index = 0
            train_label_name.append("Soccer_Ball")
        elif ("motorbike" in img_path):
            class_index = 1
            train_label_name.append("motorbike")
        elif ("dollar_bill" in img_path):
            class_index = 2
            train_label_name.append("dollar_bill")
        elif ("accordion" in img_path):
            class_index = 3
            train_label_name.append("accordion")

        train_labels = np.append(train_labels, class_index)
    return train_labels,train_label_name
def feature_extraction_from_array(arr_images,extractor):
    kp=[]
    desc=[]

    for img in arr_images:
        keypoints, descriptors = extractor.detectAndCompute(img, None)
        kp.append(keypoints)
        desc.append(descriptors)
    return kp,desc

def getDescriptors(sift, img):
    kp, des = sift.detectAndCompute(img, None)
    return des
def clusterDescriptors(descriptors, no_clusters):
    kmeans = KMeans(n_clusters = no_clusters).fit(descriptors)
    return kmeans

def task1a_linear_svm(X_input,y_output,test_input,test_output):
    print('start learning SVM.')
    lin_clf = svm.LinearSVC(max_iter=25000)
    lin_clf.fit(X_input, y_output)
    # clf = svm.SVC()
    # clf.fit(X, y)
    print('finish learning SVM.')
    print(lin_clf.fit(X_input, y_output))
    y_pred = lin_clf.predict(test_input)
    print("Accuracy: " + str(accuracy_score(test_output, y_pred)))
    print('\n')
    print(classification_report(test_output, y_pred))
    joblib.dump(lin_clf, '25k_iter_person_detector.pkl', compress=9)
def tsak1b_random_forest_tree(X_input,y_output,test_input,test_output):
    rft_clf = RandomForestClassifier(n_estimators=100)

    # Train the model using the training sets y_pred=clf.predict(X_test)
    rft_clf.fit(X_input, y_output)
    y_pred = rft_clf.predict(test_input)
    print("Accuracy:", metrics.accuracy_score(test_output, y_pred))
    joblib.dump(rft_clf, 'n_estimator_100_person_detector.pkl', compress=9)
    print(classification_report(test_output, y_pred))

    tn, fp, fn, tp=confusion_matrix(test_output, y_pred)
    print(tn, fp, fn, tp)

def load_model_plot_results(filename):
    test_posimgrgb, test_posimggray, test_posimgName = pil_load_images_from_folder("C://Users//dell//Downloads//phd//Computer vision cs 867//assignments//Assignment2//INRIA_Dataset_Samples//Test//post")
    test_negimgrgb, test_negimggray, test_negimgName = pil_load_images_from_folder("C://Users//dell//Downloads//phd//Computer vision cs 867//assignments//Assignment2//INRIA_Dataset_Samples//Test//negt")
    test_pos_fd, test_pos_hog, test_pos_label = calculate_Hog(test_posimggray, 1)
    test_neg_fd, test_neg_hog, test_neg_label = calculate_Hog(test_negimggray, 0)
    test_input_hog = np.asarray(np.vstack((test_pos_hog, test_neg_hog)))
    test_input = np.asarray(np.vstack((test_pos_fd,test_neg_fd)))
    test_output = np.asarray(np.hstack((test_pos_label,test_neg_label)))
    loaded_model = joblib.load(filename)
    y_pred = loaded_model.predict(test_input)
    print(len(y_pred))
    i=0
    while(i<len(y_pred)):
        if(test_output[i]==1):
            plt.title('Ground Truth:' + str(test_output[i]) + ' Pridicted value :' + str(y_pred[i]))
            plt.imshow(test_input_hog[i])
            plt.show()
        #plt.savefig('FAST' + str(name))

        i+=1
    print("Accuracy: " + str(accuracy_score(test_output, y_pred)))
    print('\n')
    print(classification_report(test_output, y_pred))
    print(y_pred,test_output)

def display_all_images(array_images,class_name):
    for x in range(len(array_images)):
        cv2.imshow('Image of class '+str(class_name[x]) ,array_images[x])
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def plotHistogram(im_features, no_clusters):
    x_scalar = np.arange(no_clusters)
    y_scalar = np.array([abs(np.sum(im_features[:,h], dtype=np.int32)) for h in range(no_clusters)])

    plt.bar(x_scalar, y_scalar)
    plt.xlabel("Visual Word Index")
    plt.ylabel("Frequency")
    plt.title("Complete Vocabulary Generated")
    plt.xticks(x_scalar + 0.4, x_scalar)
    plt.show()

def vstackDescriptors(descriptor_list):
    descriptors = np.array(descriptor_list[0])
    for descriptor in descriptor_list[1:]:
        descriptors = np.vstack((descriptors, descriptor))
    return descriptors

def clusterDescriptors(descriptors, no_clusters):
    kmeans = KMeans(n_clusters = no_clusters).fit(descriptors)
    return kmeans

def extractFeatures(kmeans, descriptor_list, image_count, no_clusters):
    im_features = np.array([np.zeros(no_clusters) for i in range(image_count)])
    for i in range(image_count):
        for j in range(len(descriptor_list[i])):
            feature = descriptor_list[i][j]
            feature = feature.reshape(1, 128)
            idx = kmeans.predict(feature)
            im_features[i][idx] += 1

    return im_features

def normalizeFeatures(scale, features):
    return scale.transform(features)

def svcParamSelection(X, y, kernel, nfolds):
    Cs = [0.5, 0.1, 0.15, 0.2, 0.3]
    gammas = [0.1, 0.11, 0.095, 0.105]
    param_grid = {'C': Cs, 'gamma' : gammas}
    grid_search = GridSearchCV(SVC(kernel=kernel), param_grid, cv=nfolds)
    grid_search.fit(X, y)
    grid_search.best_params_
    return grid_search.best_params_


def findSVM(im_features, train_labels, kernel):
    features = im_features
    if (kernel == "precomputed"):
        features = np.dot(im_features, im_features.T)

    params = svcParamSelection(features, train_labels, kernel, 5)
    C_param, gamma_param = params.get("C"), params.get("gamma")
    print(C_param, gamma_param)
    class_weight = {
        0: (807 / (4 * 140)),
        1: (807 / (4 * 140)),
        2: (807 / (4 * 133)),
        3: (807 / (4 * 70))
#        4: (807 / (7 * 42)),
#        5: (807 / (7 * 140)),
#        6: (807 / (7 * 142))
    }

    svm = SVC(kernel=kernel, C=C_param, gamma=gamma_param, class_weight=class_weight)
    svm.fit(features, train_labels)
    return svm

def findSVM2(im_features, train_labels, kernel):
    features = im_features
    if (kernel == "precomputed"):
        features = np.dot(im_features, im_features.T)

    params = svcParamSelection(features, train_labels, kernel, 5)
    C_param, gamma_param = params.get("C"), params.get("gamma")
    print(C_param, gamma_param)
    class_weight = {
        0: (807 / (5 * 140)),
        1: (807 / (5 * 140)),
        2: (807 / (5 * 133)),
        3: (807 / (5 * 70)),
        4: (807 / (5 * 42))
#        5: (807 / (7 * 140)),
#        6: (807 / (7 * 142))
    }

    svm = SVC(kernel=kernel, C=C_param, gamma=gamma_param, class_weight=class_weight)
    svm.fit(features, train_labels)
    return svm


def load_model_display_matrices(filename):
    test_posimgrgb, test_posimggray, test_posimgName = pil_load_images_from_folder(
        "C://Users//dell//Downloads//phd//Computer vision cs 867//assignments//Assignment2//INRIA_Dataset_Samples//Test//pos")
    test_negimgrgb, test_negimggray, test_negimgName = pil_load_images_from_folder(
        "C://Users//dell//Downloads//phd//Computer vision cs 867//assignments//Assignment2//INRIA_Dataset_Samples//Test//neg")
    test_pos_fd, test_pos_hog, test_pos_label = calculate_Hog(test_posimggray, 1)
    test_neg_fd, test_neg_hog, test_neg_label = calculate_Hog(test_negimggray, 0)
    test_input_hog = np.asarray(np.vstack((test_pos_hog, test_neg_hog)))
    test_input = np.asarray(np.vstack((test_pos_fd, test_neg_fd)))
    test_output = np.asarray(np.hstack((test_pos_label, test_neg_label)))
    loaded_model = joblib.load(filename)
    y_pred = loaded_model.predict(test_input)
    print("Accuracy: " + str(accuracy_score(test_output, y_pred)))
    print('\n')
    print(classification_report(test_output, y_pred))


def Task1():

    train_posimgrgb, train_posimggray, train_posimgName = pil_load_images_from_folder("C://Users//dell//Downloads//phd//Computer vision cs 867//assignments//Assignment2//INRIA_Dataset_Samples//Train//pos")
    train_negimgrgb, train_negimggray, train_negimgName = pil_load_images_from_folder("C://Users//dell//Downloads//phd//Computer vision cs 867//assignments//Assignment2//INRIA_Dataset_Samples//Train//neg")
    test_posimgrgb, test_posimggray, test_posimgName = pil_load_images_from_folder("C://Users//dell//Downloads//phd//Computer vision cs 867//assignments//Assignment2//INRIA_Dataset_Samples//Test//pos")
    test_negimgrgb, test_negimggray, test_negimgName = pil_load_images_from_folder("C://Users//dell//Downloads//phd//Computer vision cs 867//assignments//Assignment2//INRIA_Dataset_Samples//Test//neg")
    train_pos_fd, train_pos_hog, pos_label = calculate_Hog(train_posimggray, 1)
    train_neg_fd, train_neg_hog, neg_label = calculate_Hog(train_negimggray, 0)
    test_pos_fd, test_pos_hog, test_pos_label = calculate_Hog(test_posimggray, 1)
    test_neg_fd, test_neg_hog, test_neg_label = calculate_Hog(test_negimggray, 0)
    X_input = np.asarray(np.vstack((train_pos_fd, train_neg_fd)))
    y_output = np.asarray(np.hstack((pos_label, neg_label)))
    test_input = np.asarray(np.vstack((test_pos_fd, test_neg_fd)))
    test_output = np.asarray(np.hstack((test_pos_label, test_neg_label)))

    #task1a_linear_svm(X_input, y_output, test_input, test_output)
    tsak1b_random_forest_tree(X_input, y_output, test_input, test_output)


def plotConfusionMatrix(y_true, y_pred, classes,
                        normalize=False,
                        title=None,
                        cmap=plt.cm.Blues):
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


def plotConfusions(true, predictions,dataset_type):
    np.set_printoptions(precision=2)
    class_names=[]
    if(dataset_type==1):
        class_names = ["Soccer_Ball", "motorbike", "dollar_bill", "accordion"]
    elif(dataset_type==2):
     class_names = ["daisy", "dandelion", "roses", "sunflowers", "tulips"]


    plotConfusionMatrix(true, predictions, classes=class_names,
                        title='Confusion matrix, without normalization')

    plotConfusionMatrix(true, predictions, classes=class_names, normalize=True,
                        title='Normalized confusion matrix')

    plt.show()

def findAccuracy(true, predictions):
    print ('accuracy score: %0.3f' % accuracy_score(true, predictions))

def create_label_from_index(label_index):
    label_name=[]
    for value in label_index:

        if ( value==0.0):
           # label_name.append("daisy") = 0
            label_name.append("daisy")
        elif (value==1.0):
            #class_index = 1
            label_name.append("dandelion")
        elif (value==2.0):
            #class_index = 2
            label_name.append("roses")
        elif ( value==3.0):
            #class_index = 3
            label_name.append("sunflowers")
        elif (value==4.0):
           # class_index = 4
            label_name.append("tulips")
    return label_name

def testModel( kmeans, scale, svm, im_features, no_clusters, kernel,descriptor_list,label_index,label_name):

    image_count = len(descriptor_list)
    print(label_name,label_index)

    print(len(descriptor_list))


    name_dict = {
        "0": "Soccer_Ball",
        "1": "motorbike",
        "2": "dollar_bill",
        "3": "accordion"
    }



    #descriptors = vstackDescriptors(descriptor_list)

    test_features = extractFeatures(kmeans, descriptor_list, image_count, no_clusters)

    test_features = scale.transform(test_features)

    kernel_test = test_features
    if (kernel == "precomputed"):
        kernel_test = np.dot(test_features, im_features.T)

    predictions = [name_dict[str(int(i))] for i in svm.predict(kernel_test)]
    print("Test images classified.")

    plotConfusions(label_name, predictions,1)
    print("Confusion matrixes plotted.")

    findAccuracy(label_name, predictions)
    print("Accuracy calculated.")
    print("Execution done.")
    print("Accuracy:", metrics.accuracy_score(label_name, predictions))
    #joblib.dump(rft_clf, 'n_estimator_100_person_detector.pkl', compress=9)
    print(classification_report(label_name, predictions))

    tn, fp, fn, tp=confusion_matrix(label_name, predictions)
    print(tn, fp, fn, tp)
def testModel2( kmeans, scale, svm, im_features, no_clusters, kernel,descriptor_list,label_name):

    image_count = len(descriptor_list)
    print(label_name)

    print(len(descriptor_list))

    name_dict = {
        "0": "daisy",
        "1": "dandelion",
        "2": "roses",
        "3": "sunflowers",
        "4":"tulips"
    }



    #descriptors = vstackDescriptors(descriptor_list)

    test_features = extractFeatures(kmeans, descriptor_list, image_count, no_clusters)

    test_features = scale.transform(test_features)

    kernel_test = test_features
    if (kernel == "precomputed"):
        kernel_test = np.dot(test_features, im_features.T)


    predictions = [name_dict[str(int(i))] for i in svm.predict(kernel_test)]
    print("Test images classified.")

    label_name2=create_label_from_index(label_name)

    plotConfusions(label_name2, predictions,2)
    print("Confusion matrixes plotted.")

    findAccuracy(label_name2, predictions)
    print("Accuracy calculated.")
    print("Execution done.")
    print("Accuracy:", metrics.accuracy_score(label_name2, predictions))
    #joblib.dump(rft_clf, 'n_estimator_100_person_detector.pkl', compress=9)
    print(classification_report(label_name2, predictions))

    #tn, fp, fn, tp=confusion_matrix(label_name2, predictions)
    #print(tn, fp, fn, tp)


def Task2_a():
    number_of_cluster=50

    array_images_obj_dataset= getFiles(False,"D://Objects_Dataset_Images//train//")
    images_gray,images_rgb,class_index,class_name=read_images_from_array(array_images_obj_dataset)
    kernel = "precomputed"
    sift = cv2.SIFT_create()
    kp,desc=feature_extraction_from_array(images_gray,sift)
    print("Extracting Features")
    desc_list = vstackDescriptors(desc)
    print("Descriptors vstacked.")
    k_mean = clusterDescriptors(desc_list,number_of_cluster)
    print("Descriptors clustered.")
    image_count = len(images_gray)
    im_features = extractFeatures(k_mean, desc, image_count, number_of_cluster)
    print("Images features extracted.")
    scale = StandardScaler().fit(im_features)
    im_features = scale.transform(im_features)
    print("Train images normalized.")
    plotHistogram(im_features, number_of_cluster)
    print("Features histogram plotted.")
    svm = findSVM(im_features, class_index, kernel)
    print("SVM fitted.")
    print("Training completed.")
    print("Test images loaded.")
    test_images_path = getFiles(False, 'D://Objects_Dataset_Images//test//')
    test_images_gray, test_images_rgb, test_class_index, test_class_name = read_images_from_array(test_images_path)
    test_kp,test_desc=feature_extraction_from_array(test_images_gray,sift)
    print("Extracting Features")
    testModel(k_mean, scale, svm, im_features, number_of_cluster, kernel,test_desc,test_class_index,test_class_name)



def Task2_b():
    number_of_cluster=100
    #flower_photos2
    array_images_obj_dataset= getFiles(False,"D://flower_photos2//")
    images_gray,class_index=read_images_from_array2(array_images_obj_dataset)
    X_train, X_test, y_train, y_test = train_test_split(images_gray, class_index, test_size = 0.20, random_state = 42)
    images_gray = X_train
    test_images_gray=X_test
    test_class_name=y_test


    kernel = "precomputed"
    sift = cv2.SIFT_create()

    kp,desc=feature_extraction_from_array(images_gray,sift)

    print("Extracting Features")
    desc_list = vstackDescriptors(desc)
    print("Descriptors vstacked.")
    k_mean = clusterDescriptors(desc_list,number_of_cluster)
    print("Descriptors clustered.")
    image_count = len(images_gray)
    im_features = extractFeatures(k_mean, desc, image_count, number_of_cluster)
    print("Images features extracted.")
    scale = StandardScaler().fit(im_features)
    im_features = scale.transform(im_features)
    print("Train images normalized.")
    plotHistogram(im_features, number_of_cluster)
    print("Features histogram plotted.")
    svm = findSVM2(im_features, y_train, kernel)
    print("SVM fitted.")
    print("Training completed.")
    print("Test Started.")
    test_kp,test_desc=feature_extraction_from_array(test_images_gray,sift)
    print("Extracting Features")
    testModel2(k_mean, scale, svm, im_features, number_of_cluster, kernel,test_desc,test_class_name)






def Task2():
    #--Call to the Function for performing the BOVW against object detection dataset
    #Task2_a()
    #--Call to the Function for performing the BOVW against Flower detection dataset

    Task2_b()

def main():

    #--Note: The assignement task were decomposed into small tasks containing code segment for that specific functionality
    #Task1 and Task2 are the main function performing the required functionality of assignment.

    #--Call to the function containing the code for executing task 1 of assignment
    #Task1()



    #--Load the trained model and display the HOG of image with the predicted and actual lable value
    #load_model_plot_results('person_detector.pkl')

    #--load the model and display the performance matix of the model on the test data
    #load_model_display_matrices('person_detector.pkl')

    # --Call to the function containing the code for executing task 2 of assignment
    Task2()


if __name__ == '__main__':
    main()
