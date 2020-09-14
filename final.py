import cv2
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA

from sklearn.model_selection import train_test_split
import os

class Visual_BOW():
    def __init__(self, k=20, dictionary_size=50):
        self.k = k  # number of SIFT features to extract from every image
        self.dictionary_size = dictionary_size  # size of your "visual dictionary" (k in k-means)
        self.n_tests = 5  # how many times to re-run the same algorithm (to obtain average accuracy)
        self.curr_test = 1
        self.colors = {}

    def extract_sift_features(self):
        '''
        To Do:
            - load/read the Caltech-101 dataset
            - go through all the images and extract "k" SIFT features from every image
            - divide the data into training/testing (70% of images should go to the training set, 30% to testing)
        Useful:
            k: number of SIFT features to extract from every image
        Output:
            train_features: list/array of size n_images_train x k x feature_dim
            train_labels: list/array of size n_images_train
            test_features: list/array of size n_images_test x k x feature_dim
            test_labels: list/array of size n_images_test
        '''

        subfolders = os.listdir('./101_ObjectCategories')
        images_arr=[]
        labels=[]
        self.colors = {}
        for folders, image_type in enumerate(subfolders):
            if(folders):
                images_in_subfolder = os.listdir('./101_ObjectCategories/'+image_type)
                self.colors[image_type] = ((folders+10)/120, (folders+10)/120, 1-(folders+20)/200)
                if(image_type=='BACKGROUND_GOOGLE'):
                    continue
                for img in images_in_subfolder:
                    cv2_img = cv2.imread('./101_ObjectCategories/'+image_type+'/'+img)
                    cv2_gray_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2GRAY)
                    sift = cv2.xfeatures2d.SIFT_create()
                    (key_points, descriptors) = sift.detectAndCompute( cv2_gray_img, None)
                    if(key_points):
                        images_arr.append(np.array(descriptors[:self.k]))
                        labels.append(image_type)
        train_features, test_features, train_labels, test_labels = train_test_split(images_arr, labels, test_size=0.3)
        print('finished feature extraction')
        return train_features, train_labels, test_features, test_labels


    def create_dictionary(self, features):
        '''
        To Do:
            - go through the list of features
            - flatten it to be of size (n_images x k) x feature_dim (from 3D to 2D)
            - use k-means algorithm to group features into "dictionary_size" groups
        Useful:
            dictionary_size: size of your "visual dictionary" (k in k-means)
        Input:
            features: list/array of size n_images x k x feature_dim
        Output:
            kmeans: trained k-means object (algorithm trained on the flattened feature list)
        '''
        flattned_list = []
        for image in features: 
                for sift_feature in image:
                    flattned_list.append(sift_feature)
        kmeans = KMeans(n_clusters=self.dictionary_size, random_state=0).fit(flattned_list)
        print('finished kmeans')
        return kmeans

    def convert_features_using_dictionary(self, kmeans, features):
        '''
        To Do:
            - go through the list of features (images)
            - for every image go through "k" SIFT features that describes it
            - every image will be described by a single vector of length "dictionary_size"
            and every entry in that vector will indicate how many times a SIFT feature from a particular
            "visual group" (one of "dictionary_size") appears in the image. Non-appearing features are set to zeros.
        Input:
            features: list/array of size n_images x k x feature_dim
        Output:
            features_new: list/array of size n_images x dictionary_size
        '''
        features_new=[]
        kmeans_centers = list(kmeans.cluster_centers_)
        for image in features:
            feature_from_dic = [0] * self.dictionary_size
            for descriptor in image:
                closet_center_index = kmeans.predict([descriptor])
                feature_from_dic[closet_center_index[0]] += 1
            features_new.append(feature_from_dic)   

        return features_new

    def train_svm(self, inputs, labels):
        '''
        To Do:
            - train an SVM classifier using the data
            - return the trained object
        Input:
            inputs: new features (converted using the dictionary) of size n_images x dictionary_size
            labels: list/array of size n_images
        Output:
            clf: trained svm classifier object (algorithm trained on the inputs/labels data)
        '''
        clf = svm.SVC()
        clf.fit(inputs, labels)
        print('finished clf')
        return clf

    def test_svm(self, clf, inputs, labels):
        '''
        To Do:
            - test the previously trained SVM classifier using the data
            - calculate the accuracy of your model
        Input:
            clf: trained svm classifier object (algorithm trained on the inputs/labels data)
            inputs: new features (converted using the dictionary) of size n_images x dictionary_size
            labels: list/array of size n_images
        Output:
            accuracy: percent of correctly predicted samples
        '''
        predictions = clf.predict(inputs)
        correct = 0
        total = len(predictions)
        for i in range(total):
            if(predictions[i]==labels[i]):
                correct+=1
        accuracy = correct / total
        return accuracy

    def save_plot(self, features, labels):
        '''
        To Do:
            - perform PCA on your features
            - use only 2 first Principle Components to visualize the data (scatter plot)
            - color-code the data according to the ground truth label
            - save the plot
        Input:
            features: new features (converted using the dictionary) of size n_images x dictionary_size
            labels: list/array of size n_images
        '''
        #pca
        pca = PCA(n_components=2)
        pca.fit(features)
        components = pca.components_.transpose()
        transformed = np.dot(features,components)
        feature1, feature2 = zip(*transformed)
        
        #plot
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.title('Features with after PC=2 tranformation')


        fig = plt.figure()
        ax = fig.add_subplot(111)
        for i in range(len(feature1)):
            ax.plot(feature1[i], feature2[i], 'o', color=self.colors[labels[i]])
        

        if not os.path.exists('./plots'):
            os.makedirs('./plots')

        dir_name = f'./plots/plot-k={self.k}-dict_size={self.dictionary_size}'  
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        fname=f'plt-t{self.curr_test}'
        plt.savefig(dir_name+'/'+fname+'.png')
        self.curr_test+=1

        #change to be directory based ^
        return 0



############################################################################
################## DO NOT MODIFY ANYTHING BELOW THIS LINE ##################
############################################################################

    def algorithm(self):
        # This is the main function used to run the program
        # DO NOT MODIFY THIS FUNCTION
        accuracy = 0.0
        for i in range(self.n_tests):
            train_features, train_labels, test_features, test_labels = self.extract_sift_features()
            kmeans = self.create_dictionary(train_features)
            train_features_new = self.convert_features_using_dictionary(kmeans, train_features)
            classifier = self.train_svm(train_features_new, train_labels)
            test_features_new = self.convert_features_using_dictionary(kmeans, test_features)
            accuracy += self.test_svm(classifier, test_features_new, test_labels)
            self.save_plot(test_features_new, test_labels)
        accuracy /= self.n_tests
        return accuracy

if __name__ == "__main__":
    alg = Visual_BOW(k=20, dictionary_size=50)
    accuracy = alg.algorithm()
    print("Final accuracy of the model is:", accuracy)