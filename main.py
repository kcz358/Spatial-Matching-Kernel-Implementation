import argparse
from glob import glob
import os
from os.path import exists

import cv2
import faiss
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from sklearn import svm
from sklearn.metrics import accuracy_score

from dataset import Caltech101

dataset_path = "./caltech-101"

def parse_argument():
    parser = argparse.ArgumentParser(description="Pyramid Level Argument Parser")
    
    # Add the --level or -L argument
    parser.add_argument('--level', '-L', type=int, default=2,
                        help="Level of the pyramid you want to use")
    
    args = parser.parse_args()
    return args

# Extracting sift features from each image of the dataset
def extract_features(dataset):
    features = {}
    x = np.empty([0,128])
    des_list = []
    #sift = cv2.SIFT_create()
    sift = cv2.xfeatures2d.SIFT_create()
    for idx in tqdm(range(len(dataset))):
        
        img, label = dataset[idx]
        image_path = dataset.image_path[idx]
        
        step_size = 15
        kp = [cv2.KeyPoint(x, y, step_size) for x in range(0, img.shape[0], step_size) for y in range(0, img.shape[1], step_size)]
        kp, des = sift.compute(img, kp)
        #import pdb; pdb.set_trace()d
        #kp, des = sift.detectAndCompute(img, None)
        # Key point with shape (1,2)
        # des with shape (N, 128)
        kp = [p.pt for p in kp]
        kp = np.array(kp)
        

        if des is None:
            print("Getting image with no sift keypoints detected...")
            print("Replace with a zero vector with coordinates(-1, -1)")
            des = np.zeros((1,128))
            kp = np.array([[-1,-1]])
        
        features[image_path] = [des, kp, label]
    
        des_list.append(des)
        
    x = np.concatenate(des_list, axis=0)
    return x, features

def vectorize(L, M, features, dataset):
    bins = np.arange(M+1)
    weights = [1 / (2 ** L) if l == 0 else 1 / (2 ** (L - l + 1)) for l in range(L+1)]
    
    vector_features = []
    labels = []
    for idx in tqdm(range(len(dataset))):
        img, label = dataset[idx]
        image_path = dataset.image_path[idx]
        des, kp, label, I = features[image_path]
        vector_feature = []
        labels.append([label])
        
        for l in range(L+1):
            # In total 4**l grids, so split into 2**l segments on each side
            L_x = [int(i / 2**l  * img.shape[0]) for i in range(2**l + 1)]
            L_y = [int(i / 2**l  * img.shape[1]) for i in range(2**l + 1)]
            for i in range(len(L_x) - 1):
                for j in range(len(L_y) - 1):
                    features_in_cells = np.where((L_x[i] <= kp[:,0]) & 
                                                (kp[:,0] <= L_x[i+1]) &
                                                (L_y[j] <= kp[:,1]) &
                                                (kp[:,1] <= L_y[j+1]))[0]
                    
                    hist, _ = np.histogram(np.array(I)[features_in_cells], bins=bins)
                    # Normalize the bins
                    hist = hist - np.mean(hist) / (np.std(hist) + 1e-6)
                    vector_feature.append([hist * weights[l]])
                
        vector_feature = np.concatenate(vector_feature, axis=1)
        features[image_path].append(vector_feature)
        vector_features.append(vector_feature)
        #import pdb; pdb.set_trace()
    vector_features = np.concatenate(vector_features, axis=0)
    labels = np.concatenate(labels, axis=0)
    return vector_features, labels

if __name__ == "__main__":
    np.random.seed(42)
    
    args = parse_argument()
    L = args.level
    
    if not exists("./cache"):
        os.makedirs("./cache")
    
    if not exists(f"./cache/L{L}"):
        os.makedirs(f"./cache/L{L}")
    
    if exists(f"./cache/L{L}/train_vector_features.npy"):
        print("Using Cache")
        
        train_vector_features = np.load(f"./cache/L{L}/train_vector_features.npy")
        train_labels = np.load(f"./cache/L{L}/train_labels.npy")

        test_vector_features = np.load(f"./cache/L{L}/test_vector_features.npy")
        test_labels = np.load(f"./cache/L{L}/test_labels.npy")
    else:
        image_path = glob(dataset_path + "/101_ObjectCategories/**/*.jpg", recursive=True)
        print("Sampled 30 images for training and the rest for testing")
        # Dictionary with (cls, [images])
        image_cls_path = {}
        for i in image_path:
            cls = i.split("/")[-2]
            image_cls_path[cls] = image_cls_path.get(cls, []) + [i]
        
        train_image_path = []
        test_image_path = []
        
        for k, v in image_cls_path.items():
            sampled_idx = np.random.choice(np.arange(len(v)), size=30, replace=False)
            not_sampled_idx = np.setdiff1d(np.arange(len(v)), sampled_idx)
            train_image_path += list(np.array(v)[sampled_idx])
            test_image_path += list(np.array(v)[not_sampled_idx])
        print(f'Training size : {len(train_image_path)}')
        print(f'Testing size : {len(test_image_path)}')
        
        train_dataset = Caltech101(dataset_path, train_image_path)
        test_dataset = Caltech101(dataset_path, test_image_path)
        
        print("Extracting sift features from the training images")
        x, train_features = extract_features(train_dataset)
        print("Extracting sift detector from the testing images")
        _, test_features = extract_features(test_dataset)
        
        print("Start Clustering")
        M = 200
        niter = 50
        verbose = True
        d = x.shape[1]
        kmeans = faiss.Kmeans(d, M, niter=niter, verbose=verbose)
        kmeans.train(x)
        
        print("Finish Clustering. Assign codebooks to train features in image")
        for k,v in tqdm(train_features.items()):
            D, I = kmeans.index.search(v[0], 1)
            v.append(I)
            
        print("Vectorize the training set through histogram counting")
        train_vector_features, train_labels = vectorize(L, M, train_features, train_dataset)
        
        np.save(f"./cache/L{L}/train_vector_features.npy", train_vector_features)
        np.save(f"./cache/L{L}/train_labels.npy", train_labels)
        
        del x, train_features

        print("Assign codebooks to features in image")
        for k,v in tqdm(test_features.items()):
            D, I = kmeans.index.search(v[0], 1)
            v.append(I)
            
        print("Vectorize the testing set through histogram counting")
        test_vector_features, test_labels = vectorize(L, M, test_features, test_dataset)
        
        np.save(f"./cache/L{L}/test_vector_features.npy", test_vector_features)
        np.save(f"./cache/L{L}/test_labels.npy", test_labels)
    
    print(f"Total features shape : {train_vector_features.shape}")
    print(f"Total label shape : {train_labels.shape}")
    
    print(f"Total features shape : {test_vector_features.shape}")
    print(f"Total label shape : {test_labels.shape}")
    print("One vs Rest SVM training")
    clf = svm.LinearSVC(random_state=0, C = 0.001)
    clf.fit(train_vector_features, train_labels)
    
    train_pred = clf.predict(train_vector_features)
    training_accuracy = accuracy_score(train_pred, train_labels)
    
    print(f"Training Accuracy : {training_accuracy}")
    
    test_pred = clf.predict(test_vector_features)
    testing_accuracy = accuracy_score(test_pred, test_labels)
    
    print(f"Testing Accuracy : {testing_accuracy}")