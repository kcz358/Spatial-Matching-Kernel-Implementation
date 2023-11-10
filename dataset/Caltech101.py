from glob import glob

import torch
from torchvision import transforms
import cv2
from PIL import Image

class Caltech101(torch.utils.data.Dataset):
    def __init__(self, dataset_path, image_path, use_opencv = True):
        '''
        dataset_path : path to caltech-101
        '''
        super().__init__()
        #self.image_path = glob(dataset_path + "101_ObjectCategories/**/*.jpg", recursive=True)
        self.image_path = image_path
        self.cls = glob(dataset_path + "/101_ObjectCategories/**")
        self.cls = [c.split("/")[-1] for c in self.cls]
        self.cls2idx = {}
        for idx, c in enumerate(self.cls):
            self.cls2idx[c] = idx
        
        self.transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.ToTensor()
        ])
        
        self.use_opencv = use_opencv
    
    def __getitem__(self, index):
        img_path = self.image_path[index]
        label = self.cls2idx[img_path.split('/')[-2]]
        if self.use_opencv:
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            return img, label
        img = Image.open(img_path)
        img = self.transform(img)
        
        return img, label
    
    def __len__(self):
        return len(self.image_path)