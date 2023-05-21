import torch.utils.data as data
import os
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import DataLoader



PATH_GT = './data/GT'
PATH_HZ = 'C:\\Users\\alptk\\OneDrive\\Desktop\\data\\hazy'

class Custom_Dataset(data.Dataset):
    def __init__(self, mode='train'):
        super(Custom_Dataset, self).__init__()
        path_g = './data/GT'
        path_hz = './data/hazy'
        self.mode = mode
        test_images_gt, train_images_gt = [], []
        test_images_hz, train_images_hz = [], []
        tmpList = []

        for file in os.listdir(path_g):
            img = Image.open(os.path.join(path_g, file))
            tmpList.append(img)
        
        test_images_gt = tmpList[45:]
        train_images_gt = tmpList[:45]

        tmpList = []

        for file in os.listdir(path_hz):
            img = Image.open(os.path.join(path_hz, file))
            tmpList.append(img)
        
        test_images_hz = tmpList[45:]
        train_images_hz = tmpList[:45]

        if mode == 'train':
            self.images_gt, self.images_hz = self.augmentation(train_images_gt, train_images_hz)
        else:
            self.images_gt = test_images_gt
            self.images_hz = test_images_hz
        
        tmpList = []
    
    def __getitem__(self, index):
        
        hazy = self.images_hz[index]
        gt = self.images_gt[index]

        hazy = transforms.ToTensor()(hazy)
        gt = transforms.ToTensor()(gt)

        return gt, hazy

    def __len__(self):
        return len(self.images_gt)
    
    def augmentation(self, gt_list, hazy_list):

        gt_list_augmented, hazy_list_augmented = [], []
        horizontal_flip = transforms.RandomHorizontalFlip(p=1)
        vertical_flip = transforms.RandomVerticalFlip(p=1)
        crop = transforms.RandomCrop(size=(224,312))
        rotate =  transforms.RandomRotation(degrees=66)
        for gt, hazy in zip(gt_list, hazy_list):
            
            gt_list_augmented.append(gt)
            hazy_list_augmented.append(hazy)

            gt_hf = horizontal_flip(gt)
            hazy_hf = horizontal_flip(hazy)

            gt_list_augmented.append(gt_hf)
            hazy_list_augmented.append(hazy_hf)
            
            gt_vf = vertical_flip(gt)
            hazy_vf = vertical_flip(hazy)

            gt_list_augmented.append(gt_vf)
            hazy_list_augmented.append(hazy_vf)

            gt_crop = crop(gt)
            hazy_crop = crop(hazy)

            gt_list_augmented.append(gt_crop)
            hazy_list_augmented.append(hazy_crop)


            gt_rot = rotate(gt)
            hazy_rot = rotate(hazy)

            gt_list_augmented.append(gt_rot)
            hazy_list_augmented.append(hazy_rot)

        return gt_list_augmented, hazy_list_augmented
            

""" if __name__ == "__main__":
        
    loader = DataLoader(dataset=Custom_Dataset(mode='train'), batch_size=1, shuffle=True)

    for i in range (len(loader)):
        iterator = iter(loader)
        x,y = next(iterator) """
       