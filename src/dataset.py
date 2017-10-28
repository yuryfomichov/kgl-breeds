import torch.utils.data as data
from PIL import Image
import os
import os.path
import torchvision.transforms as transforms

class BreedsDataset(data.Dataset):
    def __init__(self, data, target, image_folder, is_train):
        self.image_folder = image_folder
        self.imgs_ids = data
        self.targets = target
        self.is_train = is_train

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.image_folder, self.imgs_ids[index] + '.jpg')).convert('RGB')
        img_transformator = self._train_image_transform() if self.is_train else self._val_image_transform()
        img = img_transformator(img)
        return img, self.targets[index]

    def __len__(self):
        return len(self.targets)

    def _train_image_transform(self):
        transform = transforms.Compose([
            transforms.Scale(256),
            transforms.RandomCrop(192),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])
        return transform

    def _val_image_transform(self):
        transform = transforms.Compose([
            transforms.Scale(256),
            transforms.CenterCrop(192),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])
        return transform