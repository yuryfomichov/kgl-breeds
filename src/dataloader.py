import torch.utils.data.dataloader as dataloader
import torch as torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from dataset import BreedsDataset
import os

class BreedsLoader(object):
    def __init__(self, params):
        self.validation_size = params.get("validation_size",0.085)
        self.batch_size = params.get("batch_size", 200)
        self.num_workers = params.get("num_workers", 8 if torch.cuda.is_available() else 0)
        self.shuffle = params.get("shuffle", True)
        self.data_dir = 'data'
        self.train_folder = 'train'
        self.test_folder = 'test'
        self.labels_file = 'labels.csv'
        self._load_data()
        self._load_submission_data()

    def get_train_loader(self):
        return self._get_loader(self.train_data, True, False)

    def get_val_loader(self):
        return self._get_loader(self.val_data, False, False)

    def get_test_loader(self):
        return self._get_loader(self.test_data, False, False)

    def get_submission_loader(self):
         return self._get_loader(self.submission_data, False, True)

    def _get_loader(self, data, drop_last = False, is_test = True):
        imageFolder = '%s/%s/' % (self.data_dir, self.test_folder if is_test else self.train_folder)
        loader = dataloader.DataLoader(BreedsDataset(data[0], data[1], imageFolder, is_train= not is_test),
                                       batch_size=self.batch_size,
                                       shuffle=self.shuffle,
                                       num_workers=self.num_workers,
                                       drop_last=drop_last,
                                       pin_memory=torch.cuda.is_available())
        return loader

    def _load_data(self):
        data = pd.read_csv('data/labels.csv')
        ids = data['id'].values
        labels = self._convert_lables(data['breed'].values)
        (X_train, x_temp, y_train, y_temp) = train_test_split(ids, labels, test_size=self.validation_size,stratify=labels)
        self.train_data = (X_train, y_train)

        (X_val, X_test, y_val, y_test) = train_test_split(x_temp, y_temp, test_size=0.5,stratify=y_temp)
        self.val_data = (X_val, y_val)
        self.test_data = (X_test, y_test)

    def _load_submission_data(self):
        imageFolder = '%s/%s/' % (self.data_dir, self.test_folder)
        images = np.array(np.char.replace(os.listdir(imageFolder), '.jpg', ''))
        labels = np.zeros(images.shape)
        self.submission_data = (images, labels)

    def _convert_lables(self, labels):
        label_encoder = preprocessing.LabelEncoder()
        label_encoder.fit(self.get_breeds());
        return label_encoder.transform(labels);

    def get_breeds(self):
        return [
            'affenpinscher',
            'afghan_hound',
            'african_hunting_dog',
            'airedale',
            'american_staffordshire_terrier',
            'appenzeller',
            'australian_terrier',
            'basenji',
            'basset',
            'beagle',
            'bedlington_terrier',
            'bernese_mountain_dog',
            'black-and-tan_coonhound',
            'blenheim_spaniel',
            'bloodhound',
            'bluetick',
            'border_collie',
            'border_terrier',
            'borzoi',
            'boston_bull',
            'bouvier_des_flandres',
            'boxer',
            'brabancon_griffon',
            'briard',
            'brittany_spaniel',
            'bull_mastiff',
            'cairn',
            'cardigan',
            'chesapeake_bay_retriever',
            'chihuahua',
            'chow',
            'clumber',
            'cocker_spaniel',
            'collie',
            'curly-coated_retriever',
            'dandie_dinmont',
            'dhole',
            'dingo',
            'doberman',
            'english_foxhound',
            'english_setter',
            'english_springer',
            'entlebucher',
            'eskimo_dog',
            'flat-coated_retriever',
            'french_bulldog',
            'german_shepherd',
            'german_short-haired_pointer',
            'giant_schnauzer',
            'golden_retriever',
            'gordon_setter',
            'great_dane',
            'great_pyrenees',
            'greater_swiss_mountain_dog',
            'groenendael',
            'ibizan_hound',
            'irish_setter',
            'irish_terrier',
            'irish_water_spaniel',
            'irish_wolfhound',
            'italian_greyhound',
            'japanese_spaniel',
            'keeshond',
            'kelpie',
            'kerry_blue_terrier',
            'komondor',
            'kuvasz',
            'labrador_retriever',
            'lakeland_terrier',
            'leonberg',
            'lhasa',
            'malamute',
            'malinois',
            'maltese_dog',
            'mexican_hairless',
            'miniature_pinscher',
            'miniature_poodle',
            'miniature_schnauzer',
            'newfoundland',
            'norfolk_terrier',
            'norwegian_elkhound',
            'norwich_terrier',
            'old_english_sheepdog',
            'otterhound',
            'papillon',
            'pekinese',
            'pembroke',
            'pomeranian',
            'pug',
            'redbone',
            'rhodesian_ridgeback',
            'rottweiler',
            'saint_bernard',
            'saluki',
            'samoyed',
            'schipperke',
            'scotch_terrier',
            'scottish_deerhound',
            'sealyham_terrier',
            'shetland_sheepdog',
            'shih-tzu',
            'siberian_husky',
            'silky_terrier',
            'soft-coated_wheaten_terrier',
            'staffordshire_bullterrier',
            'standard_poodle',
            'standard_schnauzer',
            'sussex_spaniel',
            'tibetan_mastiff',
            'tibetan_terrier',
            'toy_poodle',
            'toy_terrier',
            'vizsla',
            'walker_hound',
            'weimaraner',
            'welsh_springer_spaniel',
            'west_highland_white_terrier',
            'whippet',
            'wire-haired_fox_terrier',
            'yorkshire_terrier']

