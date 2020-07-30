import logging

import pandas
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision import transforms

from classification import parameters
from data import load_images, get_images_paths, prepare_data_for_classification
import cv2
import numpy as np

from data_transform import random_noise, random_rotation, transform_data, horizontal_flip

_logger = logging.getLogger('kws')


def get_data_transform_functions():
    """
    Define which functions should be used to transform data
    :return: list with functions
    """
    function_list = [
        random_rotation,
        # random_noise,
        horizontal_flip
    ]
    return function_list


transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
        transforms.RandomRotation(degrees=15),
        transforms.RandomHorizontalFlip(),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
])


class PlantDataset(Dataset):
    def __init__(self, output_dir, plants_names, growth_stages, train=True, use_growth=False):
        images_dict = load_images(get_images_paths(output_dir, plants_names, growth_stages))

        self.train = train
        self.data, self.labels = prepare_data_for_classification(images_dict, use_growth=use_growth)
        numeric_labels, string_labels = pandas.factorize(self.labels)
        self.label_mapping = {string_label: numeric_label
                              for numeric_label, string_label in zip(np.unique(numeric_labels), string_labels)}
        self.label_mapping_backwards = {numeric_label: string_label
                                        for numeric_label, string_label in zip(np.unique(numeric_labels), string_labels)}

        _logger.info('Applying transforms to data')
        self.train_data, self.test_data, self.train_label, self.test_label = \
            train_test_split(self.data, self.labels, test_size=parameters['test_size'], random_state=parameters['seed'],
                             shuffle=True)

        self.data, self.labels = (self.train_data, self.train_label) if self.train \
            else (self.test_data, self.test_label)

        self.data, self.labels = transform_data(self.data, self.labels, get_data_transform_functions(), trials=0)
        self.labels = [self.label_mapping[x] for x in self.labels]

        # shape = (864, 576, 3)
        shape = (432, 288, 3)
        # shape = (216, 144, 3)

        for i, image in enumerate(self.data):
            self.data[i] = cv2.resize(image, dsize=(shape[0], shape[1]))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = transform(self.data[idx])
        return sample, self.labels[idx], self.label_mapping_backwards[self.labels[idx]]

    def get_label_name(self, idx):
        return self.label_mapping_backwards[idx]

