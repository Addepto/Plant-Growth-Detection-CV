import logging

import pandas
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from classification import parameters
from data import load_images, get_images_paths, prepare_data_for_classification
import cv2

from data_transform import random_noise, random_rotation, transform_data

_logger = logging.getLogger('kws')


def get_data_transform_functions():
    """
    Define which functions should be used to transform data
    :return: list with functions
    """
    function_list = [
        random_rotation,
        random_noise,
    ]
    return function_list


class PlantDataset(Dataset):
    def __init__(self, output_dir, plants_names, growth_stages, train=True, use_growth=False):
        images_dict = load_images(get_images_paths(output_dir, plants_names, growth_stages))

        self.train = train
        self.data, self.labels = prepare_data_for_classification(images_dict, use_growth=use_growth)
        _logger.info('Applying transforms to data')
        self.data, self.labels = transform_data(self.data, self.labels, get_data_transform_functions())

        shape = (864, 576, 3)

        for i, image in enumerate(self.data):
            self.data[i] = cv2.resize(image, dsize=(shape[0], shape[1]))
        self.labels = pandas.factorize(self.labels)[0]

        self.train_data, self.test_data, self.train_label, self.test_label = \
            train_test_split(self.data, self.labels, test_size=parameters['test_size'], random_state=parameters['seed'],
                             shuffle=True)

    def __len__(self):
        return len(self.train_data) if self.train else len(self.test_data)

    def __getitem__(self, idx):
        source_images, source_labels = (self.train_data, self.train_label) if self.train \
            else (self.test_data, self.test_label)

        return source_images[idx], source_labels[idx]

