import logging

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from tqdm import tqdm

from data_transform import random_rotation, random_noise, horizontal_flip, transform_data
from features import calculate_descriptors, hu_moments, zernike_moments, haralick

_logger = logging.getLogger('kws')

parameters = {
    'seed': 5,
    'num_trees': 10,
    'test_size': 0.1,
    'kfold_splits': 10,
    'scoring': 'accuracy'
}

models = {
    'KNN': KNeighborsClassifier(),
    'Decistion Tree': DecisionTreeClassifier(random_state=parameters['seed']),
    'Random Forest': RandomForestClassifier(n_estimators=parameters['num_trees'], random_state=parameters['seed'])
}


def get_descriptor_parts_functions():
    function_list = [
        hu_moments,
        haralick,
        zernike_moments
    ]
    return function_list


def get_data_transform_functions():
    function_list = [
        random_rotation,
        random_noise,
        horizontal_flip
    ]
    return function_list


def prepare_data(loaded_segmented_dict):
    data = []
    labels = []
    for plant_name, growth_stages in loaded_segmented_dict.items():
        for growth_stage, image_path_list in growth_stages.items():
            for _, image in tqdm(image_path_list):
                data.append(image)
                labels.append(f'{plant_name}.{growth_stage}')
                # labels.append(f'{plant_name}')
    return data, labels


def run_classification(loaded_segmented_dict):
    _logger.info('Preparing data for training and testing')
    data, labels = prepare_data(loaded_segmented_dict)
    _logger.info('Applying transforms to data')
    data, labels = transform_data(data, labels, get_data_transform_functions())
    _logger.info('Preparing feature descriptors')
    descriptors = calculate_descriptors(data, get_descriptor_parts_functions())
    _logger.info('Splitting data')
    train_data, test_data, train_label, test_label = \
        train_test_split(descriptors, labels, test_size=parameters['test_size'], random_state=parameters['seed'],
                         shuffle=True)
    _logger.info('Running classification')
    results = []
    names = []
    for classifier_name, classifier in models.items():
        _logger.info(f'Cross validating {classifier_name}')
        kfold = KFold(n_splits=parameters['kfold_splits'], random_state=parameters['seed'], shuffle=True)
        cross_validation = cross_val_score(classifier, train_data, train_label, cv=kfold, scoring=parameters['scoring'])
        results.append(cross_validation)
        names.append(classifier_name)
        _logger.info("%s: %f (%f)" % (classifier_name, cross_validation.mean(), cross_validation.std()))

    means = [cross_validation.mean() for cross_validation in results]
    best = means.index(max(means))
    classifier_name = names[best]

    classifier = models[classifier_name]
    classifier.fit(train_data, train_label)
    prediction = classifier.predict(test_data)
    precision = precision_score(test_label, prediction, labels=np.unique(labels), average='micro')
    _logger.info(f'Precision of {classifier_name} is {precision}')
    precision = precision_score(test_label, prediction, labels=np.unique(labels), average='macro')
    _logger.info(f'Precision of {classifier_name} is {precision}')
