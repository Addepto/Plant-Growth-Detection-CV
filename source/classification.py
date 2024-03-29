import logging

import numpy as np
import pandas
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from tqdm import tqdm

from data_transform import random_rotation, random_noise, horizontal_flip, transform_data
from features import calculate_descriptors, hu_moments, zernike_moments, haralick, local_binary_patterns

_logger = logging.getLogger('kws')

# parameters for the classifiers
parameters = {
    'seed': 5,
    # 'seed': random.randint(0, 10000),
    'num_trees': 10,
    'test_size': 0.1,
    'kfold_splits': 10,
    'scoring': 'accuracy',
    'n_jobs': -1,
    'error_score': 'raise'
}

# tested models
models = {
    'KNN': KNeighborsClassifier(),
    'Decistion Tree': DecisionTreeClassifier(random_state=parameters['seed']),
    'Random Forest': RandomForestClassifier(n_estimators=parameters['num_trees'], random_state=parameters['seed']),
    'LGBM': LGBMClassifier(objective='multiclass', num_class=3, max_depth=-1, learning_rate=0.1, num_leaves=31,
                           boosting_type='dart', n_estimators=100),
}


def get_descriptor_parts_functions():
    """
    Define which functions should be used to create image descriptor
    :return: list with functions
    """
    function_list = [
        hu_moments,
        haralick,
        zernike_moments,
        local_binary_patterns
    ]
    return function_list


def get_data_transform_functions():
    """
    Define which functions should be used to transform data
    :return: list with functions
    """
    function_list = [
        random_rotation,
        random_noise,
        horizontal_flip
    ]
    return function_list


def prepare_data(loaded_segmented_dict, use_growth=False, searched_plant='Beta vulgaris'):
    """
    Prepare data - so divide into classes
    :param loaded_segmented_dict: images to consider
    :param use_growth: should be use use growth in label
    :param searched_plant: main plant to be recognized
    :return: data list nad label list
    """
    data = []
    labels = []
    for plant_name, growth_stages in loaded_segmented_dict.items():
        for growth_stage, image_path_list in growth_stages.items():
            for _, image in tqdm(image_path_list):
                if not use_growth:
                    data.append(image)
                # labels.append(f'{plant_name}.{growth_stage}')
                if plant_name == searched_plant:
                    if use_growth:
                        data.append(image)
                        labels.append(f'{plant_name}.{growth_stage}')
                    else:
                        labels.append(f'{plant_name}')
                elif not use_growth:
                    labels.append('other')
    return data, labels


def test_LGBM(predicted_labels, test_labels):
    """
    Testing function for LGBM Classifier
    :param predicted_labels: labels returned by the classifier
    :param test_labels: ground truth labels
    :return: score of the classifier
    """
    counter = 0
    for i in range(len(test_labels)):
        if int(test_labels[i]) == predicted_labels[i].argmax():
            counter += 1
    score = counter / len(test_labels)
    return score


def run_classification(loaded_segmented_dict, use_growth=False):
    """
    Test classifiers
    :param loaded_segmented_dict: masked images
    :param use_growth: should the growth be used as a label
    """
    _logger.info('Seed: {seed}'.format(seed=parameters['seed']))
    _logger.info('Preparing data for training and testing')
    data, labels = prepare_data(loaded_segmented_dict, use_growth=use_growth)
    _logger.info('Applying transforms to data')
    data, labels = transform_data(data, labels, get_data_transform_functions())
    _logger.info('Preparing feature descriptors')
    labels = pandas.factorize(labels)[0]
    descriptors = calculate_descriptors(data, get_descriptor_parts_functions())
    _logger.info('Splitting data')
    train_data, test_data, train_label, test_label = \
        train_test_split(descriptors, labels, test_size=parameters['test_size'], random_state=parameters['seed'],
                         shuffle=True)
    _logger.info('Running classification')
    results = []
    names = []
    for classifier_name, classifier in models.items():
        if classifier_name == 'LGBM':
            continue
        _logger.info(f'Cross validating {classifier_name}')
        kfold = KFold(n_splits=parameters['kfold_splits'], random_state=parameters['seed'], shuffle=True)
        cross_validation = cross_val_score(classifier, train_data, train_label, cv=kfold, scoring=parameters['scoring'],
                                           n_jobs=parameters['n_jobs'], error_score=parameters['error_score'])
        results.append(cross_validation)
        names.append(classifier_name)
        _logger.info("%s: %f (%f)" % (classifier_name, cross_validation.mean(), cross_validation.std()))

    means = [cross_validation.mean() for cross_validation in results]
    best = means.index(max(means))
    classifier_name = names[best]

    eval_classifier(classifier_name, labels, test_data, test_label, train_data, train_label)
    eval_classifier('LGBM', labels, test_data, test_label, train_data, train_label)


def eval_classifier(classifier_name, labels, test_data, test_label, train_data, train_label):
    """
    Eval the classifier
    :param classifier_name: classifier name
    :param labels: ground-truth labels
    :param test_data: test data after split
    :param test_label: test labels after split
    :param train_data: training data after split
    :param train_label: training labels after split
    """
    classifier = models[classifier_name]
    classifier.fit(train_data, train_label)

    if classifier_name == 'LGBM':
        prediction = classifier.predict(test_data, raw_score=True)
        precision = test_LGBM(prediction, test_label)
        _logger.info(f'Precision of {classifier_name} is {precision}')
    else:
        prediction = classifier.predict(test_data)
        precision = precision_score(test_label, prediction, labels=np.unique(labels), average='micro')
        _logger.info(f'Precision of {classifier_name} is {precision}')
        precision = precision_score(test_label, prediction, labels=np.unique(labels), average='macro')
        _logger.info(f'Precision of {classifier_name} is {precision}')
