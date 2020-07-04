import logging

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from tqdm import tqdm

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


def classify():
    pass


def train():
    pass


def get_descriptor_parts_functions():
    function_list = [
        hu_moments,
        haralick,
        zernike_moments
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
    return data, labels


def run_classification(loaded_segmented_dict):
    _logger.info('Preparing data for training and testing')
    data, labels = prepare_data(loaded_segmented_dict)
    _logger.info('Prepare feature descriptors')
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
