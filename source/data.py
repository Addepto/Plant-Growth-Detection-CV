import logging
import os

import cv2
from tqdm import tqdm

from config import DEBUG

_logger = logging.getLogger('kws')


def get_images_paths(input_dir: str, plants_names: list, growth_stages: list):
    """
    Get paths for the images according to the input_dir
    :param input_dir: Where is the root of the dataset
    :param plants_names: Which plants should be loaded
    :param growth_stages: Which growth stages should be loaded
    :return: dict with image paths
    """
    images_paths = {}

    for plant_name in plants_names:
        images_paths.setdefault(plant_name, {})
        for growth_stage in growth_stages:
            images_paths[plant_name].setdefault(growth_stage, [])

            input_plant_growth_dir = os.path.join(input_dir, plant_name, growth_stage)
            image_names = os.listdir(input_plant_growth_dir)
            if DEBUG:
                image_names = [image_names[0]]
            images_paths[plant_name][growth_stage] = \
                [os.path.join(input_plant_growth_dir, image_name)
                 for image_name in image_names if os.path.isfile(os.path.join(input_plant_growth_dir, image_name))]

    return images_paths


def load_images(images_dict: dict):
    """
    Load images using path dict
    :param images_dict: dict with image paths per plant per grow stage
    :return: dict with loaded images
    """
    loaded_images = {}
    for plant_name, growth_stages in images_dict.items():
        loaded_images.setdefault(plant_name, {})
        for growth_stage, image_path_list in growth_stages.items():
            loaded_images[plant_name].setdefault(growth_stage, [])
            _logger.info(f'Loading images for {plant_name} - {growth_stage}')
            for image_path in tqdm(image_path_list):
                image = cv2.imread(image_path)
                loaded_images[plant_name][growth_stage].append((image_path, image))
    return loaded_images


def prepare_data_for_classification(loaded_segmented_dict, use_growth=False, searched_plant='Beta vulgaris'):
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
                else:
                    if use_growth:
                        data.append(image)
                        labels.append(f'{plant_name}.{growth_stage}')
                    else:
                        labels.append('other')
    return data, labels
