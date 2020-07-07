import random

from skimage import transform
from skimage import util


def random_rotation(image):
    """
    Apply rotation to the image
    :param image: input data
    :return: transformed image
    """
    degree = random.uniform(-45, 45)
    return transform.rotate(image, degree).astype('uint8') * 255


def random_noise(image):
    """
    Apply noise to the image
    :param image: input data
    :return: transformed image
    """
    return util.random_noise(image).astype('uint8') * 255


def horizontal_flip(image):
    """
    Flip image horizontally
    :param image: input data
    :return: transformed image
    """
    return image[:, ::-1, :]


def transform_data(image_list, label_list, data_transform_functions, trials=2):
    """
    Transform input images using given transform functions
    :param image_list: list of images
    :param label_list: labels matching the images
    :param data_transform_functions: functions to be used as transform
    :param trials: how many times the operation should be repeated
    :return: new image list and new labels list
    """
    result_image_list = image_list
    result_label_list = label_list

    for i in range(trials):
        current_result_image_list = []
        current_result_label_list = []

        for image, label in zip(result_image_list, result_label_list):
            method = random.choice(data_transform_functions)
            transformed_image = method(image)
            current_result_image_list.append(transformed_image)
            current_result_label_list.append(label)

        result_image_list = image_list + current_result_image_list
        result_label_list = label_list + current_result_label_list

    return result_image_list, result_label_list
