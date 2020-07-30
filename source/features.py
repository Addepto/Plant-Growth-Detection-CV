import cv2
import numpy as np
from mahotas import features
from sklearn.preprocessing import MinMaxScaler


def hu_moments(image):
    """
    Calculate shape descriptors
    :param image: image to be used
    :return: HuMoments
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.HuMoments(cv2.moments(image)).flatten()


def haralick(image):
    """
    Calculate haralick features
    :param image: image to be used
    :return: haralick features
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return features.haralick(image).mean(axis=0)


def zernike_moments(image, radius=50):
    """
    Calculate zernike descriptor
    :param image: image to be used
    :param radius: the radius of polynomial
    :return: zernike moments
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.bitwise_not(image)
    image[image > 0] = 255
    return features.zernike_moments(image, radius)


def local_binary_patterns(image, radius=40, points=7):
    """
    Calculate local binary patterns
    :param image: image to be used
    :param radius: radius for lbp
    :param points: number of points for lbp
    :return: local binary patterns
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return features.lbp(image, radius=radius, points=points)


def calculate_descriptor(image, descriptor_component_functions):
    """
    Calculate descriptor for the image
    :param image: image to be processed
    :param descriptor_component_functions: list of functions to use to calculate the final descriptor
    :return: feature descriptor
    """

    descriptor = []
    for descriptor_component_function in descriptor_component_functions:
        descriptor.append(descriptor_component_function(image))

    return np.hstack(descriptor)


def calculate_descriptors(image_list, descriptor_component_functions):
    """
    Calculate descriptor for every image in image_list
    :param image_list: images to be processed
    :param descriptor_component_functions: list of functions to use to calculate the final descriptor
    :return: processed images = list of descriptors
    """
    result = []
    for image in image_list:
        result.append(calculate_descriptor(image, descriptor_component_functions))

    scaler = MinMaxScaler(feature_range=(0, 1))
    rescaled_features = scaler.fit_transform(result)

    return rescaled_features
