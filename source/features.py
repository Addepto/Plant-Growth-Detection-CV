import cv2
import numpy as np
from mahotas import features


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


def calculate_descriptor(image):
    """
    Calculate descriptor for the image
    :param image: image to be processed
    :return: feature descriptor
    """
    moments = hu_moments(image)
    texture = haralick(image)

    return np.hstack([moments, texture])


def calculate_descriptors(image_list):
    """
    Calculate descriptor for every image in image_list
    :param image_list:
    :return:
    """
    result = []
    for image in image_list:
        result.append(calculate_descriptor(image))
    return result
