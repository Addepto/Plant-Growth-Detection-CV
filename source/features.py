
import cv2
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