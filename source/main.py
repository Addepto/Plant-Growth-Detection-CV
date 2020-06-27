import logging
import os

import cv2
import numpy as np
from plantcv import plantcv
from tqdm import tqdm

import cli
from kws_logging import config_logger

DEBUG = False
VISUALIZE = False
_logger = logging.getLogger('kws')


def get_images_paths(input_dir: str, plants_names: list, growth_stages: list):
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


def segment(image):
    scale = 0.25
    resized = cv2.resize(image, dsize=None, fx=scale, fy=scale)
    if VISUALIZE:
        cv2.imshow('a', resized)
        cv2.waitKey(0)

    masked_a = plantcv.rgb2gray_lab(rgb_img=resized, channel='a')
    if VISUALIZE:
        cv2.imshow('a', masked_a)
        cv2.waitKey(0)

    masked_b = plantcv.rgb2gray_lab(rgb_img=resized, channel='b')
    if VISUALIZE:
        cv2.imshow('b', masked_b)
        cv2.waitKey(0)

    e = plantcv.rgb2gray_hsv(rgb_img=resized, channel='h')
    if VISUALIZE:
        cv2.imshow('e', e)
        cv2.waitKey(0)

    maskedh_thresh = plantcv.threshold.binary(gray_img=masked_a, threshold=120,
                                              max_value=255, object_type='dark')
    if VISUALIZE:
        cv2.imshow('x', maskedh_thresh)
        cv2.waitKey(0)

    e = plantcv.rgb2gray_hsv(rgb_img=resized, channel='s')
    if VISUALIZE:
        cv2.imshow('e', e)
        cv2.waitKey(0)

    maskeds_thresh = plantcv.threshold.binary(gray_img=masked_a, threshold=150,
                                              max_value=255, object_type='light')
    if VISUALIZE:
        cv2.imshow('x', maskeds_thresh)
        cv2.waitKey(0)

    hs = plantcv.logical_or(bin_img1=maskedh_thresh, bin_img2=maskeds_thresh)
    if VISUALIZE:
        cv2.imshow('x', hs)
        cv2.waitKey(0)

    maskeda_thresh = plantcv.threshold.binary(gray_img=masked_a, threshold=120,
                                              max_value=255, object_type='dark')
    if VISUALIZE:
        cv2.imshow('a', maskeda_thresh)
        cv2.waitKey(0)
    maskeda_thresh1 = plantcv.threshold.binary(gray_img=masked_a, threshold=150,
                                               max_value=255, object_type='light')
    if VISUALIZE:
        cv2.imshow('b', maskeda_thresh1)
        cv2.waitKey(0)
    maskedb_thresh = plantcv.threshold.binary(gray_img=masked_b, threshold=165,
                                              max_value=255, object_type='light')
    if VISUALIZE:
        cv2.imshow('c', maskedb_thresh)
        cv2.waitKey(0)

    ab1 = plantcv.logical_or(bin_img1=maskeda_thresh, bin_img2=maskedb_thresh)
    if VISUALIZE:
        cv2.imshow('a', ab1)
        cv2.waitKey(0)
    ab = plantcv.logical_or(bin_img1=maskeda_thresh1, bin_img2=ab1)
    if VISUALIZE:
        cv2.imshow('b', ab)
        cv2.waitKey(0)

    ab_fill = plantcv.fill(bin_img=ab, size=500)
    if VISUALIZE:
        cv2.imshow('ab', ab_fill)
        cv2.waitKey(0)

    mask = plantcv.logical_and(ab_fill, hs)
    if VISUALIZE:
        cv2.imshow('mask', mask)
        cv2.waitKey(0)

    masked = plantcv.apply_mask(img=resized, mask=mask, mask_color='white')
    if VISUALIZE:
        cv2.imshow('res', masked)
        cv2.waitKey(0)
    return resized, masked


def create_output_directories(loaded_images_dict, output_dir):
    for plant_name in loaded_images_dict.keys():
        for growth_stage in loaded_images_dict[plant_name].keys():
            final_dir_path = os.path.join(output_dir, plant_name, growth_stage)
            os.makedirs(final_dir_path, exist_ok=True)


def combine_images(image1, image2):
    return np.hstack((image1, image2))


def run_segmentation(loaded_images_dict, output_dir, should_combine_images=False):
    for plant_name, growth_stages in loaded_images_dict.items():
        for growth_stage, image_tuple_list in growth_stages.items():
            _logger.info(f'Segmenting images for {plant_name} - {growth_stage}')
            for image_path, image in tqdm(image_tuple_list):
                image_name = os.path.basename(image_path)
                output_path = os.path.join(output_dir, plant_name, growth_stage, image_name)
                _logger.info(f'Running segmentation for the image: {image_name}')
                # if image_name != 'Beta-vulgaris_Cotyledon_Substrat1_07102019_darker (1).JPG':
                #     continue
                # if image_name != 'Beta-vulgaris_CotyledonPhase_Substrat1_27082019 (5).JPG':
                #     continue
                source, result  = segment(image)
                _logger.info(f'Writing result to: {output_path}')
                if should_combine_images:
                    result = combine_images(source, result)
                cv2.imwrite(output_path, result)


def run():
    parser = cli.create_parser()
    args = parser.parse_args()

    config_logger(args.output_dir, 'kws')

    _logger.info('Getting image paths from: {input_dir} '
                 '(plant names: {plant_names}, '
                 'growth stages: {growth_stages}).'.format(input_dir=args.input_dir,
                                                           plant_names=args.plants_names,
                                                           growth_stages=args.growth_stages))
    images_dict = get_images_paths(args.input_dir, args.plants_names, args.growth_stages)
    _logger.info('Loading images...')
    loaded_images_dict = load_images(images_dict)
    _logger.info('Creating image directories in: {output_dir}'.format(output_dir=args.output_dir))
    create_output_directories(loaded_images_dict, args.output_dir)
    _logger.info('Running segmentation...')
    run_segmentation(loaded_images_dict, args.output_dir, should_combine_images=True)

    # sample_image = loaded_images_dict['Beta vulgaris']['Cotyledon'][0]
    # segment(sample_image)


if __name__ == '__main__':
    run()
