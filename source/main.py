import logging
import os

import cv2
import numpy as np
from plantcv import plantcv
from tqdm import tqdm

import cli
from classification import run_classification
from kws_logging import config_logger

DEBUG = False
VISUALIZE = False
_logger = logging.getLogger('kws')


def setup_plantcv(should_debug=False):
    plantcv.params.line_thickness = 3
    if should_debug:
        plantcv.params.debug = 'print'
        plantcv.params.debug_outdir = os.path.join(os.path.dirname(__file__), 'debug')
        _logger.info(f'Plantcv debugging is turned on. All results will be saved to: {plantcv.params.debug_outdir}')
        if os.path.exists(plantcv.params.debug_outdir):
            os.rmdir(plantcv.params.debug_outdir)
        os.makedirs(plantcv.params.debug_outdir)


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
    masked_v = plantcv.rgb2gray_hsv(rgb_img=image, channel='v')
    average_v = np.average(masked_v)

    masked_a = plantcv.rgb2gray_lab(rgb_img=image, channel='a')
    if VISUALIZE:
        cv2.imshow('a', masked_a)
        cv2.waitKey(0)

    masked_b = plantcv.rgb2gray_lab(rgb_img=image, channel='b')
    if VISUALIZE:
        cv2.imshow('b', masked_b)
        cv2.waitKey(0)

    masked_h = plantcv.rgb2gray_hsv(rgb_img=image, channel='h')
    if VISUALIZE:
        cv2.imshow('h', masked_h)
        cv2.waitKey(0)

    maskedh_thresh1 = plantcv.threshold.binary(gray_img=masked_h, threshold=25,
                                              max_value=255, object_type='light')
    if VISUALIZE:
        cv2.imshow('x', maskedh_thresh1)
        cv2.waitKey(0)
    maskedh_thresh2 = plantcv.threshold.binary(gray_img=masked_a, threshold=120,
                                              max_value=255, object_type='dark')
    if VISUALIZE:
        cv2.imshow('x', maskedh_thresh2)
        cv2.waitKey(0)

    maskedh_thresh = plantcv.logical_or(bin_img1=maskedh_thresh1, bin_img2=maskedh_thresh2)
    if VISUALIZE:
        cv2.imshow('x', maskedh_thresh)
        cv2.waitKey(0)

    masked_s = plantcv.rgb2gray_hsv(rgb_img=image, channel='s')
    if VISUALIZE:
        cv2.imshow('s', masked_s)
        cv2.waitKey(0)

    threshold_s = 250 if average_v > 90 else 50
    maskeds_thresh1 = plantcv.threshold.binary(gray_img=masked_s, threshold=threshold_s,
                                              max_value=255, object_type='light')
    if VISUALIZE:
        cv2.imshow('x', maskeds_thresh1)
        cv2.waitKey(0)

    maskeds_thresh2 = plantcv.threshold.binary(gray_img=masked_a, threshold=150,
                                              max_value=255, object_type='light')
    if VISUALIZE:
        cv2.imshow('x', maskeds_thresh2)
        cv2.waitKey(0)

    maskeds_thresh = plantcv.logical_or(bin_img1=maskeds_thresh1, bin_img2=maskeds_thresh2)
    if VISUALIZE:
        cv2.imshow('x', maskeds_thresh)
        cv2.waitKey(0)

    hs = plantcv.logical_or(bin_img1=maskedh_thresh, bin_img2=maskeds_thresh)

    hs_fill = plantcv.fill(bin_img=hs, size=500)
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

    mask = hs_fill
    threshold = 0.3
    image_size = mask.shape[0] * mask.shape[1]
    mask = mask if np.sum(mask) / 255 <= image_size * threshold else plantcv.logical_and(ab_fill, hs_fill)
    if VISUALIZE:
        cv2.imshow('mask', mask)
        cv2.waitKey(0)

    masked = plantcv.apply_mask(img=image, mask=mask, mask_color='white')
    if VISUALIZE:
        cv2.imshow('res', masked)
        cv2.waitKey(0)
    return masked, mask


def postprocess_mask(image, masked, mask):
    id_objects, obj_hierarchy = plantcv.find_objects(masked, mask)
    height, width, _ = masked.shape
    x = width // 4
    y = height // 4
    roi_size = min(height, width) // 2
    roi1, roi_hierarchy = plantcv.roi.rectangle(img=masked, x=x, y=y, h=roi_size, w=roi_size)
    roi_objects, hierarchy, kept_mask, obj_area = plantcv.roi_objects(img=image, roi_contour=roi1,
                                                                      roi_hierarchy=roi_hierarchy,
                                                                      object_contour=id_objects,
                                                                      obj_hierarchy=obj_hierarchy,
                                                                      roi_type='partial')
    obj, image_mask = plantcv.object_composition(img=image, contours=roi_objects, hierarchy=hierarchy)

    if VISUALIZE:
        cv2.imshow('source', image)
        cv2.waitKey(0)
    if image_mask is not None and VISUALIZE:
        cv2.imshow('watershed', image_mask)
        cv2.waitKey(0)
    new_masked = plantcv.apply_mask(img=image, mask=image_mask, mask_color='white')
    return new_masked, image_mask


def instance_segmentation(image, mask):
    watershed = plantcv.watershed_segmentation(image, mask, distance=75)
    return watershed


def analyze_plant(mask):
    new_mask = plantcv.dilate(mask, ksize=9, i=1)
    skeleton = plantcv.morphology.skeletonize(new_mask)
    pruned, seg_img, edge_objects = plantcv.morphology.prune(skel_img=skeleton, size=100, mask=new_mask)
    leaf_obj, stem_obj = plantcv.morphology.segment_sort(skel_img=skeleton,
                                                         objects=edge_objects,
                                                         mask=new_mask)
    segmented_img, labeled_img = plantcv.morphology.segment_id(skel_img=skeleton,
                                                               objects=leaf_obj,
                                                               mask=new_mask)
    cv2.imshow('Seg', labeled_img)
    cv2.waitKey(0)


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
                # if image_name != 'Beta-vulgaris_CotyledonPhase_Substrat3_27082019 (23).JPG':
                #     continue
                scale = 0.25
                resized = cv2.resize(image, dsize=None, fx=scale, fy=scale)
                _logger.info('Redoing segmentation - too many not needed values')
                result, mask = segment(resized)
                result, mask = postprocess_mask(resized, result, mask)
                # instance_segmentation(result, mask)
                # analyze_plant(mask)
                _logger.info(f'Writing result to: {output_path}')
                if should_combine_images:
                    result = combine_images(resized, result)
                cv2.imwrite(output_path, result)


def run():
    parser = cli.create_parser()
    args = parser.parse_args()
    config_logger(args.output_dir, 'kws')
    setup_plantcv(args.debug)

    _logger.info('Getting image paths from: {input_dir} '
                 '(plant names: {plant_names}, '
                 'growth stages: {growth_stages}).'.format(input_dir=args.input_dir,
                                                           plant_names=args.plants_names,
                                                           growth_stages=args.growth_stages))
    images_dict = get_images_paths(args.input_dir, args.plants_names, args.growth_stages)
    _logger.info('Loading images...')
    loaded_images_dict = load_images(images_dict)
    # _logger.info('Creating image directories in: {output_dir}'.format(output_dir=args.output_dir))
    # create_output_directories(loaded_images_dict, args.output_dir)
    # _logger.info('Running segmentation...')
    # run_segmentation(loaded_images_dict, args.output_dir, should_combine_images=args.combine_output_images)

    _logger.info('Loading segmented images...')
    loaded_segmented_dict = load_images(get_images_paths(args.output_dir, args.plants_names, args.growth_stages))
    run_classification(loaded_images_dict)


    # sample_image = loaded_images_dict['Beta vulgaris']['Cotyledon'][0]
    # segment(sample_image)


if __name__ == '__main__':
    run()
