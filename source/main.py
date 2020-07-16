import logging
import os

import cv2
import numpy as np
from plantcv import plantcv
from tqdm import tqdm

import cli
from classification import run_classification
from data import get_images_paths, load_images
from kws_logging import config_logger

_logger = logging.getLogger('kws')


def setup_plantcv(should_debug=False):
    """
    Initialize some params in plantcv
    :param should_debug: Should we use the debug setup - saving all operations as images
    """
    plantcv.params.line_thickness = 3
    if should_debug:
        plantcv.params.debug = 'print'
        plantcv.params.debug_outdir = os.path.join(os.path.dirname(__file__), 'debug')
        _logger.info(f'Plantcv debugging is turned on. All results will be saved to: {plantcv.params.debug_outdir}')
        if os.path.exists(plantcv.params.debug_outdir):
            os.rmdir(plantcv.params.debug_outdir)
        os.makedirs(plantcv.params.debug_outdir)


def segment(image):
    """
    Find plants in the image using plantcv.
    :param image: image that should be taken into consideration
    :return: masked image and mask used for masking
    """
    masked_v = plantcv.rgb2gray_hsv(rgb_img=image, channel='v')
    average_v = np.average(masked_v)

    masked_a = plantcv.rgb2gray_lab(rgb_img=image, channel='a')

    masked_b = plantcv.rgb2gray_lab(rgb_img=image, channel='b')

    masked_h = plantcv.rgb2gray_hsv(rgb_img=image, channel='h')

    maskedh_thresh1 = plantcv.threshold.binary(gray_img=masked_h, threshold=25,
                                               max_value=255, object_type='light')
    maskedh_thresh2 = plantcv.threshold.binary(gray_img=masked_a, threshold=120,
                                               max_value=255, object_type='dark')

    maskedh_thresh = plantcv.logical_or(bin_img1=maskedh_thresh1, bin_img2=maskedh_thresh2)

    masked_s = plantcv.rgb2gray_hsv(rgb_img=image, channel='s')

    threshold_s = 250 if average_v > 90 else 50
    maskeds_thresh1 = plantcv.threshold.binary(gray_img=masked_s, threshold=threshold_s,
                                               max_value=255, object_type='light')

    maskeds_thresh2 = plantcv.threshold.binary(gray_img=masked_a, threshold=150,
                                               max_value=255, object_type='light')

    maskeds_thresh = plantcv.logical_or(bin_img1=maskeds_thresh1, bin_img2=maskeds_thresh2)

    hs = plantcv.logical_or(bin_img1=maskedh_thresh, bin_img2=maskeds_thresh)

    hs_fill = plantcv.fill(bin_img=hs, size=500)

    maskeda_thresh = plantcv.threshold.binary(gray_img=masked_a, threshold=120,
                                              max_value=255, object_type='dark')
    maskeda_thresh1 = plantcv.threshold.binary(gray_img=masked_a, threshold=150,
                                               max_value=255, object_type='light')
    maskedb_thresh = plantcv.threshold.binary(gray_img=masked_b, threshold=165,
                                              max_value=255, object_type='light')

    ab1 = plantcv.logical_or(bin_img1=maskeda_thresh, bin_img2=maskedb_thresh)
    ab = plantcv.logical_or(bin_img1=maskeda_thresh1, bin_img2=ab1)

    ab_fill = plantcv.fill(bin_img=ab, size=500)

    mask = hs_fill
    threshold = 0.3
    image_size = mask.shape[0] * mask.shape[1]
    mask = mask if np.sum(mask) / 255 <= image_size * threshold else plantcv.logical_and(ab_fill, hs_fill)

    masked = plantcv.apply_mask(img=image, mask=mask, mask_color='white')

    return masked, mask


def postprocess_mask(image, masked, mask):
    """
    Do some post processing - usually the plant is in the image center, so try to use this info
    :param image: image to be masked
    :param masked: masked image
    :param mask: mask used to create masked image
    :return: new masked image and mask
    """
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

    new_masked = plantcv.apply_mask(img=image, mask=image_mask, mask_color='white')
    return new_masked, image_mask


def instance_segmentation(image, mask):
    """
    Perform instance segmentation
    :param image: input image
    :param mask: mask for the image
    :return: segmented output
    """
    watershed = plantcv.watershed_segmentation(image, mask, distance=75)
    return watershed


def analyze_plant(mask):
    """
    Perform morphological analysis of a plant
    :param mask: mask to analyze
    :return: segmented image
    """
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
    return segmented_img


def create_output_directories(loaded_images_dict, output_dir):
    """
    Create output directories in same manner as the input
    :param loaded_images_dict: loaded images
    :param output_dir: root of the output
    """
    for plant_name in loaded_images_dict.keys():
        for growth_stage in loaded_images_dict[plant_name].keys():
            final_dir_path = os.path.join(output_dir, plant_name, growth_stage)
            os.makedirs(final_dir_path, exist_ok=True)


def combine_images(image1, image2):
    """
    Just combine to images for visualization purposes
    :param image1:
    :param image2:
    :return: combined image
    """
    return np.hstack((image1, image2))


def run_segmentation(loaded_images_dict, output_dir, should_combine_images=False):
    """
    Perform segmentation and write output as images
    :param loaded_images_dict: dict with images per plant per growth stage
    :param output_dir: where to save images (root)
    :param should_combine_images: should the result be wrote as base_image | masked image
    """
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
    """
    Run the processing
    """
    parser = cli.create_parser()
    args = parser.parse_args()
    config_logger(args.output_dir, 'kws')
    setup_plantcv(args.debug)

    _logger.info('Redo segmentation set to: {redo_segmentation}'.format(redo_segmentation=args.redo_segmentation))

    if args.redo_segmentation:
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
        run_segmentation(loaded_images_dict, args.output_dir, should_combine_images=args.combine_output_images)

    _logger.info('Loading segmented images...')
    loaded_segmented_dict = load_images(get_images_paths(args.output_dir, args.plants_names, args.growth_stages))
    run_classification(loaded_segmented_dict)
    run_classification(loaded_segmented_dict, use_growth=True)

    # sample_image = loaded_images_dict['Beta vulgaris']['Cotyledon'][0]
    # segment(sample_image)


if __name__ == '__main__':
    run()
