import os
import json
import random
import shutil

import cv2
import numpy as np
from detectron2.structures import BoxMode

from utils import get_contour, prep_dir, resize_save, label2idx


def read_data_old(root_path):
    pass
    img_dir = 'img'
    mask_dir = 'mask'

    img_path = os.path.join(root_path, img_dir)
    mask_path = os.path.join(root_path, mask_dir)

    categories = [p.path for p in os.scandir(root_path)]

    assert os.path.exists(root_path)
    # assert os.path.exists(img_path)
    # assert os.path.exists(mask_path)

    img_types = {p.name: p.path for p in os.scandir(img_path)}
    mask_types = {p.name: p.path for p in os.scandir(mask_path)}

    assert img_types.keys() == mask_types.keys()

    images = {}
    masks = {}

    total_imgs = 0

    for key in img_types.keys():
        img_sub_dir = img_types[key]
        mask_sub_dir = mask_types[key]

        sub_images = {p.name: p.path for p in os.scandir(img_sub_dir)}
        sub_mask = {p.name: p.path for p in os.scandir(mask_sub_dir)}

        assert sub_images.keys() == sub_mask.keys()

        assert len(sub_mask) == len(sub_images)

        images[key] = sub_images
        masks[key] = sub_mask

        total_imgs += len(sub_images)

    # print(total_imgs)
    return images, masks


def read_data(root_path):
    pass
    img_dir = 'image'
    mask_dir = 'mask'
    img_path = os.path.join(root_path, img_dir)
    mask_path = os.path.join(root_path, mask_dir)

    # images = {p.name: {sub_path.name: {p.name: p.path for p in os.scandir(sub_path.path)}\
    #                    for sub_path in os.scandir(p.path)} for p in os.scandir(img_path)}
    # masks = {p.name: {sub_path.name: {p.name: p.path for p in os.scandir(sub_path.path)} \
    #                   for sub_path in os.scandir(p.path)} for p in os.scandir(mask_path)}

    images = {}
    masks = {}

    total = 0
    category_id = 0

    # print('categories:')
    for category in os.scandir(img_path):
        cat_name = category.name
        for sub_category in os.scandir(category):
            sub_category_name = sub_category.name

            # print('{} {}'.format(category.name, sub_category.name))
            # TODO: category_id += 1
            for image in os.scandir(sub_category):
                image_name = image.name

                image_path = image.path
                mask_img_path = os.path.join(mask_path, cat_name, sub_category_name, image_name)

                assert os.path.exists(mask_img_path), mask_img_path

                full_category = '{}_{}'.format(cat_name, sub_category_name)

                if full_category in images:
                    images[full_category].append(image_path)
                else:
                    images[full_category] = [image_path]

                if full_category in masks:
                    masks[full_category].append(mask_img_path)
                else:
                    masks[full_category] = [mask_img_path]

                # images[category_id] = image_path
                # masks[category_id] = mask_path

                total += 1
            # TODO
            category_id += 1

    # print(category_id, total)
    return images, masks


def get_data_dict(images, masks, debug, classes=3):
    dataset_dicts = []
    print('NUMB_CLASSES:', classes)

    idx = 0

    for category, (k, v) in enumerate(images.items()):

        sub_masks = {os.path.split(m)[-1]: m for m in masks[k]}

        for path in v:

            _, name = os.path.split(path)

            record = {}

            height, width = cv2.imread(path).shape[:2]

            # print('orig:', height, width)

            record["file_name"] = path
            record["image_id"] = idx
            record["height"] = height
            record["width"] = width
            record['name'] = os.path.split(path)[-1]

            assert name in sub_masks

            mask_path = sub_masks[name]

            mask = cv2.imread(mask_path)
            contours = get_contour(mask)

            assert len(contours) > 0

            objs = []
            for c in contours:
                c = np.squeeze(c)

                xmin, ymin = np.min(c[:, :1]), np.min(c[:, 1:])
                xmax, ymax = np.max(c[:, :1]), np.max(c[:, 1:])

                _c = []
                for x, y in zip(c[:, :1], c[:, 1:]):
                    _c.append(int(x))
                    _c.append(int(y))

                if classes == 3:
                    label = category
                elif classes == 3:
                    label = category + 1
                elif classes == 4:
                    label = 1
                elif classes == 9:
                    label = category
                else:
                    raise ValueError('{} is not supported, only 1,3,4'.format(classes))

                obj = {
                    "bbox": [xmin, ymin, xmax, ymax],
                    "bbox_mode": BoxMode.XYXY_ABS,
                    "segmentation": [_c],
                    "category_id": label,
                }
                objs.append(obj)

            record["annotations"] = objs

            dataset_dicts.append(record)

            idx += 1

    if debug:
        dataset_dicts = [d for d in dataset_dicts if
                         d['name'] == 'Agrostemma-githago_Cotyledon_Substrat1_12022020 (30).JPG']

        # draw(dataset_dicts[0]['file_name'], dataset_dicts[0]['annotations'][0]['segmentation'][0])

    return dataset_dicts


def test_read():
    pass

    # root_path = './subset'
    root_path = './subset'
    root_path = ''
    images, masks = read_data(root_path)

    resize_save(masks, 'masks')


def read_jsons(img_path):
    assert os.path.exists(img_path)

    data = []

    c = 0

    for i, category in enumerate(os.scandir(img_path)):

        for image in os.scandir(category):

            if os.path.splitext(image.path)[-1].lower() in ('.jpg'):
                image_path = image.path

                img_name_path = os.path.splitext(image_path)[0]

                json_path = '{}.json'.format(img_name_path)

                assert os.path.exists(json_path), json_path

                with open(json_path, 'r') as f:
                    json_data = json.load(f)

                data.append(json_data)

                c += 1

    return data


def read_data_and_save_mask(root_path):
    pass

    mask_path = ''

    prep_dir(mask_path)

    save_new_masks_path = ''

    images = {}

    masks = {p.name.split_annotations('.')[0]: p.path for p in os.scandir(mask_path)}

    # print('categories:')
    for category in os.scandir(root_path):
        cat_name = category.name

        for sub_category in os.scandir(category):
            sub_category_name = sub_category.name

            full_cat = os.path.join(cat_name, sub_category_name)

            for image in os.scandir(sub_category):
                image_name = image.name

                images[image_name.split_annotations('.')[0]] = full_cat

    for k, v in masks.items():
        assert k in images

        save_root = os.path.join(save_new_masks_path, images[k])

        prep_dir(save_root)

        save_full = os.path.join(save_root, k + '.png')

        shutil.copy2(masks[k], save_full)


def assert_content(orig, ref):

    def read_folders(root_path):

        images = {}

        for category in os.scandir(root_path):
            cat_name = category.name

            for sub_category in os.scandir(category):
                sub_category_name = sub_category.name

                full_cat = os.path.join(cat_name, sub_category_name)

                for image in os.scandir(sub_category):
                    image_name = image.name

                    images[image_name.split_annotations('.')[0]] = full_cat

        return images

    orig_imgms = read_folders(orig).keys()
    ref_images = read_folders(ref).keys()

    r = orig_imgms - ref_images

    assert len(r) == 0
    print('content is ok')

if __name__ == '__main__':
    DEBUG = False

    read_data_and_save_mask('')
    assert_content('',
                   '', )
