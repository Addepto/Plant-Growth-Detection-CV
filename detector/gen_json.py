import base64
import random
from collections import defaultdict

import cv2
import numpy as np
import os

import json

from read_data import get_contour, read_jsons

from utils import res_single, resize, draw, get_contour, prep_dir, resize_save, label2idx, TRUE_LABELS, get_categories


def gen_json(img_path, category, cont):
    pass

    img = cv2.imread(img_path)

    h, w = img.shape[:2]

    shapes = []
    for c in cont:
        obj = {
            "label": category,
            "points": c.tolist(),
            "group_id": None,
            "shape_type": "polygon",
            "flags": {}
        }

        shapes.append(obj)

    with open(img_path, 'rb') as f:
        img_data = base64.b64encode(f.read()).decode("utf-8")

    data = {
        "version": "4.5.5",
        "flags": {},
        "shapes": shapes,
        "imagePath": img_path,
        "imageData": img_data,
        "imageHeight": h,
        "imageWidth": w
    }

    return data


def save_json(image_path, json_data):
    pass

    root, name = os.path.split(image_path)
    name, ext = os.path.splitext(name)

    json_path = '{}/{}.{}'.format(root, name, 'json')

    with open(json_path, 'w') as f:
        pass
        print(json_path)
        json.dump(json_data, f)


def read_dataset(root_path='/home/volodymyr/Desktop/add/kws_poc/detector/dataset/current',
                 to_fix='/home/volodymyr/Desktop/add/kws_poc/detector/to_fix'):
    os.path.exists(root_path)

    images_path = os.path.join(root_path, 'imgs')
    mask_path = os.path.join(root_path, 'masks')

    assert os.path.exists(images_path)
    assert os.path.exists(mask_path)

    images = {p.name: [s.path for s in os.scandir(p.path) if os.path.splitext(s.path)[-1] in ('.JPG')] for p in
              os.scandir(images_path)}
    masks = {p.name: [s.path for s in os.scandir(p.path)] for p in os.scandir(mask_path)}

    tot = 0

    if to_fix is not None:
        assert os.path.exists(to_fix)

        with open(to_fix, 'r') as f:
            lines = set([l.strip() for l in f.readlines()])

    for cat, paths in images.items():

        sub_masks = {os.path.split(m)[-1]: m for m in masks[cat]}
        for image_path in paths:
            pass

            if image_path in lines:
                _, name = os.path.split(image_path)
                assert name in sub_masks.keys(), '{} not in {}'.format(name, sub_masks)
                mask_path = sub_masks[name]

                mask = cv2.imread(mask_path)

                contours = get_contour(mask)

                json_data = gen_json(image_path, cat, contours)

                save_json(image_path, json_data)

                tot += 1

            # print()
    print('total processed:', tot)


def test():
    pass
    c = [[320, 242],
         [
             538,
             246
         ],
         [
             536,
             417
         ],
         [
             309,
             411
         ]
         ]

    img_path = '/home/volodymyr/Desktop/add/kws_poc/detector/dataset/current/imgs/Agrostemma githago_Cotyledon/Agrostemma-githago_Cotyledon_Substrat1_12022020 (10).JPG'
    mask_path = '/home/volodymyr/Desktop/add/kws_poc/detector/dataset/current/masks/Agrostemma githago_Cotyledon/Agrostemma-githago_Cotyledon_Substrat1_12022020 (10).JPG'

    img = cv2.imread(img_path)
    mask = cv2.imread(mask_path)

    contours = get_contour(mask)

    jd = gen_json(img_path, 'l', contours)

    save_json(img_path, jd)

    # c = np.array(c)
    # cv2.drawContours(img, [c], -1, (0, 0, 255), 2)
    # cv2.imshow('',img)
    # cv2.waitKey()


def test_er():
    pass
    path = '/home/volodymyr/Desktop/add/kws_poc/detector/dataset/current/masks/Agrostemma githago_Foliage/Agrostemma-githago_Foliage_Substrat1_28022020 (327).JPG'

    im = cv2.imread(path)

    ks = 3

    kernel = np.ones((ks, ks), np.uint8)

    erosion = cv2.erode(im, kernel, iterations=1)
    # opening = cv2.morphologyEx(im, cv2.MORPH_OPEN, kernel)

    cv2.imshow('', erosion)
    cv2.waitKey()


def split_images(data_dict, test_proc, logger=None):
    def get_annot(images, data_dict):

        # imgs2annot = {}
        annotations = []
        for img in images:
            # imgs2annot[img] = []
            for annot in data_dict['annotations']:
                if annot['image_id'] == img:
                    # imgs2annot[img].append(annot)
                    annotations.append(annot)

            # if len(imgs2annot[img]) < 1:
            #    print('warning, empty image_id: {}'.format(img))

        # return imgs2annot
        return annotations

    assert isinstance(test_proc, int)
    assert test_proc < 50

    train_data, test_data = [], []

    cat2img = defaultdict()
    ann_id2annot = defaultdict()

    for a in data_dict['annotations']:

        l = a['category_id']

        if l in cat2img:
            cat2img[l].append(a['image_id'])
        else:
            cat2img[l] = [a['image_id']]

        ann_id2annot[a['id']] = a

    cat2img = {k: set(v) for k, v in cat2img.items()}

    fractions = {k: int(len(v) * (test_proc / 100)) for k, v in cat2img.items()}

    for k, v in cat2img.items():
        test_subset = set(random.sample(list(v), k=fractions[k]))

        train_subset = set([i for i in v if i not in test_subset])

        assert len(train_subset.union(test_subset) - v) == 0

        train_annot = get_annot(train_subset, data_dict)
        test_annot = get_annot(test_subset, data_dict)

        train_data.extend(train_annot)
        test_data.extend(test_annot)

    if logger is not None:
        logger.info('Split data_dict into {} and {} train/test subsets'.format(len(train_data), len(test_data)))

    return train_data, test_data


def split_annotations(data_dict, test_proc):
    train_data, test_data = [], []

    label_numb_inst = {}

    for a in data_dict['annotations']:

        l = a['category_id']

        if l in label_numb_inst:
            label_numb_inst[l] += 1
        else:
            label_numb_inst[l] = 1

    fractions = {k: int(v * (test_proc / 100)) for k, v in label_numb_inst.items()}

    data = {k: [] for k in label_numb_inst.keys()}

    for a in data_dict['annotations']:
        data[a['category_id']].append(a)

    for k, v in data.items():
        test = {i['id']: i for i in random.sample(v, fractions[k])}

        train = [val for val in v if val['id'] not in test]
        test = [v for k, v in test.items()]

        train_data.extend(train)
        test_data.extend(test)

    return train_data, test_data


def prep_datadict(data_path, save_path, test_proc=10, skip_unfocused=True, label_type=1):
    def prep_categories(categories, skip_unfocused=True):
        cat = [{"id": v, "name": k, "supercategory": "background"} for k, v in categories.items() if
               skip_unfocused and k != 'unfocus']

        return cat

    raw_json = read_jsons(data_path)

    assert os.path.exists(save_path)

    images = []
    annotations = []

    ann_ndx = 0
    for ndx, rj in enumerate(raw_json):
        pass

        img_name = os.path.split(rj['imagePath'])[-1]
        # img_name = "/".join(rj['imagePath'].split('/')[-2:])

        image = {"id": ndx, "width": rj['imageWidth'], "height": rj['imageHeight'], "file_name": img_name,
                 "license": None, "flickr_url": "", "coco_url": None, "date_captured": None}

        images.append(image)

        for ann in rj['shapes']:
            label = ann['label']

            if skip_unfocused and label == 'unfocus':
                continue

            label_ndx = label2idx(label_type)[label]

            c = np.array(ann['points'])

            xmin, ymin = np.min(c[:, :1]), np.min(c[:, 1:])
            xmax, ymax = np.max(c[:, :1]), np.max(c[:, 1:])

            bbox = [xmin, ymin, xmax, ymax]
            area = (xmax - xmin) * (ymax - ymin)

            polygon = []
            for x, y in zip(c[:, :1], c[:, 1:]):
                polygon.append(int(x))
                polygon.append(int(y))

            annotation = {"id": ann_ndx,
                          "image_id": ndx,
                          "category_id": label_ndx,
                          "segmentation": [polygon],
                          "area": area,
                          "bbox": bbox,
                          "iscrowd": 0}
            annotations.append(annotation)

            ann_ndx += 1

    categories = prep_categories(label2idx(label_type))

    data_dict = {"images": images, "annotations": annotations, 'categories': categories}

    train_annot, test_annot = split_annotations(data_dict, test_proc)

    data_dict['annotations'] = train_annot
    train_data_dict = data_dict

    with open('{}/train_data_dict.json'.format(save_path), 'w') as f:
        json.dump(train_data_dict, f, cls=NpEncoder)

    data_dict['annotations'] = test_annot
    test_data_dict = data_dict

    with open('{}/test_data_dict.json'.format(save_path), 'w') as f:
        json.dump(test_data_dict, f, cls=NpEncoder)


def dev():
    raw_json = read_jsons('./dataset/current/imgs')
    print()


def relabel(data_dict, label_type, unfocus=False):
    def lblndx2newlbl(ndx, cur_labels_all):
        true_label = TRUE_LABELS[ndx]
        new_label = cur_labels_all[true_label]
        return new_label

    annotations = data_dict['annotations']

    all_labels_all = label2idx(label_type, unfocus)
    cur_labs = get_categories(label_type, unfocus)

    new_annots = []

    for annot in annotations:

        if not unfocus and annot['category_id'] == 10:
            continue

        # print(annot['category_id'], lblndx2newlbl(annot['category_id'], all_labels_all))
        annot['category_id'] = lblndx2newlbl(annot['category_id'], all_labels_all)
        new_annots.append(annot)

    ids = set([v for k, v in all_labels_all.items()])

    new_categories = [{'id': i, 'name': cur_labs[i - 1], 'supercategory': None} for i in ids]
    # new_categories.append({'id': 0, 'name': '_background_', 'supercategory': None})

    data_dict['categories'] = new_categories
    data_dict['annotations'] = new_annots

    return data_dict


def prepare_data(data_dict, label_type, train_test_dd, unfocus=False, proc=15, logger=None):
    assert os.path.exists(data_dict), data_dict

    prep_dir(train_test_dd)

    save = '{}/{}'.format(train_test_dd, label_type)

    if os.path.exists(save):
        files = [p.name for p in os.scandir(save)]
        assert 'train_data_dict.json' in files, '\'train_data_dict.json\' not found at: {}'.format(save)
        assert 'test_data_dict.json' in files, '\'test_data_dict.json\' not found at: {}'.format(save)

        save_train = '{}/train_data_dict.json'.format(save)
        save_test = '{}/test_data_dict.json'.format(save)

        if logger is not None:
            logger.info('Using previously generated json data_dict files')

        return save_train, save_test
    else:

        with open(data_dict, 'r') as f:
            data_dict = json.load(f)

        data_dict = relabel(data_dict, label_type, unfocus)

        # train_annot, test_annot = split_annotations(data_dict, proc)
        train_annot, test_annot = split_images(data_dict, proc, logger)

        assert len({i['image_id'] for i in train_annot}.intersection({i['image_id'] for i in test_annot})) == 0, \
            'image/s in train subset intersect/s with image/s in test subset'

        data_dict_train = data_dict.copy()
        data_dict_test = data_dict.copy()

        data_dict_train['annotations'] = train_annot
        data_dict_test['annotations'] = test_annot

        prep_dir(save)

        if logger is not None:
            logger.info('Generating a new json data_dict files')

        if save is not None:
            save_train = '{}/train_data_dict.json'.format(save)
            with open(save_train, 'w') as f:
                json.dump(data_dict_train, f, )

            save_test = '{}/test_data_dict.json'.format(save)
            with open(save_test, 'w') as f:
                json.dump(data_dict_test, f, )

        return save_train, save_test


if __name__ == '__main__':
    # test_er()

    # relabel
    # a, b = relabel_and_split('./dataset/current/all_out/annotations.json', label_type=1)
    # print()
    with open('/home/volodymyr/Desktop/add/kws_poc/detector/dataset/current/all_out/annotations.json', 'r') as f:
        data_dict = json.load(f)

    data_dict = relabel(data_dict, 3, False)
    split_images(data_dict, 30)
