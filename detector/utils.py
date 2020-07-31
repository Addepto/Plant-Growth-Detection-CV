import argparse
import copy
import json
import math
import os

import cv2
import numpy as np
import torch

from detectron2.data import detection_utils as utils
import detectron2.data.transforms as T

args = None



def get_args():
    list_of_models = ['COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml',
                      "Misc/cascade_mask_rcnn_R_50_FPN_3x.yaml"]
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', type=str, choices=['train', 'test'],
                        help='train or test')

    parser.add_argument('--data_dict', type=str, default='./dataset/current/all_out/annotations.json')
    parser.add_argument('--train_test_dd', type=str, default='./dataset/current/json_data/current', help=
    'root path to train_datadict.json and test_datadict.json. If None they will be generated\
     based on --data_dict')

    parser.add_argument('-o', '--output', type=str, default='./output', help='output dir')
    parser.add_argument('--model_path', type=str, default=None, help='output dir')
    parser.add_argument('--model_type', type=str, choices=list_of_models,
                        default=list_of_models[0])

    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--unfocused', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--override_tt', action='store_true', help='override train/test data_dicts')
    parser.add_argument('--sf', action='store_true', help='skip unfocused class')

    parser.add_argument('--test_split', type=int, default=30,
                        help='split --data_dict into subsets with data_dict - test and test ')
    parser.add_argument('--label_type', type=int, choices=[1, 2, 3], required=True)
    parser.add_argument('--thrsh', type=float, default=0.8)
    parser.add_argument('--lr', type=float, default=1e-3)

    # parser.add_argument('--sum', dest='accumulate', action='store_const',
    #                    const=sum, default=max,
    #                    help='sum the integers (default: find the max)')

    global args
    args = parser.parse_args()
    check_args(args)

    return args


def check_args(args):
    if args.mode == 'test' and args.model_path is None:
        raise ValueError('provide model path for evaluation')
    return True


def resize(img_path):
    assert os.path.exists(img_path)

    img_types = {p.name: p.path for p in os.scandir(img_path)}

    for key in img_types.keys():
        img_sub_dir = img_types[key]

        for p in os.scandir(img_sub_dir):
            path = p.path

            img = cv2.imread(path)

            img = cv2.resize(img, (864, 576), interpolation=cv2.INTER_CUBIC)

            cv2.imwrite(path, img)


def mapper(dataset_dict):
    dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
    # can use other ways to read image
    image = utils.read_image(dataset_dict["file_name"], format="BGR")
    # can use other augmentations
    transform = T.Resize((800, 800)).get_transform(image)
    image = torch.from_numpy(transform.apply_image(image).transpose(2, 0, 1))
    annos = [
        utils.transform_instance_annotations(annotation, [transform], image.shape[1:])
        for annotation in dataset_dict.pop("annotations")
    ]
    return {
       # create the format that the model expects
       "image": image,
       "instances": utils.annotations_to_instances(annos, image.shape[1:])
    }

def res_single(path, override=True, save_to=None):
    img = cv2.imread(path)

    h, w = img.shape[:2]
    dw, dh = (864, 576)

    if dw != w and dh != h:
        img = cv2.resize(img, (dw, dh), interpolation=cv2.INTER_CUBIC)
    else:
        pass

    if override:
        cv2.imwrite(path, img)
    elif save_to is not None:

        _, name = os.path.split(path)

        img_path = os.path.join(save_to, name)

        cv2.imwrite(img_path, img)
    else:
        raise Exception('error')


def get_contour(mask, cont_len_thresh=100):
    try:

        imgray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(imgray, 250, 255, 0)

        assert len(mask.shape) == 3
        h, w = mask.shape[:2]

        thresh = cv2.rectangle(thresh, (0, 0), (w, h), (255, 255, 255), thickness=4)

        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        cont = []

        for c in contours:
            cand = c.shape[0]
            if cand > cont_len_thresh:
                cont.append(c.squeeze().astype(np.int32))

        # mask = cv2.drawContours(mask, cont, -1, (0, 255, 0), 3)
        return np.array(cont)
    except Exception as e:

        print('error:', e)
        return None


def draw(path, c):
    assert os.path.exists(path)

    print(path)
    if isinstance(c, list):
        c = np.array(c)

    xmin, ymin = np.min(c[:, :1]), np.min(c[:, 1:])
    xmax, ymax = np.max(c[:, :1]), np.max(c[:, 1:])

    img = cv2.imread(path)
    #
    # cv2.drawContours(img, [c], -1, (0, 244, 0), 2)
    cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 244, 0), 2)

    cv2.imshow('', img)
    cv2.waitKey()


def prep_dir(p):
    if not os.path.exists(p):
        os.makedirs(p)
    return p


def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
    if interArea == 0:
        return 0
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou


def pred2tg(annot_box, annot_class, list_pred):
    pass

    idx = None
    max_iou = 0
    for i, (pred_bb, pred_cls) in enumerate(list_pred):
        iou = bb_intersection_over_union(annot_box, pred_bb)
        if iou > max_iou:
            max_iou = iou
            idx = i

    if idx is not None:
        pred_box, pred_class = list_pred[idx]
        if pred_class == annot_class:
            return 1
        else:
            return 0
    else:
        return 0


def cast_cont(mask, cont):
    pass
    s = 2
    st, end = 0, 2
    for c in range(int(len(cont) / 2)):
        # x, y  = int(x), int(y)
        # mask[x][y] = 255
        a = cont[st:end]

        p1, p2 = a
        p1 = int(p1[0]), int(p1[1])
        p2 = int(p2[0]), int(p2[1])

        cv2.rectangle(mask, p1, p2, (255, 255, 255), 1)
        st += 2
        end += 2
        print()


from PIL import Image, ImageDraw


def shape_to_mask(
        img_shape, points, shape_type=None, line_width=10, point_size=5
):
    mask = np.zeros(img_shape[:2], dtype=np.uint8)
    mask = Image.fromarray(mask)
    draw = ImageDraw.Draw(mask)
    xy = [tuple(point) for point in points]
    if shape_type == "circle":
        assert len(xy) == 2, "Shape of shape_type=circle must have 2 points"
        (cx, cy), (px, py) = xy
        d = math.sqrt((cx - px) ** 2 + (cy - py) ** 2)
        draw.ellipse([cx - d, cy - d, cx + d, cy + d], outline=1, fill=1)
    elif shape_type == "rectangle":
        assert len(xy) == 2, "Shape of shape_type=rectangle must have 2 points"
        draw.rectangle(xy, outline=1, fill=1)
    elif shape_type == "line":
        assert len(xy) == 2, "Shape of shape_type=line must have 2 points"
        draw.line(xy=xy, fill=1, width=line_width)
    elif shape_type == "linestrip":
        draw.line(xy=xy, fill=1, width=line_width)
    elif shape_type == "point":
        assert len(xy) == 1, "Shape of shape_type=point must have 1 points"
        cx, cy = xy[0]
        r = point_size
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=1, fill=1)
    else:
        assert len(xy) > 2, "Polygon must have points more than 2"
        draw.polygon(xy=xy, outline=1, fill=1)
    mask = np.array(mask, dtype=bool)
    return mask


def crop(img, annot):
    pts = []
    slide = 2
    st, end = 0, 2
    for i in range(int(len(annot[0]) / 2)):
        pts.append(annot[0][st:end])
        st += slide
        end += slide

    pts = np.array(pts).astype('uint8')

    mask = shape_to_mask(img.shape[:2], pts, 'polygon').astype('uint8') * 255

    dst = cv2.bitwise_and(img, img, mask=mask)

    bg = np.ones_like(img, np.uint8) * 255
    cv2.bitwise_not(bg, bg, mask=mask)
    dst2 = bg + dst

    return dst2


def prep_clean_masks():
    json_data_path = '/home/volodymyr/Desktop/add/kws_poc/detector/dataset/current/test_data_dict.json'
    img_all = '/home/volodymyr/Desktop/add/kws_poc/detector/dataset/current/all_out'

    save_to = '/home/volodymyr/Desktop/add/kws_poc/detector/dataset/test_test'
    assert os.path.exists(json_data_path)

    with open(json_data_path, 'r')as f:
        data = json.load(f)

        id2image = {i['id']: i['file_name'] for i in data['images']}

        saved_imgs = {v: 0 for k, v in id2image.items()}

        acc = 0
        total = len(data['annotations'])

        for record in data['annotations']:
            img_id = record['image_id']

            file_name = id2image[img_id]

            img_path = os.path.join(img_all, file_name)

            bbox = record['bbox']
            annotations = record['segmentation']
            cat_id = record['category_id']

            im = cv2.imread(img_path)

            cv2.rectangle(im, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (243, 33, 231), 2)

            cv2.imshow('', im)
            cv2.waitKey()

            mask = crop(im, annotations)
            # crop1(image)
            cv2.imwrite(
                '/home/volodymyr/Desktop/add/kws_poc/detector/dataset/test_test/JPEGImages/Beta-vulgaris_Cotyledon_Substrat3_11102019 (3).jpg',
                mask)
            return


def resize_save(images, sub_fold='imgs', save='./dataset/current'):
    prep_dir(save)

    save_to_imgs = os.path.join(save, sub_fold)

    for key, val in images.items():
        category_dir = prep_dir(os.path.join(save_to_imgs, key))

        for img_path in val:
            res_single(img_path, False, category_dir)


TRUE_LABELS = {
    1: "Agrostemma githago_Cotyledon",
    2: "Agrostemma githago_Intermediate",
    3: "Agrostemma githago_Foliage",
    4: "Beta vulgaris_Cotyledon",
    5: "Beta vulgaris_Intermediate",
    6: "Beta vulgaris_Foliage",
    7: "Crepis setosa_Cotyledon",
    8: "Crepis setosa_Intermediate",
    9: "Crepis setosa_Foliage",
    10: "unfocus"}

LABEL_TO_IDX_1 = {
    'Agrostemma githago_Cotyledon': 1,
    'Agrostemma githago_Intermediate': 2,
    'Agrostemma githago_Foliage': 3,
    'Beta vulgaris_Cotyledon': 4,
    'Beta vulgaris_Intermediate': 5,
    'Beta vulgaris_Foliage': 6,
    'Crepis setosa_Cotyledon': 7,
    'Crepis setosa_Intermediate': 8,
    'Crepis setosa_Foliage': 9}

LABEL_TO_IDX_2 = {
    'Agrostemma githago_Cotyledon': 1,
    'Agrostemma githago_Intermediate': 1,
    'Agrostemma githago_Foliage': 1,
    'Beta vulgaris_Cotyledon': 2,
    'Beta vulgaris_Intermediate': 2,
    'Beta vulgaris_Foliage': 2,
    'Crepis setosa_Cotyledon': 1,
    'Crepis setosa_Intermediate': 1,
    'Crepis setosa_Foliage': 1}

LABEL_TO_IDX_3 = {
    'Agrostemma githago_Cotyledon': 1,
    'Agrostemma githago_Intermediate': 2,
    'Agrostemma githago_Foliage': 3,

    'Beta vulgaris_Cotyledon': 1,
    'Beta vulgaris_Intermediate': 2,
    'Beta vulgaris_Foliage': 3,

    'Crepis setosa_Cotyledon': 1,
    'Crepis setosa_Intermediate': 2,
    'Crepis setosa_Foliage': 3}


def label2idx(label_type=1, unfocused=True):
    if label_type == 1:
        if unfocused:
            LABEL_TO_IDX_1['unfocus'] = 10
        return LABEL_TO_IDX_1
    elif label_type == 2:
        if unfocused:
            LABEL_TO_IDX_2['unfocus'] = 3
        return LABEL_TO_IDX_2
    elif label_type == 3:
        if unfocused:
            LABEL_TO_IDX_3['unfocus'] = 4
        return LABEL_TO_IDX_3
    else:
        raise Exception('no such ndx: ', label_type)


def labeltype2nclass(label_type, unfocused=False):
    if label_type == 1:
        return 10 if unfocused else 9
    elif label_type == 2:
        return 3 if unfocused else 2
    elif label_type == 3:
        return 4 if unfocused else 3


def get_categories(label_type, unfocused=False):
    if label_type == 1:
        ret = [
            'Agrostemma githago_Cotyledon',
            'Agrostemma githago_Intermediate',
            'Agrostemma githago_Foliage',
            'Beta vulgaris_Cotyledon',
            'Beta vulgaris_Intermediate',
            'Beta vulgaris_Foliage',
            'Crepis setosa_Cotyledon',
            'Crepis setosa_Intermediate',
            'Crepis setosa_Foliage']
        if unfocused:
            ret.append('unfocuse')
        return ret
    elif label_type == 2:
        ret = ['Not beta vulgaris', 'Beta vulgaris', ]
        if unfocused:
            ret.append('unfocuse')
        return ret
    elif label_type == 3:
        ret = [
            'Cotyledon',
            'Intermediate',
            'Foliage']
        if unfocused:
            ret.append('unfocuse')
        return ret


if __name__ == '__main__':
    prep_clean_masks()
