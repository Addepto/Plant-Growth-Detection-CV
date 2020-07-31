import json
import os
import cv2

from detectron2.data.datasets import register_coco_instances
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.utils.logger import setup_logger
import torch

from utils import get_args, labeltype2nclass, get_categories, pred2tg, mapper

args = get_args()

DEBUG = args.debug
OUTPUT = '{}_dev'.format(args.output) if DEBUG else args.output
TO_TRAIN = True if args.mode.lower() == 'train' else False

model_iter = 'init'
if not TO_TRAIN:
    pass
    model_iter = os.path.split(args.model_path)[-1].split('_')[-1].split('.')[0]

OUTPUT = OUTPUT if TO_TRAIN else '{}'.format(os.path.split(args.model_path)[0])
OUTPUT_LOGS = '{}/{}_{}_log.txt'.format(OUTPUT, 'train' if TO_TRAIN else 'eval', model_iter)

logger = setup_logger(output=OUTPUT_LOGS)
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, build_detection_test_loader
from detectron2.engine import DefaultTrainer

from gen_json import prepare_data

TRAIN = True if TO_TRAIN else False
EVAL = False if TO_TRAIN else True

RESUME = args.resume

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_CLASSES = labeltype2nclass(args.label_type, args.unfocused)
THRSH = args.thrsh

MODEL_YML = args.model_type

MODEL_PATH = args.model_path

root_images_path = './dataset/current/all_out'
data_dict = args.data_dict

train_datadict_path, test_datadict_path = prepare_data(data_dict=data_dict, label_type=args.label_type,
                                                       train_test_dd=args.train_test_dd,
                                                       unfocus=args.unfocused, proc=args.test_split,
                                                       logger=logger)

assert train_datadict_path is not None and test_datadict_path is not None

register_coco_instances("burak_train", {}, train_datadict_path, root_images_path)
register_coco_instances("burak_test", {}, test_datadict_path, root_images_path)

cat = get_categories(args.label_type, args.unfocused)
metadata = MetadataCatalog.get("burak_test").set(thing_classes=cat)


def get_model_config(model_yml, is_train, model_path=None, debug=False, device='cpu', training_output=None):
    logger.info("DEBUG: {}".format(DEBUG))
    logger.info('label_type: {}'.format(args.label_type))
    for c in cat:
        logger.info('\t- {}'.format(c))
    logger.info('model_type: {}'.format(args.model_type))
    logger.info("Numb classes: {}".format(NUM_CLASSES))
    logger.info("Unfocused: {}".format(args.unfocused))
    logger.info("Output dir: {}".format(OUTPUT))

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(model_yml))

    if model_path:
        cfg.MODEL.WEIGHTS = model_path
    else:
        model_zoo.get_checkpoint_url(model_yml)

    cfg.DATASETS.TRAIN = ("burak_train",)
    cfg.DATASETS.TEST = ("burak_test",)

    cfg.MODEL.ROI_HEADS.NUM_CLASSES = NUM_CLASSES
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256

    if device == 'cpu':
        logger.info('DEVICE: {}'.format(device))
        cfg.MODEL.DEVICE = 'cpu'

    if is_train:

        cfg.OUTPUT_DIR = training_output
        if debug:
            cfg.DATALOADER.NUM_WORKERS = 1
            cfg.SOLVER.IMS_PER_BATCH = 1
            cfg.SOLVER.BASE_LR = args.lr
            cfg.SOLVER.WARMUP_ITERS = 5000
            cfg.SOLVER.MAX_ITER = 10000
            cfg.SOLVER.STEPS = (3000, 6000)
            cfg.SOLVER.GAMMA = 0.1
            cfg.SOLVER.CHECKPOINT_PERIOD = 2500
            cfg.MODEL.ROI_HEADS.NUM_CLASSES = NUM_CLASSES
            cfg.TEST.EVAL_PERIOD = 500000
            cfg.OUTPUT_DIR = OUTPUT
        else:
            cfg.DATALOADER.NUM_WORKERS = 4
            cfg.SOLVER.IMS_PER_BATCH = 6
            cfg.SOLVER.BASE_LR = args.lr
            cfg.SOLVER.WARMUP_ITERS = 5000
            cfg.SOLVER.MAX_ITER = 50000
            cfg.SOLVER.STEPS = (15000, 22500)
            cfg.SOLVER.GAMMA = 0.1
            cfg.SOLVER.CHECKPOINT_PERIOD = 2500
            cfg.MODEL.ROI_HEADS.NUM_CLASSES = NUM_CLASSES
            cfg.TEST.EVAL_PERIOD = 50000

        os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    else:
        logger.info("Model path: {}".format(MODEL_PATH))

        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = THRSH

    return cfg


def train():
    cfg = get_model_config(model_yml=MODEL_YML, is_train=True, model_path=MODEL_PATH, device=DEVICE, debug=DEBUG,
                           training_output=OUTPUT)

    trainer = DefaultTrainer(cfg, mapper=mapper)
    trainer.resume_or_load(resume=RESUME)

    trainer.train()


def eval():
    cfg = get_model_config(model_yml=MODEL_YML, is_train=False, model_path=MODEL_PATH, device=DEVICE, debug=DEBUG,
                           training_output=OUTPUT)
    predictor = DefaultPredictor(cfg)
    evaluator = COCOEvaluator("burak_test", cfg, False, output_dir=args.output)
    val_loader = build_detection_test_loader(cfg, "burak_test")
    inference_on_dataset(predictor.model, val_loader, evaluator)


def test():
    cfg = get_model_config(model_yml=MODEL_YML, is_train=False, model_path=MODEL_PATH)
    predictor = DefaultPredictor(cfg)
    img_path = ''
    assert os.path.exists(img_path)
    im = cv2.imread(img_path)
    outputs = predictor(im)
    v = Visualizer(im[:, :, ::-1],
                   metadata=metadata,
                   scale=1)
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    im = v.get_image()[:, :, ::-1]
    print(outputs)
    cv2.imshow('', im)
    cv2.waitKey()


def prep_dir(p):
    if not os.path.exists(p):
        os.makedirs(p)


def cust_eval():
    cfg = get_model_config(model_yml=MODEL_YML, is_train=False, model_path=MODEL_PATH, device=DEVICE, debug=DEBUG,
                           training_output=OUTPUT)
    predictor = DefaultPredictor(cfg)

    save_to = os.path.join(OUTPUT, 'results', model_iter)
    logger.info('Evaluation logs: {}'.format(save_to))
    save_to_dir = os.path.join(save_to, 'JPEGImages')
    prep_dir(save_to_dir)

    with open(test_datadict_path, 'r')as f:
        data = json.load(f)

        id2image = {i['id']: i['file_name'] for i in data['images']}

        logger.info('Evaluation on: {} images, {} objects'.format(len(id2image), len(data['annotations'])))

        saved_imgs = {v: 0 for k, v in id2image.items()}

        acc = 0
        total = len(data['annotations'])

        logger.info('Running the custom evaluation...')

        for record in data['annotations']:
            img_id = record['image_id']

            file_name = id2image[img_id]

            img_path = os.path.join(root_images_path, file_name)

            bbox = record['bbox']
            cat_id = record['category_id']

            im = cv2.imread(img_path)

            outputs = predictor(im)

            outputs = outputs["instances"].to("cpu")

            pred_boxes_classes = [[[int(i) for i in bb.tolist()], int(cls) + 1] for bb, cls in
                                  zip(outputs._fields['pred_boxes'], outputs._fields['pred_classes'])]

            cls = [c for _, c in pred_boxes_classes]

            v = Visualizer(im[:, :, ::-1],
                           metadata=metadata,
                           scale=1)
            v = v.draw_instance_predictions(outputs)

            im = v.get_image()[:, :, ::-1]

            im_o = im.copy()

            # for bb, cc in pred_boxes_classes:
            #     x1, y1, x2, y2 = bb
            #
            #     cv2.rectangle(im_o, (x1, y1), (x2, y2), (0, 225, 0), 2)
            #
            #     t = 'True class: {}'.format(cat[cat_id])
            #
            #     # xt, yt = int(bbox[0]), int(bbox[3])
            #     xt, yt = 25, 25
            #
            #     cv2.putText(im_o, t, (xt, yt), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 225, 225), 2,
            #                 cv2.LINE_AA)
            # print('{}, predicted class: {}'.format(t, cat[cat_id]))

            new_file_name = '{}/{}_{}.png'.format(save_to, id2image[img_id], saved_imgs[file_name])

            # print('save to:', new_file_name)
            cv2.imwrite(new_file_name, im_o)

            bbox_c = bbox.copy()

            bbox_c[2] += bbox_c[0]
            bbox_c[3] += bbox_c[1]

            acc += pred2tg(bbox_c, cat_id, pred_boxes_classes)

            saved_imgs[file_name] += 1

            print('true category: {}, found category: {}, file: {}'.format(cat_id, cls, id2image[img_id]))

            # cv2.imshow('', im_o)
            # cv2.waitKey()
        logger.info('total classification accuracy: {}'.format(round(acc / total, 3)))


if TRAIN:
    train()
if EVAL:
    # vis2()
    # vis()
    # test()
    # eval()
    cust_eval()
