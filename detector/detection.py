import json
import os
import cv2

import torch
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import build_detection_test_loader
from detectron2.engine import DefaultTrainer
from detectron2.data import MetadataCatalog

from utils import get_categories, pred2tg, mapper


class Detection(object):

    def __init__(self, args,to_train, numb_classes, logger, output,model_iter ):
        self.args = args
        self.cat = get_categories(args.label_type, args.unfocused)
        self.metadata = MetadataCatalog.get("burak_test").set(thing_classes=self.cat)
        self.numb_classes = numb_classes
        self.logger = logger
        self.output = output
        self.to_train = to_train
        self.debug = self.args.debug
        self.model_iter = model_iter
        self.cfg = self.get_model_config()

    def get_model_config(self):

        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.logger.info("DEBUG: {}".format(self.debug))
        self.logger.info('label_type: {}'.format(self.args.label_type))
        for c in self.cat:
            self.logger.info('\t- {}'.format(c))
        self.logger.info('model_type: {}'.format(self.args.model_yml))
        self.logger.info("Numb classes: {}".format(self.numb_classes))
        self.logger.info("Unfocused: {}".format(self.args.unfocused))
        self.logger.info("Output dir: {}".format(self.output))

        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file(self.args.model_yml))

        if self.args.model_path:
            cfg.MODEL.WEIGHTS = self.args.model_path
        else:
            model_zoo.get_checkpoint_url(self.args.model_yml)

        cfg.DATASETS.TRAIN = ("burak_train",)
        cfg.DATASETS.TEST = ("burak_test",)

        cfg.MODEL.ROI_HEADS.NUM_CLASSES = self.numb_classes
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256

        if device == 'cpu':
            self.logger.info('DEVICE: {}'.format(device))
            cfg.MODEL.DEVICE = 'cpu'

        if self.to_train:

            cfg.OUTPUT_DIR = self.output
            cfg.MODEL.ROI_HEADS.NUM_CLASSES = self.numb_classes
            if self.debug:
                cfg.DATALOADER.NUM_WORKERS = 1
                cfg.SOLVER.IMS_PER_BATCH = 1
                cfg.SOLVER.BASE_LR = self.args.lr
                cfg.SOLVER.WARMUP_ITERS = 500
                cfg.SOLVER.MAX_ITER = 1000
                cfg.SOLVER.STEPS = (300, 600)
                cfg.SOLVER.GAMMA = 0.1
                cfg.SOLVER.CHECKPOINT_PERIOD = 250
                cfg.TEST.EVAL_PERIOD = 50000

            else:
                cfg.DATALOADER.NUM_WORKERS = 4
                cfg.SOLVER.IMS_PER_BATCH = self.args.batch
                cfg.SOLVER.BASE_LR = self.args.lr
                cfg.SOLVER.WARMUP_ITERS = 5000
                cfg.SOLVER.MAX_ITER = 50000
                cfg.SOLVER.STEPS = (15000, 22500)
                cfg.SOLVER.GAMMA = 0.1
                cfg.SOLVER.CHECKPOINT_PERIOD = 2500

                cfg.TEST.EVAL_PERIOD = 50000

            os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
        else:
            self.logger.info("Model path: {}".format(self.args.model_path))

            cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.args.thrs

        return cfg

    def train(self):
        mapper = None
        trainer = DefaultTrainer(self.cfg, mapper=mapper)
        trainer.resume_or_load(resume=self.args.resume)
        trainer.train()

    def eval(self):
        predictor = DefaultPredictor(self.cfg)
        evaluator = COCOEvaluator("burak_test", self.cfg, False, output_dir=self.args.output)
        val_loader = build_detection_test_loader(self.cfg, "burak_test")
        inference_on_dataset(predictor.model, val_loader, evaluator)

    def test(self):
        """test single image """
        predictor = DefaultPredictor(self.cfg)
        img_path = 'PATH TO IMAGE'
        assert os.path.exists(img_path)
        im = cv2.imread(img_path)
        outputs = predictor(im)
        v = Visualizer(im[:, :, ::-1],
                       metadata=self.metadata,
                       scale=1)
        v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        im = v.get_image()[:, :, ::-1]
        print(outputs)
        cv2.imshow('', im)
        cv2.waitKey()

    @staticmethod
    def prep_dir(p):
        if not os.path.exists(p):
            os.makedirs(p)

    def cust_eval(self):
        predictor = DefaultPredictor(self.cfg)

        save_to = os.path.join(self.output, 'results', self.model_iter)
        self.logger.info('Evaluation logs: {}'.format(save_to))
        save_to_dir = os.path.join(save_to, 'JPEGImages')
        self.prep_dir(save_to_dir)

        with open(self.test_datadict_path, 'r')as f:
            data = json.load(f)

            id2image = {i['id']: i['file_name'] for i in data['images']}

            self.logger.info('Evaluation on: {} images, {} objects'.format(len(id2image), len(data['annotations'])))

            saved_imgs = {v: 0 for k, v in id2image.items()}

            acc = 0
            total = len(data['annotations'])

            self.logger.info('Running the custom evaluation...')

            for record in data['annotations']:
                img_id = record['image_id']

                file_name = id2image[img_id]

                img_path = os.path.join(self.root_images_path, file_name)

                bbox = record['bbox']
                cat_id = record['category_id']

                im = cv2.imread(img_path)

                outputs = predictor(im)

                outputs = outputs["instances"].to("cpu")

                pred_boxes_classes = [[[int(i) for i in bb.tolist()], int(cls) + 1] for bb, cls in
                                      zip(outputs._fields['pred_boxes'], outputs._fields['pred_classes'])]

                cls = [c for _, c in pred_boxes_classes]

                v = Visualizer(im[:, :, ::-1],
                               metadata=self.metadata,
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
            self.logger.info('total classification accuracy: {}'.format(round(acc / total, 3)))
