import os
from utils import get_args
from gen_json import prepare_data
from utils import labeltype2nclass
from detection import Detection
from detectron2.utils.logger import setup_logger
# initialize the logger before the detectron2
args = get_args()
output = '{}_dev'.format(args.output) if args.debug else args.output
is_train = True if args.mode.lower() == 'train' else False
model_iter = 'init'
if not is_train:
    model_iter = os.path.split(args.model_path)[-1].split('_')[-1].split('.')[0]
output_logs = '{}/{}_{}_log.txt'.format(output, 'train' if is_train else 'eval', model_iter)


logger = setup_logger(output=output_logs)
from detectron2.data.datasets import register_coco_instances

if __name__ == '__main__':

    args = get_args()

    numb_classes = labeltype2nclass(args.label_type, args.unfocused)


    _MODEL_PATH = args.model_path
    _RESUME = args.resume

    root_images_path = args.root_images
    data_dict = args.data_dict

    train_datadict_path, test_datadict_path = prepare_data(data_dict=data_dict, label_type=args.label_type,
                                                           train_test_dd=args.train_test_dd,
                                                           unfocus=args.unfocused, proc=args.test_split,
                                                           logger=logger)

    assert train_datadict_path is not None and test_datadict_path is not None

    register_coco_instances("burak_train", {}, train_datadict_path, root_images_path)
    register_coco_instances("burak_test", {}, test_datadict_path, root_images_path)

    det = Detection(args, is_train, numb_classes, logger, output, model_iter)

    if is_train:
        det.train()
    if args.eval_after_train or not is_train:
        # Default COCO evaluation, calculates AP metric
        # det.eval()
        # Custom COCO evaluation, calculates accuracy metric
        det.cust_eval(root_images_path,test_datadict_path)
