import logging
from tools.eval_wrapper import eval_lane
import os
from collections import OrderedDict
import torch
from torch.nn.parallel import DistributedDataParallel
from tools.train_net import Trainer
# import detectron2.utils.comm as comm
# from detectron2.data import MetadataCatalog, build_detection_train_loader
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, hooks, launch
# from detectron2.utils.events import EventStorage
# from detectron2.evaluation import (
#     COCOEvaluator,
#     COCOPanopticEvaluator,
#     DatasetEvaluators,
#     LVISEvaluator,
#     PascalVOCDetectionEvaluator,
#     SemSegEvaluator,
#     verify_results,
# )
# from detectron2.modeling import GeneralizedRCNNWithTTA
# from detectron2.utils.logger import setup_logger

from adet.data.dataset_mapper import DatasetMapperWithBasis
from adet.config import get_cfg
from adet.checkpoint import AdetCheckpointer
from adet.evaluation import TextEvaluator
from tools.parsing import parsingNet
# from detectron2.data.datasets import register_coco_instances
def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)

    # rank = comm.get_rank()
    # setup_logger(cfg.OUTPUT_DIR, distributed_rank=rank, name="adet")

    return cfg
def main(args):
    cfg = setup(args)
    model = Trainer.build_model(cfg)
    net = parsingNet(pretrained = False, backbone=model,cls_dim = (200+1,18, 4),
                    use_aux=False).cuda()
    AdetCheckpointer(net, save_dir=cfg.OUTPUT_DIR).resume_or_load(
        cfg.MODEL.WEIGHTS, resume=args.resume
    )
    
    eval_lane(net, 'culane', '/home/ghr/CULANEROOT', '/home/ghr/CULANEROOT/own_test_result', 200, False, False)
if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )