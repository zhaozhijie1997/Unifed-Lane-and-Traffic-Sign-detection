import torch, os, datetime
import numpy as np


from .dist_utils import dist_print, dist_tqdm, is_main_process, DistSummaryWriter
from .factory import get_metric_dict, get_loss_dict, get_optimizer, get_scheduler
from .metrics import MultiLabelAcc, AccTopk, Metric_mIoU, update_metrics, reset_metrics

from .common import merge_config, save_model, cp_projects
from .common import get_work_dir, get_logger

import time


def inference(data_label, seg_label,use_aux,cls_out,seg_out):
    if use_aux:
        
        # img, cls_label, seg_label = data_label
        cls_label, seg_label = data_label, seg_label[:,-38:-1,:]
        seg_out = seg_out[:,:,-38:-1,:]
        # import pdb;pdb.set_trace()
        return {'cls_out': cls_out, 'cls_label': cls_label, 'seg_out':seg_out, 'seg_label': seg_label}
    else:
        # img, cls_label = data_label
        cls_label = cls_label.cuda()
        return {'cls_out': cls_out, 'cls_label': cls_label}


def resolve_val_data(results, use_aux):
    results['cls_out'] = torch.argmax(results['cls_out'], dim=1)
    if use_aux:
        results['seg_out'] = torch.argmax(results['seg_out'], dim=1)
    return results


def calc_loss(loss_dict, results):
    loss = 0
    
    for i in range(len(loss_dict['name'])):

        data_src = loss_dict['data_src'][i]

        datas = [results[src] for src in data_src]

        # import pdb; pdb.set_trace()

        loss_cur = loss_dict['op'][i](*datas)

        #if global_step % 20 == 0:
        #    logger.add_scalar('loss/'+loss_dict['name'][i], loss_cur, global_step)

        loss += loss_cur * loss_dict['weight'][i]
    # if np.isnan(loss):
    #     import pdb;pdb.set_trace()
    return loss
