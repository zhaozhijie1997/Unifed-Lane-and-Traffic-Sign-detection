import os, json, torch, scipy
from scipy.special import softmax
import numpy as np
from adet.data.datasets.constant import culane_row_anchor
def generate_lines(out,path=None, shape=[470,800], name='test1', griding_num=200, localization_type='rel', flip_updown=False):
    
    col_sample = np.linspace(0, shape[1] - 1, griding_num)
    col_sample_w = col_sample[1] - col_sample[0]
    
    for j in range(len(out)):
        out_j = out[j].data.cpu().numpy()

        # out_j = out_j[:, ::-1, :]
        # prob = scipy.special.softmax(out_j[:-1, :, :], axis=0)
        # idx = np.arange(griding_num) + 1
        # idx = idx.reshape(-1, 1, 1)
        # loc = np.sum(prob * idx, axis=0)
        # out_j = np.argmax(out_j, axis=0)
        # loc[out_j == griding_num] = 0
        # out_j = loc
        if flip_updown:
            out_j = out_j[:, ::-1, :]
        if localization_type == 'abs':
            out_j = np.argmax(out_j, axis=0)
            out_j[out_j == griding_num] = -1
            out_j = out_j + 1
        elif localization_type == 'rel':
            # prob = scipy.special.softmax(out_j[:-1, :, :], axis=0)
            prob = softmax(out_j[:-1, :, :], axis=0)
            idx = np.arange(griding_num) + 1
            idx = idx.reshape(-1, 1, 1)
            loc = np.sum(prob * idx, axis=0)
            out_j = np.argmax(out_j, axis=0)
            loc[out_j == griding_num] = 0
            out_j = loc
        else:
            raise NotImplementedError
        # name = names[j]
        output_path = 'test'
        # path = path.split('/')[4:]
        # path = path.split('/')[1]
        # paths = '/'.join(path)

        # paths = output_path+'/'+path
        # import pdb;pdb.set_trace()

        line_save_path = os.path.join(output_path, 'test0408.lines.txt')
        # import pdb;pdb.set_trace()
        save_dir, _ = os.path.split(line_save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        with open(line_save_path, 'w') as fp:
            # if int(np.sum(out_j))==0:
            #     fp.write('\n')
            # else:
            for i in range(out_j.shape[1]):
                if np.sum(out_j[:, i] != 0) > 2:
                    for k in range(out_j.shape[0]):
                        if out_j[k, i] > 0:
                            # import pdb;pdb.set_trace()
                            # ppp = (int(out_j[k, i] * col_sample_w * 800 / 800) - 1, int(288 * (culane_row_anchor[len(culane_row_anchor)-1-k]/470)) - 1 )
                            
                            # fp.write('%d %d ' % (int(out_j[k, i] * col_sample_w ) - 1, int(culane_row_anchor[len(culane_row_anchor)-1-k]) - 1))
                            fp.write('%d %d ' % (int(out_j[k, i] * col_sample_w * 800 / 800) - 1, int(470 - k * 10) - 1))
                            # fp.write('%d %d ' % ppp)
                    fp.write('\n')
                   