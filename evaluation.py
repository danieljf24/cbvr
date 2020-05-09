from __future__ import print_function
import os
import sys
import csv
import pickle

import numpy
import time
import numpy as np
import torch
from collections import OrderedDict

from utils.common import makedirsforfile, checkToSkip
from simpleknn.bigfile import BigFile

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=0):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / (.0001 + self.count)

    def __str__(self):
        """String representation for logging
        """
        # for values that should be recorded exactly e.g. iteration number
        if self.count == 0:
            return str(self.val)
        # for stats
        return '%.4f (%.4f)' % (self.val, self.avg)


class LogCollector(object):
    """A collection of logging objects that can change from train to val"""

    def __init__(self):
        # to keep the order of logged variables deterministic
        self.meters = OrderedDict()

    def update(self, k, v, n=0):
        # create a new meter if previously not recorded
        if k not in self.meters:
            self.meters[k] = AverageMeter()
        self.meters[k].update(v, n)

    def __str__(self):
        """Concatenate the meters in one log line
        """
        s = ''
        for i, (k, v) in enumerate(self.meters.iteritems()):
            if i > 0:
                s += '  '
            s += k + ' ' + str(v)
        return s

    def tb_log(self, tb_logger, prefix='', step=None):
        """Log using tensorboard
        """
        for k, v in self.meters.iteritems():
            tb_logger.log_value(prefix + k, v.val, step=step)


def encode_data(model, data_loader, log_step=10, logging=print):
    """Encode all images and captions loadable by `data_loader`
    """
    batch_time = AverageMeter()

    # switch to evaluate mode
    model.val_start()

    end = time.time()

    # numpy array to keep all the embeddings
    video_embs = None
    video_ids_list = []
    for i, (videos, ids, idxs) in enumerate(data_loader):

        # compute the embeddings
        video_emb = model.forward_emb(videos, volatile=True)

        # initialize the numpy arrays given the size of the embeddings
        if video_embs is None:
            video_embs = np.zeros((len(data_loader.dataset), video_emb.size(1)))

        # preserve the embeddings by copying from gpu and converting to numpy
        video_embs[list(idxs)] = video_emb.data.cpu().numpy().copy()
        video_ids_list.extend(ids)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % log_step == 0:
            logging('Test: [{0}/{1}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    .format(
                        i, len(data_loader), batch_time=batch_time))
        del videos

    return video_embs, video_ids_list


def cal_rel_index(rele_file):
    with open(rele_file,'r') as csvfile:
        data=[]
        csv_reader=csv.reader(csvfile)
        for line in csv_reader:
            data.append(line)
    return data


def re_cal_scores(scores, rel_index, n, sumofdata):

    scores_list = []
    for i, index in enumerate(rel_index):
        data = []
        data.append(scores[i])
        if i < sumofdata:
            for j in index[:n]:
                data.append(scores[int(j)])
            scores_list.append(sum(data) / len(data))
        else:
            scores_list.append(data)
    
    # test video has no available relations
    scores_list.append(scores[len(rel_index): len(scores)])

    return scores_list



def score2result(scores, test_video_list, cand_video_list, rel_index, n):
    video2predrank = {}
    n_rows, n_column = scores.shape
    assert n_rows == len(test_video_list)
    assert n_column == len(cand_video_list)
    for i, test_video in enumerate(test_video_list):
        score_list = scores[i]
        if rel_index is not None:
            sumofdata = n_column - n_rows
            score_list = re_cal_scores(score_list, rel_index, n, sumofdata)
        cand_video_score_list = zip(cand_video_list, score_list)
        sorted_cand_video_score_list = sorted(cand_video_score_list, key=lambda v:v[1], reverse=True)
        #video2predrank[test_video] = [x[0] for x in sorted_cand_video_score_list]
        predrank = [x[0] for x in sorted_cand_video_score_list]
        predrank.remove(test_video)
        video2predrank[test_video] = predrank
    return video2predrank


def score2result_fusion(scores, test_video_list, cand_video_list):
    video2predrank = {}
    n_rows, n_column = scores.shape
    assert n_rows == len(test_video_list)
    assert n_column == len(cand_video_list)
    for i, test_video in enumerate(test_video_list):
        score_list = scores[i]
        cand_video_score_list = zip(cand_video_list, score_list)
        sorted_cand_video_score_list = sorted(cand_video_score_list, key=lambda v:v[1], reverse=True)
        predrank = [x[0] for x in sorted_cand_video_score_list]
        predrank.remove(test_video)
        video2predrank[test_video] = predrank
    return video2predrank


def do_predict(test_video_emd, test_video_list, cand_video_emd, cand_video_list, rel_index=None, n=5, output_dir=None, overwrite=0, no_imgnorm=False):

    if no_imgnorm:
        scores = cal_score(test_video_emd, cand_video_emd, measure='cosine')
    else:
        scores = cal_score(test_video_emd, cand_video_emd, measure='dot')

    video2predrank = score2result(scores, test_video_list, cand_video_list, rel_index, n)

    if output_dir is not None:
        output_file = os.path.join(output_dir, 'pred_scores_matrix.pth.tar')
        if checkToSkip(output_file, overwrite):
            sys.exit(0)
        makedirsforfile(output_file)
        torch.save({'scores': scores, 'test_videos': test_video_list, 'cand_videos': cand_video_list}, output_file)
        print("write score matrix into: " + output_file)

    return video2predrank


# def cal_error(images, captions, measure='cosine', n_caption=2):
#     """
#     Images->Text (Image Annotation)
#     Images: (5N, K) matrix of images
#     Captions: (5N, K) matrix of captions
#     """
#     idx = range(0, images.shape[0], n_caption)
#     im = images[idx, :]
#     if measure == 'cosine':
#         errors = -1*numpy.dot(captions, im.T)

#     return errors


def cal_score(video_1, video_2, measure='cosine'):
    if measure == 'cosine':
        # l2 normalization
        import sklearn.preprocessing as preprocessing
        video_1 = preprocessing.normalize(video_1, norm='l2')
        video_2 = preprocessing.normalize(video_2, norm='l2')
        scores = numpy.dot(video_1, video_2.T)
    elif measure == 'dot':
        scores = numpy.dot(video_1, video_2.T)

    return scores
