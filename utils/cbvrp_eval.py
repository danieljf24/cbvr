# __author__ = 'Hulu_Research'

import csv
import numpy as np

def read_csv(filepath):
    """
    read csv file.
    :param file_path: the path of the csv file
    :return: list of list
    """
    reader = csv.reader(open(filepath, 'r'))
    data = []
    for x in reader:
        data.append(x)
    return data

def read_csv_to_dict(filepath):
    """
    read csv file.
    :param file_path: the path of the csv file
    :return: list of list
    """
    reader = csv.reader(open(filepath, 'r'))
    video2gtrank = {}
    for x in reader:
        video2gtrank[x[0]] = x[1:]
    return video2gtrank


def eval_recall(ground, predict, top_k):
    """
    Compute the recall metric using in CBVRP-ACMMM-2018 Challenge.
    :param ground: A list of indices represent real relevant show (ids) for current show (id).
    :param predict: A list of indices represent predicted relevant show (ids) for current show (id).
    :param top_k: max top_k = 500
    :return: recall, a float.

    """
    predict = predict[:top_k]
    intersect = [x for x in predict if x in ground]
    recall = float(len(intersect)) / len(ground)
    return recall

def mean_recall_hit_k(gdir, pdir, top_k):
    """
    Compute the mean recall@k & mean hit@k metric over a whole val/test set.
    :param gdir: the dir (path) of ground truth file.
    :param pdir: the dir (path) of prediction file.
    :param top_k: max top_k = 500
    :return: mean_recall_k, mean_hit_k, both are floats.
    """
    recall_k = 0.0
    hit_k = 0
    predict_set = read_csv(pdir)
    ground_set = read_csv(gdir)
    
    for i in range(len(predict_set)):
        predict = [int(x) for x in predict_set[i]]
        ground = [int(x) for x in ground_set[i]]
        recall = eval_recall(ground[1:], predict[1:], top_k)
        recall_k = recall_k + recall
        if recall > 0:
            hit_k = hit_k + 1
    mean_recall_k = float(recall_k) / len(predict_set)
    mean_hit_k = float(hit_k) / len(predict_set)
    return mean_recall_k, mean_hit_k



def recall_k(gdir, pdir, top_k=[50, 100, 200, 300]):
    mean_recall_k_list = []
    for recall_k in top_k:
        mean_recall_k, _ = mean_recall_hit_k(gdir, pdir, recall_k)
        mean_recall_k_list.append(round(mean_recall_k,3))
    return mean_recall_k_list



def hit_k(gdir, pdir, top_k=[5, 10, 20, 30]):
    mean_hit_k_list = []
    for hit_k in top_k:
        _, mean_hit_k = mean_recall_hit_k(gdir, pdir, hit_k)
        mean_hit_k_list.append(round(mean_hit_k,3))
    return mean_hit_k_list


def mean_recall_hit_k_own(gdict, pdict, top_k):
    """
    Compute the mean recall@k & mean hit@k metric over a whole val/test set.
    :param gdir: the dir (path) of ground truth file.
    :param pdir: the dir (path) of prediction file.
    :param top_k: max top_k = 500
    :return: mean_recall_k, mean_hit_k, both are floats.
    """
    recall_k = 0.0
    hit_k = 0
    assert len(gdict) == len(pdict), '%d != %d' % (len(gdict), len(pdict))
    
    for i, video in enumerate(gdict.keys()):
        predict = [int(x) for x in pdict[video]]
        ground = [int(x) for x in gdict[video]]
        recall = eval_recall(ground, predict, top_k)
        recall_k = recall_k + recall
        if recall > 0:
            hit_k = hit_k + 1
    mean_recall_k = float(recall_k) / len(gdict.keys())
    mean_hit_k = float(hit_k) / len(gdict.keys())
    return mean_recall_k, mean_hit_k

def recall_k_own(gdict, pdict, top_k=[50, 100, 200, 300]):
    mean_recall_k_list = []
    for recall_k in top_k:
        mean_recall_k, _ = mean_recall_hit_k_own(gdict, pdict, recall_k)
        mean_recall_k_list.append(mean_recall_k)
    return mean_recall_k_list


def hit_k_own(gdict, pdict, top_k=[5, 10, 20, 30]):
    mean_hit_k_list = []
    for hit_k in top_k:
        _, mean_hit_k = mean_recall_hit_k_own(gdict, pdict, hit_k)
        mean_hit_k_list.append(mean_hit_k)
    return mean_hit_k_list


# Evaluation script example.
if __name__ == "__main__":
    track = 'track_1_shows'
    fname = 'c3d-pool5'

    gdir = './%s/relevance_val.csv'%(track)
    pdir = './%s/predict_val_%s.csv'%(track, fname)

    print('hit_k rate for %s'%(fname))
    mean_hit_k_list = []
    for hit_k in [5, 10, 20, 30, 40, 50]:
        _, mean_hit_k = mean_recall_hit_k(gdir, pdir, hit_k)
        mean_hit_k_list.append(round(mean_hit_k,3))
        #print('%.3f' % mean_hit_k)
    print(mean_hit_k_list)


    print('recall_k rate for %s'%(fname))
    mean_recall_k_list = []
    for recall_k in [50, 100, 200, 300, 400, 500]:
        mean_recall_k, _ = mean_recall_hit_k(gdir, pdir, recall_k)
        mean_recall_k_list.append(round(mean_recall_k,3))
        #print('%.3f' % mean_recall_k)
    print(mean_recall_k_list)
    
