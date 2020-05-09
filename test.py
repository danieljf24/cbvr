import os
import sys
import time
import json

import torch

import data
from model import ReLearning
from evaluation import encode_data, do_predict, cal_rel_index

import argparse
import logging
import tensorboard_logger as tb_logger

from simpleknn.bigfile import BigFile
from utils.generic_utils import Progbar
from utils.common import makedirsforfile, checkToSkip, ROOT_PATH
from utils.util import read_video_set, write_csv, read_dict, write_csv_video2rank, get_count
from utils.cbvrp_eval import read_csv_to_dict, hit_k_own, recall_k_own


def main():
    # Hyper Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("--rootpath", default=ROOT_PATH, type=str, help="rootpath (default: %s)" % ROOT_PATH)
    parser.add_argument('--collection', default='track_1_shows', type=str, help='collection')
    parser.add_argument('--checkpoint_path', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
    parser.add_argument("--test_set", default="val", type=str, help="val or test")
    parser.add_argument('--batch_size', default=128, type=int, help='Size of a training mini-batch.')
    parser.add_argument("--overwrite", default=0, type=int,  help="overwrite existing file (default: 0)")
    parser.add_argument('--strategy', default=1, type=int, help='1: use Strategy 1, 2: use Strategy 2')
    parser.add_argument('--n', default=5, type=int, help='top n relevant videos of a candidate video')

    opt = parser.parse_args()
    print(json.dumps(vars(opt), indent = 2))


    assert opt.test_set in ['val', 'test']
    output_dir = os.path.dirname(opt.checkpoint_path.replace('/cv/', '/results/%s/' % opt.test_set ))
    if opt.strategy == 2:
        output_dir = os.path.join(output_dir, 'strategy_%d_n_%d' % (opt.strategy, opt.n))
    output_file = os.path.join(output_dir,'pred_video2rank.csv')
    if checkToSkip(output_file, opt.overwrite):
        sys.exit(0)
    makedirsforfile(output_file)


    if opt.strategy == 2:
        rele_index_path = os.path.join(opt.rootpath, opt.collection, 'rel_index.csv')
        if not os.path.exists(rele_index_path):
            get_count(os.path.join(opt.rootpath, opt.collection))
        rel_index = cal_rel_index(rele_index_path)
    else:
        rel_index = None

    # reading data
    train_video_set_file = os.path.join(opt.rootpath, opt.collection, 'split', 'train.csv')
    val_video_set_file = os.path.join(opt.rootpath, opt.collection, 'split', 'val.csv')
    train_video_list = read_video_set(train_video_set_file)
    val_video_list = read_video_set(val_video_set_file)
    if opt.test_set ==  'test':
        test_video_set_file = os.path.join(opt.rootpath, opt.collection, 'split', 'test.csv' )
        test_video_list = read_video_set(test_video_set_file)


    # optionally resume from a checkpoint
    print("=> loading checkpoint '{}'".format(opt.checkpoint_path))
    checkpoint = torch.load(opt.checkpoint_path)
    options = checkpoint['opt']

    # set feature reader
    video_feat_path = os.path.join(opt.rootpath, opt.collection, 'FeatureData', options.feature)
    video_feats = BigFile(video_feat_path)

 
    # Construct the model
    if opt.test_set == 'val':
        val_rootpath = os.path.join(opt.rootpath, opt.collection, 'relevance_val.csv')
        val_video2gtrank = read_csv_to_dict(val_rootpath)
        val_feat_loader = data.get_feat_loader(val_video_list, video_feats, opt.batch_size, False, 1)
        cand_feat_loader = data.get_feat_loader(train_video_list + val_video_list, video_feats, opt.batch_size, False, 1)
    elif opt.test_set == 'test':
        val_feat_loader = data.get_feat_loader(test_video_list, video_feats, opt.batch_size, False, 1)
        cand_feat_loader = data.get_feat_loader(train_video_list + val_video_list + test_video_list, video_feats, opt.batch_size, False, 1)
    
    model = ReLearning(options)
    model.load_state_dict(checkpoint['model'])
    val_video_embs, val_video_ids_list = encode_data(model, val_feat_loader, options.log_step, logging.info)
    cand_video_embs, cand_video_ids_list = encode_data(model, cand_feat_loader, options.log_step, logging.info)

    
    video2predrank = do_predict(val_video_embs, val_video_ids_list, cand_video_embs, cand_video_ids_list, rel_index, opt.n, output_dir=output_dir, overwrite=1, no_imgnorm=options.no_imgnorm)
    write_csv_video2rank(output_file, video2predrank)

    if opt.test_set ==  'val':
        hit_top_k = [5, 10, 20, 30]
        recall_top_k = [50, 100, 200, 300]
        hit_k_scores = hit_k_own(val_video2gtrank, video2predrank, top_k=hit_top_k)
        recall_K_scores = recall_k_own(val_video2gtrank, video2predrank, top_k=recall_top_k)

        # output val performance
     
        print('# Using Strategy %d for relevance prediction:' %  (opt.strategy))
        print('best performance on validation:')
        print('hit_top_k', [round(x,3) for x in hit_k_scores])
        print('recall_top_k', [round(x,3) for x in recall_K_scores])
        with open(os.path.join(output_dir,'perf.txt'), 'w') as fout:
            fout.write('best performance on validation:')
            fout.write('\nhit_top_k: ' + ", ".join(map(str, [round(x,3) for x in hit_k_scores])))
            fout.write('\necall_top_k: ' + ", ".join(map(str, [round(x,3) for x in recall_K_scores])))


if __name__ == '__main__':
    main()
