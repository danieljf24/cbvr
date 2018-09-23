import os
import sys
import time
import json
import shutil

import torch

import data
from model import ReLearning
from evaluation import AverageMeter, LogCollector, encode_data, do_predict

import argparse
import logging
import tensorboard_logger as tb_logger

from simpleknn.bigfile import BigFile
from utils.generic_utils import Progbar
from utils.util import read_video_set, write_csv, read_dict
from utils.common import ROOT_PATH, checkToSkip, makedirsforfile
from utils.cbvrp_eval import read_csv_to_dict, hit_k_own, recall_k_own



def main():
    # Hyper Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("--rootpath", default=ROOT_PATH, type=str, help="rootpath (default: %s)" % ROOT_PATH)
    parser.add_argument("--overwrite", default=0, type=int,  help="overwrite existing file (default: 0)")
    parser.add_argument('--collection', default='track_1_shows', type=str, help='collection')

    parser.add_argument('--feature', default='inception-pool3', type=str, help="video feature.")
    parser.add_argument('--embed_size', default=1024, type=int, help='Dimensionality of the video embedding.')

    parser.add_argument('--loss', default='mrl', type=str, help='loss function.')
    parser.add_argument("--cost_style", default='sum', type=str,  help="cost_style (sum|mean)")
    parser.add_argument('--max_violation', action='store_true', help='Use max instead of sum in the rank loss.')
    parser.add_argument('--margin', default=0.2, type=float, help='Rank loss margin.')
    parser.add_argument('--grad_clip', default=2., type=float, help='Gradient clipping threshold.')
    parser.add_argument('--optimizer', default='adam', type=str, help='optimizer. (adam|rmsprop)')
    parser.add_argument('--learning_rate', default=.001, type=float, help='Initial learning rate.')
    parser.add_argument('--lr_decay', default=0.99, type=float, help='learning rate decay after each epoch')

    parser.add_argument('--num_epochs', default=50, type=int, help='Number of training epochs.')
    parser.add_argument('--batch_size', default=32, type=int, help='Size of a training mini-batch.')
    parser.add_argument('--workers', default=2, type=int, help='Number of data loader workers.')
    parser.add_argument('--log_step', default=100, type=int, help='Number of steps to print and record the log.')

    parser.add_argument('--measure', default='cosine', help='Similarity measure used (cosine|order)')
    parser.add_argument('--no_imgnorm', action='store_true', help='Do not normalize the image embeddings.')
    parser.add_argument('--postfix', default='run_0', type=str, help='')

    # augmentation for frame-level features
    parser.add_argument('--stride', default='1', type=str, help='stride=1 means no frame-level data augmentation (default: 1)')
    # augmentation for video-level features
    parser.add_argument('--aug_prob', default=0.0, type=float, 
        help='aug_prob=0 means no frame-level data augmentation, aug_prob=0.5 means half of video use augmented features(default: 0.0)')
    parser.add_argument('--perturb_intensity', default=1.0, type=float, help='perturbation intensity, epsilon  in Eq.2 (default: 1.0)')
    parser.add_argument('--perturb_prob', default=0.5, type=float, help='perturbation probability, p in Eq.2 (default: 0.5)')


    opt = parser.parse_args()
    print json.dumps(vars(opt), indent = 2)

    visual_info = 'feature_%s_embed_size_%d_no_imgnorm_%s' % (opt.feature, opt.embed_size, opt.no_imgnorm)
    loss_info = '%s_%s_margin_%.1f_max_violation_%s_%s' % (opt.loss, opt.measure, opt.margin, opt.max_violation, opt.cost_style)
    optimizer_info = '%s_lr_%.5f_%.2f_bs_%d' % ( opt.optimizer, opt.learning_rate, opt.lr_decay, opt.batch_size)
    data_argumentation_info = 'frame_stride_%s_video_prob_%.1f_perturb_intensity_%.5f_perturb_prob_%.2f' % (opt.stride, opt.aug_prob, opt.perturb_intensity, opt.perturb_prob)


    opt.logger_name = os.path.join(opt.rootpath, opt.collection, 'cv', 'ReLearning', visual_info, loss_info, optimizer_info, data_argumentation_info, opt.postfix)
    if checkToSkip(os.path.join(opt.logger_name,'model_best.pth.tar'), opt.overwrite):
        sys.exit(0)
    if checkToSkip(os.path.join(opt.logger_name,'val_perf.txt'), opt.overwrite):
        sys.exit(0)
    makedirsforfile(os.path.join(opt.logger_name,'model_best.pth.tar'))

    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
    tb_logger.configure(opt.logger_name, flush_secs=5)


    # reading data
    train_video_set_file = os.path.join(opt.rootpath, opt.collection, 'split', 'train.csv' )
    val_video_set_file = os.path.join(opt.rootpath, opt.collection, 'split', 'val.csv' )
    train_video_list = read_video_set(train_video_set_file)
    val_video_list = read_video_set(val_video_set_file)

    train_rootpath = os.path.join(opt.rootpath, opt.collection, 'relevance_train.csv')
    val_rootpath = os.path.join(opt.rootpath, opt.collection, 'relevance_val.csv')
    val_video2gtrank = read_csv_to_dict(val_rootpath)

    stride_list = map(int, opt.stride.strip().split('-'))
    opt.sum_subs = sum(stride_list)
    if opt.aug_prob <= 0:
        opt.feature = "avg-" + opt.feature + "-stride%s" %  opt.stride

    video_feat_path = os.path.join(opt.rootpath, opt.collection, 'FeatureData', opt.feature)
    video_feats = BigFile(video_feat_path)
    opt.feature_dim = video_feats.ndims


    # Load data loaders
    if opt.sum_subs > 1:
        video2subvideo_path = os.path.join(video_feat_path, 'video2subvideo.txt')
        video2subvideo = read_dict(video2subvideo_path)
        train_loader = data.get_video_da_loader(train_rootpath, video_feats, opt, opt.batch_size, True, opt.workers, 
                video2subvideo, opt.sum_subs, feat_path=video_feat_path)
    else:
        train_loader = data.get_video_da_loader(train_rootpath, video_feats, opt, opt.batch_size, True, opt.workers, feat_path=video_feat_path)
    val_feat_loader = data.get_feat_loader(val_video_list, video_feats, opt.batch_size, False, 1)
    cand_feat_loader = data.get_feat_loader(train_video_list + val_video_list, video_feats, opt.batch_size, False, 1)
    
    # Construct the model
    model = ReLearning(opt)

    # Train the Model
    best_rsum = 0
    best_hit_k_scores = 0
    best_recall_K_scoress = 0
    no_impr_counter = 0
    lr_counter = 0
    fout_val_perf_hist = open(os.path.join(opt.logger_name,'val_perf_hist.txt'), 'w')

    for epoch in range(opt.num_epochs):

        # train for one epoch
        print "\nEpoch: ", epoch + 1
        print "learning rate: ", get_learning_rate(model.optimizer)
        train(opt, train_loader, model, epoch)

        # evaluate on validation set
        rsum, hit_k_scores, recall_K_scores = validate(val_feat_loader, cand_feat_loader, model, val_video2gtrank, log_step=opt.log_step, opt=opt)

        # remember best R@ sum and save checkpoint
        is_best = rsum > best_rsum
        best_rsum = max(rsum, best_rsum)
        if is_best:
            best_hit_k_scores = hit_k_scores
            best_recall_K_scoress = recall_K_scores
        print 'current perf: ', rsum
        print 'best perf: ', best_rsum
        print 'current hit_top_k: ', [round(x,3) for x in hit_k_scores]
        print 'current recall_top_k: ', [round(x,3) for x in recall_K_scores]
        fout_val_perf_hist.write("epoch_%d %f\n" % (epoch, rsum))
        fout_val_perf_hist.flush()

        save_checkpoint({
            'epoch': epoch + 1,
            'model': model.state_dict(),
            'best_rsum': best_rsum,
            'opt': opt,
            'Eiters': model.Eiters,
        }, is_best, filename='checkpoint_epoch_%s.pth.tar' % epoch, prefix=opt.logger_name + '/')

        lr_counter += 1
        decay_learning_rate(opt, model.optimizer, opt.lr_decay)
        if not is_best:
            # Early stop occurs if the validation performance 
            # does not improve in ten consecutive epochs. 
            no_impr_counter += 1
            if no_impr_counter > 10:
                print ("Early stopping happened")
                break

            # when the validation performance has decreased after an epoch,
            # we divide the learning rate by 2 and continue training;
            # but we use each learning rate for at least 3 epochs
            if lr_counter > 2:
                decay_learning_rate(opt, model.optimizer, 0.5)
                lr_counter = 0
        else:
            # lr_counter = 0
            no_impr_counter = 0
    
    fout_val_perf_hist.close()
    # output val performance
    print json.dumps(vars(opt), indent = 2)
    print '\nbest performance on validation:'
    print 'hit_top_k', [round(x,3) for x in best_hit_k_scores]
    print 'recall_top_k', [round(x,3) for x in best_recall_K_scoress]
    with open(os.path.join(opt.logger_name,'val_perf.txt'), 'w') as fout:
        fout.write('best performance on validation:')
        fout.write('\nhit_top_k: ' + ", ".join(map(str, [round(x,3) for x in best_hit_k_scores])))
        fout.write('\necall_top_k: ' + ", ".join(map(str, [round(x,3) for x in best_recall_K_scoress])))



    # generate and run the shell script for test
    templete = ''.join(open( 'TEMPLATE_eval.sh'  ).readlines())
    striptStr = templete.replace('@@@rootpath@@@', opt.rootpath)
    striptStr = striptStr.replace('@@@collection@@@', opt.collection)
    striptStr = striptStr.replace('@@@overwrite@@@', str(opt.overwrite))
    striptStr = striptStr.replace('@@@model_path@@@', opt.logger_name)

    runfile = 'do_eval_%s.sh' % opt.collection
    open( runfile, 'w' ).write(striptStr+'\n')
    os.system('chmod +x %s' % runfile)
    os.system('./%s' % runfile) 


def train(opt, train_loader, model, epoch):
    # average meters to record the training statistics
    batch_time = AverageMeter()
    data_time = AverageMeter()
    train_logger = LogCollector()

    # switch to train mode
    model.train_start()

    progbar = Progbar(train_loader.dataset.length)
    end = time.time()
    for i, train_data in enumerate(train_loader):

        # measure data loading time
        data_time.update(time.time() - end)

        # make sure train logger is used
        model.logger = train_logger

        # Update the model
        b_size, loss = model.train_emb(*train_data)
        # print loss
        progbar.add(b_size, values=[("loss", loss)])

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # Record logs in tensorboard
        tb_logger.log_value('epoch', epoch, step=model.Eiters)
        tb_logger.log_value('step', i, step=model.Eiters)
        tb_logger.log_value('batch_time', batch_time.val, step=model.Eiters)
        tb_logger.log_value('data_time', data_time.val, step=model.Eiters)
        model.logger.tb_log(tb_logger, step=model.Eiters)



def validate(val_feat_loader, cand_feat_loader, model, video2gtrank, log_step=100, opt=None):
    # compute the encoding for all the validation images and captions
    val_video_embs, val_video_ids_list = encode_data(model, val_feat_loader, log_step, logging.info)
    cand_video_embs, cand_video_ids_list = encode_data(model, cand_feat_loader, log_step, logging.info)

    video2predrank = do_predict(val_video_embs, val_video_ids_list, cand_video_embs, cand_video_ids_list, output_dir=None, overwrite=0, no_imgnorm=opt.no_imgnorm)
    hit_top_k = [5, 10, 20, 30]
    recall_top_k = [50, 100, 200, 300]
    hit_k_scores = hit_k_own(video2gtrank, video2predrank, top_k=hit_top_k)
    recall_K_scores = recall_k_own(video2gtrank, video2predrank, top_k=recall_top_k)

    for i, k in enumerate(hit_top_k):
        tb_logger.log_value('hit_%d' % k, hit_k_scores[i], step=model.Eiters)
    for i, k in enumerate(recall_top_k):
        tb_logger.log_value('recall_%d' % k, recall_K_scores[i], step=model.Eiters)
    currscore = recall_K_scores[1]

    return currscore, hit_k_scores, recall_K_scores


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', prefix=''):
    torch.save(state, prefix + filename)
    if is_best:
        shutil.copyfile(prefix + filename, prefix + 'model_best.pth.tar')


def decay_learning_rate(opt, optimizer, decay):
    """decay learning rate to the last LR"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr']*decay


def get_learning_rate(optimizer):
    """decay learning rate to the last LR"""
    lr_list = []
    for param_group in optimizer.param_groups:
        lr_list.append(param_group['lr'])
    return lr_list


if __name__ == '__main__':
    main()
