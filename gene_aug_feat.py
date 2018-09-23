import os
import sys
import numpy as np

from utils.generic_utils import Progbar
from utils.common import ROOT_PATH, checkToSkip, makedirsforfile
from simpleknn.bigfile import BigFile
from simpleknn.txt2bin import process as text2bin
from data_augmenter import Frame_Level_Augmenter


def process(opt):

    rootpath = opt.rootpath
    collection = opt.collection
    feature = opt.feature
    stride = opt.stride
    overwrite = opt.overwrite
    pooling_style = opt.pooling_style


    feat_path = os.path.join(rootpath, collection, "FeatureData", feature)

    output_dir = os.path.join(rootpath, collection, "FeatureData", '%s-' % pooling_style + feature + "-stride%s" %  stride)
    feat_combined_file = os.path.join(output_dir, "id_feat.txt")
    if checkToSkip(os.path.join(output_dir, "feature.bin"), overwrite):
        sys.exit(0)
    makedirsforfile(feat_combined_file)

    print "Generate augmented frame-level features and operate mean pooling..."

    feat_data = BigFile(feat_path)
    video2fmnos = {}
    for frame_id in feat_data.names:
        data = frame_id.strip().split("_")
        video_id = '_'.join(data[:-1])
        fm_no = data[-1]
        video2fmnos.setdefault(video_id, []).append(int(fm_no))

    video2frames = {}
    for video_id, fmnos in video2fmnos.iteritems():
        for fm_no in sorted(fmnos):
            video2frames.setdefault(video_id, []).append(video_id + "_" + str(fm_no))
    

    stride = map(int, stride.strip().split('-'))
    f_auger = Frame_Level_Augmenter(stride)

    video2subvideo = {}
    fout = open(feat_combined_file, 'w')
    progbar = Progbar(len(video2frames))
    for video in video2frames:
        frame_ids = video2frames[video]

        # output the while video level feature
        video2subvideo.setdefault(video, []).append(video)
        reanme, feats  = feat_data.read(frame_ids)
        if pooling_style == 'avg':
            feat_vec = np.array(feats).mean(axis=0)
        elif pooling_style == 'max':
            feat_vec = np.array(feats).max(axis=0)
        fout.write(video + " " + " ".join(map(str,feat_vec)) + '\n')

    
        # output the sub video level feature
        counter = 0
        aug_index = f_auger.get_aug_index(len(frame_ids))  # get augmented frame list
        for sub_index in aug_index:
            sub_frames = [frame_ids[idx] for idx in sub_index]
            reanme, sub_feats  = feat_data.read(sub_frames)
            
            if pooling_style == 'avg':
                feat_vec = np.array(sub_feats).mean(axis=0)
            elif pooling_style == 'max':
                feat_vec = np.array(sub_feats).max(axis=0)

            video2subvideo.setdefault(video, []).append(video + "_sub%d" % counter)
            fout.write(video + "_sub%d" % counter + " " + " ".join(map(str,feat_vec)) + '\n')
            counter += 1
        progbar.add(1)

    fout.close()

    f = open(os.path.join(output_dir, "video2subvideo.txt"),'w')  
    f.write(str(video2subvideo))  
    f.close()  

    text2bin(len(feat_vec), [feat_combined_file], output_dir, 1)
    os.system('rm %s' % feat_combined_file)



def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    from optparse import OptionParser
    parser = OptionParser(usage="""usage: %prog [options]""")
    parser.add_option("--rootpath", default=ROOT_PATH, type="string", help="rootpath (default: %s)" % ROOT_PATH)
    parser.add_option("--collection", default="", type="string", help="collection name")
    parser.add_option("--feature", default="", type="string", help="feature name")
    parser.add_option("--stride", default="2", type="str", help="stride for frame-level data augmentation")
    parser.add_option("--overwrite", default=0, type="int", help="overwrite existing file (default=0)")
    parser.add_option("--pooling_style", default='avg', type="str", help="pooling style: avg(average pooling), max(max pooling)")

    (options, args) = parser.parse_args(argv)
    return process(options)

if __name__ == "__main__":
    sys.exit(main())
