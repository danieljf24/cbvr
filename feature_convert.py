import os
import sys
import numpy as np

from utils.generic_utils import Progbar
from utils.common import ROOT_PATH, checkToSkip, makedirsforfile
from simpleknn.txt2bin import process as text2bin


def process(options, collection, feat_name):
    overwrite = options.overwrite
    rootpath = options.rootpath

    feature_dir = os.path.join(rootpath, collection, 'feature')
    resdir = os.path.join(rootpath, collection, 'FeatureData', feat_name)

    train_csv = os.path.join(rootpath, collection, 'split', 'train.csv')
    val_csv = os.path.join(rootpath, collection, 'split', 'val.csv')
    test_csv = os.path.join(rootpath, collection, 'split', 'test.csv')

    train_val_test_set = []
    train_val_test_set.extend(map(str.strip, open(train_csv).readlines()))
    train_val_test_set.extend(map(str.strip, open(val_csv).readlines()))
    train_val_test_set.extend(map(str.strip, open(test_csv).readlines()))
    
    target_feat_file = os.path.join(resdir, 'id.feature.txt')
    if checkToSkip(os.path.join(resdir,'feature.bin'), overwrite):
        sys.exit(0)
    makedirsforfile(target_feat_file)

    frame_count = []
    print 'Processing %s - %s' % (collection, feat_name)
    with open(target_feat_file, 'w') as fw_feat:
        progbar = Progbar(len(train_val_test_set))
        for d in train_val_test_set:
            feat_file = os.path.join(feature_dir, d, '%s-%s.npy'%(d,feat_name))
            feats = np.load(feat_file)
            if len(feats.shape) == 1:  # video level feature
                dim = feats.shape[0]
                fw_feat.write('%s %s\n' % (d, ' '.join(['%.6f'%x for x in feats])))
            elif len(feats.shape) == 2:  # frame level feature
                frames, dim = feats.shape
                frame_count.append(frames)
                for i in range(frames):
                    frame_id = d+'_'+str(i)
                    fw_feat.write('%s %s\n' % (frame_id, ' '.join(['%.6f'%x for x in feats[i]])))
            progbar.add(1)

    text2bin(dim, [target_feat_file], resdir, 1)
    os.system('rm %s' % target_feat_file)


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    from optparse import OptionParser
    parser = OptionParser(usage="""usage: %prog [options] collection featname""")
    parser.add_option("--overwrite", default=0, type="int", help="overwrite existing file (default=0)")
    parser.add_option("--rootpath", default=ROOT_PATH, type="string", help="rootpath (default: %s)" % ROOT_PATH)
    
    (options, args) = parser.parse_args(argv)
    if len(args) < 2:
        parser.print_help()
        return 1

    return process(options, args[0], args[1])

if __name__ == '__main__':
    sys.exit(main())

