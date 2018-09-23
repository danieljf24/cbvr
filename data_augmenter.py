import os
import random
import numpy as np
from simpleknn.bigfile import BigFile

# Augmentation for frame-level features
class Frame_Level_Augmenter(object):

    def __init__(self, stride=2, n_frame_threshold=20):
        self.stride = stride
        self.n_frame_threshold = n_frame_threshold

    def get_aug_index(self, n_vecs):
        if type(self.stride) is int:
            self.stride = [self.stride]

        aug_index = []
        # aug_index.append(range(n_vecs)) # keep original frame level features
        if n_vecs < self.n_frame_threshold:
            return aug_index
        for stride in self.stride:
            for i in range(stride):
                sub_index = range(n_vecs)[i::stride]
                aug_index.append(sub_index)
        return aug_index

    def get_aug_feat(self, frm_feat):
        n_vecs = len(frm_feat)
        aug_index = self.get_aug_index(n_vecs)

        aug_feats = []
        for index in aug_index:
            org_feat = [frm_feat[x] for x in index]
            aug_feats.append(org_feat)
        return aug_feats


    def aug_index_choice(self, n_vecs):
        return random.choice(self.get_aug_index(n_vecs))


    def aug_feat_choice(self, frm_feat):
        n_vecs = len(frm_feat)
        aug_index_choice = self.aug_index_choice(n_vecs)
        aug_feat = [frm_feat[x] for x in aug_index_choice]
        return aug_feat

        


# Augmentation for video-level features
class Video_Level_Augmenter(object):

    def __init__(self, feat_path=None, feat_reader=None, perturb_intensity=1, perturb_prob=0.5, n_sample=10000, step_size=500, mean=None, std=None):
        self.feat_reader = feat_reader
        self.perturb_intensity = perturb_intensity
        self.perturb_prob = perturb_prob
        self.step_size = step_size

        if mean is None or std is None:
            self.n_dims = feat_reader.ndims
            mean_std_file = os.path.join(feat_path, "mean_std.txt")
            if not os.path.exists(mean_std_file):
                # calculate the mean and std
                print "calculating the mean and std ..."
                if len(feat_reader.names) <= n_sample:
                    self.sampled_videos = feat_reader.names
                else:
                    self.sampled_videos = random.sample(feat_reader.names, n_sample)
                self.mean, self.std = self.__get_mean_std()
                if not os.path.exists(mean_std_file):
                    with open(mean_std_file, 'w') as fout:
                        fout.write(" ".join(map(str, self.mean)) + "\n")
                        fout.write(" ".join(map(str, self.std)) + "\n")
            else:
                with open(mean_std_file) as fin:
                    self.mean = map(float, fin.readline().strip().split(" "))
                    self.std =  map(float, fin.readline().strip().split(" "))
        else:
            self.n_dims = len(mean)
            self.mean = mean
            self.std = std

        # initialize mask
        self.__init_mask()


    def __get_mean_std(self):
        mean = []
        std = []
        for i in range(0, self.n_dims, self.step_size):
            vec_list = []
            for video in self.sampled_videos:
                feat_vec = self.feat_reader.read_one(video)
                # using the subvec to accelerate calculation
                vec_list.append(feat_vec[i:min(self.step_size+i, self.n_dims)])
            mean.extend(np.mean(vec_list, 0))
            std.extend(np.std(vec_list, 0))
        return np.array(mean), np.array(std)

    def __init_mask(self):
        self.mask = np.zeros(self.n_dims)
        self.mask[:int(self.n_dims*self.perturb_prob)] = 1

    def __shuffle_mask(self):
        random.shuffle(self.mask)

    def get_aug_feat(self, vid_feat):
        self.__shuffle_mask()
        perturbation = (np.random.randn(self.n_dims)*self.std + self.mean) * self.perturb_intensity * self.mask
        aug_feat =  vid_feat + perturbation
        return aug_feat




if __name__ == "__main__":

    # test frame level augmentation
    feats = np.random.randn(11, 4)
    n_vecs = feats.shape[0]
    for stride in [2, [2,3]]:
        f_auger = Frame_Level_Augmenter(stride)
        print f_auger.get_aug_index(n_vecs)
        # print f_auger.get_aug_feat(feats)
        print [len(a) for a in f_auger.get_aug_feat(feats)]

    # test video level augmentation
    rootpath = '/home/daniel/VisualSearch/hulu'
    collection = 'track_1_shows'
    feature = 'c3d-pool5'
    feat_path = os.path.join(rootpath, collection, "FeatureData", feature)
    feat_reader = BigFile(feat_path)

    v_auger = Video_Level_Augmenter(feat_path, feat_reader, perturb_intensity=1, perturb_prob=0.5)
    vid_feat = feat_reader.read_one(random.choice(feat_reader.names))
    aug_feat = v_auger.get_aug_feat(vid_feat)