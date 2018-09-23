import os
import csv
import random
import numpy as np
from data_augmenter import Frame_Level_Augmenter, Video_Level_Augmenter

import torch
import torch.utils.data as data


def read_videopair(input_file):
    print 'reading data from:', input_file
    videopairlist = []
    reader = csv.reader(open(input_file, 'r'))
    for data in reader:
        if data[1] == "":
            continue
        video = data[0]
        for video2 in data[1:]:
            videopairlist.append((video, video2))
    return videopairlist


def collate_fn(data):
    videos_1, videos_2, inds = zip(*data)

    # Merge videos (convert tuple of 2D tensor to 3D tensor)
    videos_1 = torch.stack(videos_1, 0)
    videos_2 = torch.stack(videos_2, 0)

    return videos_1, videos_2, inds


# using data argumentation on the fly (training is too slow, so we discard it)
# class Dataset_frame_da(data.Dataset):

#     def __init__(self, data_path, frame_feats, video2frames, stride=2):
#         self.videopairlist = read_videopair(data_path)
#         self.frame_feats = frame_feats
#         self.video2frames = video2frames
#         self.sub_length = len(self.videopairlist)
        
#         if type(stride) is int:
#             self.length = self.sub_length * stride
#         else:
#             self.length = self.sub_length * sum(stride)

#         self.f_auger = Frame_Level_Augmenter(stride)

    
#     def get_aug_pool_feat(self, vidoe_id):
#         frm_feat = [self.frame_feats.read_one(fid) for fid in self.video2frames[vidoe_id]]
#         frm_feat = self.f_auger.aug_feat_choice(frm_feat)
#         return np.array(frm_feat).mean(axis=0)


#     def __getitem__(self, index):
#         vidoe_id_1, video_id_2 = self.videopairlist[index%self.sub_length]

#         video_1 = torch.Tensor(self.get_aug_pool_feat(vidoe_id_1))
#         video_2 = torch.Tensor(self.get_aug_pool_feat(video_id_2))

#         return video_1, video_2, index

#     def __len__(self):
#         return self.length


# def get_frame_da_loader(data_path, frame_feats, opt, batch_size=100, shuffle=True, num_workers=2, video2frames=None, stride=5):
#     """Returns torch.utils.data.DataLoader for custom coco dataset."""
#     dset = Dataset_frame_da(data_path, frame_feats, video2frames, stride)

#     data_loader = torch.utils.data.DataLoader(dataset=dset,
#                                               batch_size=batch_size,
#                                               shuffle=shuffle,
#                                               pin_memory=True,
#                                               collate_fn=collate_fn)
#     return data_loader





class PrecompDataset_video_da(data.Dataset):

    def __init__(self, data_path, video_feats, video2subvideo, n_subs, aug_prob=0, perturb_intensity=0.01, perturb_prob=0.5, feat_path=None):
        self.videopairlist = read_videopair(data_path)
        self.video_feats = video_feats
        self.sub_length = len(self.videopairlist)
        self.video2subvideo = video2subvideo
        self.length = self.sub_length * n_subs
        self.n_subs = n_subs

        self.aug_prob = aug_prob
        self.perturb_intensity = perturb_intensity
        self.perturb_prob = perturb_prob
        if self.aug_prob > 0:
            self.length = int(self.length / aug_prob)
            self.v_auger = Video_Level_Augmenter(feat_path, video_feats, perturb_intensity=perturb_intensity, perturb_prob=perturb_prob)

    def __getitem__(self, index):
        vidoe_id_1, video_id_2 = self.videopairlist[index%self.sub_length]

        if self.n_subs > 1:
            vidoe_id_1 = random.choice(self.video2subvideo[vidoe_id_1])
            video_id_2 = random.choice(self.video2subvideo[video_id_2])

        video_1 = self.video_feats.read_one(vidoe_id_1)
        video_2 = self.video_feats.read_one(video_id_2)

        if self.aug_prob > 0: # Adding tiny perturbations for data argumentation
            if random.random() < self.aug_prob:
                video_1 = self.v_auger.get_aug_feat(video_1)
                video_2 = self.v_auger.get_aug_feat(video_2)

        video_1 = torch.Tensor(video_1)
        video_2 = torch.Tensor(video_2)

        return video_1, video_2, index

    def __len__(self):
        return self.length




def get_video_da_loader(data_path, video_feats, opt, batch_size=100, shuffle=True, num_workers=2, video2subvideo=None, n_subs=1, feat_path=""):
    dset = PrecompDataset_video_da(data_path, video_feats, video2subvideo, n_subs, 
        aug_prob=opt.aug_prob, perturb_intensity=opt.perturb_intensity, perturb_prob=opt.perturb_prob, feat_path=feat_path)

    data_loader = torch.utils.data.DataLoader(dataset=dset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              pin_memory=True,
                                              collate_fn=collate_fn)
    return data_loader        





# for validation and test
class FeatDataset(data.Dataset):
    """
    Load precomputed video features
    """

    def __init__(self, videolist, video_feats):
        self.video_feats = video_feats
        self.videolist = videolist
        self.length = len(videolist)

    def __getitem__(self, index):
        vidoe_id = self.videolist[index]
        video = torch.Tensor(self.video_feats.read_one(vidoe_id))
        return video, vidoe_id, index

    def __len__(self):
        return self.length


def collate_fn_feat(data):

    videos, ids, idxs = zip(*data)

    # Merge videos (convert tuple of 2D tensor to 3D tensor)
    videos = torch.stack(videos, 0)

    return videos, ids, idxs


def get_feat_loader(videolist, video_feats, batch_size=100, shuffle=False, num_workers=2):

    dset = FeatDataset(videolist, video_feats)
    data_loader = torch.utils.data.DataLoader(dataset=dset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              pin_memory=True,
                                              collate_fn=collate_fn_feat)
    return data_loader