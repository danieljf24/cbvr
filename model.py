import torch
import torch.nn as nn
import torch.nn.init
import torchvision.models as models
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from torch.nn.utils.clip_grad import clip_grad_norm
import numpy as np
from collections import OrderedDict


def l2norm(X):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=1, keepdim=True).sqrt()
    X = torch.div(X, norm)
    return X

###################################################
##########        Model Structure        ##########   
###################################################
class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()

    def load_state_dict(self, state_dict):
        """Copies parameters. overwritting the default one to
        accept state_dict from Full model
        """
        own_state = self.state_dict()
        new_state = OrderedDict()
        for name, param in state_dict.items():
            if name in own_state:
                new_state[name] = param

        super(BaseModel, self).load_state_dict(new_state)



class EncoderVideo(BaseModel):

    def __init__(self, opt):
        super(EncoderVideo, self).__init__()
        self.embed_size = opt.embed_size
        self.no_imgnorm = opt.no_imgnorm

        self.fc = nn.Linear(opt.feature_dim, opt.embed_size)


        self.init_weights()

    def init_weights(self):
        """Xavier initialization for the fully connected layer
        """
        r = np.sqrt(6.) / np.sqrt(self.fc.in_features +
                                  self.fc.out_features)
        self.fc.weight.data.uniform_(-r, r)
        self.fc.bias.data.fill_(0)


    def forward(self, images):
        """Extract image feature vectors."""
        # assuming that the precomputed features are already l2-normalized

        features = self.fc(images)
        if not self.no_imgnorm:
            features = l2norm(features)

        return features



def cosine_sim(im, s):
    """Cosine similarity between all the image and sentence pairs
    """
    return im.mm(s.t())



class ContrastiveLoss(nn.Module):
    """
    Compute contrastive loss
    """

    def __init__(self, margin=0, measure=False, max_violation=False, cost_style='sum'):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.cost_style = cost_style
        if  measure == 'cosine':
            self.sim = cosine_sim
        else:
            print('measure %s is not supported')

        self.max_violation = max_violation

    def forward(self, im, s):
        # compute image-sentence score matrix
        scores = self.sim(im, s)
        diagonal = scores.diag().view(im.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        # compare every diagonal score to scores in its column
        # caption retrieval
        cost_s = (self.margin + scores - d1).clamp(min=0)
        # compare every diagonal score to scores in its row
        # image retrieval
        cost_im = (self.margin + scores - d2).clamp(min=0)

        # clear diagonals
        mask = torch.eye(scores.size(0)) > .5
        I = Variable(mask)
        if torch.cuda.is_available():
            I = I.cuda()
        cost_s = cost_s.masked_fill_(I, 0)
        cost_im = cost_im.masked_fill_(I, 0)

        # keep the maximum violating negative for each query
        if self.max_violation:
            cost_s = cost_s.max(1)[0]
            cost_im = cost_im.max(0)[0]

        if self.cost_style == 'sum':
            cost = cost_s.sum() + cost_im.sum()
        elif self.cost_style == 'mean':
            cost = cost_s.mean() + cost_im.mean()
        return cost


class ReLearning(object):

    def __init__(self, opt):
        self.grad_clip = opt.grad_clip
        self.img_enc = EncoderVideo(opt)

        print(self.img_enc)
        if torch.cuda.is_available():
            self.img_enc.cuda()
            cudnn.benchmark = True
        
        # Loss and Optimizer
        if opt.loss == 'mrl':
            self.criterion = ContrastiveLoss(margin=opt.margin,
                                             measure=opt.measure,
                                             max_violation=opt.max_violation,
                                             cost_style=opt.cost_style)


        params = list(self.img_enc.parameters())
        self.params = params
        
        if opt.optimizer == 'adam':
            self.optimizer = torch.optim.Adam(params, lr=opt.learning_rate)
        elif opt.optimizer == 'rmsprop':
            self.optimizer = torch.optim.RMSprop(params, lr=opt.learning_rate)
        else:
            print('optimizer %s is not supported' % self.optimizer)

        self.Eiters = 0

    def state_dict(self):
        state_dict = [self.img_enc.state_dict()]
        return state_dict

    def load_state_dict(self, state_dict):
        self.img_enc.load_state_dict(state_dict[0])

    def train_start(self):
        """switch to train mode
        """
        self.img_enc.train()

    def val_start(self):
        """switch to evaluate mode
        """
        self.img_enc.eval()


    def forward_emb(self, videos, volatile=False):
        """Compute the image and caption embeddings
        """
        # Set mini-batch dataset
        videos = Variable(videos, volatile=volatile)
        if torch.cuda.is_available():
            videos = videos.cuda()

        # Forward
        videos_emb = self.img_enc(videos)
        return videos_emb


    def forward_loss(self, videos_emb_1, videos_emb_2, **kwargs):
        """Compute the loss given pairs of image and caption embeddings
        """
        loss = self.criterion(videos_emb_1, videos_emb_2)
        # self.logger.update('Le', loss.data[0], videos_emb_1.size(0))
        return loss

    def train_emb(self, videos_1, videos_2, ids=None, *args):
        """One training step given images and captions.
        """
        self.Eiters += 1

        # zero the gradient buffers
        self.optimizer.zero_grad()   

        # compute the embeddings
        videos_emb_1 = self.forward_emb(videos_1)
        videos_emb_2 = self.forward_emb(videos_2)

        # measure accuracy and record loss
        # self.optimizer.zero_grad()
        loss = self.forward_loss(videos_emb_1, videos_emb_2)
        # loss_value = loss.item()
        loss_value = loss.data[0]

        # compute gradient and do SGD step
        loss.backward()
        if self.grad_clip > 0:
            clip_grad_norm(self.params, self.grad_clip)
        self.optimizer.step()

        return videos_emb_1.size(0), loss_value