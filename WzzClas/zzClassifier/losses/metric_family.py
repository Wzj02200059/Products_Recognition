from torchvision import models
import torch.nn.functional as F
from torch import nn
import torch
import inspect
from torch.nn import Parameter

from zzClassifier.core import Registry
from zzClassifier.core import MODEL_REGISTRY, Basic
from zzClassifier.models.resnet import ResNet, Bottleneck
from zzClassifier.losses.metric_family import Arcface

# registry all func in models
BASICMODEL_BACKBONE_REGISTRY = Registry('BASICMODEL_BACKBONE')
defined_model_function_list = inspect.getmembers(models, inspect.isfunction)
for name, func in defined_model_function_list:
    BASICMODEL_BACKBONE_REGISTRY.register(func)
# BASICMODEL_BACKBONE_REGISTRY.register(models.mobilenet_v2)

class GeM(nn.Module):

    def __init__(self, p=3.0, eps=1e-6, freeze_p=False):
        super(GeM, self).__init__()
        self.p = p if freeze_p else Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return F.adaptive_avg_pool2d(x.clamp(min=self.eps).pow(self.p),
                            (x.size(-2), x.size(-1))).pow(1. / self.p)

    def __repr__(self):
        if isinstance(self.p, float):
            p = self.p
        else:
            p = self.p.data.tolist()[0]
        return self.__class__.__name__ +\
               '(' + 'p=' + '{:.4f}'.format(p) +\
               ', ' + 'eps=' + str(self.eps) + ')'


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

def weights_init_xavier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.xavier_uniform_(m.weight)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)
def margin_softmax(x, w, label, scale=64.0, margin=0.4):
    # x shape is batch_size x K, w shape is N * K
    cos = F.linear(x, w)
    batch_size = x.shape[0]
    gt = cos[torch.arange(0, batch_size), label].view(-1, 1)
    final_gt = torch.where(gt > 0, gt - margin, gt)
    cos.scatter_(1, label.view(-1, 1), final_gt)
    return cos * scale

@MODEL_REGISTRY.register()
class RESNET_Model(Basic):
    def __init__(self, cfg, pretrained=True):
        super(RESNET_Model, self).__init__(cfg)
        self.in_planes = 2048
        self.neck_planes = 2048
        self.cfg = cfg
        self.backbone_name = cfg.model.backbone
        self.pretrained = pretrained
        self.num_classes = cfg.dataset.num_class
        self.model = ResNet(last_stride=cfg.model.last_stride,
                               block=Bottleneck,
                               layers=[3, 4, 6, 3])
        self.gem = GeM()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.gmp = nn.AdaptiveMaxPool2d(1)
        self.fcneck = nn.Linear(self.in_planes, self.neck_planes, bias=False)   
        self.fcneck.apply(weights_init_xavier)
        self.fcneck_bn = nn.BatchNorm1d(self.neck_planes)
        self.fcneck_bn.bias.requires_grad_(False) # no shift
        self.fcneck_bn.apply(weights_init_kaiming)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.2)
        self.classifier = Arcface(self.neck_planes, self.num_classes,
                                      s=30, m=0.3)
        self.classifier.apply(weights_init_classifier)

    def forward(self, x, label=None):  # label is unused if self.cos_layer == 'no'
        x = self.model(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        feat = self.gem(x)
        feat = feat.view(feat.shape[0], -1)  # flatten to (bs, 2048)

        if self.training:
            global_feat = self.fcneck(feat)
            global_feat = self.fcneck_bn(global_feat)
            # global_feat = self.relu(global_feat)
            global_feat_cls = self.dropout(global_feat)
            logits = self.classifier(global_feat_cls, label)
            return logits, feat
        else:
            return feat
