from torchvision import models
from torch import nn
import inspect

from zzClassifier.core import Registry
from zzClassifier.core import MODEL_REGISTRY, Basic


# registry all func in models
BASICMODEL_BACKBONE_REGISTRY = Registry('BASICMODEL_BACKBONE')
defined_model_function_list = inspect.getmembers(models, inspect.isfunction)
for name, func in defined_model_function_list:
    BASICMODEL_BACKBONE_REGISTRY.register(func)
# BASICMODEL_BACKBONE_REGISTRY.register(models.mobilenet_v2)

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

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)

@MODEL_REGISTRY.register()
class RejectModel(Basic):
    def __init__(self, cfg, pretrained=True):
        super(RejectModel, self).__init__(cfg)
        self.in_planes = 512
        self.cfg = cfg
        self.num_classes = cfg.DATASET.CLASSES_NUM
        self.backbone_name = cfg.MODEL.BACKBONE
        self.pretrained = pretrained
        self.channel_trans = None
        self.model = nn.Sequential(*(list(self._get_model().children())[:-2]))
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)  # no shift
        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.bottleneck.apply(weights_init_kaiming)
        self.classifier.apply(weights_init_classifier)

    def forward(self, x):
        embd = self.model(x)
        global_feat = self.gap(embd)
        global_feat = global_feat.view(global_feat.shape[0], -1)
        feat = self.bottleneck(global_feat)  # normalize for angular softmax
        output = self.classifier(feat)
        return output

    def _get_model(self):
        # load the pretrained model torchvision provided will speed up the training session a lot
        kargs = {}
        model = BASICMODEL_BACKBONE_REGISTRY.get(
            self.backbone_name)(pretrained=self.pretrained, **kargs)

        # modify the last layer by mumber of classes
        if isinstance(model, models.resnet.ResNet) or \
                isinstance(model, models.inception.Inception3) or \
                isinstance(model, models.shufflenetv2.ShuffleNetV2) or \
                'googlenet' in self.backbone_name:
            pass
            # model.fc = nn.Linear(model.fc.in_features, 555)

        elif isinstance(model, models.mobilenet.MobileNetV2) or \
                'alexnet' in self.backbone_name or \
                isinstance(model, models.mnasnet.MNASNet) or \
                isinstance(model, models.vgg.VGG):
            # model.classifier is a Sequential, and last layer is fc layer to predict results
            model.classifier[-1] = nn.Linear(in_features=model.classifier[-1].in_features,
                                             out_features=self.num_classes, bias=True)
        elif isinstance(model, models.densenet.DenseNet):
            model.classifier = nn.Linear(in_features=model.classifier.in_features,
                                         out_features=self.num_classes, bias=True)
        elif isinstance(model, models.squeezenet.SqueezeNet):
            # model.classifier is a Sequential, and the second layer is the Conv2d layer to predict results
            model.classifier[1] = nn.Conv2d(in_channels=model.classifier[1].in_channels,
                                            out_channels=self.num_classes, kernel_size=(1, 1), stride=(1, 1))
        else:
            raise NotImplementedError(
                'Do not support the backbone {} for BasicModel now.'.format(self.backbone_name))
        return model

    @classmethod
    def get_backbone_list(cls):
        return list(BASICMODEL_BACKBONE_REGISTRY._obj_map.keys())
