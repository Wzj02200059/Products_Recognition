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
    def __init__(self, options, pretrained=True):
        super(RejectModel, self).__init__(options)
        self.in_planes = 512
        self.options = options
        self.num_classes = options['num_classes']
        self.backbone_name = options['backbone']
        self.pretrained = pretrained
        self.channel_trans = None
        self.model = nn.Sequential(*(list(self._get_model().children())[:-2]))
        self.gap = nn.AdaptiveAvgPool2d(1)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)  # no shift
        self.bottleneck.apply(weights_init_kaiming)

        # self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        # self.classifier.apply(weights_init_classifier)

    def forward(self, x):
        embd = self.model(x)
        global_feat = self.gap(embd)
        global_feat = global_feat.view(global_feat.shape[0], -1)
        feat = self.bottleneck(global_feat)  # normalize for angular softmax
        # output = self.classifier(feat)
        return feat

    def _get_model(self):
        kargs = {}
        model = BASICMODEL_BACKBONE_REGISTRY.get(
            self.backbone_name)(pretrained=self.pretrained, **kargs)
        # modify the last layer by mumber of classes
        if isinstance(model, models.resnet.ResNet):
            pass
        else:
            raise NotImplementedError(
                'Do not support the backbone {} for RejectModel now.'.format(self.backbone_name))
        return model

    @classmethod
    def get_backbone_list(cls):
        return list(BASICMODEL_BACKBONE_REGISTRY._obj_map.keys())
