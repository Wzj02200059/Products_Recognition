import os
import logging
import torch
from torch import nn

from zzClassifier.core import Registry

MODEL_REGISTRY = Registry('MODEL_NAME')

def build_model(options, pretrained=True):
    model_generator = options['model_name']
    return MODEL_REGISTRY.get(model_generator)(options, pretrained)


def load_model(model_path, model, strict=True):
    # only load the state_dict
    if os.path.isfile(model_path):
        state_dict = torch.load(model_path)
        model.load_state_dict(state_dict['state_dict'], strict=strict)
    else:
        err_msg = 'Inputted model path {} does not exists.'.format(model_path)
        raise Exception(err_msg)
    return model


def get_model_list():
    return list(MODEL_REGISTRY._obj_map.keys())


def get_backbone_list(model_name):
    return MODEL_REGISTRY.get(model_name).get_backbone_list()


class Basic(nn.Module):
    def __init__(self, options):
        self.options = options
        super(Basic, self).__init__()

    def load_state_dict(self, state_dict, strict=True):
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in state_dict.items()
                           if k in model_dict and model_dict[k].size() == v.size()}

        not_loaded_keys = [
            k for k in state_dict.keys() if k not in pretrained_dict.keys()]
        extra_keys = [k for k in pretrained_dict.keys() if k not in state_dict.keys()]

        model_dict.update(pretrained_dict)
        super(Basic, self).load_state_dict(model_dict)

    @classmethod
    def get_backbone_list(cls):
        raise NotImplementedError()