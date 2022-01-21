"""Extract features from a model layer"""

from collections import defaultdict
import torch
import torch.nn as nn
import torchvision


class FeatureExtractor(nn.Module):
    def __init__(self, model, target_layers):
        super().__init__()
        self.model = model
        self.target_layers = target_layers
        self._features - defaultdict(list)

        layer_dict = dict([*self.model.named_modules()])
        for layer_name in self.target_layers:
            layer = layer_dict[layer_name]
            layer.register_forward_hook(self.hook_fn(layer_name))

    def hook_fn(self, layer_name):
        def hook(module, input, output):
            self._features[layer_name] = output
        return hook
    
    def forward(self, x):
        _ = self.model(x)
        return self._features