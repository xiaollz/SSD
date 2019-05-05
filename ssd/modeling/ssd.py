import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F

from ssd.modeling.multibox_loss import MultiBoxLoss
from ssd.module import L2Norm
from ssd.module.prior_box import PriorBox
from ssd.utils import box_utils

from ssd.modeling.ssd_fcos_loss import make_fcos_loss_evaluator
from ssd.modeling.fcos_inference import make_fcos_postprocessor


class SSD(nn.Module):
    def __init__(self, cfg,
                 vgg: nn.ModuleList,
                 extras: nn.ModuleList,
                 classification_headers: nn.ModuleList,
                 regression_headers: nn.ModuleList,
                 centerness_headers: nn.ModuleList):
        """Compose a SSD model using the given components.
        """
        super(SSD, self).__init__()
        self.cfg = cfg
        self.num_classes = cfg.MODEL.NUM_CLASSES
        self.vgg = vgg
        self.extras = extras
        self.classification_headers = classification_headers
        self.regression_headers = regression_headers
        self.centerness_headers = centerness_headers
        self.l2_norm = L2Norm(512, scale=20)
        self.reset_parameters()

        # add evaluator & inference
        loss_evaluator = make_fcos_loss_evaluator(cfg)
        self.loss_evaluator = loss_evaluator
        box_selector_test = make_fcos_postprocessor(cfg)
        self.box_selector_test = box_selector_test

    def reset_parameters(self):
        def weights_init(m):
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

        self.vgg.apply(weights_init)
        self.extras.apply(weights_init)
        self.classification_headers.apply(weights_init)
        self.regression_headers.apply(weights_init)
        self.centerness_headers.apply(weights_init)

    def forward(self, x, targets=None):
        sources = []
        confidences = []
        locations = []
        centerness = []
        for i in range(23):
            x = self.vgg[i](x)
        s = self.l2_norm(x)  # Conv4_3 L2 normalization
        sources.append(s)

        # apply vgg up to fc7
        for i in range(23, len(self.vgg)):
            x = self.vgg[i](x)
        sources.append(x)

        for k, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True)
            if k % 2 == 1:
                sources.append(x)
        
        # add points computation, here, points as locations in fcos.py
        points = self.compute_locations(sources)

        for (x, l, c, ct) in zip(sources, self.regression_headers, self.classification_headers,
                                self.centerness_headers):
            locations.append(l(x).permute(0, 2, 3, 1).contiguous())
            confidences.append(c(x).permute(0, 2, 3, 1).contiguous())
            centerness.append(ct(x).permute(0, 2, 3, 1).contiguous())

        confidences = torch.cat([o.view(o.size(0), -1) for o in confidences], 1)
        locations = torch.cat([o.view(o.size(0), -1) for o in locations], 1)
        centerness = torch.cat([o.view(o.size(0), -1) for o in centerness], 1)

        confidences = confidences.view(confidences.size(0), -1, self.num_classes)
        locations = locations.view(locations.size(0), -1, 4)
        centerness = centerness.view(locations.size(0), -1, 1)
        
        # fcos inference & loss
        if not self.training:
            # when evaluating, decode predictions
            confidences = F.softmax(confidences, dim=2)
            boxes = box_utils.convert_locations_to_boxes(
                locations, self.priors, self.cfg.MODEL.CENTER_VARIANCE, self.cfg.MODEL.SIZE_VARIANCE
            )
            boxes = box_utils.center_form_to_corner_form(boxes)
            return boxes, {} # mind the return value
        else:
            # when training, compute losses
            regression_loss, classification_loss, centerness_loss = 
                            self.loss_evaluator(points, confidences, locations, centerness, targets)
            loss_dict = dict(
                regression_loss=regression_loss,
                classification_loss=classification_loss,
                centerness_loss=centerness_loss
            )
            return None, loss_dict
    
    def compute_locations(self, features):
    locations = []
    for level, feature in enumerate(features):
        h, w = feature.size()[-2:]
        locations_per_level = self.compute_locations_per_level(
            h, w, self.cfg.STRIDES[level], # add STRIDES in config
            feature.device
        )
        locations.append(locations_per_level)
    return locations

    def compute_locations_per_level(self, h, w, stride, device):
        shifts_x = torch.arange(
            0, w * stride, step=stride,
            dtype=torch.float32, device=device
        )
        shifts_y = torch.arange(
            0, h * stride, step=stride,
            dtype=torch.float32, device=device
        )
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
        shift_x = shift_x.reshape(-1)
        shift_y = shift_y.reshape(-1)
        locations = torch.stack((shift_x, shift_y), dim=1) + stride // 2
        return locations

    def init_from_base_net(self, model):
        vgg_weights = torch.load(model, map_location=lambda storage, loc: storage)
        self.vgg.load_state_dict(vgg_weights, strict=True)

    def load(self, model):
        self.load_state_dict(torch.load(model, map_location=lambda storage, loc: storage))

    def save(self, model_path):
        torch.save(self.state_dict(), model_path)
