import torch.nn as nn
import torch
import numpy as np
from typing import List, Tuple
import torch.nn.functional as F

from ..utils import box_utils
from collections import namedtuple
GraphPath = namedtuple("GraphPath", ['s0', 'name', 's1'])  #


class SSD(nn.Module):
    def __init__(self, num_classes: int, base_net: nn.ModuleList, source_layer_indexes: List[int],
                 extras: nn.ModuleList, classification_headers: nn.ModuleList,
                 regression_headers: nn.ModuleList, is_test=False, config=None, device=None, debug_dk=None):
        """Compose a SSD model using the given components.
        """
        super(SSD, self).__init__()

        self.num_classes = num_classes
        self.base_net = base_net
        self.source_layer_indexes = source_layer_indexes
        self.extras = extras
        self.classification_headers = classification_headers
        self.regression_headers = regression_headers
        self.is_test = is_test
        self.config = config
        self.debug_dk = debug_dk
        #print( "self.debug_dk=", self.debug_dk )

        # register layers in source_layer_indexes by adding them to a module list
        self.source_layer_add_ons = nn.ModuleList([t[1] for t in source_layer_indexes
                                                   if isinstance(t, tuple) and not isinstance(t, GraphPath)])
        if device:
            self.device = device
        else:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if is_test:
            self.config = config
            self.priors = config.priors.to(self.device)
            
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        confidences = []
        locations = []
        start_layer_index = 0
        header_index = 0
        
        if self.debug_dk == "dump":
            print( "x.shape=", x.shape, "source_layer_indexes=", self.source_layer_indexes )
        
        lcnt = 0
        for end_layer_index in self.source_layer_indexes:
            if isinstance(end_layer_index, GraphPath):
                path = end_layer_index
                end_layer_index = end_layer_index.s0
                added_layer = None
            elif isinstance(end_layer_index, tuple):
                added_layer = end_layer_index[1]
                end_layer_index = end_layer_index[0]
                path = None
            else:
                added_layer = None
                path = None
            for layer in self.base_net[start_layer_index: end_layer_index]:
                if self.debug_dk == "dump" and lcnt == 0:
                    if len(x.data.cpu().numpy().reshape(-1)) % 16 == 0:
                        np.savetxt( "./logs/" + "in_conv.csv", x.data.cpu().numpy().reshape(-1, 16), fmt='%.9f', delimiter=',' )
                    else:
                        np.savetxt( "./logs/" + "in_conv.csv", x.data.cpu().numpy().reshape(-1, 10), fmt='%.9f', delimiter=',' )
                
                if self.debug_dk == "dump":# and lcnt <= 5:
                    tmp = x
                    if lcnt == 0: print( "x[0, 0:3, 124:127, 290:293]=", x[0, 0:3, 124:127, 290:293] ) # inp 確認済み
                    for ii, ll in enumerate(layer):
                        tmp = ll(tmp)
                        print( "ii=", ii, "tmp.shape=", tmp.shape )
                        if len(tmp.data.cpu().numpy().reshape(-1)) % 16 == 0:
                            np.savetxt( "./logs/" + str(lcnt) + "_conv_" + str(ii) + ".csv", tmp.data.cpu().numpy().reshape(-1, 16), fmt='%.9f', delimiter=',' )
                        else:
                            np.savetxt( "./logs/" + str(lcnt) + "_conv_" + str(ii) + ".csv", tmp.data.cpu().numpy().reshape(-1, 10), fmt='%.9f', delimiter=',' )
                        if ii == 0 and False:
                            wei = ll.weight.data.cpu().numpy()
                            print( "wei.shape=", wei.shape ) #=> (32, 3, 3, 3)
                            if lcnt == 0:
                                print( "wei[21]=", wei[21] ) #=> wei 確認済み
                                print( "tmp[0, 21, 63, 146]=", tmp[0, 21, 63, 146] ) #=> tensor(2.8134, device='cuda:0')
                                print( "np.sum(x[0, 0:3, 124:127, 290:293] * wei[21])=", np.sum(x.data.cpu().numpy()[0, 0:3, 124:127, 290:293] * wei[21]) ) #=> np.sum(x[0, 0:3, 124:127, 290:293] * wei[21])= 2.081288
                            if wei.size % 16 == 0:
                                np.savetxt( "./logs/" + str(lcnt) + "_conv_" + str(ii) + "_wei.csv", wei.reshape(-1, 16), fmt='%.9f', delimiter=',' )
                            else:
                                np.savetxt( "./logs/" + str(lcnt) + "_conv_" + str(ii) + "_wei.csv", wei.reshape(-1, 10), fmt='%.9f', delimiter=',' )
                
                x = layer(x)
                
                if self.debug_dk == "dump":
                    print( "lcnt=", lcnt, "x.shape=", x.shape, "header_index=", header_index )
                    if len(x.data.cpu().numpy().reshape(-1)) % 16 == 0:
                        np.savetxt( "./logs/" + str(lcnt) + "_conv.csv", x.data.cpu().numpy().reshape(-1, 16), fmt='%.9f', delimiter=',' )
                    else:
                        np.savetxt( "./logs/" + str(lcnt) + "_conv.csv", x.data.cpu().numpy().reshape(-1, 10), fmt='%.9f', delimiter=',' )
                    lcnt += 1
            if added_layer:
                y = added_layer(x)
            else:
                y = x
            if path:
                sub = getattr(self.base_net[end_layer_index], path.name)
                for layer in sub[:path.s1]:
                    x = layer(x)
                y = x
                for layer in sub[path.s1:]:
                    x = layer(x)
                end_layer_index += 1
            start_layer_index = end_layer_index
            confidence, location = self.compute_header(header_index, y)
            header_index += 1
            confidences.append(confidence)
            locations.append(location)

        for layer in self.base_net[end_layer_index:]:
            if self.debug_dk == "dump":
                print( "layer=", layer )
            x = layer(x)

        for ii, layer in enumerate(self.extras):
            if self.debug_dk == "dump":
                x0 = layer[0](x)
                x1 = layer[1](x0)
                if len(x1.data.cpu().numpy().reshape(-1)) % 16 == 0:
                    np.savetxt( "./logs/" + str(ii) + "_extra_x1.csv", x1.data.cpu().numpy().reshape(-1, 16), fmt='%.9f', delimiter=',' )
                else:
                    np.savetxt( "./logs/" + str(ii) + "_extra_x1.csv", x1.data.cpu().numpy().reshape(-1, 10), fmt='%.9f', delimiter=',' )
                x2 = layer[2](x1)
                x3 = layer[3](x2)
                if len(x3.data.cpu().numpy().reshape(-1)) % 16 == 0:
                    np.savetxt( "./logs/" + str(ii) + "_extra_x3.csv", x3.data.cpu().numpy().reshape(-1, 16), fmt='%.9f', delimiter=',' )
                else:
                    np.savetxt( "./logs/" + str(ii) + "_extra_x3.csv", x3.data.cpu().numpy().reshape(-1, 10), fmt='%.9f', delimiter=',' )
            
            x = layer(x)
            
            if self.debug_dk == "dump":
                print( "x.shape=", x.shape, "ii=", ii )
                print( "layer=", layer )
                print( "layer[0]=", layer[0] )
                if len(x.data.cpu().numpy().reshape(-1)) % 16 == 0:
                    np.savetxt( "./logs/" + str(ii) + "_extra.csv", x.data.cpu().numpy().reshape(-1, 16), fmt='%.9f', delimiter=',' )
                else:
                    np.savetxt( "./logs/" + str(ii) + "_extra.csv", x.data.cpu().numpy().reshape(-1, 10), fmt='%.9f', delimiter=',' )
            confidence, location = self.compute_header(header_index, x)
            header_index += 1
            confidences.append(confidence)
            locations.append(location)

        confidences = torch.cat(confidences, 1)
        locations = torch.cat(locations, 1)
        if len(confidences.data.cpu().numpy().reshape(-1)) % 16 == 0:
            np.savetxt( "./logs/" + "concat1.csv", confidences.data.cpu().numpy().reshape(-1, 16), fmt='%.9f', delimiter=',' )
        else:
            np.savetxt( "./logs/" + "concat1.csv", confidences.data.cpu().numpy().reshape(-1, 10), fmt='%.9f', delimiter=',' )
        if len(locations.data.cpu().numpy().reshape(-1)) % 16 == 0:
            np.savetxt( "./logs/" + "concat2.csv", locations.data.cpu().numpy().reshape(-1, 16), fmt='%.9f', delimiter=',' )
        else:
            np.savetxt( "./logs/" + "concat2.csv", locations.data.cpu().numpy().reshape(-1, 10), fmt='%.9f', delimiter=',' )
        
        if self.is_test:
            confidences = F.softmax(confidences, dim=2)
            if self.debug_dk == "dump":
                print( "softmax: confidences.shape=", confidences.shape )
                if len(confidences.data.cpu().numpy().reshape(-1)) % 16 == 0:
                    np.savetxt( "./logs/" + "softmax.csv", confidences.data.cpu().numpy().reshape(-1, 16), fmt='%.9f', delimiter=',' )
                elif len(confidences.data.cpu().numpy().reshape(-1)) % 10 == 0:
                    np.savetxt( "./logs/" + "softmax.csv", confidences.data.cpu().numpy().reshape(-1, 10), fmt='%.9f', delimiter=',' )
                else:
                    np_softmax = confidences.data.cpu().numpy()
                    rest = 10 - np_softmax.size % 10
                    np.savetxt( "./logs/" + "softmax.csv", np.concatenate([np_softmax.reshape(-1), np.zeros((rest))]).reshape(-1, 10), fmt='%.9f', delimiter=',' )
            boxes = box_utils.convert_locations_to_boxes(
                locations, self.priors, self.config.center_variance, self.config.size_variance
            )
            if len(boxes.data.cpu().numpy().reshape(-1)) % 16 == 0:
                np.savetxt( "./logs/" + "boxes1.csv", boxes.data.cpu().numpy().reshape(-1, 16), fmt='%.9f', delimiter=',' )
            else:
                np.savetxt( "./logs/" + "boxes1.csv", boxes.data.cpu().numpy().reshape(-1, 10), fmt='%.9f', delimiter=',' )
            boxes = box_utils.center_form_to_corner_form(boxes)
            if len(boxes.data.cpu().numpy().reshape(-1)) % 16 == 0:
                np.savetxt( "./logs/" + "boxes2.csv", boxes.data.cpu().numpy().reshape(-1, 16), fmt='%.9f', delimiter=',' )
            else:
                np.savetxt( "./logs/" + "boxes2.csv", boxes.data.cpu().numpy().reshape(-1, 10), fmt='%.9f', delimiter=',' )
            return confidences, boxes
        else:
            return confidences, locations

    def compute_header(self, i, x):
        confidence = self.classification_headers[i](x)
        if self.debug_dk == "dump":
            print( "i=", i, "x.shape=", x.shape, "confidence.shape=", confidence.shape )
            if len(confidence.data.cpu().numpy().reshape(-1)) % 16 == 0:
                np.savetxt( "./logs/" + str(i) + "_conf.csv", confidence.data.cpu().numpy().reshape(-1, 16), fmt='%.9f', delimiter=',' )
            elif len(confidence.data.cpu().numpy().reshape(-1)) % 10 == 0:
                np.savetxt( "./logs/" + str(i) + "_conf.csv", confidence.data.cpu().numpy().reshape(-1, 10), fmt='%.9f', delimiter=',' )
            else:
                conf = confidence.data.cpu().numpy().reshape(-1)
                rest = 10 - conf.size % 10
                np.savetxt( "./logs/" + str(i) + "_conf.csv", np.concatenate([conf, np.zeros((rest))]).reshape(-1, 10), fmt='%.9f', delimiter=',' )
        confidence = confidence.permute(0, 2, 3, 1).contiguous()
        confidence = confidence.view(confidence.size(0), -1, self.num_classes)

        location = self.regression_headers[i](x)
        if self.debug_dk == "dump":
            print( "i=", i, "location.shape=", location.shape )
            if len(location.data.cpu().numpy().reshape(-1)) % 16 == 0:
                np.savetxt( "./logs/" + str(i) + "_loc.csv", location.data.cpu().numpy().reshape(-1, 16), fmt='%.9f', delimiter=',' )
            elif len(location.data.cpu().numpy().reshape(-1)) % 10 == 0:
                np.savetxt( "./logs/" + str(i) + "_loc.csv", location.data.cpu().numpy().reshape(-1, 10), fmt='%.9f', delimiter=',' )
            else:
                loc = location.data.cpu().numpy().reshape(-1)
                rest = 10 - loc.size % 10
                np.savetxt( "./logs/" + str(i) + "_loc.csv", np.concatenate([loc, np.zeros((rest))]).reshape(-1, 10), fmt='%.9f', delimiter=',' )
        location = location.permute(0, 2, 3, 1).contiguous()
        location = location.view(location.size(0), -1, 4)

        return confidence, location

    def init_from_base_net(self, model):
        self.base_net.load_state_dict(torch.load(model, map_location=lambda storage, loc: storage), strict=True)
        self.source_layer_add_ons.apply(_xavier_init_)
        self.extras.apply(_xavier_init_)
        self.classification_headers.apply(_xavier_init_)
        self.regression_headers.apply(_xavier_init_)

    def init_from_pretrained_ssd(self, model):
        state_dict = torch.load(model, map_location=lambda storage, loc: storage)
        state_dict = {k: v for k, v in state_dict.items() if not (k.startswith("classification_headers") or k.startswith("regression_headers"))}
        model_dict = self.state_dict()
        model_dict.update(state_dict)
        self.load_state_dict(model_dict)
        self.classification_headers.apply(_xavier_init_)
        self.regression_headers.apply(_xavier_init_)

    def init(self):
        self.base_net.apply(_xavier_init_)
        self.source_layer_add_ons.apply(_xavier_init_)
        self.extras.apply(_xavier_init_)
        self.classification_headers.apply(_xavier_init_)
        self.regression_headers.apply(_xavier_init_)

    def load(self, model):
        self.load_state_dict(torch.load(model, map_location=lambda storage, loc: storage))

    def save(self, model_path):
        torch.save(self.state_dict(), model_path)


class MatchPrior(object):
    def __init__(self, center_form_priors, center_variance, size_variance, iou_threshold):
        self.center_form_priors = center_form_priors
        self.corner_form_priors = box_utils.center_form_to_corner_form(center_form_priors)
        self.center_variance = center_variance
        self.size_variance = size_variance
        self.iou_threshold = iou_threshold

    def __call__(self, gt_boxes, gt_labels):
        if type(gt_boxes) is np.ndarray:
            gt_boxes = torch.from_numpy(gt_boxes)
        if type(gt_labels) is np.ndarray:
            gt_labels = torch.from_numpy(gt_labels)
        boxes, labels = box_utils.assign_priors(gt_boxes, gt_labels,
                                                self.corner_form_priors, self.iou_threshold)
        boxes = box_utils.corner_form_to_center_form(boxes)
        locations = box_utils.convert_boxes_to_locations(boxes, self.center_form_priors, self.center_variance, self.size_variance)
        return locations, labels


def _xavier_init_(m: nn.Module):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)
