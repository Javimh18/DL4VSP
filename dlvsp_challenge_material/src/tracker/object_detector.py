import torch
import torch.nn.functional as F

from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

import numpy as np

@staticmethod
def _soft_nms(boxes, scores, IoU_thresh=0.5, score_thresh=0.5):
    """
    Apply soft non-maximum suppression algorithm on detected bounding boxes.
    """
    indices = np.arange(len(scores))
    for i, box in enumerate(boxes):
        for j in indices[i+1:]:
            IoU = _compute_IoU(box, boxes[j])
            if IoU > IoU_thresh:
                scores[j] = scores[j] * (1-IoU)
    
    keep = scores > score_thresh
    return boxes[keep], scores[keep]

@staticmethod
def _compute_IoU(box1, box2):
    """
    Compute the Intersection over Union (IoU) of two bounding boxes.
    """
    # Coordenadas de las cajas
    x1, y1, x2, y2 = box1
    x1_p, y1_p, x2_p, y2_p = box2

    # Área de la intersección
    xi1 = max(x1, x1_p)
    yi1 = max(y1, y1_p)
    xi2 = min(x2, x2_p)
    yi2 = min(y2, y2_p)
    inter_area = max(xi2 - xi1, 0) * max(yi2 - yi1, 0)

    # Área de las cajas
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2_p - x1_p) * (y2_p - y1_p)

    # Unión de las áreas
    union_area = box1_area + box2_area - inter_area

    # IoU
    IoU = inter_area / union_area

    return IoU

class FRCNN_FPN(FasterRCNN):

    def __init__(self, num_classes, nms_thresh=0.5):
        backbone = resnet_fpn_backbone('resnet50', False)
        super(FRCNN_FPN, self).__init__(backbone, num_classes)

        # self.roi_heads.nms_thresh = nms_thresh

    def detect(self, img):
        device = list(self.parameters())[0].device
        img = img.to(device)

        detections = self(img)[0]
        
        boxes = detections['boxes'].detach().cpu().numpy()
        scores = detections['scores'].detach().cpu().numpy()
        boxes, scores = _soft_nms(boxes, scores, IoU_thresh=0.35, score_thresh=0.5)
        boxes, scores = torch.from_numpy(boxes), torch.from_numpy(scores)

        return boxes, scores
    

class ObjDetector():
    def __init__(self, model_path=None, num_classes=2, nms_thresh=0.3):
        if model_path:
            self.model = torch.load(model_path)
        else:
            self.model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")

            # get number of input features for the classifier
            in_features = self.model.roi_heads.box_predictor.cls_score.in_features
            # replace the pre-trained head with a new one
            self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
            self.model.roi_heads.nms_thresh = nms_thresh

            
            # now get the number of input features for the mask classifier
            in_features_mask = self.model.roi_heads.mask_predictor.conv5_mask.in_channels
            hidden_layer = self.model.roi_heads.mask_predictor.conv5_mask.out_channels
            # and replace the mask predictor with a new one
            self.model.roi_heads.mask_predictor = MaskRCNNPredictor(
                in_features_mask,
                hidden_layer,
                num_classes
            )
        
    def detect(self, img):
        device = list(self.parameters())[0].device
        img = img.to(device)

        detections = self(img)[0]

        return detections['boxes'].detach().cpu(), detections['scores'].detach().cpu()