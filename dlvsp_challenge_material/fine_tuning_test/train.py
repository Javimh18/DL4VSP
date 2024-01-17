import utils
import os
import sys
import random

import torch 
from torchvision.transforms import v2 as T
from torchvision.transforms import functional as F

from engine import train_one_epoch, evaluate
from dataset import PennFudanDataset, MOT16ObjDetectMasked
from model import ObjDetector


sys.path.append(os.path.join(os.getcwd(), ".."))
from src.tracker.data_obj_detect import MOT16ObjDetect
from src.tracker.object_detector import FRCNN_FPN, ObjDetector

def get_transform(train):
    transforms = []
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    transforms.append(T.ToDtype(torch.float, scale=True))
    transforms.append(T.ToPureTensor())
    return T.Compose(transforms)

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):           
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

class RandomHorizontalFlip(object):
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            height, width = image.shape[-2:]
            image = image.flip(-1)
            bbox = target["boxes"]
            bbox[:, [0, 2]] = width - bbox[:, [2, 0]]
            target["boxes"] = bbox
            if "masks" in target:
                target["masks"] = target["masks"].flip(-1)
        return image, target

class ToTensor(object):
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target

def obj_detect_transforms(train):
    transforms = []
    # converts the image, a PIL image, into a PyTorch Tensor
    transforms.append(ToTensor())
    if train:
        # during training, randomly flip the training images
        # and ground-truth for data augmentation
        transforms.append(RandomHorizontalFlip(0.5))
    return Compose(transforms)

def collate_fn(batch):
    return tuple(zip(*batch))

if __name__ == '__main__':
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    obj_detect_nms_thresh = 0.35
    obj_detect_model_file = os.path.join("../models/faster_rcnn_fpn.model")
    
    # use our dataset and defined transformations
    dataset = MOT16ObjDetectMasked(root='../data/MOT16/train', transforms=obj_detect_transforms(train=True))

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=3,
        shuffle=True,
        num_workers=os.cpu_count(),
        collate_fn=collate_fn
    )

    # get the model using our helper function
    #obj_detect = ObjDetector(model_path="../models/maskrcnn_model.pth").model
    # obj_detect = ObjDetector(num_classes=2, nms_thresh=obj_detect_nms_thresh).model
    
    obj_detect = FRCNN_FPN(num_classes=2, nms_thresh=obj_detect_nms_thresh)
    obj_detect_state_dict = torch.load(obj_detect_model_file,map_location=lambda storage, loc: storage)
    obj_detect.load_state_dict(obj_detect_state_dict)
    obj_detect.eval()     # set to evaluation mode
    obj_detect.to(device) # load detector to GPU or CPU
    

    # move model to the right device
    obj_detect.to(device)

    # construct an optimizer
    params = [p for p in obj_detect.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params,
        lr=0.005,
        momentum=0.9,
        weight_decay=0.0005
    )

    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=3,
        gamma=0.1
    )

    num_epochs = 1

    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(obj_detect, optimizer, data_loader, device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        # evaluate(model.model, data_loader_test, device=device)
        
    torch.save(obj_detect.state_dict(), "../models/finetuned_faster_rcnn.model")