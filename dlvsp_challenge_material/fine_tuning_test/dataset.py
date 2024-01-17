import os
import torch
from torchvision.io import read_image
from torchvision.ops.boxes import masks_to_boxes
import configparser
import csv
from PIL import Image, ImageDraw

from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F
from torchvision import transforms

import numpy as np

import os
import torch


class PennFudanDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "PNGImages"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "PedMasks"))))

    def __getitem__(self, idx):
        # load images and masks
        img_path = os.path.join(self.root, "PNGImages", self.imgs[idx])
        mask_path = os.path.join(self.root, "PedMasks", self.masks[idx])
        img = read_image(img_path)
        mask = read_image(mask_path)
        # instances are encoded as different colors
        obj_ids = torch.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]
        num_objs = len(obj_ids)
        
        """
        # pass to 8-bit mask then to binary mask
        mask = Image.open(mask_path)
        mask_array = np.array(mask)
        scaled_image_array = ((mask_array - mask_array.min()) / (mask_array.max() - mask_array.min()) * 255).astype(np.uint8)
        eight_bit_mask = Image.fromarray(scaled_image_array)
        mask = transforms.ToTensor()(eight_bit_mask)
        binary_mask = (mask != 0).float() # different classes can be just merged into a single one (person) as True and background as False
        """

        # split the color-encoded mask into a set
        # of binary masks
        masks = (mask == obj_ids[:, None, None]).to(dtype=torch.uint8)

        # get bounding box coordinates for each mask
        boxes = masks_to_boxes(masks)

        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)

        image_id = idx
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        # Wrap sample and targets into torchvision tv_tensors:
        img = tv_tensors.Image(img)

        target = {}
        target["boxes"] = tv_tensors.BoundingBoxes(boxes, format="XYXY", canvas_size=F.get_size(img))
        target["masks"] = tv_tensors.Mask(masks)
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)
    
    
def listdir_nohidden(path):
    for f in os.listdir(path):
        if not f.startswith('.'):
            yield f

class MOT16ObjDetectMasked(torch.utils.data.Dataset):
    """ Data class for the Multiple Object Tracking Dataset
    """

    def __init__(self, root, transforms=None, vis_threshold=0.25):
        self.root = root
        self.transforms = transforms
        self._vis_threshold = vis_threshold
        self._classes = ('background', 'pedestrian')
        self._img_paths = []
        ########################## CHANGED FOR THE CHALLENGE ##########################
        self._mask_paths = []
        ########################## CHANGED FOR THE CHALLENGE ##########################

        for f in listdir_nohidden(root):
            path = os.path.join(root, f)
            config_file = os.path.join(path, 'seqinfo.ini')

            assert os.path.exists(config_file), \
                'Path does not exist: {}'.format(config_file)

            config = configparser.ConfigParser()
            config.read(config_file)
            seq_len = int(config['Sequence']['seqLength'])
            im_ext = config['Sequence']['imExt']
            im_dir = config['Sequence']['imDir']
            ########################## CHANGED FOR THE CHALLENGE ##########################
            mask_dir = os.path.join(path, 'seg_ins')
            ########################## CHANGED FOR THE CHALLENGE ##########################

            _imDir = os.path.join(path, im_dir)

            for i in range(1, seq_len + 1):
                if os.path.exists(os.path.join(mask_dir)): # use only the directories with segmented data
                    img_path = os.path.join(_imDir, f"{i:06d}{im_ext}")
                    assert os.path.exists(img_path), \
                        f'Path does not exist: {img_path}'
                    self._img_paths.append(img_path)
                    ########################## CHANGED FOR THE CHALLENGE ##########################
                    mask_path = os.path.join(mask_dir, f"{i:06d}.png")
                    self._mask_paths.append(mask_path)
                    
            self._mask_paths = list(sorted(self._mask_paths))
            self._img_paths = list(sorted(self._img_paths))
            
            ########################## CHANGED FOR THE CHALLENGE ##########################

    @property
    def num_classes(self):
        return len(self._classes)

    def _get_annotation(self, idx):
   
        # get the image and the mask
        img_path = self._img_paths[idx]
        img_size = Image.open(img_path).size

        # get the bounding boxes 
        gt_file = os.path.join(os.path.dirname(os.path.dirname(img_path)), 'gt', 'gt.txt')
        bounding_boxes = []
        file_index = int(os.path.basename(img_path).split('.')[0])

        with open(gt_file, "r") as inf:
            reader = csv.reader(inf, delimiter=',')
            for row in reader:
                visibility = float(row[8])
                if int(row[0]) == file_index and int(row[6]) == 1 and int(row[7]) == 1 and visibility >= self._vis_threshold:
                    bb = {}
                    bb['bb_left'] = int(row[2])
                    bb['bb_top'] = int(row[3])
                    bb['bb_width'] = int(row[4])
                    bb['bb_height'] = int(row[5])
                    bb['visibility'] = visibility

                    bounding_boxes.append(bb)
                 
        # creating the masks from the bounding boxes   
        masks = []
        for bb in bounding_boxes:
            x0 = bb['bb_left']
            y0 = bb['bb_top']
            h = bb['bb_height']
            w = bb['bb_width']
            
            # convert W,H to x1,y1 -> (x0, y0, x1, y1)
            x1 = x0 + w
            y1 = y0 + h
            
            mask_img = Image.new('L', img_size, 0)
            ImageDraw.Draw(mask_img, 'L').rectangle([x0,y0,x1,y1], fill=(255))
            
            masks.append(mask_img)
        # once we have synthesized our masks, we cast them to tensors
        masks = torch.concat([tv_tensors.Mask(transforms.PILToTensor()(mask_img), dtype=torch.bool) for mask_img in masks])    
            
        num_objs = len(bounding_boxes)
        boxes = torch.zeros((num_objs, 4), dtype=torch.float32)
        visibilities = torch.zeros((num_objs), dtype=torch.float32)

        # creating the bounding boxes as tensors
        for i, bb in enumerate(bounding_boxes):
            # Make pixel indexes 0-based, should already be 0-based (or not)
            x1 = bb['bb_left'] - 1
            y1 = bb['bb_top'] - 1
            # This -1 accounts for the width (width of 1 x1=x2)
            x2 = x1 + bb['bb_width'] - 1
            y2 = y1 + bb['bb_height'] - 1

            boxes[i, 0] = x1
            boxes[i, 1] = y1
            boxes[i, 2] = x2
            boxes[i, 3] = y2
            visibilities[i] = bb['visibility']
            
        # there is only one class
        labels = torch.ones((num_objs, ), dtype=torch.int64)   
        
        image_id = idx
        area = (boxes[:,3] - boxes[:,1] * boxes[:,2] - boxes[:,0])   
        
        # suppose that all the labels are not crowd
        iscrowd = torch.zeros((num_objs, ), dtype=torch.int64)
        target = {}
        target['boxes'] = boxes
        target["masks"] = masks
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        return target

    def __getitem__(self, idx):
        # load images ad masks
        img_path = self._img_paths[idx]
        img = Image.open(img_path)

        target = self._get_annotation(idx)

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self._img_paths)