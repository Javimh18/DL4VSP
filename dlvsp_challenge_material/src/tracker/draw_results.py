import torch
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks
from torchvision.io import read_image
from torchvision.transforms import v2 as T
import sys
import os


sys.path.append(os.path.join(os.getcwd(), ".."))
from object_detector import FRCNN_FPN, ObjDetector

#image = read_image("../../fine_tuning_test/data/PennFudanPed/PNGImages/FudanPed00046.png")
image = read_image("../../data/MOT16/train/MOT16-04/img1/000001.jpg")

# obj_detect_model_file = os.path.join("../../models/maskrcnn_model.pth")
obj_detect_model_file = os.path.join("../../models/finetuned_masked_rcnn.model")
def get_transform(train):
    transforms = []
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    transforms.append(T.ToDtype(torch.float, scale=True))
    transforms.append(T.ToPureTensor())
    return T.Compose(transforms)

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    eval_transform = get_transform(train=False)
    obj_detect_nms_thresh = 0.3
    
    
    model = ObjDetector(nms_thresh=0.5).model
    obj_detect_state_dict = torch.load(obj_detect_model_file,map_location=lambda storage, loc: storage)
    model.load_state_dict(obj_detect_state_dict)
    model.to(device)
    
    
    """
    model = FRCNN_FPN(num_classes=2, nms_thresh=obj_detect_nms_thresh)
    obj_detect_state_dict = torch.load(obj_detect_model_file,map_location=lambda storage, loc: storage)
    model.load_state_dict(obj_detect_state_dict)
    model.to(device)
    """

    model.eval()
    with torch.no_grad():
        x = eval_transform(image)
        # convert RGBA -> RGB and move to device
        x = x[:3, ...].to(device)
        predictions = model([x, ])
        pred = predictions[0]


    image = (255.0 * (image - image.min()) / (image.max() - image.min())).to(torch.uint8)
    image = image[:3, ...]
    pred_labels = [f"pedestrian: {score:.3f}" for label, score in zip(pred["labels"], pred["scores"])]
    pred_boxes = pred["boxes"].long()
    output_image = draw_bounding_boxes(image, pred_boxes, pred_labels, colors="red")

    output_image = T.ToPILImage()(output_image).save("obj_detect_results/res_finetuned_masked_000699.jpg")
    
    