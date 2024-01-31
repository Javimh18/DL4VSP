import torch
from torchvision import transforms
from PIL import Image
import time
import numpy as np

class BoundingBoxCrop(transforms.Compose):
    def __init__(self, bounding_boxes):
        self.bounding_boxes = bounding_boxes
        transforms_list = []
        super(BoundingBoxCrop, self).__init__(transforms_list)

    def crop_box(self, image, box):
        x0, y0, x1, y1 = box
        return image.crop((x0, y0, x1, y1))

    def __call__(self, image):
        crops = list(map(lambda box: self.crop_box(image, box), self.bounding_boxes))
        crop_tensors = list(map(lambda crop: transforms.ToTensor()(crop), crops))
        crops = torch.stack(crop_tensors, dim=0)

        return crops
    
def get_crops(frame, bounding_boxes):
    crops = []
    for bbox in bounding_boxes:
        crops.append(frame.crop(bbox))
      
    crops_tensors = []  
    for crop in crops:
        crops_tensors.append(transforms.ToTensor()(crop))
        
    return torch.stack(crops_tensors, dim=0)

if __name__ == '__main__':
    # Ejemplo de uso
    image_path = 'data/MOT16/test/MOT16-95/img1/000001.jpg'
    bounding_boxes = np.array([
        [100, 150, 300, 350],
        [200, 250, 400, 450],
        [100, 150, 300, 350],
        [200, 250, 400, 450],
        [100, 150, 300, 350],
        [200, 250, 400, 450],
        [100, 150, 300, 350],
        [200, 250, 400, 450],
        [100, 150, 300, 350],
        [200, 250, 400, 450],
        [100, 150, 300, 350],
        [200, 250, 400, 450],
        [100, 150, 300, 350],
        [200, 250, 400, 450],
        [100, 150, 300, 350],
        [200, 250, 400, 450],
        [100, 150, 300, 350],
        [200, 250, 400, 450],
        [100, 150, 300, 350],
        [200, 250, 400, 450]
    ])

    image = Image.open(image_path)
    start_t_map = time.time()
    custom_transform = BoundingBoxCrop(bounding_boxes)
    result_map = custom_transform(image)
    print("Map result shape:", result_map.shape)
    print("Map Time: ", time.time() - start_t_map)
    
    start_t_loop = time.time()
    result_loop = get_crops(image, bounding_boxes)
    print("Loop result shape:", result_map.shape)
    print("Loop Time: ", time.time() - start_t_loop)
    
