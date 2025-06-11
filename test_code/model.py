import torch
from torchvision import transforms
import torch.nn as nn
import numpy as np
from Unet import *

class Model(object):
    def __init__(self, model_path, device='cpu'):
        self.device = torch.device(device)
        self.model = AttU_Net(img_ch= 3, output_ch=11).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

        self.BOX_COORD_NUM = 4
        self.score_threshold = 0.6
        self.keypoint_count = 0

    def prepare(self):
        return None

    def predict(self, image):
        """
        image: numpy array with shape (H, W, C), RGB
        """

        #Giả sử `image` là một đối tượng PIL.Image
        transform = transforms.Compose([
            transforms.ToTensor(),  # Chuyển ảnh PIL (HWC, 0–255) -> tensor (CHW, 0–1)
            transforms.Normalize(mean=[0.485, 0.456, 0.406],  # giống input_mean
                         std=[0.229, 0.224, 0.225])   # giống input_std
        ])

        input_tensor = transform(image).unsqueeze(0).to(self.device) # Them batch dimension (1, C, H, W)

        with torch.no_grad():
            output_tensor = self.model(input_tensor) # (1, C, H, W)

        output_data = output_tensor.squeeze(0).cpu().numpy().transpose(1,2,0) # HWC 
        return output_data
