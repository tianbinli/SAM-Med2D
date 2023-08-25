import os
import subprocess
subprocess.run("cd /home/xlab-app-center/FastSAM/ && pip install -e .", shell=True)
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os, sys
from scipy import ndimage
import torch
import random
from argparse import Namespace
from segment_anything.predictor_sammed import SammedPredictor
from segment_anything import sam_model_registry
import openxlab
from openxlab.model import download
try:
    openxlab.login(ak="k76vxnebvv058mggrmz1", sk="ygne5jobler7a4z0v3ddvxm91dwk8vzmg32obqnv")
except:
    pass
import json

data = {
    "ak": "k76vxnebvv058mggrmz1",
    "sk": "ygne5jobler7a4z0v3ddvxm91dwk8vzmg32obqnv"
}
os.system("mkdir ~/.openxlab")
with open('~/.openxlab/config.json', 'w') as file:
    json.dump(data, file)
os.system("mkdir /home/xlab-app-center/pretrain_model")
os.system("cd /home/xlab-app-center/pretrain_model && wget https://download.openxlab.org.cn/models/litianbin/SAM-Med2D/weight/sam-med2d_b.pth")
os.system("cd /home/xlab-app-center/pretrain_model && wget https://download.openxlab.org.cn/models/litianbin/SAM-Med2D/weight/sam_vit_b_01ec64.pth")
os.system("cd /home/xlab-app-center/pretrain_model && wget https://download.openxlab.org.cn/models/litianbin/SAM-Med2D/weight/sam_vit_h_4b8939.pth")
os.system("cd /home/xlab-app-center/pretrain_model && wget https://download.openxlab.org.cn/models/litianbin/SAM-Med2D/weight/sam_vit_l_0b3195.pth")
os.system("cd /home/xlab-app-center/pretrain_model && wget https://download.openxlab.org.cn/models/litianbin/SAM-Med2D/weight/FastSAM-x.pt")
os.system("cd /home/xlab-app-center/pretrain_model && wget https://download.openxlab.org.cn/models/litianbin/SAM-Med2D/weight/sam_hq_vit_l.pth")
os.system("cd /home/xlab-app-center/pretrain_model && wget https://download.openxlab.org.cn/models/litianbin/SAM-Med2D/weight/sam_hq_vit_h.pth")

def draw_mask(mask, draw, random_color=False):
    if random_color:
        color = (random.randint(0, 255), random.randint(
            0, 255), random.randint(0, 255), 153)
    else:
        color = (30, 144, 255, 153)

    nonzero_coords = np.transpose(np.nonzero(mask))

    for coord in nonzero_coords:
        draw.point(coord[::-1], fill=color)


def draw_point(point, draw, r=5):
    show_point = []
    for point, label in point:
        x,y = point
        if label == 1:
            draw.ellipse((x-r, y-r, x+r, y+r), fill='green')
        elif label == 0:
            draw.ellipse((x-r, y-r, x+r, y+r), fill='red')

def load_model(img_size, encoder_adapter, model_type, ckpt, device="cuda"):
    args = Namespace()
    args.device = device
    args.image_size = img_size
    args.encoder_adapter = encoder_adapter
    args.sam_checkpoint = ckpt  #sam_vit_b.pth  sam-med2d_b.pth
    model = sam_model_registry[model_type](args).to(args.device)
    model.eval()
    predictor = SammedPredictor(model)
    return predictor

class Segment_Serious_Models():
    def __init__(self, device0="cuda", device1="cuda"):
        self.device0 = device0
        self.device1 = device1
        self.sam_med2d_b_device = device1
        self.fast_sam_device=device1

    def get_model(self, model_name):
        if not hasattr(self, model_name):
            if model_name == "sam_med2d_b":
                self.sam_med2d_b = load_model(256, True, "vit_b", "/home/xlab-app-center/pretrain_model/sam-med2d_b.pth", self.device1)
            elif model_name == "sam_vit_b":
                self.sam_vit_b = load_model(1024, False, 'vit_b', "/home/xlab-app-center/pretrain_model/sam_vit_b_01ec64.pth", self.device1)
            elif model_name == "sam_vit_l":
                self.sam_vit_l = load_model(1024, False, 'vit_l', "/home/xlab-app-center/pretrain_model/sam_vit_l_0b3195.pth", self.device1)
            elif model_name == "sam_vit_h":
                self.sam_vit_h = load_model(1024, False, 'vit_h', "/home/xlab-app-center/pretrain_model/sam_vit_h_0b3195.pth", self.device1)
            elif model_name == "fast_sam":
                from fastsam import FastSAM, FastSAMPrompt 
                self.fast_sam = FastSAM("/home/xlab-app-center/pretrain_model/FastSAM-x.pt")
            elif model_name == "sam_hq_vit_l":
                self.sam_hq_vit_l = load_model(1024, False, "vit_l", "/home/xlab-app-center/pretrain_model/sam_hq_vit_l.pth", self.device0)
            elif model_name == "sam_hq_vit_h":
                self.sam_hq_vit_h = load_model(1024, False, "vit_h", "/home/xlab-app-center/pretrain_model/sam_hq_vit_h.pth", self.device0)
            else:
                print("no model_name : " + str(model_name))
        else:
            return getattr(self, model_name)

    def run_sammed(self, input_image, selected_points, last_mask):
        image_pil = Image.fromarray(input_image)#.convert("RGB")
        image = input_image
        H,W,_ = image.shape
        predictor = self.get_model("sam_med2d_b")
        predictor.set_image(image)
        centers = np.array([a for a,b in selected_points ])
        point_coords = centers
        point_labels = np.array([b for a,b in selected_points ])

        masks, _, logits = predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        mask_input = last_mask,
        multimask_output=True 
        ) 

        mask_image = Image.new('RGBA', (H,W), color=(0, 0, 0, 0))
        mask_draw = ImageDraw.Draw(mask_image)
        for mask in masks:
            draw_mask(mask, mask_draw, random_color=False)
        image_draw = ImageDraw.Draw(image_pil)

        draw_point(point_labels,image_draw)

        image_pil = image_pil.convert('RGBA')
        image_pil.alpha_composite(mask_image)
        last_mask = torch.sigmoid(torch.as_tensor(logits, dtype=torch.float, device=self.sam_med2d_b_device))
        return [(image_pil, mask_image), last_mask]


    def run_sam_b(self, input_image, selected_points):
        image_pil = Image.fromarray(input_image)
        image = input_image
        H,W,_ = image.shape
        predictor = self.get_model("sam_vit_b")
        predictor.set_image(image)
        centers = np.array([a for a,b in selected_points ])
        point_coords = centers
        point_labels = np.array([b for a,b in selected_points ])

        masks, _, _ = predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        multimask_output=True 
        ) 
        mask_image = Image.new('RGBA', (H,W), color=(0, 0, 0, 0))
        mask_draw = ImageDraw.Draw(mask_image)
        for mask in masks:
            draw_mask(mask, mask_draw, random_color=False)
        image_draw = ImageDraw.Draw(image_pil)
        draw_point(point_coords,image_draw)
        image_pil = image_pil.convert('RGBA')
        image_pil.alpha_composite(mask_image)
        return [image_pil, mask_image]

    def run_sam_l(self, input_image, selected_points):
        image_pil = Image.fromarray(input_image)
        image = input_image
        H,W,_ = image.shape
        predictor = self.get_model("sam_vit_b")
        predictor.set_image(image)
        centers = np.array([a for a,b in selected_points ])
        point_coords = centers
        point_labels = np.array([b for a,b in selected_points ])

        masks, _, _ = predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        multimask_output=True 
        ) 
        mask_image = Image.new('RGBA', (H,W), color=(0, 0, 0, 0))
        mask_draw = ImageDraw.Draw(mask_image)
        for mask in masks:
            draw_mask(mask, mask_draw, random_color=False)
        image_draw = ImageDraw.Draw(image_pil)
        draw_point(point_coords,image_draw)
        image_pil = image_pil.convert('RGBA')
        image_pil.alpha_composite(mask_image)
        return [image_pil, mask_image]


    def run_sam_h(self, input_image, selected_points):
        image_pil = Image.fromarray(input_image)
        image = input_image
        H,W,_ = image.shape
        predictor = self.get_model("sam_vit_h")
        predictor.set_image(image)
        centers = np.array([a for a,b in selected_points ])
        point_coords = centers
        point_labels = np.array([b for a,b in selected_points ])

        masks, _, _ = predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        multimask_output=True 
        ) 
        mask_image = Image.new('RGBA', (H,W), color=(0, 0, 0, 0))
        mask_draw = ImageDraw.Draw(mask_image)
        for mask in masks:
            draw_mask(mask, mask_draw, random_color=False)
        image_draw = ImageDraw.Draw(image_pil)
        draw_point(point_coords,image_draw)
        image_pil = image_pil.convert('RGBA')
        image_pil.alpha_composite(mask_image)
        return [image_pil, mask_image]

    def run_hq_sam_l(self, input_image, selected_points):
        image_pil = Image.fromarray(input_image)
        image = input_image
        H,W,_ = image.shape
        predictor = self.get_model("sam_hq_vit_l")
        predictor.set_image(image)
        centers = np.array([a for a,b in selected_points ])
        point_coords = centers
        point_labels = np.array([b for a,b in selected_points ])

        masks, _, _ = predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        multimask_output=True 
        ) 
        mask_image = Image.new('RGBA', (H,W), color=(0, 0, 0, 0))
        mask_draw = ImageDraw.Draw(mask_image)
        for mask in masks:
            draw_mask(mask, mask_draw, random_color=False)
        image_draw = ImageDraw.Draw(image_pil)
        draw_point(point_coords,image_draw)
        image_pil = image_pil.convert('RGBA')
        image_pil.alpha_composite(mask_image)
        return [image_pil, mask_image]

    def run_hq_sam_h(self, input_image, selected_points):
        image_pil = Image.fromarray(input_image)
        image = input_image
        H,W,_ = image.shape
        predictor = self.get_model("sam_hq_vit_h")
        predictor.set_image(image)
        centers = np.array([a for a,b in selected_points ])
        point_coords = centers
        point_labels = np.array([b for a,b in selected_points ])

        masks, _, _ = predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        multimask_output=True 
        ) 
        mask_image = Image.new('RGBA', (H,W), color=(0, 0, 0, 0))
        mask_draw = ImageDraw.Draw(mask_image)
        for mask in masks:
            draw_mask(mask, mask_draw, random_color=False)
        image_draw = ImageDraw.Draw(image_pil)
        draw_point(point_coords,image_draw)
        image_pil = image_pil.convert('RGBA')
        image_pil.alpha_composite(mask_image)
        return [image_pil, mask_image]

    def run_fast_sam(self, input_image, selected_points):
        image_pil = Image.fromarray(input_image)
        predictor = self.get_model("fast_sam")
        H,W = image_pil.size
        if hasattr(self, "fast_sam_device"):
            device=self.fast_sam_device
        else:
            device="cuda"
        
        everything_results = predictor(
            image_pil,
            device=device,
            retina_masks=True,
            imgsz=1024,
            conf=0.4,
            iou=0.9    
            )
        centers = np.array([a for a,b in selected_points ])
        point_coords = centers
        point_labels = np.array([b for a,b in selected_points ])
        prompt_process = FastSAMPrompt(image_pil, everything_results, device=device)
        point_labels= point_labels.tolist()
        point_coords= point_coords.tolist()
        annotations = prompt_process.point_prompt(points=point_coords, pointlabel=point_labels)
        if isinstance(annotations[0], dict):
            annotations = [annotation['segmentation'] for annotation in annotations]
        if isinstance(annotations, torch.Tensor):
            annotations = annotations.cpu().numpy()
        # if isinstance(annotations[0], np.ndarray):
        #         annotations = torch.from_numpy(annotations)
        mask_image = Image.new('RGBA', (H,W), color=(0, 0, 0, 0))
        mask_draw = ImageDraw.Draw(mask_image)
        for mask in annotations:
            draw_mask(mask, mask_draw, random_color=False)
        image_draw = ImageDraw.Draw(image_pil)
        draw_point(point_coords,image_draw)
        image_pil = image_pil.convert('RGBA')
        image_pil.alpha_composite(mask_image)
        return [image_pil, mask_image]
    

