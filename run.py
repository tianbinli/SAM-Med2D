import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os, sys
from scipy import ndimage
from FastSAM.fastsam import FastSAM, FastSAMPrompt 
 #torch.device("cuda" if torch.cuda.is_available() else "cpu")
import torch
import random
from argparse import Namespace
from segment_anything.predictor_sammed import SammedPredictor
from segment_anything import sam_model_registry

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
    for p in point:
        x,y = p
        draw.ellipse((x-r, y-r, x+r, y+r), fill='green')

def load_model(img_size, encoder_adapter, model_type, ckpt, device="cuda:0"):
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
    def __init__(self, ):
        self.sam_med2d_b = load_model(256, True, "vit_b", "pretrain_model/sam-med2d_b.pth", "cuda:1")
        self.sam_vit_b = load_model(1024, False, 'vit_b', "SAM_facebook/sam_vit_b_01ec64.pth", "cuda:1")
        self.fast_sam = FastSAM("FastSAM-x.pt")
        self.fast_sam_device="cuda:1"
        
        self.sam_vit_l = load_model(1024, False, 'vit_l', "SAM_facebook/sam_vit_l_0b3195.pth", "cuda:1")
        # self.sam_vit_h = load_model(1024, False, 'vit_h', "SAM_facebook/sam_vit_h_4b8939.pth", "cuda:1")
        self.sam_hq_vit_l = load_model(1024, False, "vit_l", "sam_hq_vit_l.pth", "cuda:0")
        # self.sam_hq_vit_h = load_model(1024, False, "vit_h", "sam_hq_vit_h.pth", "cuda:0")

    def run_sammed(self, input_image, task_type):
        # load image
        image_pil = input_image["image"].convert("RGB")
        image = np.array(image_pil)
        H,W,_ = image.shape
        predictor = self.sam_med2d_b
        predictor.set_image(image)
        scribble = np.array(input_image["mask"])
        if task_type == 'scribble_point':
            scribble = scribble.transpose(2, 1, 0)[0]
            labeled_array, num_features = ndimage.label(scribble >= 255)
            centers = ndimage.center_of_mass(scribble, labeled_array, range(1, num_features+1))
            centers = np.array(centers)
            point_coords = centers
            point_labels = np.ones(point_coords.shape[0])

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

        else:
            print("task_type:{} error!".format(task_type))


    def run_sam_b(self, input_image, task_type):
        image_pil = input_image["image"].convert("RGB")
        image = np.array(image_pil)
        H,W,_ = image.shape
        predictor = self.sam_vit_b
        predictor.set_image(image)

        scribble = np.array(input_image["mask"])
        if task_type == 'scribble_point':
            scribble = scribble.transpose(2, 1, 0)[0]
            labeled_array, num_features = ndimage.label(scribble >= 255)
            centers = ndimage.center_of_mass(scribble, labeled_array, range(1, num_features+1))
            centers = np.array(centers)
            point_coords = centers
            point_labels = np.ones(point_coords.shape[0])

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

        else:
            print("task_type:{} error!".format(task_type))

    def run_sam_l(self, input_image, task_type):
        image_pil = input_image["image"].convert("RGB")
        image = np.array(image_pil)
        H,W,_ = image.shape
        predictor = self.sam_vit_b
        predictor.set_image(image)

        scribble = np.array(input_image["mask"])
        if task_type == 'scribble_point':
            scribble = scribble.transpose(2, 1, 0)[0]
            labeled_array, num_features = ndimage.label(scribble >= 255)
            centers = ndimage.center_of_mass(scribble, labeled_array, range(1, num_features+1))
            centers = np.array(centers)
            point_coords = centers
            point_labels = np.ones(point_coords.shape[0])

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

        else:
            print("task_type:{} error!".format(task_type))


    def run_sam_h(self, input_image, task_type):
        image_pil = input_image["image"].convert("RGB")
        image = np.array(image_pil)
        H,W,_ = image.shape
        predictor = self.sam_vit_h
        predictor.set_image(image)

        scribble = np.array(input_image["mask"])
        if task_type == 'scribble_point':
            scribble = scribble.transpose(2, 1, 0)[0]
            labeled_array, num_features = ndimage.label(scribble >= 255)
            centers = ndimage.center_of_mass(scribble, labeled_array, range(1, num_features+1))
            centers = np.array(centers)
            point_coords = centers
            point_labels = np.ones(point_coords.shape[0])

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

        else:
            print("task_type:{} error!".format(task_type))

    def run_hq_sam_l(self, input_image, task_type):
        image_pil = input_image["image"].convert("RGB")
        image = np.array(image_pil)
        H,W,_ = image.shape
        predictor = self.sam_hq_vit_l
        predictor.set_image(image)

        scribble = np.array(input_image["mask"])
        if task_type == 'scribble_point':
            scribble = scribble.transpose(2, 1, 0)[0]
            labeled_array, num_features = ndimage.label(scribble >= 255)
            centers = ndimage.center_of_mass(scribble, labeled_array, range(1, num_features+1))
            centers = np.array(centers)
            point_coords = centers
            point_labels = np.ones(point_coords.shape[0])

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

        else:
            print("task_type:{} error!".format(task_type))

    def run_hq_sam_h(self, input_image, task_type):
        image_pil = input_image["image"].convert("RGB")
        image = np.array(image_pil)
        H,W,_ = image.shape
        predictor = self.sam_hq_vit_h
        predictor.set_image(image)

        scribble = np.array(input_image["mask"])
        if task_type == 'scribble_point':
            scribble = scribble.transpose(2, 1, 0)[0]
            labeled_array, num_features = ndimage.label(scribble >= 255)
            centers = ndimage.center_of_mass(scribble, labeled_array, range(1, num_features+1))
            centers = np.array(centers)
            point_coords = centers
            point_labels = np.ones(point_coords.shape[0])

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

        else:
            print("task_type:{} error!".format(task_type))

    def run_fast_sam(self, input_image, task_type):
        image_pil = input_image["image"].convert("RGB")
        predictor = self.fast_sam
        H,W = image_pil.size
        if hasattr(self, "fast_sam_device"):
            device=self.fast_sam_device
        else:
            device="cuda:0"
        
        everything_results = predictor(
            image_pil,
            device=device,
            retina_masks=True,
            imgsz=1024,
            conf=0.4,
            iou=0.9    
            )

        scribble = np.array(input_image["mask"])
        if task_type == 'scribble_point':
            scribble = scribble.transpose(2, 1, 0)[0]
            labeled_array, num_features = ndimage.label(scribble >= 255)
            centers = ndimage.center_of_mass(scribble, labeled_array, range(1, num_features+1))
            centers = np.array(centers)
            point_coords = centers
            point_labels = np.ones(point_coords.shape[0])
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

        else:
            print("task_type:{} error!".format(task_type))
    

