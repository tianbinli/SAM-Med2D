import gradio as gr
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import cv2
import torch
import os, sys
import warnings
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os, sys
from fastsam import FastSAM, FastSAMPrompt 
import random
from argparse import Namespace
from segment_anything.predictor_sammed import SammedPredictor
from segment_anything import sam_model_registry
from run_old import *

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


# points color and marker
colors = [(255, 0, 0), (0, 255, 0)]
markers = [1, 5]

# image examples
# in each list, the first element is image path,
# the second is id (used for original_image State),
# the third is an empty list (used for selected_points State)

with gr.Blocks() as demo:
    with gr.Row():
        gr.Markdown(
            '''# SAM-Med2D!🚀'''
            # SAM-Med2D produces high quality object masks from input prompts such as points or boxes, and it can be used to generate masks for al in an image. More information can be found in [**Official Project**](https://segment-anything.com/).
            # [![Duplicate this Space](https://huggingface.co/datasets/huggingface/badges/raw/main/duplicate-this-space-sm.svg)](https://huggingface.co/spaces/AIBoy1993/segment_anything_webui?duplicate=true)
            # '''
        )
        with gr.Row():
            # select model
            model_type = gr.Dropdown(["SAM-Med2D_base" ], value='SAM-Med2D_base', label="Select Model")
            # select compare model
    # Segment image
    with gr.Tab(label='Image'):
        with gr.Row().style(equal_height=True):
            with gr.Column():
                # input image
                original_image = gr.State(value=None)   # store original image without points, default None
                input_image = gr.Image(type="numpy")
                # point prompt
                with gr.Column():
                    selected_points = gr.State([])      # store points
                    last_mask = gr.State(None) 
                    last_mask_ = gr.State(None) 
                    last_mask = gr.State(None) 
                    last_mask = gr.State(None) 
                    last_mask = gr.State(None) 
                    with gr.Row():
                        gr.Markdown('You can click on the image to select points prompt. Default: foreground_point.')
                        undo_button = gr.Button('Undo point')
                    radio = gr.Radio(['foreground_point', 'background_point'], label='point labels')
                button = gr.Button("Auto!")
            # show the image with mask
            gallery_sammed = gr.Gallery(
                label="SAMMED Generated images", show_label=True, elem_id="gallery_sammed").style(preview=True, grid_cols=2,object_fit="scale-down")
    with gr.Row():
        with gr.Column():
            gallery_sam_b = gr.Gallery(
                label="SAM-B Generated images", show_label=True, elem_id="gallery_sam_b").style(preview=True, grid_cols=2,object_fit="scale-down")
        with gr.Column():
            gallery_sam_l = gr.Gallery(
                label="SAM-L Generated images", show_label=True, elem_id="gallery_sam_l").style(preview=True, grid_cols=2,object_fit="scale-down")
    with gr.Row():
        with gr.Column():
            gallery_hq_sam_l = gr.Gallery(
                label="HQ-SAM Generated images", show_label=True, elem_id="gallery_hq_sam_l").style(preview=True, grid_cols=2,object_fit="scale-down")
        with gr.Column():
            gallery_fast_sam = gr.Gallery(
                label="FastSAM Generated images", show_label=True, elem_id="gallery_fast_sam").style(preview=True, grid_cols=2,object_fit="scale-down")
    def process_example(img):
        return img, [], None
    
    all_imgs = [img for img in os.listdir("Dataset_Demo/random_select_5image/images/")]
    all_masks = [img for img in os.listdir("Dataset_Demo/random_select_5image/masks/")]
    img_mask_dict={}
    for img in all_imgs[:300]:
        basename = ".".join(img.split(".")[:-1])
        img_mask_dict[img] = [os.path.join("Dataset_Demo/random_select_5image/images/", img)]
        for mask_path in all_masks:
            if mask_path[:len(basename)] == basename:
                img_mask_dict[img].append(os.path.join("Dataset_Demo/random_select_5image/masks/", mask_path))


    for key,vals in img_mask_dict.items():
        with gr.Row():
            # input_examples = [os.path.join(demo_image_root_path, img) for img in os.listdir(demo_image_root_path)]
            with gr.Column():
                gr.Examples(examples=vals, label=None, inputs=[input_image], outputs=[original_image, selected_points,last_mask], fn=process_example, run_on_click=True, examples_per_page=15,)
    # with gr.Row():
    #     demo_image_root_path= "Dataset_Demo/images/"
    #     input_examples = [os.path.join(demo_image_root_path, img) for img in os.listdir(demo_image_root_path)]
    #     with gr.Column():
    #         gr.Examples(examples=image_examples, inputs=[input_image], outputs=[original_image, selected_points,last_mask], fn=process_example, run_on_click=True)


    # once user upload an image, the original image is stored in `original_image`
    def store_img(img):
        return img, [], None  # when new image is uploaded, `selected_points` should be empty
    input_image.upload(
        store_img,
        [input_image],
        [original_image, selected_points, last_mask]
    )

    # user click the image to get points, and show the points on the image
    def get_point(img, sel_pix, point_type, evt: gr.SelectData):
        if point_type == 'foreground_point':
            sel_pix.append((evt.index, 1))   # append the foreground_point
        elif point_type == 'background_point':
            sel_pix.append((evt.index, 0))    # append the background_point
        else:
            sel_pix.append((evt.index, 1))    # default foreground_point
        # draw points
        for point, label in sel_pix:
            cv2.drawMarker(img, point, colors[label], markerType=markers[label], markerSize=20, thickness=5)
        # if img[..., 0][0, 0] == img[..., 2][0, 0]:  # BGR to RGB
        #     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img if isinstance(img, np.ndarray) else np.array(img)
    input_image.select(
        get_point,
        [input_image, selected_points, radio],
        [input_image],
    )

    # undo the selected point
    def undo_points(orig_img, sel_pix):
        if isinstance(orig_img, int):   # if orig_img is int, the image if select from examples
            temp = cv2.imread(image_examples[orig_img][0])
            temp = cv2.cvtColor(temp, cv2.COLOR_BGR2RGB)
        else:
            temp = orig_img.copy()
        # draw points
        if len(sel_pix) != 0:
            sel_pix.pop()
            for point, label in sel_pix:
                cv2.drawMarker(temp, point, colors[label], markerType=markers[label], markerSize=20, thickness=5)
        if temp[..., 0][0, 0] == temp[..., 2][0, 0]:  # BGR to RGB
            temp = cv2.cvtColor(temp, cv2.COLOR_BGR2RGB)
        return temp if isinstance(temp, np.ndarray) else np.array(temp)
    undo_button.click(
        undo_points,
        [original_image, selected_points],
        [input_image]
    )
    segment_models = Segment_Serious_Models()
    # button image
    button.click(segment_models.run_sammed, inputs=[original_image, selected_points, last_mask],
                 outputs=[gallery_sammed, last_mask])\
    .then(fn=segment_models.run_sam_b, inputs=[original_image, selected_points], outputs=gallery_sam_b)\
    .then(fn=segment_models.run_sam_l, inputs=[original_image, selected_points], outputs=gallery_sam_l)\
    .then(fn=segment_models.run_hq_sam_l, inputs=[original_image, selected_points], outputs=gallery_hq_sam_l)\
    .then(fn=segment_models.run_fast_sam, inputs=[original_image, selected_points], outputs=gallery_fast_sam)\
    # .then(fn=segment_models.run_hq_sam, inputs=[original_image, selected_points], outputs=gallery_hq_sam)\
    # .then(fn=segment_models.run_sam_h, inputs=[original_image, selected_points], outputs=gallery_sam_h)\


import socket
start_port=5901
while True:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('0.0.0.0', start_port))
            break
    except OSError:
        start_port += 1
demo.queue().launch(debug=True, server_name='0.0.0.0', server_port=start_port)