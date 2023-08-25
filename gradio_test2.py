import gradio as gr
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import cv2
import torch
import torchvision
import os, sys
import warnings
from scipy import ndimage
from run import *

segment_models = Segment_Serious_Models()
block = gr.Blocks()
block = block.queue()
with block:
    with gr.Row():
        with gr.Column():
            # input_image = gr.Image(source='upload', type="pil", value="Dataset_Demo/images/amos_0507_31.png", tool="sketch", brush_radius=10)
            input_image = gr.Image(source='upload', type="pil", value="Dataset_Demo/images/amos_0507_31.png", tool="sketch", brush_radius=10, brush_color="#00ffee")
            with gr.Row():
                task_type = gr.Dropdown(["scribble_point"], value="scribble_point", label="task_type")
                label_type = gr.Gradio(["positive_point", "negative_point"], value="positive_point", label="label_type")
            run_button = gr.Button(label="Run")
        with gr.Column():
            gallery_sammed = gr.Gallery(
                label="SAMMED Generated images", show_label=True, elem_id="gallery_sammed").style(preview=True, grid=2,object_fit="scale-down")
    with gr.Row():
        with gr.Column():
            gallery_sam_b = gr.Gallery(
                label="SAM-B Generated images", show_label=True, elem_id="gallery_sam_b").style(preview=True, grid=2,object_fit="scale-down")
        with gr.Column():
            gallery_sam_l = gr.Gallery(
                label="SAM-L Generated images", show_label=True, elem_id="gallery_sam_l").style(preview=True, grid=2,object_fit="scale-down")
    with gr.Row():
        with gr.Column():
            gallery_hq_sam_l = gr.Gallery(
                label="HQ-SAM Generated images", show_label=True, elem_id="gallery_hq_sam_l").style(preview=True, grid=2,object_fit="scale-down")
        with gr.Column():
            gallery_fast_sam = gr.Gallery(
                label="FastSAM Generated images", show_label=True, elem_id="gallery_fast_sam").style(preview=True, grid=2,object_fit="scale-down")

    
    with gr.Row():
        demo_image_root_path= "Dataset_Demo/images/"
        input_examples = [os.path.join(demo_image_root_path, img) for img in os.listdir(demo_image_root_path)]
        with gr.Column():
            gr.Examples(input_examples, inputs=input_image)

    # run_button.click(fn=segment_models.run_sam_h, inputs=[input_image, task_type], outputs=gallery_sam_h)\
    run_button.click(fn=segment_models.run_sammed, inputs=[input_image, task_type], outputs=gallery_sammed)\
    .then(fn=segment_models.run_sam_b, inputs=[input_image, task_type], outputs=gallery_sam_b)\
    .then(fn=segment_models.run_fast_sam, inputs=[input_image, task_type], outputs=gallery_fast_sam)\
    .then(fn=segment_models.run_sam_l, inputs=[input_image, task_type], outputs=gallery_sam_l)\
    .then(fn=segment_models.run_hq_sam_l, inputs=[input_image, task_type], outputs=gallery_hq_sam_l)\
    # .then(fn=segment_models.run_hq_sam, inputs=[input_image, task_type], outputs=gallery_hq_sam)\
    # .then(fn=segment_models.run_sam_h, inputs=[input_image, task_type], outputs=gallery_sam_h)\
    
    #.then(fn=run_sammed, inputs=[input_image, task_type], outputs=gallery)

#block.launch(debug=True, share=True, show_error=True)
import socket
start_port=20001
while True:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('0.0.0.0', start_port))
            break
    except OSError:
        start_port += 1
block.queue(api_open=False,max_size=5).launch(debug=True, server_name='0.0.0.0', server_port=start_port)