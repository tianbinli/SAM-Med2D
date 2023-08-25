from segment_anything import sam_model_registry, SamPredictor
import torch.nn as nn
import torch
import argparse
import os
import time
from tqdm import tqdm
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torch.nn import functional as F
from thop import clever_format, profile
from torchsummary import summary




def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=1, help="batch size")
    parser.add_argument("--image_size", type=int, default=256, help="image_size")
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument("--resume", type=str, default="SAM-Med2D/pretrain_model/sam-med2d_b.pth", help="load resume")
    parser.add_argument("--model_type", type=str, default="vit_b", help="sam model_type")
    parser.add_argument("--sam_checkpoint", type=str, default="SAM-Med2D/pretrain_model/sam_vit_b.pth", help="sam checkpoint")
    parser.add_argument("--multimask", type=bool, default=True, help="ouput multimask")
    parser.add_argument("--encoder_adapter", type=bool, default=True, help="use adapter")
    args = parser.parse_args()
    if args.resume is not None:
        args.sam_checkpoint = None
    return args

def to_device(batch_input, rank):
    device_input = {}
    for key, value in batch_input.items():
        if value is not None:
            if key=='image' or key=='label':
                device_input[key] = value.float().to(rank)
            elif type(value) is list or type(value) is torch.Size:
                 device_input[key] = value
            else:
                device_input[key] = value.to(rank)
        else:
            device_input[key] = value
            
    return device_input


def postprocess_masks(low_res_masks, image_size, original_size):
    ori_h, ori_w = original_size
    masks = F.interpolate(
        low_res_masks,
        (image_size, image_size),
        mode="bilinear",
        align_corners=False,
        )
    
    if ori_h < image_size and ori_w < image_size:
        top = (image_size - ori_h) // 2
        left = (image_size - ori_w) // 2
        masks = masks[..., top : ori_h + top, left : ori_w + left]
        pad = (top, left)
    else:
        masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
        pad = None 
    return masks, pad

def prompt_decoder(args, batched_input, ddp_model, image_embeddings, multimask = False):
    if  batched_input["point_coords"] is not None:
        points = (batched_input["point_coords"], batched_input["point_labels"])
    else:
        points = None

    with torch.no_grad():
        sparse_embeddings, dense_embeddings = ddp_model.prompt_encoder(
            points=points,
            boxes=batched_input.get("boxes", None),
            masks=batched_input.get("mask_inputs", None),
        )

        low_res_masks, iou_predictions = ddp_model.mask_decoder(
            image_embeddings = image_embeddings,
            image_pe = ddp_model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask,
        )
    
    if multimask:
        max_values, max_indexs = torch.max(iou_predictions, dim=1)
        max_values = max_values.unsqueeze(1)
        iou_predictions = max_values
        low_res = []
        for i, idx in enumerate(max_indexs):
            low_res.append(low_res_masks[i:i+1, idx])
        low_res_masks = torch.stack(low_res, 0)
    masks = F.interpolate(low_res_masks,(args.image_size, args.image_size), mode="bilinear", align_corners=False,)
    return masks, low_res_masks, iou_predictions, points


def setting_prompt_none(batched_input):
    batched_input["point_coords"] = None
    batched_input["point_labels"] = None
    batched_input["boxes"] = None
    # batched_input["mask_inputs"] = None
    return batched_input


def main(args):

    print('*'*100)
    for key, value in vars(args).items():
        print(key + ': ' + str(value))
    print('*'*100)

    batched_input = {}
    input_data = torch.randn(1, 3, args.image_size, args.image_size).to(args.device) 
    batched_input["image"] = input_data
    batched_input["point_coords"] = None
    batched_input["point_labels"] = None
    batched_input["boxes"] = None
    batched_input = to_device(batched_input, args.device)
    model = sam_model_registry[args.model_type](args).to(args.device) 
    model_encoder = model.image_encoder
    model_prompt = model.prompt_encoder
    model_decoder = model.mask_decoder
    summary(model_encoder, input_size=(3, args.image_size, args.image_size))
    flops, params   = profile(model_encoder, (input_data, ), verbose=False)
    #--------------------------------------------------------#
    #   flops * 2是因为profile没有将卷积作为两个operations
    #   有些论文将卷积算乘法、加法两个operations。此时乘2
    #   有些论文只考虑乘法的运算次数，忽略加法。此时不乘2
    #   本代码选择乘2，参考YOLOX。
    #--------------------------------------------------------#
    flops           = flops * 2
    flops, params   = clever_format([flops, params], "%.4f")
    print(f'Total GFLOPS: {flops}')
    print(f'Total params: {params}')

    model.eval()
    num_iterations = 100  # 运行前向传播的总次数
    total_time = 0
    for _ in range(num_iterations):
        start_time = time.time()
        with torch.no_grad():
            image_embeddings = model.image_encoder(batched_input["image"])
            masks, low_res_masks, iou_predictions, points = prompt_decoder(args, batched_input, model, image_embeddings, multimask = args.multimask)
        end_time = time.time()
        total_time += end_time - start_time

    fps = num_iterations / total_time
    print("FPS: {:.2f}".format(fps))

if __name__ == '__main__':
    args = parse_args()
    main(args)

    

