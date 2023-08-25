# SAM-Med2D

## Requirement
Please choose the appropriate version of PyTorch based on your CUDA version. The version configuration we are using is as follows:
- torch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1
- albumentations==1.3.0
- opencv-python==4.7.0.72
- Apex

## Dataset overview
SAM-Med2D was trained and tested on a dataset that includes **(4.6M images)** and **(19.7M masks)**. This dataset covers 10 medical data modalities, 4 anatomical structures + lesions, and 31 major human organs. To our knowledge, this is currently the largest and most diverse medical image segmentation dataset in terms of quantity and coverage of categories.

![Image text](https://github.com/uni-medical/SAM-Med2D/blob/main/images/dataset.png)

## SAM-Med2D overview
The pipeline of SAM-Med2D. We freeze the image encoder and incorporate learnable adapter layers in each Transformer block to acquire domain-specific knowledge in the medical field. We fine-tune the prompt encoder using point, Bbox, and mask information, while updating the parameters of the mask decoder through interactive training.

![Image text](https://github.com/uni-medical/SAM-Med2D/blob/main/images/farmework.png)


## Get Started

1. Download the model checkpoint. Place it under the `SAM-Med2D/pretrain_model/` folder

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom"></th>
<th valign="bottom">sam_vit_b</th>
<th valign="bottom">ft-sam_vit_b</th>
<th valign="bottom">sam-med2d_vit_b</th>
<!-- TABLE BODY -->
<tr><td align="left">pre-trained checkpoint</td>
<td align="center"><a href="https://drive.google.com/file/d/1_U26MIJhWnWVwmI5JkGg2cd2J6MvkqU-/view?usp=drive_link">download</a></td>
<td align="center"><a href="https://drive.google.com/file/d/1J4qQt9MZZYdv1eoxMTJ4FL8Fz65iUFM8/view?usp=drive_link">download</a></td>
<td align="center"><a href="https://drive.google.com/file/d/1ARiB5RkSsWmAB_8mqWnwDF8ZKTtFwsjl/view?usp=drive_link">download</a></td>
</tbody></table>

2. Prepare your own dataset and refer to the samples in `SAM-Med2D/Dataset_Demo` to replace them according to your specific scenario.

3. Fine-tuning based on pre-trained parameters.

```bash
python train.py
```
- work_dir: Specifies the working directory for the training process. Default value is "SAM-Med2D/workdir".
- image_size: Default value is 256.
- mask_num: Specify the number of masks corresponding to one image, with a default value of 5.
- data_path: Dataset directory, for example: `SAM-Med2D/Dataset_Demo`.
- resume: Pretrained weight file, ignore "sam_checkpoint" if present.
- sam_checkpoint: load sam checkpoint
- iter_point: Mask decoder iterative runs.
- multimask: Determines whether to output multiple masks. Default value is True.
- encoder_adapter: Whether to fine-tune the Adapter layer, set to False only for fine-tuning the decoder.

4. Get prediction result.

```bash
python test.py
```

- batch_size: 1.
- image_size: Default value is 256.
- boxes_prompt: Use Bbox prompt to get segmentation results. 
- point_num: Specifies the number of points. Default value is 1.
- iter_point: Specifies the number of iterations for point prompts.
- encoder_adapter: Set to True if using SAM-Med2D's pretrained weights.
- save_pred: Whether to save the prediction results.
- prompt_path: Is there a fixed prompt, if not generated at current prediction time.

5. Visualization demo

For more details see our paper.

![Image text](https://github.com/uni-medical/SAM-Med2D/blob/main/images/Results.png)


6. Jupyter-notebook

You can run it locally using predictor_example.ipynb, which is used to view the prediction results of different prompts

## Acknowledgements
- We are grateful to medical workers and dataset owners for making public datasets available to the community.
- Thanks to FAIR for open-sourcing their code: [segment anything](https://github.com/facebookresearch/segment-anything).

## Reference

```

```
