"""
COCO dataset which returns image_id for evaluation.
"""
import json
import os
import random
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.utils.data
import torchvision
from PIL import Image
from pycocotools import mask as coco_mask
from pycocotools.coco import COCO
from util.box_ops import box_cxcywh_to_xyxy, box_iou
import datasets.transforms_coco as T
__all__ = ['build']

class CocoDetection(torch.utils.data.Dataset):
    def __init__(self, root_path, image_set, transforms, return_masks):
        if image_set not in {'train', 'val'}:
            raise ValueError('COCO quantitative evaluation uses val2017; use infer.py for unannotated test images')
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(return_masks)
        split = 'train2017' if image_set == 'train' else 'val2017'
        self.img_folder = root_path / split
        self.coco = COCO(root_path / f'annotations/person_keypoints_{split}.json')
        self.all_imgIds = sorted(self.coco.getImgIds())
        if image_set == 'train':
            self.all_imgIds = [image_id for image_id in self.all_imgIds
                               if any(a.get('num_keypoints', 0) > 0 for a in
                                      self.coco.loadAnns(self.coco.getAnnIds(imgIds=image_id)))]

    def __len__(self):
        return len(self.all_imgIds)

    def __getitem__(self, idx):
        image_id = self.all_imgIds[idx]
        annotations = self.coco.loadAnns(self.coco.getAnnIds(imgIds=image_id))
        image = Image.open(self.img_folder / self.coco.loadImgs(image_id)[0]['file_name']).convert('RGB')
        image, target = self.prepare(image, {'image_id': image_id, 'annotations': annotations})
        if self._transforms is not None:
            image, target = self._transforms(image, target)
        return image, target


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


class ConvertCocoPolysToMask(object):
    def __init__(self, return_masks=False):
        self.return_masks = return_masks

    def __call__(self, image, target):
        w, h = image.size

        img_array = np.array(image)
        if len(img_array.shape) == 2:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
            image = Image.fromarray(img_array)
        image_id = target["image_id"]
        image_id = torch.tensor([image_id])
        anno = target["annotations"]
        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]
        anno = [obj for obj in anno if obj['num_keypoints'] != 0]
        keypoints = [obj["keypoints"] for obj in anno]
        boxes = [obj["bbox"] for obj in anno]
        keypoints = torch.as_tensor(keypoints, dtype=torch.float32).reshape(-1, 17, 3)
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)
        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)
        if self.return_masks:
            segmentations = [obj["segmentation"] for obj in anno]
            masks = convert_coco_poly_to_mask(segmentations, h, w)
        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        keypoints = keypoints[keep]
        if self.return_masks:
            masks = masks[keep]
        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        if self.return_masks:
            target["masks"] = masks
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints
        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]
        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])
        return image, target


def make_coco_transforms(image_set, fix_size=False, strong_aug=False, args=None):
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # config the params for data aug
    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
    max_size = 1333
    scales2_resize = [400, 500, 600]
    scales2_crop = [384, 600]

    # update args from config files
    scales = getattr(args, 'data_aug_scales', scales)
    max_size = getattr(args, 'data_aug_max_size', max_size)
    scales2_resize = getattr(args, 'data_aug_scales2_resize', scales2_resize)
    scales2_crop = getattr(args, 'data_aug_scales2_crop', scales2_crop)

    # resize them
    data_aug_scale_overlap = getattr(args, 'data_aug_scale_overlap', None)
    if data_aug_scale_overlap is not None and data_aug_scale_overlap > 0:
        data_aug_scale_overlap = float(data_aug_scale_overlap)
        scales = [int(i * data_aug_scale_overlap) for i in scales]
        max_size = int(max_size * data_aug_scale_overlap)
        scales2_resize = [int(i * data_aug_scale_overlap) for i in scales2_resize]
        scales2_crop = [int(i * data_aug_scale_overlap) for i in scales2_crop]
    datadict_for_print = {
        'scales': scales,
        'max_size': max_size,
        'scales2_resize': scales2_resize,
        'scales2_crop': scales2_crop
    }
    if image_set == 'train':
        if fix_size:
            return T.Compose([
                T.RandomHorizontalFlip(),
                T.RandomResize([(max_size, max(scales))]),
                normalize,
            ])

        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomSelect(
                T.RandomResize(scales, max_size=max_size),
                T.Compose([
                    T.RandomResize(scales2_resize),
                    T.RandomSizeCrop(*scales2_crop),
                    T.RandomResize(scales, max_size=max_size),
                ])
            ),
            normalize,
        ])

    if image_set in ['val', 'test']:


        return T.Compose([
            T.RandomResize([max(scales)], max_size=max_size),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')


def build(image_set, args):
    root = Path(args.coco_path)
    dataset = CocoDetection(root, image_set, transforms=make_coco_transforms(image_set, fix_size=getattr(args, "fix_size", False), strong_aug=args.strong_aug, args=args),
                            return_masks=args.masks)

    return dataset

