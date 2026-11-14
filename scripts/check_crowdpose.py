"""Validate CrowdPose files and report split sizes before training."""
import argparse
import hashlib
import json
from pathlib import Path
from PIL import Image


def check(root, verify_images=False):
    report={}
    ids_by_split={}
    for split in ['train','val','test']:
        path=root/'json'/f'crowdpose_{split}.json'
        data=json.loads(path.read_text(encoding='utf-8'))
        images={x['id']:x for x in data['images']}
        if len(images)!=len(data['images']):
            raise ValueError(f'Duplicate image IDs in {path}')
        ann_ids=set()
        for ann in data['annotations']:
            if ann['id'] in ann_ids or ann['image_id'] not in images:
                raise ValueError(f'Invalid annotation ID/reference in {path}: {ann["id"]}')
            ann_ids.add(ann['id'])
            if len(ann['keypoints'])!=42:
                raise ValueError(f'Expected 14 keypoints at annotation {ann["id"]}')
            area = ann.get('area', ann['bbox'][2] * ann['bbox'][3])
            if ann.get('num_keypoints',0)>0 and area<=0:
                raise ValueError(f'Invalid area at annotation {ann["id"]}')
        for image in images.values():
            path_image=root/'images'/image['file_name']
            if not path_image.is_file():
                raise FileNotFoundError(path_image)
            if 'crowdIndex' not in image:
                raise ValueError(f'Missing crowdIndex on image {image["id"]}')
            if verify_images:
                with Image.open(path_image) as im:
                    if im.size!=(image['width'],image['height']):
                        raise ValueError(f'Unexpected image dimensions: {path_image}')
                    im.verify()
        report[split]={'images':len(images),'annotations':len(ann_ids),
                       'annotation_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
        ids_by_split[split]=set(images)
    for a,b in [('train','val'),('train','test'),('val','test')]:
        if ids_by_split[a]&ids_by_split[b]:
            raise ValueError(f'{a}/{b} contain overlapping images')
    print(json.dumps(report,indent=2))
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path)
    parser.add_argument('--verify-images',action='store_true')
    args=parser.parse_args()
    check(args.root,args.verify_images)
