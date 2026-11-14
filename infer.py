"""Run TopoDA on ordinary RGB images and save JSON plus pose overlays."""
import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps
import torch
from torchvision.transforms import functional as TF

from main import get_args_parser, load_config, build_model_main, load_weights
from datasets.transforms_coco import resize


SKELETON_14 = [(0,1),(0,2),(2,4),(1,3),(3,5),(0,6),(1,7),(6,7),(6,8),(8,10),(7,9),(9,11),(12,13),(13,0),(13,1)]
SKELETON_17 = [(0,1),(0,2),(1,3),(2,4),(5,6),(5,7),(7,9),(6,8),(8,10),(5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16)]
COLORS = ['#e15759','#4e79a7','#59a14f','#b07aa1','#f28e2b','#76b7b2']


@torch.no_grad()
def main(cli):
    args = get_args_parser().parse_args(['-c',cli.config_file,'--dataset_file',cli.dataset_file,
                                       '--device',cli.device,'--no_backbone_pretrained'])
    load_config(args)
    device = torch.device(cli.device)
    if device.type != 'cuda' or not torch.cuda.is_available():
        raise ValueError('Image inference requires a CUDA GPU and compiled DCNv4')
    if cli.short_side <= 0 or cli.max_size < cli.short_side or not 0 <= cli.threshold <= 1:
        raise ValueError('Use positive resize dimensions and a threshold between 0 and 1')
    model, _, processors = build_model_main(args)
    model.to(device).eval()
    load_weights(model,cli.checkpoint)
    source=Path(cli.input)
    paths=sorted(p for p in source.iterdir() if p.suffix.lower() in {'.jpg','.jpeg','.png','.bmp','.webp'}) if source.is_dir() else [source]
    if not paths:
        raise ValueError(f'No supported images in {source}')
    dest=Path(cli.output_dir)
    dest.mkdir(parents=True,exist_ok=True)
    edges=SKELETON_14 if args.num_body_points==14 else SKELETON_17
    for path in paths:
        image=ImageOps.exif_transpose(Image.open(path)).convert('RGB')
        width,height=image.size
        scaled,_=resize(image,None,cli.short_side,max_size=cli.max_size)
        tensor=TF.normalize(TF.to_tensor(scaled),[.485,.456,.406],[.229,.224,.225]).to(device)
        with torch.autocast(device_type=device.type,enabled=cli.amp):
            output=model([tensor])
        pred=processors['bbox'](output,torch.tensor([[height,width]],device=device))[0]
        selected=(pred['scores']>=cli.threshold) & (pred['labels']==1)
        instances=[]
        overlay=image.copy()
        draw=ImageDraw.Draw(overlay)
        for i,(score,box,keypoints) in enumerate(zip(pred['scores'][selected],pred['boxes'][selected],pred['keypoints'][selected])):
            points=keypoints.reshape(-1,3).cpu().tolist()
            bbox=box.cpu().tolist()
            instances.append({'score':float(score),'bbox_xyxy':bbox,'keypoints':points})
            color=COLORS[i%len(COLORS)]
            draw.rectangle(bbox,outline=color,width=2)
            for a,b in edges:
                draw.line((points[a][0],points[a][1],points[b][0],points[b][1]),fill=color,width=3)
            for x,y,_ in points:
                draw.ellipse((x-3,y-3,x+3,y+3),fill=color)
        payload={'image':path.name,'width':width,'height':height,'instances':instances}
        (dest/(path.name+'.json')).write_text(json.dumps(payload,indent=2),encoding='utf-8')
        overlay.save(dest/(path.name+'.pose.png'))
        print(f'{path.name}: {len(instances)} persons')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-c','--config_file',required=True)
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--input',required=True)
    parser.add_argument('--output_dir',default='runs/inference')
    parser.add_argument('--dataset_file',choices=['coco','crowdpose','humanart'],default='crowdpose')
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--amp',action='store_true')
    parser.add_argument('--threshold',type=float,default=.3)
    parser.add_argument('--short_side',type=int,default=800)
    parser.add_argument('--max_size',type=int,default=1333)
    main(parser.parse_args())
