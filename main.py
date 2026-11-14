"""TopoDA training and evaluation entry point."""
import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler
import util.misc as utils
from util.config import Config, DictAction
from util.get_param_dicts import get_param_dict
from util.logger import setup_logger
from util.utils import ModelEma
from datasets import build_dataset, get_coco_api_from_dataset
from engine import train_one_epoch, evaluate


def get_args_parser():
    parser = argparse.ArgumentParser('TopoDA', add_help=False)
    parser.add_argument('-c', '--config_file', required=True)
    parser.add_argument('--options', nargs='+', action=DictAction)
    parser.add_argument('--dataset_file', choices=['coco', 'crowdpose', 'humanart'], default='crowdpose')
    for dataset in ['coco', 'crowdpose', 'humanart']:
        parser.add_argument(f'--{dataset}_path')
    parser.add_argument('--output_dir', default='runs/topoda')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--resume', default='')
    parser.add_argument('--pretrain_model_path', default='')
    parser.add_argument('--finetune_ignore', nargs='+', default=[])
    parser.add_argument('--backbone_weights')
    parser.add_argument('--no_backbone_pretrained', dest='backbone_pretrained', action='store_false')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--eval_split', choices=['val', 'test'], default='val')
    parser.add_argument('--max_train_steps', type=int, default=0)
    parser.add_argument('--max_eval_steps', type=int, default=0)
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--save_results', action='store_true')
    parser.add_argument('--save_log', action='store_true')
    parser.add_argument('--find_unused_params', action='store_true')
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--dist_url', default='env://')
    parser.add_argument('--fix_size', action='store_true')
    parser.add_argument('--remove_difficult', action='store_true')
    return parser


def load_config(args):
    cfg = Config.fromfile(args.config_file)
    if args.options:
        cfg.merge_from_dict(args.options)
    for key, value in cfg._cfg_dict.to_dict().items():
        if hasattr(args, key):
            raise ValueError(f'Config key {key} conflicts with a command-line argument')
        setattr(args, key, value)
    args.debug = False
    args.use_ema = getattr(args, 'use_ema', False)
    args.ema_epoch = getattr(args, 'ema_epoch', 0)
    if args.num_body_points != (14 if args.dataset_file == 'crowdpose' else 17):
        raise ValueError(f'{args.dataset_file} requires {14 if args.dataset_file == "crowdpose" else 17} keypoints')
    for key in ('max_train_steps', 'max_eval_steps', 'num_workers'):
        if getattr(args, key) < 0:
            raise ValueError(f'{key} must be nonnegative')
    return cfg


def build_model_main(args):
    import models
    from models.registry import MODULE_BUILD_FUNCS
    return MODULE_BUILD_FUNCS.get(args.modelname)(args)


def load_weights(model, path, ignore=(), strict=True):
    # All checkpoints created by this release use only tensors and primitive data.
    # Older argparse.Namespace checkpoints need conversion before loading.
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    state = checkpoint.get('model', checkpoint)
    state = utils.clean_state_dict(state)
    state = {k: v for k, v in state.items() if not any(word in k for word in ignore)}
    result = model.load_state_dict(state, strict=strict)
    print(f'Loaded {path}: {result}')
    return checkpoint


def main(args):
    cfg = load_config(args)
    if args.resume:
        args.backbone_pretrained = False
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    utils.init_distributed_mode(args)
    device = torch.device(args.device)
    if device.type != 'cuda':
        raise ValueError('TopoDA training/evaluation requires CUDA because LAFA uses DCNv4')
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for DCNv4; no CUDA GPU is visible')
    if args.amp and device.type != 'cuda':
        raise ValueError('--amp requires a CUDA device')
    data_arg = f'{args.dataset_file}_path'
    data_path = getattr(args, data_arg) or os.environ.get(f'TOPODA_{args.dataset_file.upper()}_PATH')
    if not data_path:
        raise ValueError(f'Provide --{data_arg} or TOPODA_{args.dataset_file.upper()}_PATH')
    setattr(args, data_arg, data_path)
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    logger = setup_logger(output=str(output/'info.txt'), distributed_rank=args.rank, color=False, name='topoda')
    if utils.is_main_process():
        cfg.dump(str(output/'config_cfg.py'))
        (output/'config_args_all.json').write_text(json.dumps(vars(args), indent=2), encoding='utf-8')
    model, criterion, postprocessors = build_model_main(args)
    model.to(device)
    criterion.to(device)
    checkpoint = None
    if args.resume:
        checkpoint = load_weights(model, args.resume)
    elif args.pretrain_model_path:
        load_weights(model, args.pretrain_model_path, args.finetune_ignore, strict=False)
    elif args.eval:
        raise ValueError('Evaluation requires --resume or --pretrain_model_path')
    dataset_val = build_dataset(args.eval_split, args)
    base_ds = get_coco_api_from_dataset(dataset_val)
    if len(dataset_val) == 0 or base_ds is None:
        raise ValueError('The evaluation dataset must be nonempty and expose its annotation API')
    val_sampler = DistributedSampler(dataset_val, shuffle=False) if args.distributed else torch.utils.data.SequentialSampler(dataset_val)
    loader_val = DataLoader(dataset_val, batch_size=args.batch_size, sampler=val_sampler,
                            num_workers=args.num_workers, collate_fn=utils.collate_fn, pin_memory=device.type=='cuda')
    if args.eval:
        stats, evaluator = evaluate(model, criterion, postprocessors, loader_val, base_ds, device, str(output), args=args)
        if utils.is_main_process():
            (output/'metrics.json').write_text(json.dumps(stats, indent=2), encoding='utf-8')
        if args.distributed:
            torch.distributed.destroy_process_group()
        return
    # Refuse accidental overwrites; resuming is an explicit operation.
    if (output/'checkpoint.pth').exists() and not args.resume:
        raise FileExistsError(f'{output}/checkpoint.pth exists; use --resume or a new output directory')
    ema = ModelEma(model, args.ema_decay) if args.use_ema else None
    optimizer = torch.optim.AdamW(get_param_dict(args, model), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler('cuda', enabled=args.amp)
    dataset_train = build_dataset('train', args)
    if len(dataset_train) < args.batch_size:
        raise ValueError('Training split is smaller than batch_size')
    train_sampler = DistributedSampler(dataset_train) if args.distributed else torch.utils.data.RandomSampler(dataset_train)
    loader_train = DataLoader(dataset_train, batch_size=args.batch_size, sampler=train_sampler,
                              drop_last=True, num_workers=args.num_workers, collate_fn=utils.collate_fn,
                              pin_memory=device.type=='cuda')
    if args.onecyclelr:
        steps = min(args.max_train_steps, len(loader_train)) if args.max_train_steps else len(loader_train)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, steps_per_epoch=steps, epochs=args.epochs, pct_start=0.2)
    elif args.multi_step_lr:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_drop_list)
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_drop)
    start_epoch = 0
    best_ap = -1.0
    if checkpoint is not None:
        for name in ('optimizer', 'lr_scheduler', 'epoch'):
            if name not in checkpoint:
                raise ValueError(f'Training resume checkpoint is missing {name}; use --pretrain_model_path for weights only')
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['lr_scheduler'])
        if 'scaler' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler'])
        if ema is not None and 'ema_model' in checkpoint:
            ema.module.load_state_dict(checkpoint['ema_model'])
        start_epoch = checkpoint['epoch'] + 1
        best_ap = checkpoint.get('best_ap', -1.0)
        if 'rng_state' in checkpoint:
            rng = checkpoint['rng_state']
            random.setstate(rng['python'])
            ns = rng['numpy']
            np.random.set_state((ns[0], np.asarray(ns[1], dtype=np.uint32), ns[2], ns[3], ns[4]))
            torch.set_rng_state(rng['torch'])
            if device.type == 'cuda':
                torch.cuda.set_rng_state_all(rng['cuda'])
    raw_model = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=args.find_unused_params)
        # DDP synchronizes model weights after each rank has made its EMA copy.
        # Start fresh EMA from those synchronized weights; retain restored EMA.
        if ema is not None and (checkpoint is None or 'ema_model' not in checkpoint):
            ema.set(raw_model)
    logger.info(f'Training {sum(p.numel() for p in raw_model.parameters() if p.requires_grad):,} parameters; epochs {start_epoch}..{args.epochs-1}')
    for epoch in range(start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        started = time.time()
        train_stats = train_one_epoch(model, criterion, loader_train, optimizer, device, epoch,
                                     args.clip_max_norm, lr_scheduler=scheduler, args=args,
                                     ema_m=ema, scaler=scaler, logger=logger if args.save_log else None)
        if not args.onecyclelr:
            scheduler.step()
        val_stats, _ = evaluate(model, criterion, postprocessors, loader_val, base_ds, device, str(output), args=args)
        ap = val_stats['coco_eval_keypoints_detr'][0]
        improved = ap > best_ap
        best_ap = max(best_ap, ap)
        stats = {'epoch': epoch, 'train': train_stats, 'eval': val_stats,
                 'best_ap': best_ap, 'seconds': time.time()-started,
                 'partial_training': bool(args.max_train_steps)}
        if ema is not None:
            stats['ema_eval'], _ = evaluate(ema.module, criterion, postprocessors, loader_val, base_ds, device, str(output), args=args)
        ns = np.random.get_state()
        rng = {'python': random.getstate(), 'numpy': (ns[0], ns[1].tolist(), ns[2], ns[3], ns[4]),
               'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state_all() if device.type=='cuda' else []}
        payload = {'model': raw_model.state_dict(), 'optimizer': optimizer.state_dict(),
                   'lr_scheduler': scheduler.state_dict(), 'scaler': scaler.state_dict(),
                   'epoch': epoch, 'args': vars(args), 'best_ap': best_ap, 'rng_state': rng}
        if ema is not None:
            payload['ema_model'] = ema.module.state_dict()
        if utils.is_main_process():
            temporary = output/'checkpoint.tmp.pth'
            torch.save(payload, temporary)
            temporary.replace(output/'checkpoint.pth')
            if improved:
                torch.save({'model': raw_model.state_dict(), 'epoch': epoch, 'args': vars(args), 'best_ap': best_ap}, output/'checkpoint_best.pth')
            if (epoch+1) % args.save_checkpoint_interval == 0:
                torch.save(payload, output/f'checkpoint{epoch:04}.pth')
            (output/'metrics.json').write_text(json.dumps(stats, indent=2), encoding='utf-8')
            with (output/'log.jsonl').open('a', encoding='utf-8') as f:
                f.write(json.dumps(stats)+'\n')
    if args.distributed:
        torch.distributed.destroy_process_group()


if __name__ == '__main__':
    main(argparse.ArgumentParser(parents=[get_args_parser()]).parse_args())
