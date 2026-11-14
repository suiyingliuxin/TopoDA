"""Training and dataset-specific evaluation for TopoDA."""
import json
import math
from pathlib import Path
import torch
import util.misc as utils
from util.utils import to_device


def train_one_epoch(model, criterion, data_loader, optimizer, device, epoch,
                    max_norm=0, wo_class_error=False, lr_scheduler=None,
                    args=None, logger=None, ema_m=None, scaler=None):
    model.train()
    criterion.train()
    if scaler is None:
        scaler = torch.amp.GradScaler('cuda', enabled=args.amp)
    metrics = utils.MetricLogger(delimiter='  ')
    optimizer_steps = skipped_steps = 0
    for step, (samples, targets) in enumerate(metrics.log_every(data_loader, 10, f'Epoch: [{epoch}]', logger=logger)):
        samples = samples.to(device)
        targets = [{k: to_device(v, device) for k, v in t.items()} for t in targets]
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=args.amp):
            outputs = model(samples, targets)
            loss_dict = criterion(outputs, targets)
            loss = sum(v * criterion.weight_dict[k] for k, v in loss_dict.items() if k in criterion.weight_dict)
        reduced = utils.reduce_dict({k: v.detach() for k, v in loss_dict.items()})
        loss_value = sum(v * criterion.weight_dict[k] for k, v in reduced.items() if k in criterion.weight_dict).item()
        if not math.isfinite(loss_value):
            raise FloatingPointError(f'Non-finite loss at epoch={epoch}, step={step}: {reduced}')
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm if max_norm > 0 else float('inf'))
        if not args.amp and not torch.isfinite(grad_norm):
            raise FloatingPointError(f'Non-finite gradients at epoch={epoch}, step={step}')
        scale_before = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        stepped = not args.amp or scaler.get_scale() >= scale_before
        if stepped:
            optimizer_steps += 1
            if args.onecyclelr:
                lr_scheduler.step()
            if ema_m is not None and epoch >= args.ema_epoch:
                ema_m.update(model)
        else:
            skipped_steps += 1
        metrics.update(loss=loss_value, lr=optimizer.param_groups[0]['lr'])
        if torch.isfinite(grad_norm):
            metrics.update(grad_norm=grad_norm.item())
        if args.max_train_steps and step + 1 >= args.max_train_steps:
            break
    if getattr(criterion, 'loss_weight_decay', False):
        criterion.loss_weight_decay(epoch=epoch)
    if getattr(criterion, 'tuning_matching', False):
        criterion.tuning_matching(epoch)
    metrics.synchronize_between_processes()
    return {**{k: m.global_avg for k, m in metrics.meters.items() if m.count},
            'steps': step + 1, 'optimizer_steps': optimizer_steps,
            'amp_skipped_steps': skipped_steps}


@torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, base_ds, device,
             output_dir, wo_class_error=False, args=None, logger=None):
    model.eval()
    criterion.eval()
    if args.dataset_file == 'crowdpose':
        from datasets.crowdpose_eval import CocoEvaluator
    elif args.dataset_file == 'humanart':
        from datasets.humanart_eval import CocoEvaluator
    else:
        from datasets.coco_eval import CocoEvaluator
    evaluator = CocoEvaluator(base_ds, ('bbox', 'keypoints'))
    metrics = utils.MetricLogger(delimiter='  ')
    predictions = []
    for step, (samples, targets) in enumerate(metrics.log_every(data_loader, 10, 'Test:', logger=logger)):
        samples = samples.to(device)
        with torch.autocast(device_type=device.type, enabled=args.amp):
            outputs = model(samples)
        sizes = torch.stack([t['orig_size'] for t in targets]).to(device)
        results = postprocessors['bbox'](outputs, sizes)
        batch = {int(t['image_id'].item()): r for t, r in zip(targets, results)}
        evaluator.update(batch)
        if args.save_results:
            predictions.extend(evaluator.prepare_for_coco_keypoint(batch))
        if args.max_eval_steps and step + 1 >= args.max_eval_steps:
            break
    evaluator.synchronize_between_processes()
    evaluator.accumulate()
    evaluator.summarize()
    stats = {
        'coco_eval_bbox': evaluator.coco_eval['bbox'].stats.tolist(),
        'coco_eval_keypoints_detr': evaluator.coco_eval['keypoints'].stats.tolist(),
        'evaluated_images': len(set(evaluator.img_ids)),
        'partial_evaluation': bool(args.max_eval_steps),
    }
    values = stats['coco_eval_keypoints_detr']
    stats['keypoint_metrics'] = {'AP': values[0], 'AP50': values[1], 'AP75': values[2]}
    if args.dataset_file == 'crowdpose':
        stats['keypoint_metrics'].update(AP_easy=values[8], AP_medium=values[9], AP_hard=values[10])
    if args.save_results:
        predictions = [p for group in utils.all_gather(predictions) for p in group]
        if utils.is_main_process():
            Path(output_dir, 'predictions.json').write_text(json.dumps(predictions), encoding='utf-8')
    return stats, evaluator


def inference_vis(*args, **kwargs):
    raise ValueError('Use python infer.py --help for image inference and visualization')
