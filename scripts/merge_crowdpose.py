"""Create trainval annotations without changing IDs or including test images."""
import argparse
import json
from pathlib import Path


def merge(root):
    root=Path(root)/'json'
    train=json.loads((root/'crowdpose_train.json').read_text())
    val=json.loads((root/'crowdpose_val.json').read_text())
    for key in ['images','annotations']:
        left={x['id'] for x in train[key]}
        right={x['id'] for x in val[key]}
        if left&right:
            raise ValueError(f'Train/val share {key} IDs')
        train[key].extend(val[key])
    (root/'crowdpose_trainval.json').write_text(json.dumps(train),encoding='utf-8')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root')
    merge(parser.parse_args().root)
