"""Download the official CrowdPose images and annotations with resumable chunks."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
from pathlib import Path
import subprocess
import sys
import time
import zipfile

import requests

FILE_ID = '1VprytECcLtU4tKP32SYi_7oDRbw7yUTL'
URL = f'https://drive.usercontent.google.com/download?id={FILE_ID}&export=download&confirm=t'
ANNOTATIONS = 'https://drive.google.com/drive/folders/1Ch1Cobe-6byB7sLhy8XRzOGCGTW2ssFv'
SIZE = 2402012797
SHA256 = '6b6d9d225856705f7f740008061b6f249e13b94bef249e9f30d161d196a67240'
BLOCK = 8 * 1024 * 1024


def sha256(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(4*1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()


def download(root,workers):
    root.mkdir(parents=True,exist_ok=True)
    cache=root/'.downloads'
    cache.mkdir(exist_ok=True)
    archive=cache/'images.zip'
    if not archive.exists() or sha256(archive)!=SHA256:
        def block(i):
            first=i*BLOCK
            last=min(first+BLOCK,SIZE)-1
            target=cache/f'images.{i:04}.part'
            if target.exists() and target.stat().st_size==last-first+1:
                return target
            for attempt in range(4):
                try:
                    # A distinct URL also avoids proxies caching another Range.
                    with requests.get(URL+f'&part={i}',headers={'Range':f'bytes={first}-{last}'},
                                      stream=True,timeout=(12,40)) as response:
                        response.raise_for_status()
                        if response.status_code!=206 or response.headers.get('Content-Range')!=f'bytes {first}-{last}/{SIZE}':
                            raise RuntimeError('Server did not return the requested file range')
                        payload=b''.join(response.iter_content(1024*1024))
                        if len(payload)!=last-first+1:
                            raise RuntimeError('Incomplete download range')
                        target.write_bytes(payload)
                    return target
                except (requests.RequestException,RuntimeError):
                    if attempt==3:
                        raise
                    time.sleep(attempt+1)
        # Check reachability before dispatching the remaining requests.
        block(0)
        count=(SIZE+BLOCK-1)//BLOCK
        pool=ThreadPoolExecutor(max_workers=workers)
        futures=[pool.submit(block,i) for i in range(1,count)]
        try:
            for done,future in enumerate(as_completed(futures),2):
                future.result()
                if done%16==0 or done==count:
                    print(f'Image archive blocks: {done}/{count}',flush=True)
        except BaseException:
            pool.shutdown(wait=True,cancel_futures=True)
            raise
        else:
            pool.shutdown(wait=True)
        with archive.open('wb') as out:
            for i in range(count):
                out.write((cache/f'images.{i:04}.part').read_bytes())
        if sha256(archive)!=SHA256:
            raise RuntimeError('Image archive checksum mismatch; remove .downloads/*.part and retry')
    with zipfile.ZipFile(archive) as z:
        resolved_root=root.resolve()
        for member in z.infolist():
            if not (resolved_root/member.filename).resolve().is_relative_to(resolved_root):
                raise ValueError('Unsafe archive path')
        z.extractall(root)
    annotation_root=root/'json'
    annotation_root.mkdir(exist_ok=True)
    subprocess.run([sys.executable,'-m','gdown','--no-cookies','--continue','--retries','3',
                    ANNOTATIONS,'-O',str(annotation_root)],check=True)
    print('Downloaded CrowdPose. Run scripts/check_crowdpose.py on this directory.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('data/crowdpose'))
    parser.add_argument('--workers',type=int,default=4)
    cli=parser.parse_args()
    if not 1<=cli.workers<=16:
        parser.error('--workers must be between 1 and 16')
    download(cli.root,cli.workers)
