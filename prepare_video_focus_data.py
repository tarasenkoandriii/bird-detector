#!/usr/bin/env python3
"""Додаткові повністю розмічені відеокадри; початкова validation незмінна."""
import io
import json
import hashlib
import shutil
import zipfile
from pathlib import Path

import yaml
from PIL import Image
from common import ROOT
from prepare_video_test import fingerprint, SHA, file_sha256
from tools.dataset import audit


def main():
    out = ROOT / 'datasets/bird-video-focus'
    source = ROOT / 'datasets/bird-video'
    if out.exists():
        raise SystemExit('Набір уже існує; повторне створення не потрібне.')
    archive = ROOT / 'datasets/cache-video/bird.zip'
    if file_sha256(archive) != SHA:
        raise ValueError('Не збігається SHA256 джерела LaSOT.')
    selection = json.loads((ROOT / 'tools/video_focus_annotations.json').read_text())
    if not selection['reviewed']:
        raise ValueError('Потрібна перевірена розмітка.')
    protected = []
    for folder in ['bird-expanded/val/images', 'bird-expanded/test/images',
                   'bird-video/val/images']:
        for path in (ROOT / 'datasets' / folder).glob('*.jpg'):
            protected.append(fingerprint(Image.open(path)))
    for split in ['train','val']:
        shutil.copytree(source / split, out / split)
    manifest = json.loads((source / 'source.json').read_text())
    with zipfile.ZipFile(archive) as z:
        index = {(part,Path(n).name):n for n in z.namelist() for part in Path(n).parts if part.startswith('bird-')}
        for r in selection['records']:
            if r['sequence'] not in {'bird-4','bird-12','bird-13','bird-20'}:
                raise ValueError('Заборонена додаткова послідовність.')
            blob = z.read(index[(r['sequence'], f"{r['frame']:08d}.jpg")])
            if hashlib.sha256(blob).hexdigest() != r['source_sha256']:
                raise ValueError('Не збігається SHA256 кадру.')
            im = Image.open(io.BytesIO(blob));w,h = im.size
            pixel, dh, textured = fingerprint(im)
            if any(pixel==p or (textured and t and (dh^d).bit_count()<=4) for p,d,t in protected):
                raise ValueError('Збіг додаткового кадру із захищеною вибіркою.')
            name = f"extra-{r['sequence']}-{r['frame']:08d}"
            (out/'train/images'/f'{name}.jpg').write_bytes(blob)
            lines=[]
            for x,y,bw,bh in r['boxes_xywh']:
                if min(x,y)<0 or min(bw,bh)<=0 or x+bw>w or y+bh>h:
                    raise ValueError('Рамка виходить за межі кадру.')
                lines.append(f'0 {(x+bw/2)/w:.8f} {(y+bh/2)/h:.8f} {bw/w:.8f} {bh/h:.8f}')
            (out/'train/labels'/f'{name}.txt').write_text('\n'.join(lines)+'\n')
            manifest['records'].append(dict(r,split='train',kind='positive',output_image=f'train/images/{name}.jpg',annotation='Усі видимі птахи; візуальна розмітка асистентом.'))
    (out/'data.yaml').write_text(yaml.safe_dump(dict(path=str(out),train='train/images',val='val/images',names={0:'bird'},nc=1)))
    (out/'source.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    result = audit(out)
    if result['errors']:
        raise ValueError(result['errors'])
    (out/'audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print('Готово:',out,flush=True)


if __name__=='__main__':
    main()
