"""Атомарне збереження нових звітів без неявної заміни історичних результатів."""

import hashlib
import json
import os
import tempfile
from pathlib import Path


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def normal_file_mode():
    mask = os.umask(0)
    os.umask(mask)
    return 0o666 & ~mask


def write_json(path, value, overwrite=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix='.report-', suffix='.tmp')
    try:
        with os.fdopen(descriptor, 'w') as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write('\n')
        os.chmod(temporary, normal_file_mode())
        if overwrite:
            os.replace(temporary, path)
        else:
            # На відміну від exists()+replace(), не перезапише файл, що виник у гонці.
            os.link(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def dataset_snapshot(config, split):
    """Той самий список зображень/labels, який читає локальний Ultralytics."""
    from types import SimpleNamespace
    from ultralytics.data.base import BaseDataset
    from ultralytics.data.utils import img2label_paths

    images = BaseDataset.get_img_files(SimpleNamespace(prefix='', fraction=1.0), config[split])
    root = Path(config['path']).resolve()
    records = []
    for image, label in zip(images, img2label_paths(images)):
        image = Path(image).resolve()
        try:
            name = image.relative_to(root).as_posix()
        except ValueError:
            name = image.as_posix()
        records.append(dict(image=name, image_sha256=sha256(image),
                            label_sha256=sha256(label) if Path(label).exists() else None))
    content = dict(split=split, names=config['names'], records=records)
    digest = hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()
    return dict(sha256=digest, images=len(records), **content)
