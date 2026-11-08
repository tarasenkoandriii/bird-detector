#!/usr/bin/env python3
"""Audit Roboflow YOLO detection exports, then optionally prepare bird-only data."""

import argparse, collections, hashlib, json, math, random, re, shutil
from pathlib import Path
from PIL import Image
import yaml

EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def audit(root, bird_names=("bird", "birds"), distance=4):
    cfg = yaml.safe_load((root / "data.yaml").read_text())
    names = cfg["names"]
    names = (
        dict(enumerate(names)) if isinstance(names, list) else {int(k): v for k, v in names.items()}
    )
    bird = {k for k, v in names.items() if str(v).casefold() in bird_names}
    if not bird:
        raise ValueError(f"No selected bird classes in {names}; specify --bird-names explicitly.")
    records = []
    errors = []
    warnings = []
    for split, folder in [
        ("train", "train"),
        ("val", "valid" if (root / "valid").exists() else "val"),
        ("test", "test"),
    ]:
        base = root / folder
        images = base / "images"
        labels = base / "labels"
        if not images.exists():
            if split != "test":
                errors.append(f"Missing {images}")
            continue
        seen = set()
        for img in sorted(images.rglob("*")):
            if img.suffix.lower() not in EXT:
                continue
            rel = img.relative_to(images)
            lab = (labels / rel).with_suffix(".txt")
            seen.add(lab)
            issues = []
            boxes = []
            try:
                with Image.open(img) as im:
                    im.load()
                    rgb = im.convert("RGB")
                    w, h = rgb.size
                    pixel = digest(f"{w},{h}:".encode() + rgb.tobytes())
                    gray = rgb.resize((9, 8)).convert("L")
                    px = list(
                        gray.get_flattened_data()
                        if hasattr(gray, "get_flattened_data")
                        else gray.getdata()
                    )
                    dh = sum(
                        (px[y * 9 + x] > px[y * 9 + x + 1]) << (y * 8 + x)
                        for y in range(8)
                        for x in range(8)
                    )
                    # Near-hash matching of flat images is unreliable.
                    usable = max(px) - min(px) > 15
                raw = digest(img.read_bytes())
            except Exception:
                errors.append(f"Unreadable image: {img}")
                continue
            if not lab.exists():
                issues.append("missing label (not silently treated as background)")
            else:
                for n, line in enumerate(lab.read_text().splitlines(), 1):
                    if not line.strip():
                        continue
                    try:
                        fields = line.split()
                        if len(fields) != 5:
                            raise ValueError("expected 5 detection fields")
                        c = float(fields[0])
                        x, y, bw, bh = map(float, fields[1:])
                        if not all(math.isfinite(v) for v in (c, x, y, bw, bh)):
                            raise ValueError("non-finite")
                        if not c.is_integer() or int(c) not in names:
                            raise ValueError("unknown class")
                        if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < bw <= 1 and 0 < bh <= 1):
                            raise ValueError("range")
                        if (
                            min(x - bw / 2, y - bh / 2) < -1e-6
                            or max(x + bw / 2, y + bh / 2) > 1 + 1e-6
                        ):
                            raise ValueError("box outside image")
                        if int(c) in bird:
                            boxes.append([0, x, y, bw, bh])
                    except ValueError as e:
                        issues.append(f"line {n}: {e}")
            if len({tuple(b) for b in boxes}) < len(boxes):
                issues.append("duplicate boxes")
            if issues:
                errors.append({"image": str(img), "issues": issues})
            family = re.sub(r"\.rf\.[^.]+$", "", img.stem) if ".rf." in img.stem else None
            records.append(
                dict(
                    image=str(img.resolve()),
                    label=str(lab.resolve()),
                    split=split,
                    raw=raw,
                    pixel=pixel,
                    dhash=dh,
                    hash_usable=usable,
                    family=family,
                    boxes=boxes,
                    valid=not issues,
                )
            )
        for lab in labels.rglob("*.txt") if labels.exists() else []:
            if lab not in seen:
                errors.append(f"Orphan label: {lab}")
    if not records:
        errors.append("No images")
    parent = list(range(len(records)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    pairs = []
    for i, a in enumerate(records):
        for j in range(i):
            b = records[j]
            kind = None
            if a["pixel"] == b["pixel"] or a["raw"] == b["raw"]:
                kind = "exact"
            elif a["family"] and a["family"] == b["family"]:
                kind = "roboflow_source_family"
            elif (
                a["hash_usable"]
                and b["hash_usable"]
                and (a["dhash"] ^ b["dhash"]).bit_count() <= distance
            ):
                kind = "near_candidate"
            if kind:
                parent[find(i)] = find(j)
                pairs.append(
                    dict(a=a["image"], b=b["image"], kind=kind, leakage=a["split"] != b["split"])
                )
    for i, r in enumerate(records):
        r["group"] = find(i)
    return dict(
        classes=names,
        bird_ids=sorted(bird),
        counts=dict(collections.Counter(r["split"] for r in records)),
        errors=errors,
        pairs=pairs,
        records=records,
        notes=[
            "Near duplicates are heuristic candidates, not proof. Crops/video frames may escape detection. Provide source/video grouping before scientific evaluation."
        ],
    )


def prepare(report, out, seed, limit):
    if report["errors"]:
        raise ValueError("Audit errors must be corrected in source before preparation.")
    if out.exists():
        raise ValueError("Output exists; refusing overwrite.")
    # Preserve source split; protect held-out data by removing duplicates from train first.
    priority = {"test": 0, "val": 1, "train": 2}
    groups = collections.defaultdict(list)
    for r in report["records"]:
        groups[r["group"]].append(r)
    chosen = []
    for group in groups.values():
        chosen.append(sorted(group, key=lambda r: (priority[r["split"]], r["image"]))[0])
    rng = random.Random(seed)
    manifest = []
    for split in ["train", "val", "test"]:
        rows = [r for r in chosen if r["split"] == split]
        rng.shuffle(rows)
        if limit:
            rows = rows[: dict(zip(["train", "val", "test"], limit))[split]]
        if split != "test" and not rows:
            raise ValueError(f"No {split} images after deduplication.")
        for r in rows:
            manifest.append(r)
    if not any(r["boxes"] for r in manifest if r["split"] == "train"):
        raise ValueError("No training bird boxes")
    out.mkdir(parents=True)
    for r in manifest:
        s = r["split"]
        image = Path(r["image"])
        name = r["pixel"][:16]  # stable across checkout locations
        ip = out / s / "images" / f"{name}{image.suffix.lower()}"
        lp = out / s / "labels" / f"{name}.txt"
        ip.parent.mkdir(parents=True, exist_ok=True)
        lp.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(image, ip)
        lp.write_text("".join(" ".join(map(str, b)) + "\n" for b in r["boxes"]))
        r["output_image"] = str(ip.resolve())
    cfg = {
        "path": str(out.resolve()),
        "train": "train/images",
        "val": "val/images",
        "names": {0: "bird"},
        "nc": 1,
    }
    if any(r["split"] == "test" for r in manifest):
        cfg["test"] = "test/images"
    (out / "data.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    (out / "manifest.json").write_text(
        json.dumps(
            {
                "seed": seed,
                "policy": "preserve splits; one representative per connected duplicate/candidate group, prefer test then val then train",
                "records": manifest,
            },
            indent=2,
        )
    )
    return dict(collections.Counter(r["split"] for r in manifest))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--report", type=Path, required=True)
    p.add_argument("--prepare", type=Path)
    p.add_argument("--bird-names", default="bird,birds")
    p.add_argument("--distance", type=int, default=4)
    p.add_argument("--seed", type=int, default=26)
    p.add_argument("--limit", type=int, nargs=3, metavar=("TRAIN", "VAL", "TEST"))
    a = p.parse_args()
    if not 0 <= a.distance <= 64 or (a.limit and min(a.limit) < 1):
        p.error("Invalid distance/limits")
    r = audit(
        a.source.resolve(), tuple(n.strip().casefold() for n in a.bird_names.split(",")), a.distance
    )
    a.report.parent.mkdir(parents=True, exist_ok=True)
    a.report.write_text(json.dumps(r, indent=2))
    print("Audit:", r["counts"], "errors:", len(r["errors"]), "duplicate pairs:", len(r["pairs"]))
    if a.prepare:
        print("Prepared:", prepare(r, a.prepare.resolve(), a.seed, a.limit))
    elif r["errors"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
