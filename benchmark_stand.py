#!/usr/bin/env python3
"""Контрольований CPU/MPS benchmark; тести та релізні ваги не змінюються."""

import argparse, hashlib, json, os, random, resource, subprocess, sys, time
from pathlib import Path
from common import ROOT, resolve_device, UkrainianParser
import yaml

R = ROOT / "reports/stand_benchmark"
W = ROOT / "work/stand-benchmark"
RUN = ROOT / "runs/stand-benchmark"


def write(p, x):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(x, indent=2, ensure_ascii=False))


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def worker(device):
    import torch
    from ultralytics import YOLO

    resolve_device(device)
    model = YOLO(str(W / "source.pt"))
    head = len(model.model.model) - 1
    timing = {"train": [], "validation": []}
    start = {}
    peaks = {"allocated": 0, "driver": 0}

    def sync():
        if device == "mps":
            torch.mps.synchronize()

    def begin(key):
        def cb(t):
            sync()
            start[key] = time.perf_counter()

        return cb

    def end(key):
        def cb(t):
            sync()
            timing[key].append(time.perf_counter() - start[key])

        return cb

    def bn(t):
        for m in t.model.modules():
            if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
                m.eval()

    def sample(t):
        if device == "mps":
            for key, fn in [
                ("allocated", "current_allocated_memory"),
                ("driver", "driver_allocated_memory"),
            ]:
                if hasattr(torch.mps, fn):
                    peaks[key] = max(peaks[key], getattr(torch.mps, fn)())

    model.add_callback("on_train_batch_start", bn)
    model.add_callback("on_train_batch_end", sample)
    for key, prefix in [("train", "on_train_epoch"), ("validation", "on_val")]:
        model.add_callback(prefix + "_start", begin(key))
        model.add_callback(prefix + "_end", end(key))
    t = time.perf_counter()
    model.train(
        data=str(W / "data.yaml"),
        epochs=3,
        batch=8,
        nbs=8,
        imgsz=640,
        device=device,
        workers=0,
        freeze=head,
        optimizer="AdamW",
        lr0=1e-5,
        lrf=0.1,
        warmup_epochs=0,
        warmup_bias_lr=0,
        weight_decay=0,
        mosaic=0,
        close_mosaic=0,
        scale=0,
        translate=0,
        fliplr=0,
        amp=False,
        seed=26,
        deterministic=True,
        plots=False,
        patience=0,
        project=str(RUN),
        name=device,
        save=True,
    )
    sync()
    write(
        R / f"{device}_timing.json",
        {
            "device": device,
            "timing_seconds": timing,
            "wall_seconds": time.perf_counter() - t,
            "peak_process_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2,
            "sampled_mps_peak_bytes": peaks,
            "last": str(model.trainer.last),
            "torch": torch.__version__,
            "mps_fallback_enabled": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK"),
        },
    )


def main():
    global R, W, RUN
    p = UkrainianParser(description=__doc__)
    p.add_argument("--worker", choices=["cpu", "mps"])
    p.add_argument("--name", default="primary", help="Нове ім’я для повторного заміру")
    a = p.parse_args()
    if not a.name.replace("-", "").replace("_", "").isalnum():
        p.error("Ім’я має містити лише літери, цифри, дефіс або підкреслення.")
    if a.name != "primary":
        R = ROOT / "reports" / ("stand_benchmark_" + a.name)
        W = ROOT / "work" / ("stand-benchmark-" + a.name)
        RUN = ROOT / "runs" / ("stand-benchmark-" + a.name)
    if a.worker:
        return worker(a.worker)
    R.mkdir(parents=True, exist_ok=True)
    W.mkdir(parents=True, exist_ok=True)
    status = R / "status.json"
    if status.exists():
        raise ValueError("Benchmark уже запущений; повторне виконання скасовано.")
    write(status, {"state": "running", "phase": "prepare"})
    try:
        import torch, inspect, shutil
        from ultralytics import YOLO

        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS недоступний; чесне порівняння неможливе.")
        # Перевіряємо фактичні API встановленого стеку.
        assert hasattr(torch.mps, "synchronize")
        source = ROOT / "weights/photo_reference.pt"
        shutil.copyfile(source, W / "source.pt")
        rng = random.Random(26)
        counts = {}
        for split, n in [("train", 128), ("val", 64)]:
            files = sorted((ROOT / f"datasets/bird-expanded/{split}/images").glob("*.jpg"))
            rng.shuffle(files)
            files = files[:n]
            assert len(files) == n
            (W / f"{split}.txt").write_text("".join(str(f) + "\n" for f in files))
            counts[split] = len(files)
        (W / "data.yaml").write_text(
            yaml.safe_dump(
                {"train": str(W / "train.txt"), "val": str(W / "val.txt"), "names": {0: "bird"}}
            )
        )
        digest = hashlib.sha256()
        for split in ["train", "val"]:
            for name in (W / f"{split}.txt").read_text().splitlines():
                f = Path(name)
                lab = f.parent.parent / "labels" / f"{f.stem}.txt"
                digest.update((f.name + sha(f) + sha(lab)).encode())
        release_hash = sha(ROOT / "weights/best.pt")
        write(
            R / "protocol.json",
            {
                "source_sha256": sha(source),
                "release_sha256_before": release_hash,
                "dataset_sha256": digest.hexdigest(),
                "counts": counts,
                "seed": 26,
                "epochs": 3,
                "imgsz": 640,
                "batch": 8,
                "nbs": 8,
                "optimizer": "AdamW",
                "lr0": 1e-5,
                "freeze": 23,
                "bn_statistics_frozen": True,
                "quality_evaluation": "CPU, 600 CUB validation, batch=4; test не використовується",
                "mps_api": {
                    k: hasattr(torch.mps, k)
                    for k in ["synchronize", "current_allocated_memory", "driver_allocated_memory"]
                },
                "order": ["cpu", "mps"],
            },
        )
        for device in ["cpu", "mps"]:
            write(status, {"state": "running", "phase": "training", "device": device})
            with (R / f"{device}.log").open("w") as log:
                subprocess.run(
                    [
                        sys.executable,
                        str(Path(__file__).resolve()),
                        "--worker",
                        device,
                        "--name",
                        a.name,
                    ],
                    cwd=ROOT,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
            write(
                status, {"state": "running", "phase": "cpu_reference_validation", "device": device}
            )
            with (R / f"{device}_validation.log").open("w") as log:
                subprocess.run(
                    [
                        sys.executable,
                        "evaluate.py",
                        "--data",
                        "datasets/bird-expanded/data.yaml",
                        "--weights",
                        str(RUN / device / "weights/last.pt"),
                        "--split",
                        "val",
                        "--imgsz",
                        "640",
                        "--device",
                        "cpu",
                        "--batch",
                        "4",
                        "--report",
                        str(R / f"{device}_quality.json"),
                    ],
                    cwd=ROOT,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
        results = {}
        for device in ["cpu", "mps"]:
            d = json.loads((R / f"{device}_timing.json").read_text())
            d["metrics"] = json.loads((R / f"{device}_quality.json").read_text())["metrics"]
            d["fallback_warning_seen"] = (
                "fall back" in (R / f"{device}.log").read_text()
                or "MPSFallback" in (R / f"{device}.log").read_text()
            )
            results[device] = d
        assert release_hash == sha(ROOT / "weights/best.pt")
        write(R / "results.json", results)
        s = """# Стенд CPU проти MPS

Однакові стартові ваги A, 128 train / 64 validation, 3 епохи, 640 px, batch=nbs=8, seed=26, AdamW 1e-5; навчається голова, backbone/neck і BN-статистика зафіксовані. Запуски послідовні: CPU, потім MPS. Test і релізні ваги не змінювалися.

| Пристрій | Перша train-епоха, с | Середнє епох 2–3, с | Швидкість прогрітих, зобр./с | Пікова RSS процесу, МіБ | CUB validation mAP50–95 |
|---|---:|---:|---:|---:|---:|
"""
        for device, d in results.items():
            train = d["timing_seconds"]["train"]
            warm = sum(train[1:]) / len(train[1:])
            s += f'| {device.upper()} | {train[0]:.2f} | {warm:.2f} | {128/warm:.2f} | {d["peak_process_rss_mib"]:.0f} | {100*d["metrics"]["metrics/mAP50-95(B)"]:.2f}% |\n'
        ratio = sum(results["cpu"]["timing_seconds"]["train"][1:]) / sum(
            results["mps"]["timing_seconds"]["train"][1:]
        )
        s += f"\nСпіввідношення прогрітого часу CPU/MPS: **{ratio:.2f}×**; >1 означає швидший MPS. Це один парний замір, не статистична оцінка.\n"
        s += """
Перша епоха включає первинні витрати backend, але не весь запуск процесу. Окремі train/validation-часи та повний wall-time — у JSON. RSS — максимум процесу; MPS allocated/driver — найбільші зразки після batch, не гарантований апаратний пік і не величини для додавання до RSS. CPU fallback дозволений; факт попередження записаний окремо. Навіть без попередження частина підготовки й постобробки виконується на CPU.

Якість обох last.pt після рівно трьох епох оцінена на CPU на тих самих 600 validation-зображеннях, щоб не змішувати якість навчання з відомою проблемою MPS-координат. Три епохи малого набору не доводять еквівалентності повного навчання. Стек не оновлювався; порядок запусків, нагрів і кеш можуть впливати на час.

[Протокол](protocol.json) · [Усі результати](results.json). Відтворення на підготовленому проєкті: `python benchmark_stand.py --name repeat1`. Наявний запуск не перезаписується. Нові ваги залишені лише у runs/stand-benchmark/, без заміни релізу.
"""
        (R / "README.md").write_text(s)
        write(status, {"state": "completed"})
    except BaseException as e:
        write(status, {"state": "failed", "error": str(e)})
        raise


if __name__ == "__main__":
    main()
