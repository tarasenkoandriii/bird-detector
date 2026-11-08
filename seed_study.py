#!/usr/bin/env python3
"""П'ять seed для фінального етапу D: сталі дані, без тестів і заміни ваг."""

import argparse, csv, fcntl, hashlib, importlib.metadata, json, math, os, random, shutil, statistics, subprocess, sys, time
from datetime import datetime, timezone
from pathlib import Path
from common import ROOT, UkrainianParser

SOURCE_SHA = "e0085bc7dfbb260d2411d72d42d15d66d65156a32b3b051943515b695fe4c276"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False))
    tmp.replace(path)


def stats(values):
    return {
        "n": len(values),
        "mean": statistics.mean(values) if values else None,
        "sd": statistics.stdev(values) if len(values) > 1 else None,
        "min": min(values) if values else None,
        "max": max(values) if values else None,
    }


def meets_criterion(row, baseline):
    return (
        row["cub"] >= baseline["cub"]
        and row["video"] >= baseline["video"]
        and row["background_fp_images"] <= baseline["background_fp_images"]
        and (row["cub"] > baseline["cub"] or row["video"] > baseline["video"])
    )


def aggregate(rows, baseline):
    return {
        "n": len(rows),
        "cub": stats([r["cub"] for r in rows]),
        "video": stats([r["video"] for r in rows]),
        "joint_passes": sum(meets_criterion(r, baseline) for r in rows),
        "max_background_fp_images": max((r["background_fp_images"] for r in rows), default=None),
    }


def inputs():
    videos = sorted((ROOT / "datasets/bird-video/train/images").glob("*.jpg"))
    photos = sorted((ROOT / "datasets/bird-expanded/train/images").glob("*.jpg"))
    random.Random(26).shuffle(photos)
    photos = photos[:512]
    vv = sorted((ROOT / "datasets/bird-video/val/images").glob("*.jpg"))
    cv = sorted((ROOT / "datasets/bird-expanded/val/images").glob("*.jpg"))
    if [len(videos), len(photos), len(vv), len(cv)] != [87, 512, 32, 600]:
        raise ValueError("Потрібні фіксовані набори D: 87+512 train, 32+600 validation.")
    files = videos + photos + vv + cv
    digest = hashlib.sha256()
    for image in files:
        label = image.parent.parent / "labels" / f"{image.stem}.txt"
        for file in [image, label]:
            digest.update((str(file.relative_to(ROOT)) + ":" + sha(file) + "\n").encode())
    selection = hashlib.sha256("\n".join(p.name for p in videos + photos).encode()).hexdigest()
    return {
        "dataset_sha256": digest.hexdigest(),
        "train_selection_sha256": selection,
        "train_code_sha256": sha(ROOT / "train_video.py"),
        "controller_code_sha256": sha(Path(__file__)),
        "environment": {
            n: importlib.metadata.version(n) for n in ["torch", "ultralytics", "numpy", "Pillow"]
        },
    }


def load_row(folder, seed):
    v = json.loads((folder / "video.json").read_text())
    c = json.loads((folder / "cub.json").read_text())
    b = json.loads((folder / "backgrounds.json").read_text())
    if len({v["weights_sha256"], c["weights_sha256"], b["weights_sha256"]}) != 1:
        raise ValueError("Різні ваги в оцінках одного seed.")
    return {
        "seed": seed,
        "cub": c["metrics"]["metrics/mAP50-95(B)"],
        "video": v["metrics"]["metrics/mAP50-95(B)"],
        "background_fp_images": b["images_with_false_positives"],
        "weights_sha256": v["weights_sha256"],
    }


def report(directory, plan, state):
    rows = []
    for seed in plan["seeds"]:
        folder = directory / f"seed_{seed}"
        if all((folder / f).exists() for f in ["video.json", "cub.json", "backgrounds.json"]):
            rows.append(load_row(folder, seed))
    a = plan["baseline_a"]
    summary = aggregate(rows, a)
    complete = len(rows) == len(plan["seeds"])
    summary.update(
        {
            "completed": complete,
            "planned_seeds": plan["seeds"],
            "rows": rows,
            "baseline_a": a,
            "state": state,
        }
    )
    write(directory / "summary.json", summary)

    def pct(x):
        return f"{100*x:.2f}%".replace(".", ",")

    s = f"""# Стійкість D до seed

**Стан: {state}. Завершено {len(rows)}/{len(plan['seeds'])}.**

Кожен запуск незалежно починається з однакових ваг A. Сталі дані: 87 відеоприкладів + 512 фото CUB, data-seed=26. Змінюється лише seed навчання: {', '.join(map(str,plan['seeds']))}. 3 епохи, CPU, 640 px, batch=nbs=8, AdamW 1e-5. Seed 26 повторно взято з уже завершеного D.

Це стійкість фінального етапу D від фіксованого A, не всього ланцюжка навчання. Валідації сталі: 600 CUB, 32 відеозображення (24 позитивні + 8 фонових вирізок), batch=4. Тестові вибірки не використовуються.

| Seed | CUB mAP50–95 | Відео mAP50–95 | Негативні з FP / 8 | Критерій оновлення відносно A |
|---|---:|---:|---:|---|
"""
    for row in rows:
        passes = meets_criterion(row, a)
        s += f'| {row["seed"]} | {pct(row["cub"])} | {pct(row["video"])} | {row["background_fp_images"]} | {"так" if passes else "ні"} |\n'
    s += f'\nA для порівняння: CUB **{pct(a["cub"])}**, відео **{pct(a["video"])}**, фонових помилок {a["background_fp_images"]}/8.\n'
    if complete:
        s += "\n## Розкид між seed\n\n| Метрика | Середнє | SD, в.п. | Мінімум | Максимум |\n|---|---:|---:|---:|---:|\n"
        for key, label in [("cub", "CUB"), ("video", "Відео")]:
            x = summary[key]
            sd = f'{100*x["sd"]:.2f}'.replace(".", ",")
            s += f'| {label} | {pct(x["mean"])} | {sd} | {pct(x["min"])} | {pct(x["max"])} |\n'
        s += f'\nКритерій одночасного виграшу відносно A виконується у **{summary["joint_passes"]}/{len(rows)}** запусків. SD — вибіркове стандартне відхилення між seed, не довірчий інтервал якості на нових відео.\n'
        if summary["joint_passes"] == len(rows):
            s += "\nПеревага за обома mAP повторилася в усіх перевірених seed за цих фіксованих умов.\n"
        elif summary["video"]["mean"] > a["video"] and summary["cub"]["mean"] >= a["cub"]:
            s += "\nСередні mAP кращі за A, але перевага не повторюється в кожному seed: результат залежить від випадковості навчання.\n"
        else:
            s += "\nСтійкий одночасний виграш за обома mAP не підтверджений; оцінювати D як гарантоване покращення не слід.\n"
    else:
        s += "\nСередні й остаточний висновок будуть опубліковані після всіх запусків. Часткові результати не є завершеним дослідженням.\n"
    s += """
## Контроль досліду

Зафіксовані хеші стартових ваг, навчального списку, усіх навчальних/валідаційних зображень і labels, коду та версій бібліотек. Захищені параметри перевіряються після епох. Нові ваги не підміняють реліз і найкращий seed автоматично не обирається.

Завершені оцінки повторно не запускаються. Незавершена спроба зберігається; seed можна перезапустити з A в іншому каталозі. Публікуються всі заплановані seed, а не лише вдалі.

Відтворення з підготовленими даними:

```bash
python seed_study.py
```

Протокол — [protocol.json](protocol.json), точні результати — [summary.json](summary.json); окремі оцінки та параметри — у каталогах seed_*. У великій відеопопуляції це не доводить переваги: відеовалідація охоплює лише два ролики.
"""
    (directory / "README.md").write_text(s)
    return summary


def publish(directory):
    rel = str(directory.relative_to(ROOT))
    env = os.environ
    cmd = ["git"]
    if env.get("BIRD_STUDY_GIT_NAME"):
        cmd += [
            "-c",
            "user.name=" + env["BIRD_STUDY_GIT_NAME"],
            "-c",
            "user.email=" + env["BIRD_STUDY_GIT_EMAIL"],
        ]
    subprocess.run(["git", "add", "--", rel], cwd=ROOT, check=True)
    if subprocess.run(["git", "diff", "--cached", "--quiet", "--", rel], cwd=ROOT).returncode:
        subprocess.run(
            cmd
            + [
                "commit",
                "--only",
                "-m",
                "Завершити дослідження стійкості D на п’яти seed",
                "--",
                rel,
            ],
            cwd=ROOT,
            check=True,
        )
    for attempt in range(3):
        if subprocess.run(["git", "push", "origin", "HEAD"], cwd=ROOT).returncode == 0:
            return
        time.sleep(5)
    raise RuntimeError("Не вдалося опублікувати звіт; локальні результати збережені.")


def main():
    p = UkrainianParser(description=__doc__)
    p.add_argument("--prepare-only", action="store_true")
    p.add_argument("--publish", action="store_true")
    p.add_argument("--archive-out", type=Path)
    args = p.parse_args()
    directory = ROOT / "reports/seed_study"
    directory.mkdir(parents=True, exist_ok=True)
    (ROOT / "work").mkdir(exist_ok=True)
    lock = (ROOT / "work/seed-study.lock").open("w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        p.error("Дослідження вже працює.")
    source = ROOT / "weights/photo_reference.pt"
    if sha(source) != SOURCE_SHA:
        raise ValueError("Потрібні початкові ваги A із зафіксованим хешем.")
    config = inputs()
    a_c = json.loads((ROOT / "reports/expanded_candidate_validation.json").read_text())
    a_v = json.loads((ROOT / "reports/video_training_baseline.json").read_text())
    a_b = json.loads((ROOT / "reports/video_backgrounds_baseline.json").read_text())
    plan = {
        "seeds": [26, 27, 28, 29, 30],
        "criterion": "Обидві validation mAP50–95 ≥ A, принаймні одна > A; фонові FP ≤ A.",
        "data_seed": 26,
        "source_weights_sha256": SOURCE_SHA,
        "epochs": 3,
        "cub_images": 512,
        "baseline_a": {
            "cub": a_c["metrics"]["metrics/mAP50-95(B)"],
            "video": a_v["metrics"]["metrics/mAP50-95(B)"],
            "background_fp_images": a_b["images_with_false_positives"],
        },
        **config,
    }
    pp = directory / "protocol.json"
    if pp.exists() and json.loads(pp.read_text()) != plan:
        raise ValueError("Протокол або дані змінилися; не змішуємо різні досліди.")
    write(pp, plan)
    original = ROOT / "work/video-more-photos/train.txt"
    if original.exists():
        names = [Path(x).name for x in original.read_text().splitlines()]
        if hashlib.sha256("\n".join(names).encode()).hexdigest() != plan["train_selection_sha256"]:
            raise ValueError("Дані seed 26 не збігаються з планом.")
    seed26 = directory / "seed_26"
    seed26.mkdir(exist_ok=True)
    for src, dst in [
        ("profile_d_video_validation.json", "video.json"),
        ("profile_d_cub_validation.json", "cub.json"),
        ("profile_d_backgrounds.json", "backgrounds.json"),
        ("profile_d_protocol.json", "training_protocol.json"),
        ("profile_d_protected.json", "protected.json"),
        ("profile_d_training_history.csv", "training_history.csv"),
    ]:
        if not (seed26 / dst).exists():
            shutil.copyfile(ROOT / "reports" / src, seed26 / dst)
    report(directory, plan, "підготовлено" if args.prepare_only else "виконується")
    if args.prepare_only:
        return
    snapshot = ROOT / "work/seed-study/source.pt"
    snapshot.parent.mkdir(exist_ok=True)
    if not snapshot.exists():
        shutil.copyfile(source, snapshot)
    if sha(snapshot) != SOURCE_SHA:
        raise ValueError("Невірний snapshot A.")
    try:
        for seed in plan["seeds"][1:]:
            folder = directory / f"seed_{seed}"
            folder.mkdir(exist_ok=True)
            if all((folder / f).exists() for f in ["video.json", "cub.json", "backgrounds.json"]):
                continue
            if inputs() != config:
                raise ValueError("Дані, код чи середовище змінилися під час досліду.")
            run = None
            for attempt in [1, 2]:
                name = f"d-seed-{seed}-attempt-{attempt}"
                run = ROOT / "runs/experiments" / name
                if run.exists():
                    csvfile = run / "results.csv"
                    n = len(list(csv.DictReader(csvfile.open()))) if csvfile.exists() else 0
                    if n >= 3 and (run / "weights/best.pt").exists():
                        break
                    continue
                write(
                    directory / "status.json",
                    {
                        "state": "training",
                        "seed": seed,
                        "attempt": attempt,
                        "pid": os.getpid(),
                        "updated_utc": datetime.now(timezone.utc).isoformat(),
                    },
                )
                with (folder / f"attempt_{attempt}.log").open("a") as log:
                    result = subprocess.run(
                        [
                            sys.executable,
                            str(ROOT / "train_video.py"),
                            "--model",
                            str(snapshot),
                            "--cub-images",
                            "512",
                            "--data-seed",
                            "26",
                            "--seed",
                            str(seed),
                            "--lr0",
                            "0.00001",
                            "--name",
                            name,
                        ],
                        cwd=ROOT,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                    )
                if result.returncode == 0:
                    history = run / "results.csv"
                    epochs_done = (
                        len(list(csv.DictReader(history.open()))) if history.exists() else 0
                    )
                    if epochs_done >= 3 and (run / "weights/best.pt").exists():
                        break
            else:
                raise RuntimeError(
                    f"Seed {seed}: обидві спроби незавершені; завершені seed збережені."
                )
            protocol = json.loads((run / "protocol.json").read_text())
            if (
                protocol["source_weights_sha256"] != SOURCE_SHA
                or protocol["train_selection_sha256"] != plan["train_selection_sha256"]
                or protocol["data_seed"] != 26
                or protocol["seed"] != seed
            ):
                raise ValueError("Порушено контроль досліду.")
            protected = json.loads((run / "protected_parameters.json").read_text())
            if not protected["unchanged"]:
                raise ValueError("Змінилися захищені параметри.")
            for src, dst in [
                ("protocol.json", "training_protocol.json"),
                ("protected_parameters.json", "protected.json"),
                ("results.csv", "training_history.csv"),
            ]:
                shutil.copyfile(run / src, folder / dst)
            weight = run / "weights/best.pt"
            write(
                directory / "status.json",
                {
                    "state": "evaluating_validation",
                    "seed": seed,
                    "updated_utc": datetime.now(timezone.utc).isoformat(),
                },
            )
            commands = [
                ("video.json", ["evaluate.py", "--data", "datasets/bird-video/data.yaml"]),
                ("cub.json", ["evaluate.py", "--data", "datasets/bird-expanded/data.yaml"]),
                ("backgrounds.json", ["evaluate_video_backgrounds.py"]),
            ]
            for target, command in commands:
                if (folder / target).exists():
                    continue
                command += ["--weights", str(weight), "--report", str(folder / target)]
                if target != "backgrounds.json":
                    command += [
                        "--split",
                        "val",
                        "--imgsz",
                        "640",
                        "--batch",
                        "4",
                        "--device",
                        "cpu",
                    ]
                with (folder / "evaluation.log").open("a") as log:
                    subprocess.run(
                        [sys.executable, *command],
                        cwd=ROOT,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        check=True,
                    )
            report(directory, plan, "виконується")
        summary = report(directory, plan, "завершено")
        assert summary["completed"]
        write(
            directory / "status.json",
            {
                "state": "completed",
                "completed_seeds": len(plan["seeds"]),
                "finished_utc": datetime.now(timezone.utc).isoformat(),
            },
        )
        if args.publish:
            publish(directory)
        if args.archive_out:
            args.archive_out.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                [
                    "git",
                    "archive",
                    "--format=zip",
                    "--prefix=bird-detector/",
                    "--output=" + str(args.archive_out),
                    "HEAD",
                ],
                cwd=ROOT,
                check=True,
            )
            args.archive_out.with_suffix(args.archive_out.suffix + ".sha256").write_text(
                sha(args.archive_out) + "  " + args.archive_out.name + "\n"
            )
            shutil.copyfile(directory / "README.md", args.archive_out.parent / "D_SEED_STUDY_UK.md")
    except BaseException as exc:
        write(
            directory / "status.json",
            {
                "state": "failed",
                "error": str(exc),
                "updated_utc": datetime.now(timezone.utc).isoformat(),
            },
        )
        report(directory, plan, "помилка — часткові результати збережено")
        raise


if __name__ == "__main__":
    main()
