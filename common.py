"""Спільні шляхи та вибір пристрою. Імпорт torch — після налаштування MPS."""

import os
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
ROOT = Path(__file__).resolve().parent


def resolve_device(requested):
    import torch

    torch.set_num_threads(min(6, os.cpu_count() or 1))
    if requested == "auto":
        return (
            "mps"
            if torch.backends.mps.is_available()
            else ("0" if torch.cuda.is_available() else "cpu")
        )
    if requested == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS недоступний. Використайте --device cpu або --device auto.")
    return requested


import argparse
import sys


class UkrainianParser(argparse.ArgumentParser):
    """Українська оболонка стандартного argparse."""

    def __init__(self, **kwargs):
        super().__init__(add_help=False, **kwargs)
        self.add_argument("-h", "--help", action="help", help="Показати довідку й завершити")
        self._positionals.title = "Позиційні аргументи"
        self._optionals.title = "Параметри"

    def format_help(self):
        return super().format_help().replace("usage: ", "Використання: ", 1)

    def format_usage(self):
        return super().format_usage().replace("usage: ", "Використання: ", 1)

    def error(self, message):
        for old, new in [
            ("unrecognized arguments:", "невідомі аргументи:"),
            ("the following arguments are required:", "обов’язкові аргументи:"),
            ("invalid choice:", "неприпустимий варіант:"),
            ("choose from", "доступні варіанти"),
            ("invalid int value:", "очікується ціле число:"),
            ("invalid float value:", "очікується число:"),
            ("argument ", "аргумент "),
        ]:
            message = message.replace(old, new)
        self.print_usage(sys.stderr)
        self.exit(2, f"{self.prog}: помилка: {message}\n")
