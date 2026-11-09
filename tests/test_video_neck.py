"""Легкі перевірки neck/head scope та захисту BN; без forward, навчання чи val."""
import collections
import contextlib
import io
import unittest
from pathlib import Path

import torch
import ultralytics
import yaml

from train_video_neck import (bn_state, changed_tensors, class_schema, freeze_bn, layer_groups,
                              parser, select_training_images, trainable_parameter)


class VideoNeckTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        config = Path(ultralytics.__file__).parent / 'cfg/models/26/yolo26.yaml'
        cls.groups = layer_groups(yaml.safe_load(config.read_text()))

    def test_layer_boundaries_come_from_actual_yaml(self):
        self.assertEqual(self.groups['backbone'], list(range(11)))
        self.assertEqual(self.groups['neck'], list(range(11, 23)))
        self.assertEqual(self.groups['head'], [23])
        with self.assertRaises(ValueError):
            layer_groups(dict(backbone=[[]], head=[[-1, 1, 'Classify', []]]))

    def test_scope_keeps_backbone_and_dfl_frozen(self):
        for scope in ['head', 'neck-head']:
            self.assertFalse(trainable_parameter('model.10.cv1.conv.weight', self.groups, scope))
            self.assertTrue(trainable_parameter('model.23.cv3.0.2.weight', self.groups, scope))
            self.assertFalse(trainable_parameter('model.23.dfl.conv.weight', self.groups, scope))
            self.assertFalse(trainable_parameter('model.24.weight', self.groups, scope))
        self.assertFalse(trainable_parameter('model.13.cv1.conv.weight', self.groups, 'head'))
        self.assertTrue(trainable_parameter('model.13.cv1.conv.weight', self.groups, 'neck-head'))
        self.assertTrue(trainable_parameter('model.22.cv1.conv.weight', self.groups, 'neck-head'))

    def test_raw_and_finetuned_class_schemas_are_distinct(self):
        self.assertEqual(class_schema({0: 'bird'}), 'single-bird')
        coco = {i: f'class-{i}' for i in range(80)}
        coco[14] = 'bird'
        self.assertEqual(class_schema(coco), 'coco80')
        for invalid in [{0: 'person'}, {0: 'bird', 1: 'other'}, {14: 'bird'}]:
            with self.assertRaises(ValueError):
                class_schema(invalid)

    def test_bn_eval_preserves_affine_choice_and_guard_detects_buffer_changes(self):
        model = torch.nn.Sequential(torch.nn.Conv2d(2, 2, 1), torch.nn.BatchNorm2d(2))
        model.train()
        model[1].weight.requires_grad_(False)
        snapshot = bn_state(model)
        freeze_bn(model)
        self.assertTrue(model[0].training)
        self.assertFalse(model[1].training)
        self.assertFalse(model[1].weight.requires_grad)
        self.assertTrue(model[1].bias.requires_grad)
        self.assertEqual(changed_tensors(snapshot, dict(model.named_buffers())), [])
        with torch.no_grad():
            model[1].running_mean.add_(1)
        self.assertEqual(changed_tensors(snapshot, dict(model.named_buffers())), ['1.running_mean'])

    def test_dataset_sampling_is_fixed_and_extra_multiplicity_explicit(self):
        images = [Path('normal.jpg'), Path('extra-a.jpg'), Path('extra-b.jpg')]
        photos = [Path(f'photo-{i}.jpg') for i in range(8)]
        original = list(photos)
        first = select_training_images(images, photos, cub_images=3, extra_repeat=3, data_seed=26)
        second = select_training_images(images, photos, cub_images=3, extra_repeat=3, data_seed=26)
        self.assertEqual(first, second)
        self.assertEqual(photos, original)
        counts = collections.Counter(first)
        self.assertEqual(counts[Path('extra-a.jpg')], 3)
        self.assertEqual(counts[Path('extra-b.jpg')], 3)
        self.assertEqual(counts[Path('normal.jpg')], 1)
        self.assertEqual(len(first), 10)

    def test_cli_has_separate_training_and_data_seed_and_bounded_budget(self):
        args = parser().parse_args(['--seed', '30', '--data-seed', '26', '--epochs', '3'])
        self.assertEqual((args.seed, args.data_seed, args.epochs), (30, 26, 3))
        self.assertEqual(parser().parse_args(['--epochs', '5']).epochs, 5)
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            parser().parse_args(['--epochs', '6'])


if __name__ == '__main__':
    unittest.main()
