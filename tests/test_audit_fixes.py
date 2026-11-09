"""Регресії підтверджених дефектів вибору моделі, provenance та перезапису."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import compare_profiles
import evaluate
from report_io import write_json, dataset_snapshot


class ProfileAudit(unittest.TestCase):
    def profiles(self):
        reports = {}
        for kind in ['video', 'cub', 'background']:
            expected = (dict(split='val', imgsz=640, conf=.25, post_nms_iou=.5, images=8)
                        if kind == 'background' else dict(split='val', device='cpu', imgsz=640, batch=4))
            reports[kind] = dict(expected, dataset={'sha256': kind+'-data', 'images':dict(video=32,cub=600,background=8)[kind]},
                                 evaluator={'name': kind, 'conf': .001})
        a = dict(id='A', status='виміряно', video_map50_95=.51, cub_map50_95=.81,
                 negative_images_with_fp=0, _reports=reports)
        d = copy.deepcopy(a);d.update(id='D', video_map50_95=.52, cub_map50_95=.82)
        return [a, d]

    def test_degraded_D_is_not_selected_and_tie_is_not_improvement(self):
        profiles = self.profiles()
        self.assertEqual(compare_profiles.select_profile(profiles), 'D')
        profiles[1].update(video_map50_95=.4, cub_map50_95=.7, negative_images_with_fp=5)
        self.assertEqual(compare_profiles.select_profile(profiles), 'A')
        profiles[1].update(video_map50_95=.51, cub_map50_95=.81, negative_images_with_fp=0)
        self.assertEqual(compare_profiles.select_profile(profiles), 'A')

    def test_mixed_split_size_data_or_evaluator_rejected(self):
        for field, value in [('split','test'), ('imgsz',320), ('dataset',{'sha256':'other'}),
                             ('evaluator',{'name':'other'})]:
            profiles = self.profiles();profiles[1]['_reports']['video'][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                compare_profiles.validate_comparability(profiles, allow_legacy=True)

    def test_missing_provenance_needs_explicit_legacy_mode(self):
        profiles = self.profiles()
        del profiles[1]['_reports']['cub']['dataset']
        with self.assertRaises(ValueError): compare_profiles.validate_comparability(profiles)
        self.assertFalse(compare_profiles.validate_comparability(profiles, allow_legacy=True))
        self.assertTrue(compare_profiles.validate_comparability(self.profiles()))

    def test_generated_report_does_not_claim_degraded_D_is_better(self):
        import contextlib
        import io
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);reports=root/'reports';reports.mkdir()
            for filename in ['video_training_baseline.json','expanded_candidate_validation.json',
                             'video_backgrounds_baseline.json','video_training_candidate.json',
                             'video_training_cub_validation.json','video_backgrounds_candidate.json',
                             'profile_c_video_validation.json','profile_c_cub_validation.json',
                             'profile_c_backgrounds.json','profile_d_video_validation.json',
                             'profile_d_cub_validation.json','profile_d_backgrounds.json']:
                value=json.loads((compare_profiles.R/filename).read_text())
                if filename=='profile_d_video_validation.json':value['metrics']['metrics/mAP50-95(B)']=.4
                if filename=='profile_d_cub_validation.json':value['metrics']['metrics/mAP50-95(B)']=.7
                if filename=='profile_d_backgrounds.json':value['images_with_false_positives']=5
                (reports/filename).write_text(json.dumps(value))
            output=root/'out'
            with patch.object(compare_profiles,'R',reports), patch('sys.argv',[
                    'compare_profiles.py','--allow-legacy','--out-dir',str(output)]), contextlib.redirect_stdout(io.StringIO()):
                compare_profiles.main()
            result=json.loads((output/'model_profiles.json').read_text())
            self.assertEqual(result['default_profile'],'A')
            markdown=(output/'model_profiles.md').read_text()
            self.assertNotIn('**D — основний варіант:** перевищив A',markdown)
            self.assertIn('Рекомендовано залишити A',markdown)


class ReportAudit(unittest.TestCase):
    def test_report_never_clobbered_without_opt_in(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'report.json';write_json(path, {'keep':1})
            with self.assertRaises(FileExistsError): write_json(path, {'keep':2})
            self.assertEqual(json.loads(path.read_text()), {'keep':1})
            write_json(path, {'keep':2}, overwrite=True)
            self.assertEqual(json.loads(path.read_text()), {'keep':2})

    def test_existing_evaluation_report_rejected_before_model_load(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);data=root/'data.yaml';weights=root/'weights.pt';out=root/'old.json'
            for p in [data,weights,out]:p.write_text('keep')
            with patch('sys.argv', ['evaluate.py','--data',str(data),'--weights',str(weights),'--report',str(out)]):
                with self.assertRaises(SystemExit) as exc: evaluate.main()
            self.assertEqual(exc.exception.code,2)
            self.assertEqual(out.read_text(),'keep')

    def test_label_change_changes_dataset_digest(self):
        import cv2
        import numpy as np
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);images=root/'images';labels=root/'labels';images.mkdir();labels.mkdir()
            cv2.imwrite(str(images/'sample.jpg'),np.zeros((32,32,3),dtype=np.uint8))
            label=labels/'sample.txt';label.write_text('0 0.5 0.5 0.5 0.5\n')
            config=dict(path=root,val=str(images),names={0:'bird'})
            a=dataset_snapshot(config,'val');label.write_text('0 0.5 0.5 0.25 0.25\n')
            b=dataset_snapshot(config,'val');self.assertEqual(a['images'],1)
            self.assertNotEqual(a['sha256'],b['sha256'])

if __name__=='__main__':unittest.main()
