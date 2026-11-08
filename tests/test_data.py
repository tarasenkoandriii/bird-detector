import tempfile, unittest
from pathlib import Path
from PIL import Image
from tools.dataset import audit, prepare


class DataTests(unittest.TestCase):
    def make(self, root):
        (root / "data.yaml").write_text("names: [bird, person]\n")
        for s, c in [("train", "red"), ("val", "green"), ("test", "blue")]:
            (root / s / "images").mkdir(parents=True)
            (root / s / "labels").mkdir()
            Image.new("RGB", (20, 20), c).save(root / s / "images" / f"{s}.png")
            (root / s / "labels" / f"{s}.txt").write_text("0 .5 .5 .2 .2\n1 .5 .5 .1 .1\n")

    def test_remap_and_remove_cross_split_copy(self):
        import shutil

        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self.make(root)
            shutil.copy(root / "test/images/test.png", root / "train/images/duplicate.png")
            shutil.copy(root / "test/labels/test.txt", root / "train/labels/duplicate.txt")
            report = audit(root)
            self.assertTrue(any(x["leakage"] for x in report["pairs"]))
            out = root / "clean"
            counts = prepare(report, out, 26, None)
            self.assertEqual(counts, {"train": 1, "val": 1, "test": 1})
            self.assertFalse(audit(out)["pairs"])
            for p in out.rglob("*.txt"):
                self.assertEqual(len(p.read_text().splitlines()), 1)

    def test_invalid_coordinates_block_preparation(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self.make(root)
            (root / "train/labels/train.txt").write_text("0 nan .5 .2 .2\n")
            report = audit(root)
            self.assertTrue(report["errors"])
            with self.assertRaises(ValueError):
                prepare(report, root / "clean", 26, None)


if __name__ == "__main__":
    unittest.main()
