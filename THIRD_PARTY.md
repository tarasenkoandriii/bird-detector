# Sources and licenses

## Dataset (downloaded separately)

CUB-200-2011, Wah, Branson, Welinder, Perona and Belongie, Caltech, 2011.
Original: https://www.vision.caltech.edu/datasets/cub_200_2011/
Mirror with images and boxes: https://huggingface.co/datasets/bentrevett/caltech-ucsd-birds-200-2011
Pinned revision: `1ef09e021b0b65b40337f6f285909656f407f6e0`.

The original source restricts image use to non-commercial research and education; image copyrights remain with their owners. Training images are **not included in this repository**. Preparation downloads the research subset separately. This project and its bundled experimental weights are presented as a research/educational demonstration, with no assertion of commercial suitability. Inspect the original terms before other use.

## Example video and derived preview images

**“Blue Jay at bird feeder, video”**, Matt MacGillivray, 30 June 2009.

- Commons file: https://commons.wikimedia.org/wiki/File:Blue_Jay_at_bird_feeder,_video_(3675639120).webm
- Original Flickr source: https://www.flickr.com/photos/qmnonic/3675639120/
- License: [Creative Commons Attribution 2.0](https://creativecommons.org/licenses/by/2.0/).
- Changes for `examples/input.mp4`: excerpt from 2–10 seconds, resize to 640×360, H.264 encoding, audio removed.
- Changes for `examples/output.mp4` and `frame-*.jpg`: model-generated boxes, class name and confidence overlays; still frames extracted from that result. Attribution and the CC BY 2.0 source license also apply to this example material.
- The video is for qualitative demonstration and has no manually verified per-frame detection labels. It was not used for training or numerical test metrics.

## Software

Project Python code is supplied under GNU AGPL-3.0; see LICENSE. Ultralytics YOLO and its pretrained weights are distributed under the terms described at https://www.ultralytics.com/license and https://github.com/ultralytics/ultralytics . Third-party dependencies retain their respective licenses. The video and dataset terms above are separate from the software license.
