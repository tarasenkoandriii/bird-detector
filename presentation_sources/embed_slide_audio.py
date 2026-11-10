#!/usr/bin/env python3
"""Embed a click-to-play MP3 narration clip in each presentation slide."""
from __future__ import annotations

import re
import subprocess
import sys
import zipfile
from pathlib import Path

REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
REL_AUDIO = "http://schemas.openxmlformats.org/officeDocument/2006/relationships/audio"
REL_MEDIA = "http://schemas.microsoft.com/office/2007/relationships/media"
REL_IMAGE = "http://schemas.openxmlformats.org/officeDocument/2006/relationships/image"
EMU = 914400


def player_xml(shape_id: int, number: int) -> str:
    x, y, size = round(12.78 * EMU), round(0.18 * EMU), round(0.38 * EMU)
    return (
        '<p:pic><p:nvPicPr>'
        f'<p:cNvPr id="{shape_id}" name="Озвучення {number:02d}">'
        '<a:hlinkClick r:id="" action="ppaction://media"/></p:cNvPr>'
        '<p:cNvPicPr><a:picLocks noChangeAspect="1"/></p:cNvPicPr>'
        '<p:nvPr>'
        '<a:audioFile r:link="AUDIO_REL"/>'
        '<p:extLst><p:ext uri="{DAA4B4D4-6D71-4841-9C94-3DE7FCFB9230}">'
        '<p14:media xmlns:p14="http://schemas.microsoft.com/office/powerpoint/2010/main" '
        'r:embed="MEDIA_REL"/></p:ext></p:extLst>'
        '</p:nvPr></p:nvPicPr>'
        '<p:blipFill><a:blip r:embed="IMAGE_REL"/><a:stretch><a:fillRect/></a:stretch></p:blipFill>'
        '<p:spPr><a:xfrm><a:off x="%d" y="%d"/><a:ext cx="%d" cy="%d"/></a:xfrm>'
        '<a:prstGeom prst="rect"><a:avLst/></a:prstGeom></p:spPr></p:pic>' % (x, y, size, size)
    )


def encode_audio(flac_dir: Path, mp3_dir: Path) -> None:
    mp3_dir.mkdir(parents=True, exist_ok=True)
    for number in range(1, 61):
        stem = f"slide_{number:02d}"
        source = flac_dir / f"{stem}.flac"
        target = mp3_dir / f"{stem}.mp3"
        if not source.exists():
            raise FileNotFoundError(source)
        subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-i", str(source), "-ac", "1", "-ar", "24000",
             "-codec:a", "libmp3lame", "-b:a", "64k", str(target)],
            check=True,
        )


def main(source: Path, flac_dir: Path, icon: Path, output: Path, mp3_dir: Path) -> None:
    encode_audio(flac_dir, mp3_dir)
    icon_data = icon.read_bytes()
    with zipfile.ZipFile(source) as original:
        files = {name: original.read(name) for name in original.namelist()}

    for number in range(1, 61):
        slide_name = f"ppt/slides/slide{number}.xml"
        rel_name = f"ppt/slides/_rels/slide{number}.xml.rels"
        xml = files[slide_name].decode("utf-8")
        rels = files[rel_name].decode("utf-8")
        existing_ids = re.findall(r'\bId="rId(\d+)"', rels)
        next_id = max(map(int, existing_ids), default=0) + 1
        audio_id, media_id, image_id = (f"rId{next_id + offset}" for offset in range(3))
        shape_ids = [int(value) for value in re.findall(r'<p:cNvPr\s+id="(\d+)"', xml)]
        block = (player_xml(max(shape_ids, default=1) + 1, number)
                 .replace("AUDIO_REL", audio_id)
                 .replace("MEDIA_REL", media_id)
                 .replace("IMAGE_REL", image_id))

        if "</p:spTree>" not in xml:
            raise ValueError(f"No shape tree found in {slide_name}")
        xml = xml.replace("</p:spTree>", block + "</p:spTree>", 1)
        media_target = f"../media/audio-slide-{number:02d}.mp3"
        new_rels = (
            f'<Relationship Id="{audio_id}" Type="{REL_AUDIO}" Target="{media_target}"/>'
            f'<Relationship Id="{media_id}" Type="{REL_MEDIA}" Target="{media_target}"/>'
            f'<Relationship Id="{image_id}" Type="{REL_IMAGE}" Target="../media/audio-player.png"/>'
        )
        rels = rels.replace("</Relationships>", new_rels + "</Relationships>")
        files[slide_name] = xml.encode("utf-8")
        files[rel_name] = rels.encode("utf-8")
        files[f"ppt/media/audio-slide-{number:02d}.mp3"] = (
            mp3_dir / f"slide_{number:02d}.mp3"
        ).read_bytes()

    files["ppt/media/audio-player.png"] = icon_data
    content_types = files["[Content_Types].xml"].decode("utf-8")
    if not re.search(r'<Default\s+Extension="mp3"', content_types):
        content_types = content_types.replace(
            "</Types>", '<Default Extension="mp3" ContentType="audio/mp3"/></Types>'
        )
    files["[Content_Types].xml"] = content_types.encode("utf-8")

    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as package:
        for name, data in files.items():
            package.writestr(name, data)
    print(f"Embedded 60 slide-specific audio clips: {output}")


if __name__ == "__main__":
    if len(sys.argv) != 6:
        raise SystemExit(
            "Usage: embed_slide_audio.py source.pptx flac_dir icon.png output.pptx mp3_cache_dir"
        )
    main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4]), Path(sys.argv[5]))
