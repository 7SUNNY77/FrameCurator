"""Тесты разбора ffprobe и слияния videos.csv (без вызова ffprobe).

JSON-фикстуры собраны вручную по формату ``ffprobe -print_format json`` и не являются
выводом реальных файлов.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import inventory_videos as inv  # noqa: E402

PHONE_HDR = {
    "streams": [
        {
            "codec_type": "video", "codec_name": "hevc", "width": 3840, "height": 2160,
            "r_frame_rate": "30/1", "avg_frame_rate": "17970000/599521",
            "pix_fmt": "yuv420p10le", "color_transfer": "arib-std-b67",
            "side_data_list": [{"side_data_type": "Display Matrix", "rotation": -90}],
        },
        {"codec_type": "audio", "codec_name": "aac"},
    ],
    "format": {
        "duration": "12.345678",
        "tags": {"com.apple.quicktime.make": "Apple", "com.apple.quicktime.model": "iPhone 15"},
    },
}  # fmt: skip

DRONE_SDR = {
    "streams": [
        {"codec_type": "video", "codec_name": "h264", "width": 1920, "height": 1080,
         "r_frame_rate": "30000/1001", "avg_frame_rate": "30000/1001", "pix_fmt": "yuv420p",
         "tags": {"rotate": "180"}},
        {"codec_type": "video", "codec_name": "mjpeg", "disposition": {"attached_pic": 1}},
    ],
    "format": {"duration": "180.0", "tags": {"encoder": "DJI Mini3"}},
}  # fmt: skip


def test_parse_phone_hdr():
    r = inv.parse_probe(PHONE_HDR)
    assert (r["width"], r["height"]) == (3840, 2160)
    assert r["hdr"] == 1
    assert r["rotation"] == 90  # Display Matrix -90 (против часовой) -> 90 по часовой
    assert r["device"] == "Apple iPhone 15"
    assert r["fps_nominal"] == 29.974
    assert r["duration"] == 12.346


def test_parse_drone_sdr():
    r = inv.parse_probe(DRONE_SDR)
    assert r["hdr"] == 0
    assert r["rotation"] == 180
    assert r["device"] == "dji"
    assert r["fps_nominal"] == 29.97
    assert r["codec"] == "h264"  # обложка mjpeg пропущена


def test_detect_vfr():
    cfr = [i / 30 for i in range(300)]
    assert inv.detect_vfr(cfr) == 0
    rng_gaps = [0.033] * 200 + [0.05] * 20 + [0.02] * 20
    vfr = list(__import__("itertools").accumulate(rng_gaps))
    assert inv.detect_vfr(vfr) == 1
    # Порядок декодирования (B-кадры) не важен
    assert inv.detect_vfr(cfr[::-1]) == 0


def test_video_id():
    assert inv.make_video_id("2026-09 Алтай", "DJI_0012") == "2026-09_Алтай__DJI_0012"
    assert inv.make_video_id("s1", "my clip (2)") == "s1__my_clip_2"


def test_find_videos(tmp_path):
    (tmp_path / "s1").mkdir()
    (tmp_path / "s1" / "a.MP4").touch()
    (tmp_path / "s1" / "a.LRF").touch()
    (tmp_path / "s2" / "sub").mkdir(parents=True)
    (tmp_path / "s2" / "sub" / "b.mov").touch()
    (tmp_path / "loose.mp4").touch()
    found = [(s, p.name) for s, p in inv.find_videos(tmp_path)]
    assert found == [("s1", "a.MP4"), ("s2", "b.mov")]


def test_merge_preserves_manual():
    new = [dict.fromkeys(inv.COLUMNS, "") | {"video_id": "s1__a", "session": "s1", "width": 1920}]
    old = [
        dict.fromkeys(inv.COLUMNS, "")
        | {"video_id": "s1__a", "session": "s1", "scene_type": "город", "has_people": "1",
           "width": 1280},
        dict.fromkeys(inv.COLUMNS, "") | {"video_id": "s9__gone", "session": "s9"},
    ]  # fmt: skip
    merged = inv.merge_existing(new, old)
    a = next(r for r in merged if r["video_id"] == "s1__a")
    assert a["scene_type"] == "город" and a["has_people"] == "1"
    assert a["width"] == 1920  # автоматические поля обновляются
    assert any(r["video_id"] == "s9__gone" for r in merged)
