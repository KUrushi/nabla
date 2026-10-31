#!/usr/bin/env python3
"""nabla/data/image の画像のフィクスチャを作る（issue #283）。

nabla のライブラリ本体は Python に依存しない（CLAUDE.md）。これは、フィクスチャを
作るときにだけ使う開発用スクリプトで、venv 自体はコミットしない。

    python3 -m venv /tmp/venv && /tmp/venv/bin/pip install pillow==13.0.0 numpy pypng==0.20220715.0  # numpy 2.5.4
    /tmp/venv/bin/python tests/fixtures/image/generate.py

画像（*.png / *.jpg）と、Pillow でデコードした画素値 expected.lisp を書く。
expected.lisp は (名前 (H W C) (画素値 ...)) の並びで、画素値は (H W C) の行優先。
グレースケールは C = 1、パレットは RGB に、1 ビットのグレースケールは 0 / 255 に直した値
（nabla/data/image の read-image の約束）。インタレースの PNG は Pillow が書けないので
pypng で書き、期待値は Pillow でデコードした値にする。
"""


import os

import numpy as np
import png
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
H, W = 5, 7  # H と W を変えて、縦横の取り違えを見つける


def noise(channels, seed, height=H, width=W):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(height, width, channels), dtype=np.uint8)


def with_app14(name, source, transform):
    """SOURCE の JPEG の SOI の直後に Adobe の APP14（"Adobe"、版 100、flags 0 0、色変換
    TRANSFORM）を差し込んだ NAME を書く。"""
    with open(os.path.join(HERE, source), "rb") as f:
        data = f.read()
    app14 = b"\xff\xee\x00\x0e" + b"Adobe" + b"\x00\x64" + b"\x00\x00" + b"\x00\x00" + bytes([transform])
    path = os.path.join(HERE, name)
    with open(path, "wb") as f:
        f.write(data[:2] + app14 + data[2:])
    return path


def smooth(channels):
    """JPEG 用のなめらかな画像（量子化の誤差を小さくする）。"""
    y, x = np.mgrid[0:16, 0:24]  # 4:2:0 の MCU（16 x 16）の途中で切れる幅
    planes = [x * 5 + y * 4 + 25 * c for c in range(channels)]
    return np.stack(planes, axis=-1).astype(np.uint8)


def expected(image):
    """Pillow でデコードした画素を (H W C) の uint8 にする。"""
    if image.mode in ("P", "1"):
        image = image.convert("RGB" if image.mode == "P" else "L")
    array = np.asarray(image, dtype=np.uint8)
    if array.ndim == 2:
        array = array[:, :, None]
    return array


def save(name, image, **options):
    path = os.path.join(HERE, name)
    image.save(path, **options)
    return path


def main():
    paths = []
    paths.append(save("rgb.png", Image.fromarray(noise(3, 1), "RGB")))
    paths.append(save("rgba.png", Image.fromarray(noise(4, 2), "RGBA")))
    paths.append(save("gray.png", Image.fromarray(noise(1, 3)[:, :, 0], "L")))
    paths.append(save("gray-alpha.png", Image.fromarray(noise(2, 4), "LA")))
    paths.append(save("palette.png", Image.fromarray(noise(3, 5), "RGB").quantize(colors=6)))
    paths.append(save("bilevel.png", Image.fromarray(noise(1, 6)[:, :, 0] > 127)))
    interlaced = os.path.join(HERE, "interlaced.png")
    with open(interlaced, "wb") as f:
        png.Writer(W, H, greyscale=False, interlace=True).write(f, noise(3, 7).reshape(H, W * 3))
    paths.append(interlaced)
    # 長さが 256 以上のチャンク（IDAT）を持つ PNG（チャンクの長さの読み方を確かめる）
    paths.append(save("noise.png", Image.fromarray(noise(3, 9, 20, 30), "RGB")))
    paths.append(save("baseline.jpg", Image.fromarray(smooth(3), "RGB"), quality=95, subsampling=0))
    paths.append(save("subsampled.jpg", Image.fromarray(smooth(3), "RGB"), quality=95, subsampling=2))
    paths.append(save("progressive.jpg", Image.fromarray(smooth(3), "RGB"), quality=95,
                      subsampling=0, progressive=True))
    paths.append(save("gray.jpg", Image.fromarray(smooth(1)[:, :, 0], "L"), quality=95))
    # APP14 が YCbCr（色変換 1）と言う3成分の JPEG と、APP14 の色変換 0 を持つグレースケールは読む
    paths.append(with_app14("adobe-ycbcr.jpg", "baseline.jpg", 1))
    paths.append(with_app14("adobe-gray.jpg", "gray.jpg", 0))

    # 未対応の形式（期待値なし。テストは :unsupported のコンディションを確かめる）
    with open(os.path.join(HERE, "gray16.png"), "wb") as f:
        png.Writer(W, H, greyscale=True, bitdepth=16).write(f, noise(1, 8)[:, :, 0].astype(np.uint16) * 257)
    save("cmyk.jpg", Image.fromarray(smooth(4), "CMYK"), quality=95)
    # Adobe の APP14 で色変換 0（YCbCr でない。RGB のまま）と言う3成分の JPEG
    with_app14("adobe-rgb.jpg", "baseline.jpg", 0)

    with open(os.path.join(HERE, "expected.lisp"), "w") as f:
        f.write(";;;; nabla/data/image のフィクスチャ（issue #283）の、Pillow でデコードした画素値。\n")
        f.write(";;;;\n;;;; tests/fixtures/image/generate.py で生成した。手で編集しない。\n(\n")
        for path in paths:
            with Image.open(path) as image:
                array = expected(image)
            shape = " ".join(str(d) for d in array.shape)
            values = " ".join(str(v) for v in array.reshape(-1))
            f.write(f' ("{os.path.basename(path)}" ({shape}) ({values}))\n')
        f.write(")\n")


if __name__ == "__main__":
    main()
