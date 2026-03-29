"""
download_samples.py
===================
Download public-domain sample tube-formation assay images from the
ImageJ Angiogenesis Analyzer website for testing.
"""

import os
import urllib.request

SAMPLE_DIR = os.path.join(os.path.dirname(__file__), "sample_images")

# Public-domain sample images from Gilles Carpentier's site
# (these are the demonstration images that ship with the plugin)
SAMPLES = {
    "HUVEC_phase_contrast.png": "http://image.bio.methods.free.fr/ImageJ/IMG/png/t8h-pseudo-phase.png",
    "HUVEC_fluorescence.png": "http://image.bio.methods.free.fr/ImageJ/IMG/png/fluo.png",
    "HUVEC_phase_contrast_skeleton.png": "http://image.bio.methods.free.fr/ImageJ/IMG/png/binary-huvec-fluo-tree-tr.png",
}


def download():
    os.makedirs(SAMPLE_DIR, exist_ok=True)
    for name, url in SAMPLES.items():
        dest = os.path.join(SAMPLE_DIR, name)
        if os.path.exists(dest):
            print(f"  ✓ {name} (already exists)")
            continue
        print(f"  ↓ Downloading {name} …")
        try:
            urllib.request.urlretrieve(url, dest)
            print(f"    Saved → {dest}")
        except Exception as e:
            print(f"    ✗ Failed: {e}")
    print(f"\nSample images saved to: {SAMPLE_DIR}")


if __name__ == "__main__":
    download()
