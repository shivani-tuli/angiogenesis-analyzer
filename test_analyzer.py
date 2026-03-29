"""
test_analyzer.py
================
Quick validation of the analysis pipeline using synthetic and
downloaded sample images.
"""

import os
import sys
import numpy as np
import cv2

# Make sure we can import the analyzer
sys.path.insert(0, os.path.dirname(__file__))
import analyzer as engine


def _make_synthetic_network(h: int = 400, w: int = 400) -> np.ndarray:
    """
    Draw a synthetic tube-formation network on a black background.
    Creates a realistic-ish pattern with known topology.
    """
    img = np.zeros((h, w), dtype=np.uint8)

    # Draw "tubes" as thick lines
    pts = [
        # Horizontal tubes
        ((50, 50), (350, 50)),
        ((50, 150), (350, 150)),
        ((50, 250), (350, 250)),
        ((50, 350), (350, 350)),
        # Vertical tubes
        ((50, 50), (50, 350)),
        ((150, 50), (150, 350)),
        ((250, 50), (250, 350)),
        ((350, 50), (350, 350)),
        # Diagonal connections
        ((50, 50), (150, 150)),
        ((250, 150), (350, 250)),
        ((150, 250), (250, 350)),
    ]
    for p1, p2 in pts:
        cv2.line(img, p1, p2, 255, thickness=3)

    return img


def test_synthetic():
    """Run analysis on a synthetic network image."""
    print("=" * 60)
    print("TEST 1: Synthetic network image")
    print("=" * 60)

    img = _make_synthetic_network()
    result = engine.analyze(
        img,
        image_name="synthetic_network",
        image_type="fluorescence",
        sigma=0.0,              # no blur — already clean
        percentile=50.0,
        min_object_size=10,
        max_loop_area=200,
        twig_threshold=10,
    )

    print("\nResults:")
    for k, v in result.as_dict().items():
        print(f"  {k:35s} {v}")

    # Sanity checks
    assert result.nb_junctions > 0, "Expected junctions in synthetic network"
    assert result.nb_segments > 0, "Expected segments in synthetic network"
    assert result.tot_length > 0, "Expected nonzero total length"
    print("\n✅  Synthetic test PASSED")
    return result


def test_sample_images():
    """Run analysis on any sample images that exist."""
    sample_dir = os.path.join(os.path.dirname(__file__), "sample_images")
    if not os.path.isdir(sample_dir):
        print("\n⚠  No sample_images/ directory — run download_samples.py first.")
        return

    images = [f for f in os.listdir(sample_dir)
              if f.lower().endswith(('.png', '.jpg', '.tif', '.tiff'))]
    if not images:
        print("\n⚠  No images found in sample_images/.")
        return

    results = []
    for fname in sorted(images):
        print("\n" + "=" * 60)
        print(f"TEST: {fname}")
        print("=" * 60)
        path = os.path.join(sample_dir, fname)
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"  ✗ Could not read {path}")
            continue

        # Auto-detect type
        img_type = "fluorescence" if "fluo" in fname.lower() else "phase_contrast"
        result = engine.analyze(img, image_name=fname, image_type=img_type)

        print(f"\nResults for {fname}:")
        for k, v in result.as_dict().items():
            print(f"  {k:35s} {v}")
        results.append(result)

        # Save overlay
        overlay = engine.render_overlay(img, result)
        out_path = os.path.join(sample_dir, f"analyzed_{fname}")
        cv2.imwrite(out_path, overlay)
        print(f"  → Overlay saved: {out_path}")

    if results:
        csv_path = os.path.join(sample_dir, "test_results.csv")
        engine.export_csv(results, csv_path)
        print(f"\n📊  Combined results → {csv_path}")
        print(f"\n✅  Sample image tests PASSED ({len(results)} images)")


if __name__ == "__main__":
    r = test_synthetic()
    test_sample_images()
    print("\n🎉  All tests complete!")
