# Angiogenesis Analyzer

A standalone desktop application for quantitative analysis of endothelial tube formation assays (ETFA). This tool is an independent Python reimplementation of the morphometric analysis algorithm described by Carpentier et al. (2020), built for accessibility and batch processing.

![Angiogenesis Analyzer Screenshot](docs/screenshot.png)

## Features

- **Complete Morphometric Analysis** — Segments, branches, junctions, meshes, extremities, master segments, and 20+ quantitative parameters
- **Batch Processing** — Analyze 50+ images in one click with automatic Excel export
- **Interactive Overlay** — Color-coded visualization of detected network elements
- **Crop Margins** — Exclude scale bars and legends from analysis
- **Cross-Platform** — Desktop app (macOS/Windows) and web app (any browser)
- **No ImageJ Required** — Fully standalone, zero dependencies on Java or ImageJ

## Quantitative Parameters

| Parameter | Description |
|---|---|
| Nb junctions | Number of junction clusters |
| Nb master junctions | Junctions connecting ≥3 master segments |
| Nb segments | Elements connecting two junctions |
| Nb master segments | Segments in the master tree (after pruning) |
| Nb branches | Elements connecting a junction to an extremity |
| Nb meshes | Enclosed areas in the master tree |
| Tot. length | Total length of all skeleton elements |
| Tot. branching length | Total length excluding isolated elements |
| Tot. segments length | Total length of all segments |
| Tot. master segments length | Total length of master segments |
| Tot. branches length | Total length of all branches |
| Branching interval | Mean distance between branches |
| Mesh index | Average number of meshes per junction |
| Analysed area | Total image area in pixels |

## Installation

### Desktop App (macOS)

```bash
# Clone the repository
git clone https://github.com/shivani-tuli/angiogenesis-analyzer.git
cd angiogenesis-analyzer

# Create virtual environment & install dependencies
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run the app
python app.py

# Or build a standalone .app bundle
python build.py
```

### From Source (any platform)

```bash
pip install -r requirements.txt
python app.py
```

## Usage

### Single Image Analysis
1. Open the app → **File → Open Image…**
2. Select your tube formation assay image
3. Adjust settings (image type, crop margins, etc.)
4. Click **▶ Analyze Image**

### Batch Processing
1. Click **📂 Batch Process Folder → Excel**
2. Select a folder containing your images
3. Results are automatically saved as `angiogenesis_results.xlsx` in that folder

### Keyboard Shortcuts
- **H** — Hide overlay
- **S** — Show overlay
- **B** — Blink/toggle overlay

## Algorithm

The analysis pipeline follows these steps:

1. **Preprocessing** — Grayscale conversion, Gaussian blur
2. **Segmentation** — Adaptive thresholding (phase contrast) or percentile thresholding (fluorescence)
3. **Skeletonization** — Binary thinning to single-pixel skeleton
4. **Artifact Removal** — Small loop elimination
5. **Node Detection** — Identify pixels with ≥3 neighbors, cluster into junctions
6. **Element Classification** — Classify paths as segments, branches, or twigs
7. **Master Tree Construction** — Iterative pruning for master segments/junctions
8. **Mesh Detection** — Find enclosed areas in the master tree
9. **Quantification** — Calculate all morphometric parameters

## Citation

If you use this tool in your research, please cite both:

1. This implementation:
```
Tuli, S. (2026). Angiogenesis Analyzer: A standalone Python application for
quantitative analysis of endothelial tube formation assays.
GitHub: https://github.com/shivani-tuli/angiogenesis-analyzer
```

2. The original algorithm:
```
Carpentier G, Berndt S, Ferratge S, Rasband W, Cuendet M, Uzan G, Albanese P.
Angiogenesis Analyzer for ImageJ — A comparative morphometric analysis of
"Endothelial Tube Formation Assay" and "Fibrin Bead Assay".
Sci Rep. 2020;10(1):11568. doi: 10.1038/s41598-020-67289-8
```

## Tech Stack

- **Python 3** — Core language
- **scikit-image** — Skeletonization, morphology
- **OpenCV** — Image I/O, preprocessing
- **NumPy / SciPy** — Numerical operations
- **skan** — Skeleton analysis utilities
- **Tkinter** — Desktop GUI
- **openpyxl** — Excel export
- **PyInstaller** — Desktop packaging

## License

MIT License — see [LICENSE](LICENSE) for details.
