"""
Angiogenesis Analyzer — Core Analysis Engine
=============================================
Replicates the ImageJ Angiogenesis Analyzer (Gilles Carpentier) in pure Python.

Pipeline
--------
1. Preprocess (grayscale, blur, optional background subtraction)
2. Segment   (adaptive / percentile threshold + morphological cleanup)
3. Skeletonize
4. Remove artifactual small loops
5. Detect nodes (pixels with ≥3 neighbours) → cluster into junctions
6. Classify skeleton pieces (segments, branches, twigs, isolated)
7. Iterative pruning → master tree (master segments, master junctions)
8. Mesh detection (enclosed areas)
9. Quantify 20+ parameters
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from scipy import ndimage
from skimage import morphology, filters, util


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class SkeletonElement:
    """A single piece of the skeleton (segment, branch, isolated element, etc.)."""
    element_type: str          # "segment", "branch", "twig", "isolated", "master_segment"
    pixels: np.ndarray         # Nx2 array of (row, col) coords
    length: float = 0.0        # geodesic length in pixels
    junction_ids: List[int] = field(default_factory=list)


@dataclass
class Junction:
    """A junction = cluster of adjacent node pixels."""
    junction_id: int
    pixels: np.ndarray         # Kx2 (row, col) of node pixels in cluster
    centroid: Tuple[float, float] = (0.0, 0.0)
    is_master: bool = False


@dataclass
class Mesh:
    """An enclosed area in the master tree."""
    contour: np.ndarray
    area: float = 0.0


@dataclass
class AnalysisResult:
    """All quantified parameters + overlay data."""
    # --- parameters ---
    image_name: str = ""
    analyzed_area: float = 0.0

    nb_extremities: int = 0
    nb_nodes: int = 0
    nb_junctions: int = 0
    nb_master_junctions: int = 0
    nb_master_segments: int = 0
    tot_master_segments_length: float = 0.0

    nb_meshes: int = 0
    tot_meshes_area: float = 0.0

    nb_pieces: int = 0
    nb_segments: int = 0
    nb_branches: int = 0
    nb_isolated_segments: int = 0

    tot_length: float = 0.0
    tot_branching_length: float = 0.0
    tot_segments_length: float = 0.0
    tot_branches_length: float = 0.0
    tot_isolated_length: float = 0.0

    branching_interval: float = 0.0
    mesh_index: float = 0.0
    mean_mesh_size: float = 0.0

    # --- overlay layers (boolean masks or coordinate lists) ---
    skeleton_mask: Optional[np.ndarray] = None
    segment_mask: Optional[np.ndarray] = None
    branch_mask: Optional[np.ndarray] = None
    twig_mask: Optional[np.ndarray] = None
    master_segment_mask: Optional[np.ndarray] = None
    node_mask: Optional[np.ndarray] = None
    junction_mask: Optional[np.ndarray] = None
    master_junction_mask: Optional[np.ndarray] = None
    mesh_mask: Optional[np.ndarray] = None
    extremity_mask: Optional[np.ndarray] = None
    isolated_mask: Optional[np.ndarray] = None
    binary_mask: Optional[np.ndarray] = None

    # raw element lists for advanced usage
    junctions: List[Junction] = field(default_factory=list)
    elements: List[SkeletonElement] = field(default_factory=list)
    meshes: List[Mesh] = field(default_factory=list)

    def as_dict(self) -> Dict[str, object]:
        """Return scalar parameters as an ordered dict (for table / export)."""
        return {
            "Image Name": self.image_name,
            "Analysed area": self.analyzed_area,
            "Nb extremities": self.nb_extremities,
            "Nb nodes": self.nb_nodes,
            "Nb Junctions": self.nb_junctions,
            "Nb master junctions": self.nb_master_junctions,
            "Nb master segments": self.nb_master_segments,
            "Tot. master segments length": round(self.tot_master_segments_length, 2),
            "Nb meshes": self.nb_meshes,
            "Tot. meshes area": round(self.tot_meshes_area, 2),
            "Nb pieces": self.nb_pieces,
            "Nb segments": self.nb_segments,
            "Nb branches": self.nb_branches,
            "Nb isol. seg.": self.nb_isolated_segments,
            "Tot. length": round(self.tot_length, 2),
            "Tot. branching length": round(self.tot_branching_length, 2),
            "Tot. segments length": round(self.tot_segments_length, 2),
            "Tot. branches length": round(self.tot_branches_length, 2),
            "Tot. isol. branches length": round(self.tot_isolated_length, 2),
            "Branching interval": round(self.branching_interval, 2),
            "Mesh index": round(self.mesh_index, 2),
            "Mean Mesh Size": round(self.mean_mesh_size, 2),
        }


# ---------------------------------------------------------------------------
# Neighbour helpers
# ---------------------------------------------------------------------------

_NEIGHBOUR_OFFSETS = [(-1, -1), (-1, 0), (-1, 1),
                      (0, -1),           (0, 1),
                      (1, -1),  (1, 0),  (1, 1)]


def _neighbour_count(skel: np.ndarray) -> np.ndarray:
    """Return an image where each skeleton pixel holds its 8-neighbour count."""
    kern = np.ones((3, 3), dtype=np.uint8)
    kern[1, 1] = 0
    counts = cv2.filter2D(skel.astype(np.uint8), -1, kern)
    counts[skel == 0] = 0
    return counts


def _skeleton_length(mask: np.ndarray) -> float:
    """
    Approximate skeleton length.
    Horizontal/vertical neighbour transitions count 1, diagonal count √2.
    """
    if mask.sum() == 0:
        return 0.0
    coords = np.argwhere(mask)
    if len(coords) <= 1:
        return 0.0
    # build adjacency for skeleton pixels count
    total = 0.0
    skel_set = set(map(tuple, coords))
    counted = set()
    for r, c in skel_set:
        for dr, dc in _NEIGHBOUR_OFFSETS:
            nr, nc = r + dr, c + dc
            edge = (min((r, c), (nr, nc)), max((r, c), (nr, nc)))
            if (nr, nc) in skel_set and edge not in counted:
                counted.add(edge)
                total += math.sqrt(dr * dr + dc * dc)
    return total


def _skeleton_length_fast(mask: np.ndarray) -> float:
    """Fast estimation: count pixels + add sqrt(2)-1 for diagonal connections."""
    if mask.sum() == 0:
        return 0.0

    skel = mask.astype(np.uint8)
    # Count pixels as base length
    n_pixels = int(skel.sum())

    # Count diagonal neighbours (approximate — each diagonal adj adds sqrt(2)-1 extra)
    # Simple approach: just return pixel count as length estimate (standard for skeleton)
    return float(n_pixels)


# ---------------------------------------------------------------------------
# Step 1 — Preprocessing
# ---------------------------------------------------------------------------

def preprocess(image: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """Convert to grayscale uint8 and apply Gaussian blur."""
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    if gray.dtype != np.uint8:
        gray = util.img_as_ubyte(gray)
    if sigma > 0:
        gray = cv2.GaussianBlur(gray, (0, 0), sigma)
    return gray


# ---------------------------------------------------------------------------
# Step 2 — Segmentation
# ---------------------------------------------------------------------------

def segment_phase_contrast(gray: np.ndarray, block_size: int = 51,
                           c_val: int = 10, min_size: int = 150) -> np.ndarray:
    """
    Segment a phase-contrast HUVEC image.
    Uses adaptive thresholding → morphological cleanup.
    """
    # For phase-contrast the cells are darker — invert after thresholding
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, block_size, c_val
    )
    # morphological closing to connect nearby pieces
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    # remove small objects
    binary_bool = binary.astype(bool)
    binary_bool = morphology.remove_small_objects(binary_bool, max_size=min_size - 1)
    return binary_bool.astype(np.uint8) * 255


def segment_fluorescence(gray: np.ndarray, percentile: float = 15.0,
                          min_size: int = 150) -> np.ndarray:
    """
    Segment a fluorescence HUVEC image.
    Uses a percentile-based threshold.
    """
    thresh_val = np.percentile(gray, 100 - percentile)
    _, binary = cv2.threshold(gray, int(thresh_val), 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    binary_bool = binary.astype(bool)
    binary_bool = morphology.remove_small_objects(binary_bool, max_size=min_size - 1)
    return binary_bool.astype(np.uint8) * 255


# ---------------------------------------------------------------------------
# Step 3 — Skeletonization
# ---------------------------------------------------------------------------

def skeletonize_binary(binary: np.ndarray) -> np.ndarray:
    """Thin binary mask to 1-pixel-wide skeleton."""
    skel = morphology.skeletonize(binary > 0)
    return skel.astype(np.uint8)  # 0/1


# ---------------------------------------------------------------------------
# Step 4 — Remove artifactual small loops
# ---------------------------------------------------------------------------

def remove_small_loops(skel: np.ndarray, max_loop_area: int = 100) -> np.ndarray:
    """
    Find small enclosed regions in the skeleton and break them by
    removing the shortest edge of each loop.
    """
    filled = ndimage.binary_fill_holes(skel.astype(bool))
    holes = filled & ~skel.astype(bool)
    labelled, n = ndimage.label(holes)
    for i in range(1, n + 1):
        region = labelled == i
        area = region.sum()
        if area <= max_loop_area:
            # Dilate the region slightly to touch the skeleton border
            dilated = cv2.dilate(region.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1)
            border = (dilated > 0) & (skel > 0)
            border_coords = np.argwhere(border)
            if len(border_coords) > 0:
                # Remove one pixel of the border to break the loop
                # Choose the pixel with highest neighbour count (likely safest to remove)
                nc = _neighbour_count(skel)
                best_idx = np.argmax([nc[r, c] for r, c in border_coords])
                r, c = border_coords[best_idx]
                skel[r, c] = 0
    # Re-skeletonize to clean up
    skel = morphology.skeletonize(skel.astype(bool)).astype(np.uint8)
    return skel


# ---------------------------------------------------------------------------
# Step 5 — Node / junction detection
# ---------------------------------------------------------------------------

def detect_nodes_and_junctions(skel: np.ndarray,
                                junction_merge_distance: int = 5
                                ) -> Tuple[np.ndarray, np.ndarray, List[Junction], np.ndarray]:
    """
    Returns
    -------
    node_mask : bool mask of individual node pixels (≥3 neighbours)
    extremity_mask : bool mask of extremity pixels (1 neighbour)
    junctions : list[Junction] (clustered nodes)
    junction_label_map : int image where each pixel = junction_id (0 = none)
    """
    nc = _neighbour_count(skel)
    node_mask = (nc >= 3) & (skel > 0)
    extremity_mask = (nc == 1) & (skel > 0)

    # Cluster adjacent node pixels into junctions using connected-component labelling
    # Dilate slightly to merge nodes within merge_distance
    struct = morphology.disk(max(1, junction_merge_distance // 2))
    dilated_nodes = cv2.dilate(node_mask.astype(np.uint8), struct.astype(np.uint8))
    labelled, n_junctions = ndimage.label(dilated_nodes)

    junctions: List[Junction] = []
    junction_label_map = np.zeros_like(skel, dtype=np.int32)

    for jid in range(1, n_junctions + 1):
        region = labelled == jid
        # Only keep actual node pixels in this cluster
        cluster_mask = region & node_mask
        pix = np.argwhere(cluster_mask)
        if len(pix) == 0:
            # Cluster was only from dilation — take the nearest skeleton pixel
            pix = np.argwhere(region & (skel > 0))
            if len(pix) == 0:
                continue
        centroid = tuple(pix.mean(axis=0))
        j = Junction(junction_id=jid, pixels=pix, centroid=centroid)
        junctions.append(j)
        for r, c in pix:
            junction_label_map[r, c] = jid

    return node_mask, extremity_mask, junctions, junction_label_map


# ---------------------------------------------------------------------------
# Step 6 — Trace & classify skeleton elements
# ---------------------------------------------------------------------------

def _trace_from(skel: np.ndarray, start: Tuple[int, int],
                junction_map: np.ndarray, visited: np.ndarray,
                origin_junction_id: int = 0) -> Tuple[List[Tuple[int, int]], List[int]]:
    """
    Trace a path along the skeleton from *start* until hitting another junction,
    an extremity, or a dead end.  Returns the path pixels and junction IDs
    found at endpoints (excluding the origin junction).

    Parameters
    ----------
    origin_junction_id : int
        The junction that launched this trace — skip it during endpoint detection
        so the tracer doesn't terminate immediately.
    """
    path = [start]
    visited[start[0], start[1]] = 1
    junctions_hit: List[int] = []

    r, c = start
    while True:
        found_next = False

        # Check if any neighbour is a *different* junction pixel (even if visited).
        # This lets us detect the endpoint junction of a segment.
        hit_junction = False
        for dr, dc in _NEIGHBOUR_OFFSETS:
            nr, nc_ = r + dr, c + dc
            if 0 <= nr < skel.shape[0] and 0 <= nc_ < skel.shape[1]:
                jid = junction_map[nr, nc_]
                if jid > 0 and jid != origin_junction_id and visited[nr, nc_] == 1:
                    # We reached an adjacent junction different from where we started
                    junctions_hit.append(int(jid))
                    hit_junction = True
                    break
        if hit_junction:
            break

        # Continue tracing to unvisited skeleton pixels
        for dr, dc in _NEIGHBOUR_OFFSETS:
            nr, nc_ = r + dr, c + dc
            if 0 <= nr < skel.shape[0] and 0 <= nc_ < skel.shape[1]:
                if skel[nr, nc_] > 0 and visited[nr, nc_] == 0:
                    visited[nr, nc_] = 1
                    path.append((nr, nc_))
                    jid = junction_map[nr, nc_]
                    if jid > 0 and jid != origin_junction_id:
                        junctions_hit.append(int(jid))
                        found_next = False
                        break
                    r, c = nr, nc_
                    found_next = True
                    break
        if not found_next:
            break

    return path, junctions_hit


def classify_elements(skel: np.ndarray, junctions: List[Junction],
                       junction_label_map: np.ndarray, extremity_mask: np.ndarray,
                       twig_threshold: int = 25) -> List[SkeletonElement]:
    """
    Walk along every piece of the skeleton and classify it.

    - segment  : between two junctions
    - branch   : between a junction and an extremity
    - twig     : branch shorter than twig_threshold
    - isolated : not connected to any junction
    """
    h, w = skel.shape
    visited = np.zeros_like(skel, dtype=np.uint8)

    # Mark all junction cluster pixels as visited (we start tracing from their neighbours)
    for j in junctions:
        for r, c in j.pixels:
            visited[r, c] = 1

    elements: List[SkeletonElement] = []

    # --- Traces starting from junction-adjacent skeleton pixels ---
    for j in junctions:
        for r, c in j.pixels:
            for dr, dc in _NEIGHBOUR_OFFSETS:
                nr, nc_ = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc_ < w:
                    if skel[nr, nc_] > 0 and visited[nr, nc_] == 0:
                        path, jids = _trace_from(
                            skel, (nr, nc_), junction_label_map, visited,
                            origin_junction_id=j.junction_id
                        )
                        # Prepend the starting junction
                        all_jids = [j.junction_id] + jids
                        pix = np.array(path)
                        length = float(len(pix))
                        # Classify
                        unique_jids = [x for x in all_jids if x > 0]
                        if len(unique_jids) >= 2:
                            etype = "segment"
                        elif len(unique_jids) == 1:
                            etype = "branch" if length >= twig_threshold else "twig"
                        else:
                            etype = "isolated"
                        elem = SkeletonElement(element_type=etype, pixels=pix,
                                               length=length, junction_ids=unique_jids)
                        elements.append(elem)

    # --- Isolated elements (skeleton regions not touching any junction) ---
    remaining = (skel > 0) & (visited == 0)
    labelled, n = ndimage.label(remaining)
    for i in range(1, n + 1):
        pix = np.argwhere(labelled == i)
        length = float(len(pix))
        elem = SkeletonElement(element_type="isolated", pixels=pix, length=length)
        elements.append(elem)
        for r, c in pix:
            visited[r, c] = 1

    return elements


# ---------------------------------------------------------------------------
# Step 7 — Master tree (iterative pruning)
# ---------------------------------------------------------------------------

def _find_master_tree(junctions: List[Junction],
                       elements: List[SkeletonElement]) -> Tuple[List[Junction], List[SkeletonElement]]:
    """
    Iterative pruning to find master segments and master junctions.

    A master junction is a junction that connects to ≥2 segments (not only branches).
    A master segment is a segment between two master junctions.
    """
    # Build adjacency: junction_id → list of elements
    junction_elements: Dict[int, List[SkeletonElement]] = {}
    for elem in elements:
        for jid in elem.junction_ids:
            junction_elements.setdefault(jid, []).append(elem)

    # Identify master junctions: junctions involved with ≥2 segments (or ≥3 elements total)
    master_junction_ids = set()
    for j in junctions:
        elems = junction_elements.get(j.junction_id, [])
        segment_count = sum(1 for e in elems if e.element_type == "segment")
        if segment_count >= 2 or len(elems) >= 3:
            master_junction_ids.add(j.junction_id)
            j.is_master = True

    # If no master junctions found, relax: any junction with ≥2 connections is master
    if not master_junction_ids:
        for j in junctions:
            elems = junction_elements.get(j.junction_id, [])
            if len(elems) >= 2:
                master_junction_ids.add(j.junction_id)
                j.is_master = True

    # Master segments: segments whose both endpoint junctions are master
    master_segments: List[SkeletonElement] = []
    for elem in elements:
        if elem.element_type == "segment":
            jids = [jid for jid in elem.junction_ids if jid in master_junction_ids]
            if len(jids) >= 2:
                ms = SkeletonElement(
                    element_type="master_segment",
                    pixels=elem.pixels.copy(),
                    length=elem.length,
                    junction_ids=elem.junction_ids.copy(),
                )
                master_segments.append(ms)

    master_junctions = [j for j in junctions if j.is_master]
    return master_junctions, master_segments


# ---------------------------------------------------------------------------
# Step 8 — Mesh detection
# ---------------------------------------------------------------------------

def detect_meshes(master_segments: List[SkeletonElement],
                  master_junctions: List[Junction],
                  shape: Tuple[int, int]) -> List[Mesh]:
    """
    Build a mask from master segments + master junction pixels,
    fill holes, and measure enclosed areas = meshes.
    """
    master_mask = np.zeros(shape, dtype=np.uint8)
    for ms in master_segments:
        for r, c in ms.pixels:
            master_mask[r, c] = 1
    for mj in master_junctions:
        for r, c in mj.pixels:
            master_mask[r, c] = 1

    # Dilate slightly so skeleton pixels fully close regions
    master_mask_d = cv2.dilate(master_mask, np.ones((3, 3), np.uint8), iterations=1)
    filled = ndimage.binary_fill_holes(master_mask_d)
    holes = filled & ~master_mask_d.astype(bool)

    labelled, n = ndimage.label(holes)
    meshes: List[Mesh] = []
    for i in range(1, n + 1):
        region = (labelled == i).astype(np.uint8)
        area = float(region.sum())
        if area < 10:  # skip tiny artifacts
            continue
        contours, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour = contours[0] if contours else np.array([])
        meshes.append(Mesh(contour=contour, area=area))
    return meshes


# ---------------------------------------------------------------------------
# Putting it all together
# ---------------------------------------------------------------------------

def analyze(image: np.ndarray,
            image_name: str = "untitled",
            image_type: str = "phase_contrast",
            sigma: float = 1.0,
            block_size: int = 51,
            c_val: int = 10,
            percentile: float = 15.0,
            min_object_size: int = 150,
            max_loop_area: int = 100,
            twig_threshold: int = 25,
            junction_merge_distance: int = 5,
            ) -> AnalysisResult:
    """
    Run the full Angiogenesis Analyzer pipeline on *image*.

    Parameters
    ----------
    image : BGR or grayscale uint8/uint16 image
    image_type : "phase_contrast" or "fluorescence"
    """
    result = AnalysisResult(image_name=image_name)
    h, w = image.shape[:2]
    result.analyzed_area = float(h * w)

    # 1. Preprocess
    gray = preprocess(image, sigma=sigma)

    # 2. Segment
    if image_type == "phase_contrast":
        binary = segment_phase_contrast(gray, block_size=block_size,
                                         c_val=c_val, min_size=min_object_size)
    else:
        binary = segment_fluorescence(gray, percentile=percentile,
                                       min_size=min_object_size)
    result.binary_mask = binary

    # 3. Skeletonize
    skel = skeletonize_binary(binary)
    result.skeleton_mask = skel.copy()

    # 4. Remove small loops
    skel = remove_small_loops(skel, max_loop_area=max_loop_area)

    # 5. Detect nodes & junctions
    node_mask, extremity_mask, junctions, junction_label_map = \
        detect_nodes_and_junctions(skel, junction_merge_distance=junction_merge_distance)

    result.node_mask = node_mask.astype(np.uint8)
    result.extremity_mask = extremity_mask.astype(np.uint8)

    # 6. Classify elements
    elements = classify_elements(skel, junctions, junction_label_map,
                                  extremity_mask, twig_threshold=twig_threshold)

    # 7. Master tree
    master_junctions, master_segments = _find_master_tree(junctions, elements)

    # 8. Meshes
    meshes = detect_meshes(master_segments, master_junctions, skel.shape)

    # ---- Build overlay masks ----
    segment_mask = np.zeros_like(skel, dtype=np.uint8)
    branch_mask = np.zeros_like(skel, dtype=np.uint8)
    twig_mask = np.zeros_like(skel, dtype=np.uint8)
    isolated_mask = np.zeros_like(skel, dtype=np.uint8)
    master_seg_mask = np.zeros_like(skel, dtype=np.uint8)

    for elem in elements:
        for r, c in elem.pixels:
            if elem.element_type == "segment":
                segment_mask[r, c] = 1
            elif elem.element_type == "branch":
                branch_mask[r, c] = 1
            elif elem.element_type == "twig":
                twig_mask[r, c] = 1
            elif elem.element_type == "isolated":
                isolated_mask[r, c] = 1

    for ms in master_segments:
        for r, c in ms.pixels:
            master_seg_mask[r, c] = 1

    junction_mask = np.zeros_like(skel, dtype=np.uint8)
    for j in junctions:
        for r, c in j.pixels:
            junction_mask[r, c] = 1

    master_junction_mask = np.zeros_like(skel, dtype=np.uint8)
    for mj in master_junctions:
        for r, c in mj.pixels:
            master_junction_mask[r, c] = 1

    mesh_mask = np.zeros_like(skel, dtype=np.uint8)
    for m in meshes:
        if len(m.contour) > 0:
            cv2.drawContours(mesh_mask, [m.contour], -1, 1, cv2.FILLED)

    result.segment_mask = segment_mask
    result.branch_mask = branch_mask
    result.twig_mask = twig_mask
    result.isolated_mask = isolated_mask
    result.master_segment_mask = master_seg_mask
    result.junction_mask = junction_mask
    result.master_junction_mask = master_junction_mask
    result.mesh_mask = mesh_mask

    result.junctions = junctions
    result.elements = elements + master_segments
    result.meshes = meshes

    # ---- Quantify ----
    result.nb_extremities = int(extremity_mask.sum())
    result.nb_nodes = int(node_mask.sum())
    result.nb_junctions = len(junctions)
    result.nb_master_junctions = len(master_junctions)
    result.nb_master_segments = len(master_segments)
    result.tot_master_segments_length = sum(ms.length for ms in master_segments)

    result.nb_meshes = len(meshes)
    result.tot_meshes_area = sum(m.area for m in meshes)

    segments = [e for e in elements if e.element_type == "segment"]
    branches = [e for e in elements if e.element_type == "branch"]
    twigs = [e for e in elements if e.element_type == "twig"]
    isolated = [e for e in elements if e.element_type == "isolated"]

    result.nb_segments = len(segments)
    result.nb_branches = len(branches) + len(twigs)
    result.nb_isolated_segments = len(isolated)
    result.nb_pieces = result.nb_segments + result.nb_branches + result.nb_isolated_segments

    result.tot_segments_length = sum(e.length for e in segments)
    result.tot_branches_length = sum(e.length for e in branches) + sum(e.length for e in twigs)
    result.tot_isolated_length = sum(e.length for e in isolated)
    result.tot_length = result.tot_segments_length + result.tot_branches_length + result.tot_isolated_length
    result.tot_branching_length = result.tot_segments_length + result.tot_branches_length

    if result.nb_branches > 0:
        result.branching_interval = result.tot_segments_length / result.nb_branches
    if result.nb_master_segments > 0:
        result.mesh_index = result.tot_master_segments_length / result.nb_master_segments
    if result.nb_meshes > 0:
        result.mean_mesh_size = result.tot_meshes_area / result.nb_meshes

    return result


# ---------------------------------------------------------------------------
# Overlay rendering
# ---------------------------------------------------------------------------

def render_overlay(image: np.ndarray, result: AnalysisResult,
                   show_segments: bool = True,
                   show_branches: bool = True,
                   show_twigs: bool = True,
                   show_master_segments: bool = True,
                   show_junctions: bool = True,
                   show_master_junctions: bool = True,
                   show_meshes: bool = True,
                   show_extremities: bool = True,
                   show_isolated: bool = True,
                   alpha: float = 0.7) -> np.ndarray:
    """
    Render a color-coded overlay onto the original image.

    Colour scheme (matches original ImageJ plugin):
      segments          → magenta   (255, 0, 255)
      branches          → green     (0, 255, 0)
      twigs             → cyan      (0, 255, 255)
      master segments   → orange    (255, 165, 0)
      nodes/junctions   → red+yellow
      master junctions  → red+blue
      meshes            → sky blue  (135, 206, 235) fill
      extremities       → blue      (0, 100, 255)
      isolated          → gray      (180, 180, 180)
    """
    if image.ndim == 2:
        canvas = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        canvas = image.copy()

    overlay = canvas.copy()

    def _draw(mask, color, thickness=2):
        if mask is None:
            return
        dilated = cv2.dilate(mask, np.ones((thickness, thickness), np.uint8))
        overlay[dilated > 0] = color

    if show_meshes and result.mesh_mask is not None:
        overlay[result.mesh_mask > 0] = (235, 206, 135)  # sky blue (BGR)

    if show_isolated:
        _draw(result.isolated_mask, (180, 180, 180))
    if show_segments:
        _draw(result.segment_mask, (255, 0, 255))       # magenta (BGR)
    if show_branches:
        _draw(result.branch_mask, (0, 255, 0))           # green
    if show_twigs:
        _draw(result.twig_mask, (255, 255, 0))            # cyan (BGR)
    if show_master_segments:
        _draw(result.master_segment_mask, (0, 165, 255))  # orange (BGR)

    if show_junctions and result.junction_mask is not None:
        dilated = cv2.dilate(result.junction_mask, morphology.disk(3).astype(np.uint8))
        overlay[dilated > 0] = (0, 255, 255)  # yellow ring (BGR)
        dilated2 = cv2.dilate(result.junction_mask, morphology.disk(2).astype(np.uint8))
        overlay[dilated2 > 0] = (0, 0, 255)   # red center

    if show_master_junctions and result.master_junction_mask is not None:
        dilated = cv2.dilate(result.master_junction_mask, morphology.disk(4).astype(np.uint8))
        overlay[dilated > 0] = (255, 0, 0)    # blue ring (BGR)
        dilated2 = cv2.dilate(result.master_junction_mask, morphology.disk(2).astype(np.uint8))
        overlay[dilated2 > 0] = (0, 0, 255)   # red center

    if show_extremities and result.extremity_mask is not None:
        dilated = cv2.dilate(result.extremity_mask, morphology.disk(3).astype(np.uint8))
        overlay[dilated > 0] = (255, 100, 0)  # blue (BGR)

    canvas = cv2.addWeighted(overlay, alpha, canvas, 1 - alpha, 0)
    return canvas


# ---------------------------------------------------------------------------
# Export helpers
# ---------------------------------------------------------------------------

def export_csv(results: List[AnalysisResult], path: str) -> None:
    """Export a list of results to CSV."""
    import csv
    if not results:
        return
    keys = list(results[0].as_dict().keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in results:
            writer.writerow(r.as_dict())


def export_excel(results: List[AnalysisResult], path: str) -> None:
    """Export a list of results to Excel (.xlsx)."""
    from openpyxl import Workbook
    wb = Workbook()
    ws = wb.active
    if not results:
        wb.save(path)
        return
    keys = list(results[0].as_dict().keys())
    ws.append(keys)
    for r in results:
        ws.append(list(r.as_dict().values()))
    wb.save(path)
