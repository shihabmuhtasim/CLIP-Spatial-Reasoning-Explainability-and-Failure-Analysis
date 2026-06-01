# 🔍 Spatial Blind Spots: CLIP Failures
### *A Mechanistic Study of How and Why CLIP Fails at Spatial Reasoning*

**Author:** Shihab Muhtasim · IPCVAI, Universidad Autónoma de Madrid
**Hardware:** NVIDIA RTX 3080 Ti (12 GB VRAM) · AMD Ryzen 9 5950X
**Notebook:** `CLIP_EXPLAINABILITY.ipynb`

---

## Overview

Vision-language models like CLIP are now deployed in image retrieval, visual question answering, and surveillance systems. Almost all of their success rests on learning *what* objects are — but very little on learning *where* objects are relative to each other.

*"The cat is **in front of** the dog"* and *"The cat is **behind** the dog"* contain the same words and the same objects. Only their spatial relationship differs. A model with genuine spatial understanding should treat these sentences very differently. This project investigates whether CLIP does — and finds, systematically and mechanistically, that it does not.

Using the **Visual Spatial Reasoning (VSR)** dataset and **CLIP ViT-B/32** as the primary subject, this notebook runs a chain of **7 experiments** designed to answer not just *whether* CLIP fails at spatial reasoning, but *how* and *why* it fails — and whether a differently-trained model (BLIP) shows the same failure modes.

---

## 🎯 Key Results at a Glance

| Finding | Result |
|---------|--------|
| CLIP overall accuracy on spatial tasks | **54.43%** (random = 50%) |
| ROC-AUC | **0.566** (random = 0.50) |
| Accuracy gap: 2D positional shortcut | **+23.15 pp** (63.21% vs 40.06%) — *core finding* |
| Foil test: correct caption preferred | **52.33%** (near-random even in comparative mode) |
| Text cosine similarity, opposite captions | **~0.98** (e.g. "in front of" ↔ "behind") |
| Text encoder failure begins at | **Layer 1** (similarity = 0.9977) |
| Attention on relational space between objects | **0.8%** of total attention |
| BLIP accuracy (vs CLIP 54.43%) | **61.61%** |
| BLIP foil preference (vs CLIP 52.33%) | **65.73%** |
| BLIP heuristic dependency gap (vs CLIP +23.15%) | **−6.33%** ← sign reversal |

---

## Dataset

**Visual Spatial Reasoning (VSR)** — Liu, Emerson & Collier, 2023

- **10,972** image-caption pairs total (train + val + test combined)
- Each example: a COCO image, a spatial caption, and a binary TRUE/FALSE label
- **10 relations selected** across 3 theoretically motivated groups:

| Group | Relations | What makes them hard |
|-------|-----------|---------------------|
| **2D** | left of, right of, above, below | Answerable by pixel position; no depth reasoning required |
| **3D** | in front of, behind, facing, near | Require depth, viewpoint, or occlusion understanding |
| **Containment** | inside, contains | Require fine-grained enclosure detection |

- **Working dataset: 3,386 examples** (after filtering to selected relations)
- Label split: 1,744 TRUE / 1,642 FALSE (~51.5% majority class)

---

## Models and Tools

| Component | Model | Role |
|-----------|-------|------|
| Primary | **CLIP ViT-B/32** (`openai/clip-vit-base-patch32`) | Main subject of analysis |
| Object detection | **OWL-ViT** (`google/owlvit-base-patch32`) | Bounding box extraction for positional analysis |
| Attribution | **Grad-CAM** (pytorch-grad-cam) | Visual explainability heatmaps |
| Comparison | **BLIP-ITM Large** (`Salesforce/blip-itm-large-coco`) | Alternative model with denser cross-modal supervision |

---

## Experiments

| # | Experiment | Core Question | Key Output |
|---|-----------|---------------|------------|
| **1** | Baseline accuracy | Can CLIP classify spatial captions at better than chance? | 54.43% accuracy, ROC-AUC 0.566 |
| **2.1** | Vertical-position heuristic | Does CLIP exploit the "closer = lower in frame" shortcut? | 23.15 pp accuracy gap, χ²=49.9, p≈0 |
| **2.2** | Occlusion analysis | Does bounding-box overlap (depth cue) predict correctness? | IoU distribution comparison, Mann-Whitney test |
| **3** | Foil test | When given both a caption and its spatial opposite, does CLIP prefer the right one? | 52.33% preference, mean gap = 0.0006 |
| **4.1** | Text embedding geometry | Are opposite spatial captions distinguishable as vectors? | Cosine similarity ~0.98 for all pairs |
| **4.2** | Layer-wise text evolution | At which transformer layer does spatial info collapse? | Layer 1 similarity = 0.9977 — vocabulary-level failure |
| **5.1** | Aggregate Grad-CAM maps | Does CLIP's attention systematically differ by relation type? | Average heatmaps + vertical bias per relation |
| **5.2** | Attention entropy | Is attention more focused when CLIP predicts correctly? | p=0.078 — null result; attention does not predict correctness |
| **5.3** | Per-sample Grad-CAM | What do individual attribution maps look like? | 16-image qualitative grid (8 correct, 8 incorrect) |
| **5.4** | Quantified object attention | What fraction of attention falls on subject / object / between / background? | Subject 40.7%, Object 40.4%, **Between 0.8%**, Background 31.2% |
| **6** | Cross-modal foil geometry | Is the image embedding geometrically closer to correct vs foil caption? | sim(correct) 0.2709 vs sim(foil) 0.2707 — equidistant |
| **7** | BLIP comparison | Does denser cross-modal training escape shortcut reliance? | BLIP heuristic gap = **−6.33%** vs CLIP **+23.15%** |

---

## Main Findings

### 1 — CLIP has no meaningful spatial reasoning ability
Overall accuracy of 54.43% is only 4.4 percentage points above chance across 3,386 examples. Correct and incorrect predictions have mean similarity scores differing by just 0.0001. Performance is uniformly near-chance across 2D, 3D, and containment relation groups — the failure is structural, not relation-specific.

### 2 — When CLIP appears to understand depth, it exploits a 2D visual shortcut
The vertical-position heuristic test is the central finding. In natural photography, physically closer objects tend to appear lower in the image frame due to ground-plane perspective. CLIP has learned this correlation and uses it as a proxy for "in front of":

| Condition | CLIP Accuracy |
|-----------|--------------|
| 2D heuristic **matches** caption | **63.21%** [59.48%, 66.79%] |
| 2D heuristic **contradicts** caption | **40.06%** [35.14%, 45.18%] |
| **Gap** | **+23.15 pp** |

Chi-square test: χ²=49.9, p≈0. The association is statistically certain. CLIP is *below chance* when images violate the shortcut.

### 3 — The failure originates in the text encoder at the vocabulary level
Opposite spatial captions produce nearly identical text embedding vectors:

| Relation pair | Mean cosine similarity | N |
|--------------|----------------------|---|
| "in front of" ↔ "behind" | **0.9825** ± 0.0068 | 737 |
| "above" ↔ "below" | **0.9844** ± 0.0062 | 277 |
| "left of" ↔ "right of" | **0.9715** ± 0.0099 | 210 |

Layer-wise analysis shows similarity = **0.9977 at Layer 1** — spatial direction words have no distinct identity even in CLIP's vocabulary embeddings. The transformer layers slightly pull opposite captions apart (to ~0.96 by middle layers) but the final projection reconverges to ~0.98.

### 4 — Visual attention cannot explain spatial predictions
Grad-CAM attribution reveals how CLIP actually allocates visual attention:

| Region | Mean Attention Fraction |
|--------|------------------------|
| Subject object box | 40.7% |
| Reference object box | 40.4% |
| Space **between** the two objects | **0.8%** |
| Background | 31.2% |

CLIP identifies both named objects equally but almost completely ignores the relational space between them. Attention entropy is statistically identical for correct and incorrect predictions (p=0.078) — there is no visual signal that predicts when CLIP succeeds.

### 5 — Model architecture fundamentally changes shortcut reliance
BLIP, with its cross-encoder image-text matching head trained on COCO:

| Metric | CLIP | BLIP |
|--------|------|------|
| Overall accuracy | 54.43% | **61.61%** |
| Foil preference | 52.33% | **65.73%** |
| Mean foil score gap | 0.0006 | **0.0581** (97× larger) |
| Heuristic dependency gap | **+23.15%** | **−6.33%** |

The sign reversal in heuristic dependency is the most theoretically significant result. BLIP performs *better* when the positional shortcut would mislead — confirming genuine spatial discrimination that does not rely on 2D image layout regularities.

---

## Repository Structure

```
├── CLIP_EXPLAINABILITY.ipynb     # Full research notebook (all 7 experiments)
├── README.md
└── outputs/                      # Generated on first run
    ├── exp1_baseline_results.csv
    ├── exp1_accuracy.png
    ├── exp2_heuristic.png
    ├── exp2_depth_analysis.csv
    ├── exp2_occlusion.png
    ├── exp2_occlusion_results.csv
    ├── expA_foil.png
    ├── expA_foil_results.csv
    ├── exp4_embedding_geometry.png
    ├── exp4_embedding_results.csv
    ├── exp4_layerwise_text.png
    ├── exp4_layerwise_text.csv
    ├── exp5_aggregate_attention_full.png
    ├── exp5_spatial_stats_full.csv
    ├── exp5_entropy_full.png
    ├── exp5_entropy_results_full.csv
    ├── exp3_gradcam_grid.png
    ├── exp5_object_attention.png
    ├── exp5_attention_regions.csv
    ├── exp6_crossmodal.png
    ├── exp6_crossmodal.csv
    ├── exp7_blip_comparison.png
    ├── exp7_blip_original_scores.csv
    ├── exp7_blip_foil_scores.csv
    ├── exp7_clip_blip_relation_comparison.csv
    ├── exp7_blip_heuristic_analysis.csv
    ├── synthesis_summary_table.csv
    ├── synthesis_summary_full.png
    ├── bboxes_cache.pkl          # OWL-ViT detection cache
    └── gradcam_cache_224_uint8.pkl  # Grad-CAM cache
```

---

## Setup

### Requirements

```bash
pip install torch torchvision transformers datasets
pip install pytorch-grad-cam
pip install pandas numpy matplotlib seaborn scipy scikit-learn statsmodels tqdm
pip install Pillow requests
```

### Hardware

The notebook was developed and run on an **NVIDIA RTX 3080 Ti (12 GB VRAM)**. Full execution with all 3,386 examples, Grad-CAM on every image, OWL-ViT detection, and BLIP inference takes several hours on first run. All expensive results (Grad-CAM maps, bounding boxes) are cached to disk — subsequent runs use the cache and complete in minutes.

CPU execution is supported but will be significantly slower.

---

## Running the Notebook

```bash
jupyter notebook CLIP_EXPLAINABILITY.ipynb
```

Run all cells in order. On first run, the notebook will:
1. Download CLIP, OWL-ViT, and BLIP weights from HuggingFace (~4 GB total)
2. Download the VSR dataset from HuggingFace
3. Cache ~6,200 COCO images to `~/.cache/vsr_images/`
4. Compute OWL-ViT bounding boxes for all 3,386 images (saved to `outputs/bboxes_cache.pkl`)
5. Compute Grad-CAM maps for all 3,386 images (saved to `outputs/gradcam_cache_224_uint8.pkl`)

On re-runs, all caches are loaded automatically.

---

## Experiment Reproducibility

All results are deterministic given fixed HuggingFace model weights and dataset. No training is performed — the notebook is purely evaluative. The global similarity threshold `SIMILARITY_THRESHOLD = 0.25` is set to match the empirical median of the full similarity distribution and is consistent with the VSR benchmark evaluation protocol.

---

## Relevance to Production Vision Systems

This work has direct implications for deployed computer vision systems — particularly in surveillance, autonomous vehicles, and robotics — where spatial understanding is safety-critical:

- **Shortcut exploitation is invisible from accuracy alone.** A 54% accuracy sounds almost random, but the mechanism (positional shortcut) means failure is *systematically* concentrated on specific image configurations.
- **The text encoder is the bottleneck.** Open-vocabulary detection systems built on CLIP-style encoders inherit the spatial language failure before any image is processed.
- **Model choice matters more than fine-tuning.** BLIP's −6.33% heuristic gap (vs CLIP's +23.15%) is not achievable by prompt engineering or threshold tuning on CLIP — it requires architectural cross-encoder supervision.
- **Explainability reveals the failure mode.** Grad-CAM and embedding geometry analysis pinpoint exactly where and why the failure occurs — enabling targeted fixes rather than opaque accuracy improvements.

---

## References

1. Radford et al. **"Learning Transferable Visual Models From Natural Language Supervision."** ICML 2021.
2. Liu, Emerson & Collier. **"Visual Spatial Reasoning."** TACL 2023.
3. Yuksekgonul et al. **"When and Why Vision-Language Models Behave like Bags-of-Words."** ICLR 2023.
4. Wang et al. **"SpatialCLIP: Learning 3D-aware Image Representations from Spatially Discriminative Language."** CVPR 2025.
5. Dosovitskiy et al. **"An Image is Worth 16×16 Words."** ICLR 2021.
6. Minderer et al. **"Simple Open-Vocabulary Object Detection (OWL-ViT)."** ECCV 2022.
7. Selvaraju et al. **"Grad-CAM: Visual Explanations from Deep Networks."** ICCV 2017.
8. Li et al. **"BLIP: Bootstrapping Language-Image Pre-training."** ICML 2022.
9. Gildenblat et al. **pytorch-grad-cam.** GitHub 2021.

---

## Citation

```bibtex
@misc{muhtasim2026spatialblindspot,
  title   = {Spatial Blind Spots: A Mechanistic Study of How and Why CLIP Fails at Spatial Reasoning},
  author  = {Muhtasim, Shihab},
  year    = {2026},
  note    = {IPCVAI, Universidad Autónoma de Madrid},
  url     = {https://github.com/shihabmuhtasim/CLIP-SPATIAL-EXPLAINABILITY}
}
```
