# CO₂-NN-Sleipner: Deep Learning for Carbon Storage Monitoring

**Darshita Patel | Purdue University, MPS in Geodata Science for Professionals (EAPS), May 2025 | Now: MSc AI & Data Science, University of East London**

A portfolio project using 3D U-Net deep learning to automatically detect CO₂ plumes in seismic data from the world's first commercial carbon storage site.

**Quick Links:** [Paper (Li & Li 2021)](https://doi.org/10.1029/2021JB022524) | [Sleipner Data](https://co2datashare.org/dataset/sleipner-4d-seismic-dataset)

**Context:** Carbon capture and storage (CCS) could prevent 14% of global CO₂ emissions by 2050 ([IPCC](https://www.ipcc.ch/)). But it only works if we can verify the CO₂ stays underground. This project automates the monitoring that makes CCS trustworthy and scalable.

---

## Quick Note About This Project

This project reproduces and extends research by Li & Li (2021) on neural network-based CO₂ interpretation. I implemented their 3D U-Net architecture from scratch, trained it on real seismic data from Norway's Sleipner CO₂ storage site, and explored how well deep learning can replace manual interpretation of where injected CO₂ goes underground.

**Why this matters:** Carbon capture and storage (CCS) is crucial for climate goals, but monitoring where CO₂ actually goes after injection requires analyzing massive 3D seismic volumes. Currently this is done manually - slow, inconsistent, and doesn't scale. This project explores whether we can automate it with deep learning.

---

## What This Project Is About

**The Problem:** 

Since 1996, Norway has been injecting ~1 million tons of CO₂ per year into underground rock formations beneath the North Sea at the Sleipner field (over 23 million tonnes stored by 2025). This is the **world's first commercial-scale offshore CO₂ storage project**. 

To make sure it stays there and doesn't leak, they run seismic surveys - basically taking 3D "x-rays" of the subsurface - every few years:
- **Baseline:** 1994 (before injection started)
- **Time-lapse vintages:** 1999, 2001, 2002, 2004, 2006, 2008, 2010
- Each survey gets processed multiple times as technology improves (e.g., 1994 data was reprocessed in 2001, 2007, 2010)

But here's the challenge: someone has to manually trace where the CO₂ plume is in each 3D volume across 9 internal sandstone layers (L1-L9 in the Utsira Formation). This takes weeks, is subjective (different interpreters draw different boundaries), and becomes inconsistent over time as processing techniques change.

**The Question:**

Can a neural network learn to automatically identify CO₂ plumes in seismic data? And will it work consistently across different years even when the data quality and processing varies?

---

## What I Did

### 1. Data Understanding & Preparation

**Dataset:** 4D seismic data from Sleipner CO₂ storage site (via [CO₂ DataShare](https://co2datashare.org))
- **Project:** World's first commercial offshore CCS (started 1996)
- **Location:** Norwegian North Sea, ~250km offshore
- **Storage formation:** Utsira Formation (9 internal sand layers: L1-L9)
- **Injection rate:** ~1 million tonnes CO₂/year (over 23 million tonnes stored by 2025)

**Seismic surveys:**
- **Baseline:** 1994 (pre-injection)
- **Time-lapse vintages:** 1999, 2001, 2002, 2004, 2006, 2008, 2010
- **Processing vintages:** Surveys reprocessed in 2001, 2007, 2008, 2010, 2011
- **Data format:** Near-offset stacked seismic (PSTM)
- **Volume dimensions:** 3.35km × 7.0km × 2,000ms (inline × crossline × time)

**Training labels:**
- **Source:** Sleipner 2019 Benchmark Model (Equinor)
- **Coverage:** Hand-drawn CO₂ plume boundaries in 9 sand layers for 2010
- **Assumption:** Full saturation within plume boundaries (due to lack of saturation data)
- **Resolution:** 100m lateral, ~30m vertical (~28ms two-way time)

**Key Insight from EDA:**
When you subtract the baseline (1994) from a monitoring survey (say, 2010), bright spots appear where CO₂ has accumulated. But the signal is noisy, and the "bright spot = CO₂" rule isn't always reliable due to processing artifacts.

This is why you need machine learning - the patterns are subtle and multivariate.

### 2. Model Architecture: 3D U-Net

I implemented a 3D U-Net - basically the medical imaging standard for volumetric segmentation, adapted for geophysics.

**Why U-Net?**
- **Encoder:** Learns features at multiple scales (large plume structure → fine edges)
- **Decoder:** Upsamples back to original resolution  
- **Skip connections:** Preserves spatial details lost during downsampling
- **3D convolutions:** Captures vertical and lateral continuity in the plume

**Input:** Two 3D volumes stacked (baseline + time-lapse seismic)  
**Output:** Probability map of CO₂ presence (0 = no CO₂, 1 = CO₂ present)

### 3. Training Strategy

This was tricky because we only have labels for **2010** (one year out of seven vintages).

**What I did:**

**Data augmentation through processing diversity:**
- Used 4 different combinations of baseline/monitor pairs:
  - 94p01 vs. 10p10 (different processing years)
  - 94p01 vs. 10p11 (baseline processed 2001, monitor processed 2011)
  - 94p10 vs. 10p10 (matched processing)
  - 94p10 vs. 10p11 (baseline processed 2010, monitor processed 2011)
- This forces the network to learn CO₂ signatures that are **robust to processing differences**

**Label generation:**
- Started with 720 randomly sampled 64×64×64 cubes from the 2010 volume
- Each cube has 4 processing combinations → 2,880 total samples
- Problem: ~60% were all-zero (no CO₂) → slows convergence
- Solution: Reduced no-CO₂ samples to 33% → **1,576 final samples**
- Split: 1,500 training / 76 validation

**Training details:**
- Loss function: Binary cross-entropy (pixel-wise CO₂ vs. no-CO₂)
- Optimizer: Adam (lr=0.0002)
- Batch size: 30
- Epochs: 200 (3.08 hours on one NVIDIA A30 24 GB GPU on Purdue's Gilbreth cluster, using ~23.5 GB of GPU memory)
- Normalization: Each input cube normalized separately (zero mean, unit std)

<p align="center"><img src="NNpred3D/figures/fig5_lnloss.png" width="520" alt="Training and validation loss over 200 epochs"></p>

*Training (blue) and validation (red) loss, plotted as ln(loss), over 200 epochs. The two curves stay close together, so the model is not overfitting; the brief spike around epoch 70 recovers within a few epochs.*

**The gamble:** Would a model trained only on 2010 generalize to 1999, 2001, 2004, etc.? The seismic processing was different each year, and the plume looked different (it was growing over time).

### 4. Testing Across Time

Applied the trained model to **all vintages** (1999-2010) to see if it could:
- Correctly identify CO₂ in years it never saw during training
- Handle processing-induced differences between surveys
- Show the plume growing over time (which it should, physically)

---

## Key Findings

### TL;DR: It works surprisingly well - even on years it never trained on.

The big achievement: **consistency across time** with **robustness to processing changes** - exactly what you need for long-term CCS monitoring.

### Why This Matters for CCS

Traditional 4D seismic interpretation has three major problems:

1. **Labor-intensive:** Weeks of manual interpretation per survey
2. **Inconsistent:** Different interpreters, different times → different results
3. **Processing-dependent:** Needs perfectly matched baseline/monitor pairs (expensive reprocessing)

This neural network approach solves all three:

1. **Automated:** Seconds instead of weeks
2. **Consistent:** Same algorithm, same interpretation standards across all vintages
3. **Robust:** Works even with processing mismatch (no reprocessing needed)

**Real-world impact:** There are 60+ CCS projects worldwide, each requiring decades of monitoring. Manual interpretation doesn't scale. This does.

### Quantitative Results (2010 Test Set)

**Model Performance on 2010:**
- **BCE Loss:** 4.85e-3 to 4.93e-3 across different baseline/monitor combinations
- **Training time:** 3.08 hours (200 epochs on an NVIDIA A30 24 GB GPU)
- **Inference time:** Few seconds for entire 3D volume (90 overlapping patches)
- **Robustness:** Model handles processing mismatch between vintages - predictions remain consistent even when using mismatched baseline (94p01) vs. time-lapse (10p11) pairs

**What this means:** The model achieves near-identical predictions across four different input combinations (different processing vintages), demonstrating it learned physical CO₂ signatures rather than memorizing processing artifacts.

### Qualitative Results: Predictions Match Physics

When I applied the model to earlier years (1999-2008), the predictions:

**Show plume growth over time** (small in 1999 → large by 2010)  
**Match known geology** (CO₂ accumulates in high-permeability sand layers L1-L9)  
**Are spatially continuous** (no random scattered pixels)  
**Handle noise well** (doesn't mistake processing artifacts for CO₂)  
**Capture physical behavior** (slower expansion in bottom layer L1 due to buoyancy)

**Temporal generalization example:**
- **Layer L9 (top):** 1999 shows a single bright spot (CO₂ just reached the top), expands steadily through 2010
- **Layer L5 (middle):** Gradual filling and lateral migration visible
- **Layer L1 (bottom/injection layer):** Minimal expansion over time (CO₂ migrates upward due to buoyancy)

<p align="center"><img src="NNpred3D/figures/fig9_mask_layer_year.png" width="700" alt="NN-interpreted CO2 plume in layers L9, L5 and L1 from 1999 to 2010"></p>

*Consistency test: the NN-interpreted CO₂ plume in layers L9 (top row), L5 (middle row) and L1 (base row) of the Utsira Formation, for 1999, 2001, 2004, 2006, 2008 and 2010 (left to right). Red is high CO₂ probability and blue is low. Only 2010 was used for training.*

**The cool part:** The model was trained only on 2010 but correctly predicts plume evolution in both **spatial** (different layers show different patterns) and **temporal** (earlier years show smaller plumes) dimensions.

**Detection of geological features:**
- **Feeder chimneys** (vertical migration pathways between layers)
- **Singularities** where CO₂ first enters a layer (sharp, localized bright spots)
- **Migration directions** (plumes don't just grow uniformly - they follow permeability patterns)

### Robustness to Processing Mismatch

Different seismic surveys use different processing workflows. This creates **baseline mismatch** - brightness, frequency content, and phase can all vary.

**The test:** Applied the model to 4 different baseline/monitor combinations:
- 94p01 vs. 10p10 (BCE loss: 4.86e-3)
- 94p01 vs. 10p11 (BCE loss: 4.91e-3) ← note: 10p11 has different bandwidth!
- 94p10 vs. 10p10 (BCE loss: 4.85e-3) ← best match
- 94p10 vs. 10p11 (BCE loss: 4.93e-3)

**Result:** Visually indistinguishable predictions despite processing differences. The model learned **physical patterns** (how CO₂ looks geologically) not **processing quirks** (how this specific survey was processed).

**Why this matters:** In practice, you don't want to reprocess your 1994 baseline every time you get a new survey in 2025. This model doesn't require it.

**Label vs. predictions from all four input combinations (2010, 3D view):**

| Human label | 94p01 vs 10p10 | 94p01 vs 10p11 | 94p10 vs 10p10 | 94p10 vs 10p11 |
|:---:|:---:|:---:|:---:|:---:|
| <img src="NNpred3D/figures/fig7_mask_ref.png" width="160" alt="3D CO2 label"> | <img src="NNpred3D/figures/fig7_mask_pred_2010_b01_t10.png" width="160" alt="Prediction 94p01 vs 10p10"> | <img src="NNpred3D/figures/fig7_mask_pred_2010_b01_t11.png" width="160" alt="Prediction 94p01 vs 10p11"> | <img src="NNpred3D/figures/fig7_mask_pred_2010_b10_t10.png" width="160" alt="Prediction 94p10 vs 10p10"> | <img src="NNpred3D/figures/fig7_mask_pred_2010_b10_t11.png" width="160" alt="Prediction 94p10 vs 10p11"> |

*Robustness test: the label (left) and the reconstructed CO₂ distributions from NN predictions using four different combinations of baseline and time-lapse images. Each reconstruction is a weighted sum of the overlapping patch predictions of CO₂ probability.*

**Example: 2010 Prediction vs. Human Label (top sand layer, map view)**

| Input: amplitude difference (94p10 vs 10p10) | Human-labeled mask | NN prediction |
|:---:|:---:|:---:|
| <img src="NNpred3D/figures/fig8_data_ampdif_0.png" width="230" alt="Absolute amplitude difference between 94p10 and 10p10"> | <img src="NNpred3D/figures/fig8_mask_ref_0.png" width="230" alt="Human-labeled CO2 plume from the Sleipner 2019 benchmark model"> | <img src="NNpred3D/figures/fig8_mask_pred_2010_b10_t10_0.png" width="230" alt="NN-predicted CO2 probability"> |

*Top sand layer of the Utsira Formation. Left: absolute amplitude difference between the 1994 baseline (94p10) and the 2010 monitor (10p10). Middle: the human-interpreted plume from the Sleipner 2019 benchmark model. Right: the NN's CO₂ probability, where red is high and blue is low.*

The raw amplitude difference is noisy and patchy, but the prediction is a clean, continuous plume whose outline closely follows the human label, with soft edges where the CO₂ probability tapers off. The network gets there from the seismic images alone, without being given any amplitude threshold.

### The Surprising Part: The Network Learned Like a Human Interpreter

This is the coolest finding from the paper: **the neural network doesn't just look at brightness differences** (which would be the naive approach).

Through controlled experiments, Li & Li (2021) discovered the network considers **two things simultaneously:**

1. **Amplitude difference** between baseline and monitor (bright spots = CO₂)
2. **Structural similarity** between the images (is this a real geological feature or just noise?)

**How they proved this:**

They created a test with seismic multiples (imaging artifacts that look like bright spots but aren't real). A human interpreter easily distinguishes these from real CO₂ reflections. 

When they fed the network:
- **Real CO₂ reflections** → Predicted correctly ✓
- **Multiples (artifacts)** → Correctly ignored them ✓
- **Scaled-up multiples (4x brighter)** → Still mostly ignored them ✓
- **Scaled-up baseline** (same structure, different amplitude) → High CO₂ probability ✓

**I reproduced this controlled test** on one of the training samples:

| (a) Original baseline | (c) Time-lapse, locally amplified | (d) NN prediction for (c) vs (a) |
|:---:|:---:|:---:|
| <img src="NNpred3D/figures/figD1-3_R0.png" width="230" alt="Original baseline"> | <img src="NNpred3D/figures/figD1-3_Rt_0.png" width="230" alt="Locally amplified time-lapse image"> | <img src="NNpred3D/figures/figD1-3_pm_0.png" width="230" alt="NN prediction for amplified time-lapse"> |
| **(b) Local scaling multiplier** | **(e) Baseline, locally amplified** | **(f) NN prediction for (e) vs (a)** |
| <img src="NNpred3D/figures/figD1-3_s.png" width="230" alt="Local scaling multiplier"> | <img src="NNpred3D/figures/figD1-3_Rt_1.png" width="230" alt="Locally amplified baseline image"> | <img src="NNpred3D/figures/figD1-3_pm_1.png" width="230" alt="NN prediction for amplified baseline"> |

*(a) is the original baseline and (b) the local scaling multiplier. (c) and (e) are new time-lapse images made by locally amplifying the original time-lapse and baseline images by (b). (d) and (f) are the NN predictions for (c) vs (a) and (e) vs (a).*

This means the network learned to think like a geophysicist: "Does this bright spot have the right structure to be CO₂?" Not just "Is it bright?"

This is why it generalizes across different years - it's not overfitting to processing quirks, it's recognizing physically meaningful patterns.

### 2D vs 3D U-Net Comparison

I also tested a 2D version (slice-by-slice interpretation) using sparse 2D labels (10 inline + 13 crossline slices):

| Metric | 2D U-Net | 3D U-Net (sparse labels) | 3D U-Net (full labels) |
|--------|----------|----------|----------|
| Training time | ~5 min | ~3 hours | ~3 hours |
| Inference speed | Fast | Slower | Slower |
| Spatial continuity | **Choppy** - visible striping artifacts | **Smooth** 3D volumes | **Smooth** 3D volumes |
| Label requirement | 2D slices only | 2D slices with 3D weighting | Full 3D labels |
| **Recommendation** | Quick prototyping | Production with limited labels | Production (ideal case) |

**Key finding:** Even when trained with the **same sparse 2D labels**, the 3D U-Net outperforms the 2D version:
- **No striping artifacts** (2D predictions don't connect properly between slices)
- **Better boundary continuity** (preserves plume structure in all directions)
- **Higher resolution** interpretation

**Takeaway:** If you have GPU memory (24GB+), always use 3D U-Net - even for sparse labels. The computational cost is worth it for geophysically meaningful outputs.

---

## My Implementation Workflow

Here's the actual order I worked through this project (helpful if you're reproducing the work):

### Phase 1: Understanding the Data
1. **data_set_review.ipynb** - Explored Sleipner seismic volumes, understood vintages/processing
2. **label_generation.ipynb** - Converted benchmark model labels from depth → time domain

### Phase 2: 3D U-Net Development (Main Approach)
3. **input_output_preparation_fulllabel.ipynb** - Created training patches with full 3D labels
4. **Test_training_fulllabel.ipynb** - Trained the 3D U-Net (200 epochs, ~3 hours)
5. **Test_network.ipynb** - Applied trained model to all vintages (1999-2010)
6. **Figure_plots_DKP.ipynb** - Generated publication-quality figures

### Phase 3: Sparse Label Experiment
7. **input_output_preparation_sparselabel.ipynb** - Created sparse 2D slice labels (10 inline + 13 crossline)
8. **Test_training_sparselabel.ipynb** - Trained 3D U-Net with weighted sparse labels
9. Compared results → 3D still beats 2D even with same sparse labels

### Phase 4: 2D U-Net Comparison
10. **2D/Test_training.ipynb** - Trained 2D U-Net for comparison (~5 min)
11. **2D/Figure_plot_2D.ipynb** - Documented the striping artifacts issue

The notebooks show the iterative research process - not everything worked first try!

---

## What I'd Actually Recommend

If I were deploying this at a CCS monitoring company:

### **Deploy the 3D U-Net for routine monitoring**
- **Use case:** Automatically process new seismic surveys as they come in
- **Benefit:** Reduce interpretation time from weeks to hours
- **Guardrail:** Always have a geophysicist QC the output (model is a tool, not a replacement)

### **But don't trust it blindly - add uncertainty quantification**

The current model outputs a CO₂ probability for every voxel, but no measure of how much to trust that probability. In production, I'd add:
- **Uncertainty maps** (Monte Carlo dropout, deep ensembles or Bayesian networks)
- **Calibration checks**, so a predicted 0.8 really means CO₂ about 80% of the time

Why? Because in geophysics, being wrong matters. If you miss a leak, CO₂ could escape. If you over-interpret, you waste resources drilling unnecessary monitoring wells.

### **Next steps for research**

**Short-term improvements:**
1. **Active learning:** When the model is uncertain, flag those regions for human labeling → retrain
2. **Multi-site training:** Combine Sleipner data with other CCS sites (Ketzin, In Salah) for generalization
3. **Physics-informed loss:** Add a constraint that CO₂ should flow upward (buoyancy) and spread in permeable layers

**Long-term vision:**
4. **Joint inversion:** Instead of just detecting CO₂, estimate saturation levels (how much CO₂ is there, not just where)
5. **Predictive modeling:** Given injection rate and geology, predict where CO₂ will be in 5 years

---

## Technical Details

### Architecture Specs

```
Simplified 3D U-Net (modified from Wu et al. 2019 FaultSeg3D):

Input specification:
- Channels: 2 (baseline + time-lapse seismic, stacked)
- Patch size: 64 × 64 × 64 voxels
- Spatial resolution: 25m × 25m × 8ms (inline × crossline × time)
- Bottleneck resolution: 200m × 200m × 64ms per voxel (1/8 of the input)
- Volume coverage: 1.6km × 1.6km × 512ms per patch
- Full data volume: 3.35km × 7.0km × 2,000ms

Encoder path:
- Input block: [Conv3D (3×3×3) → BatchNorm → ReLU] × 2, from 2 to 16 channels
- 3 downsampling blocks: MaxPool (2×2×2) → [Conv3D (3×3×3) → BatchNorm → ReLU] × 2
- Feature maps: 16 → 32 → 64 → 256

Bottleneck:
- 256 feature maps at 1/8 resolution (8 × 8 × 8 voxels per patch)

Decoder path:
- 3 upsampling blocks: Trilinear upsampling (×2) → concatenate skip connection → [Conv3D → BatchNorm → ReLU] × 2
- Feature maps: 256 → 64 → 32 → 16

Output:
- Conv3D (1×1×1) → Sigmoid
- Shape: [1, 64, 64, 64] (probability map, 0=no CO₂, 1=CO₂)
- Total parameters: ~3.2M (counted from mynetwork.py with the default c1 = 16)

Key modifications from standard U-Net:
1. Batch normalization before each ReLU (faster convergence)
2. A shallow 3-level design with a wide 256-channel bottleneck, keeping the model small (~3.2M parameters)
3. Two-channel input (baseline + monitor)
```

### Training Details

- **Optimizer:** Adam (lr=0.0002, weight decay not specified)
- **Batch size:** 30 samples per batch
- **Epochs:** 200 (with validation monitoring)
- **Loss:** Binary cross-entropy (pixel-wise)
- **Data normalization:** Each cube normalized separately (μ=0, σ=1)
- **Training samples:** 1,500 (33% no-CO₂, 67% with CO₂)
- **Validation samples:** 76
- **Augmentation:** Processing diversity (4 baseline/monitor combinations)
- **Hardware:** NVIDIA A30 (24 GB) on Purdue's Gilbreth HPC cluster, run as SLURM batch jobs
- **Training time:** 3.08 hours for 200 epochs
- **Framework:** PyTorch 2.5.1 (CUDA 12.1 build)

**Inference strategy:**
- Patch sampling: 3×5×6 = 90 overlapping patches per volume
- Overlap: ~40% in 3D
- Reconstruction: Weighted summation with 3D Hanning window
- Weighting: Full confidence in central 60%, tapers to zero at edges
- Runtime: Few seconds per volume

### Tools Used

- **PyTorch** (with CUDA) and **torchvision** for the model and GPU training
- **NumPy, SciPy and pandas** for data processing
- **segyio, ObsPy and xtgeo** for reading SEG-Y seismic surveys and surfaces
- **Matplotlib and Plotly** for 2D and 3D visualization
- **Jupyter** for exploratory analysis
- **SLURM** for batch GPU jobs on Purdue's Gilbreth cluster

---

## Repository Structure

```
CO2-NN-Sleipner/
│
├── 2D/                                      # 2D U-Net implementation
│   ├── CO2_identify.py                      # 2D inference script
│   ├── Figure_plot_2D.ipynb                 # 2D visualization notebook
│   ├── Test_training.ipynb                  # 2D model evaluation
│   ├── Unet_parts.py                        # 2D U-Net building blocks
│   ├── input_output_preparation.ipynb       # 2D data prep
│   ├── labeling.py                          # Label generation utilities
│   ├── mynetwork.py                         # 2D U-Net architecture
│   └── polygon_inclusion.py                 # Polygon-to-mask conversion
│
├── 3DD/                                     # 3D U-Net implementation (main model)
│   ├── CO2_identify.py                      # 3D inference script
│   ├── Figure_plots_DKP.ipynb               # 3D visualizations
│   ├── Test_network.ipynb                   # 3D network testing
│   ├── Test_training_fulllabel.ipynb        # Training with full 3D labels
│   ├── Test_training_sparselabel.ipynb      # Training with sparse 2D labels
│   ├── Unet_parts.py                        # 3D U-Net building blocks
│   ├── data_set_review-Copy1.ipynb          # Data exploration
│   ├── data_set_review.ipynb                # Dataset analysis
│   ├── input_output_preparation_fulllabel.ipynb   # Full label data prep
│   ├── input_output_preparation_sparselabel.ipynb # Sparse label data prep
│   ├── label_generation.ipynb               # Convert benchmark labels to training format
│   ├── labeling.py                          # Labeling utilities
│   ├── mynetwork.py                         # 3D U-Net architecture definition
│   └── polygon_inclusion.py                 # Geometric utilities
│
├── EXTRA_FILES/                             # Data exploration notebooks, SLURM job scripts, GPU check
│
├── NNpred3D/figures/                        # Output visualizations and results
│   ├── fig2_Mr_0.png                        # Layer predictions over time
│   ├── fig2_R0_0.png
│   ├── fig2_Rt_0.png
│   ├── fig5_lnloss.png                      # Training metrics
│   ├── fig6_R0_0.png                        # Comparison figures
│   └── fig6_R0_1.png
│
├── figures/                                 # Additional result figures
├── myenv_details.yml                        # Conda environment (PyTorch 2.5.1, CUDA 12.1)
├── Darshita_Patel_Final_Presentation.pptx   # Final project presentation
├── .gitignore
└── README.md
```

**Key directories:**
- **2D/** - 2D U-Net for slice-by-slice interpretation (faster training, lower quality)
- **3DD/** - 3D U-Net for volumetric interpretation (main approach, production-ready)
- **NNpred3D/figures/** - Paper-ready visualizations of results

**Note:** Seismic data files not included in repo (download from [CO₂ DataShare](https://co2datashare.org/dataset/sleipner-4d-seismic-dataset)). Large .dat files are not included (they are excluded in .gitignore).

---

## What I Learned

### Technical Skills

- **3D deep learning:** Implementing volumetric CNNs from scratch (harder than 2D!)
- **Domain adaptation:** Making a model trained on one year work on other years
- **Geophysics fundamentals:** How seismic works, why CO₂ shows up as bright spots
- **Handling sparse labels:** Only had ground truth for 1/7 vintages
- **PyTorch best practices:** Efficient data loading for large volumes, GPU memory management

### Research Thinking

- **Validation beyond accuracy:** In scientific ML, you need to check if predictions are **physically plausible**, not just numerically accurate
- **Generalization is hard:** Training on 2010 and testing on 1999-2008 is way harder than random train/test split
- **Interpretability matters:** In geophysics, you can't just say "the model says so" - you need to explain *why* it flagged a region as CO₂

### Domain Knowledge

- **4D seismic:** How time-lapse surveys work and why they're noisy
- **CCS monitoring challenges:** Leak detection, plume migration, regulatory requirements
- **Why manual interpretation fails:** Subjectivity, inconsistency, can't scale to 100+ CCS sites globally

---

## Things I'd Do Differently

**On the ML side:**
- **Transfer learning:** Pre-train on synthetic seismic data (physics simulators) before fine-tuning on Sleipner
- **Ensemble models:** Combine 2D and 3D U-Nets for complementary strengths
- **Better metrics:** Geophysicists care about plume *volume* and *migration rate*, not just pixel accuracy

**On the engineering side:**
- **Docker container:** Make it easier for others to reproduce (my setup is messy)
- **Weights & Biases:** Track experiments systematically instead of manual logs
- **Grad-CAM visualization:** Show *which parts* of the seismic the model is focusing on

**On the analysis side:**
- **Failure case analysis:** Where does the model mess up? (Edges of plume? Thin layers?)
- **Human-AI comparison study:** Have geophysicists interpret the same data and compare

---

## Running the Code

### 1. Get the Data

Download Sleipner 4D seismic from [CO₂ DataShare](https://co2datashare.org/dataset/sleipner-4d-seismic-dataset).

You'll need:
- Sleipner 4D Seismic Dataset (baseline + time-lapse surveys)
- Sleipner 2019 Benchmark Model (for CO₂ plume labels)

### 2. Setup Environment

```bash
git clone https://github.com/drshtrkp/CO2-NN-Sleipner.git
cd CO2-NN-Sleipner
conda env create -f myenv_details.yml -n co2nn   # PyTorch 2.5.1 + CUDA 12.1, segyio, ObsPy, ...
conda activate co2nn
```

**Requirements:**
- Python 3.10
- PyTorch 2.5.1 with CUDA 12.1
- NVIDIA GPU with 24 GB+ memory (for the 3D U-Net)
- Optional: a SLURM cluster for long training runs (see `EXTRA_FILES/job_full.sh`)

### 3. Prepare Labels (3D U-Net workflow)

```bash
cd 3DD
jupyter notebook label_generation.ipynb
```

This converts the benchmark model's CO₂ plume boundaries (depth domain) into training labels (time domain).

### 4. Prepare Training Data

**For full 3D labels:**
```bash
jupyter notebook input_output_preparation_fulllabel.ipynb
```

**For sparse 2D labels:**
```bash
jupyter notebook input_output_preparation_sparselabel.ipynb
```

This samples 64×64×64 patches from the seismic volumes and creates train/validation splits.

### 5. Train the Model

**Full label training:**
```bash
jupyter notebook Test_training_fulllabel.ipynb
```

**Sparse label training:**
```bash
jupyter notebook Test_training_sparselabel.ipynb
```

Training took 3.08 hours for 200 epochs on an NVIDIA A30 (24 GB). On a SLURM cluster, submit it as a batch job; see `EXTRA_FILES/job_full.sh` for an example.

### 6. Run Inference

Inference runs in the testing notebook, which loads the saved best model and applies it to every vintage (`CO2_identify.py` holds the helper functions it imports):
```bash
jupyter notebook Test_network.ipynb
```

### 7. Visualize Results

```bash
jupyter notebook Figure_plots_DKP.ipynb
```

Generates:
- 3D volume renderings
- Time-lapse evolution plots (1999-2010)
- Layer-by-layer CO₂ distribution maps
- Comparison with human interpretation

---

### Quick Start (2D U-Net - faster but lower quality)

If you have limited GPU memory or want to prototype quickly:

```bash
cd 2D
jupyter notebook input_output_preparation.ipynb  # Prepare 2D slices
jupyter notebook Test_training.ipynb             # Train 2D U-Net (~5 min)
jupyter notebook Figure_plot_2D.ipynb            # Visualize results
```

---

## Limitations & Context

**Important caveats:**

1. **Single-site training:** Only trained on Sleipner. Will it work on other CCS sites with different geology? Unknown.

2. **2010 labels only:** Model never sees 1999-2008 ground truth. I *qualitatively* checked predictions against published papers, but didn't have pixel-level labels for validation.

3. **Seismic resolution limits:** Can't detect CO₂ plumes smaller than ~10 meters (seismic wavelength constraint).

4. **Computational cost:** Training the 3D U-Net used ~23.5 GB of a 24 GB GPU. Smaller GPUs would need smaller batches or patches.

5. **No uncertainty quantification:** Current model is deterministic. Production version should have confidence intervals.

---

## Future Work

If I continue this project (or someone else picks it up):

### Short-term
- [ ] Add dropout-based uncertainty estimates
- [ ] Test on Ketzin/In Salah datasets (different geology)
- [ ] Implement attention mechanisms (focus on plume edges)
- [ ] Real-time inference pipeline (process new surveys automatically)
- [ ] Quantitative metrics: Dice/IoU and precision/recall against the 2010 labels, plus plume area per vintage
- [ ] Combined Dice + BCE loss to handle the small fraction of CO₂ voxels
- [ ] Mixed-precision (AMP) training to cut GPU memory and training time
- [ ] Fixed random seeds and a single config file, so every experiment reruns with one command

### Long-term
- [ ] Multi-task learning: Simultaneously predict CO₂ + estimate saturation
- [ ] Physics-informed neural networks: Enforce fluid flow equations
- [ ] Generative models: Predict future plume migration
- [ ] Explainable AI: Visualize what features the network learned

---

## Why I Built This

**Academic motivation:**  
I wanted to apply deep learning to a real scientific problem (not just Kaggle competitions). CCS monitoring is a perfect domain because:
- **High-impact:** Climate change mitigation depends on CCS scalability
- **Data-rich:** Decades of seismic surveys sitting around
- **Technically challenging:** 3D data, sparse labels, domain constraints

**Career motivation:**  
I'm interested in roles at the intersection of ML and Earth sciences (energy transition, climate tech). This project shows I can:
- Read research papers and implement them
- Work with messy real-world geophysical data
- Think about deployment challenges (not just accuracy metrics)

**Personal motivation:**  
Honestly, I think it's cool that you can "see" CO₂ underground with sound waves, and that a neural network can learn to interpret that. Geophysics is underrated in the ML community.

---

## Coursework Connection

This project applies concepts from:

- **PHYS 570 (Geophysics):** Seismic wave propagation, reflection coefficients
- **CS 373 (Deep Learning):** CNNs, U-Net architecture, data augmentation
- **STAT 512 (Applied Regression):** Model validation, confidence intervals
- **EAPS 591 (Carbon Storage):** CCS monitoring techniques, regulatory requirements

It's rewarding to see how different classes fit together in a real project.

---

## Citation

This work reproduces and extends:

> Li, B., & Li, Y. E. (2021). Neural network-based CO₂ interpretation from 4D Sleipner seismic images. *Journal of Geophysical Research: Solid Earth*, 126(12), e2021JB022524. https://doi.org/10.1029/2021JB022524

## My Implementation

- Reproduced the core 3D U-Net architecture and training methodology
- Verified the temporal consistency results across multiple vintages
- Explored the model's interpretation mechanism through controlled tests
- Documented the workflow for future CCS monitoring applications

### What I Implemented in Code

- **3D U-Net in PyTorch** (`Unet_parts.py`, `mynetwork.py`): 3 encoder and 3 decoder levels, 16 → 32 → 64 → 256 channels, trilinear upsampling with skip connections, batch normalization and a sigmoid output; ~3.2M parameters.
- **Seismic data pipeline:** read SEG-Y surveys with segyio and ObsPy, checked which baseline/monitor pairs are compatible across vintages (`EXTRA_FILES/seismic_pairwise_compatibility.csv`), and converted the benchmark plume model from depth to time (`label_generation.ipynb`).
- **Label generation:** built 3D CO₂ masks from the interpreted plume boundaries with a point-in-polygon (winding number) test (`polygon_inclusion.py`, `labeling.py`).
- **Balanced patch sampling:** sampled 64³ training cubes and cut all-zero cubes from ~60% to 33% of the training set.
- **Training loop:** binary cross-entropy (weighted in the sparse-label experiment), Adam, saving the best checkpoint by validation loss, hyperparameter runs through a `RunBuilder`/`RunManager` loop, and `nn.DataParallel` for GPU training.
- **HPC runs:** SLURM batch scripts for Purdue's Gilbreth GPU cluster (`EXTRA_FILES/job_full.sh`, `job_sparse.sh`) and a GPU check script (`system_info.py`).
- **Full-volume inference:** 90 overlapping patches blended with a 3D Hanning window (`patch_combine_3D` in `CO2_identify.py`), so there are no seams at patch edges.
- **Controlled interpretation tests:** locally amplified time-lapse and baseline images to test what the network responds to (see Key Findings).
- **Ablations:** sparse vs full labels, and 2D vs 3D U-Net.

---

## Acknowledgments

- **Prof. Yunyue Elita Li** (Purdue EAPS, committee co-chair) for research guidance and access to Sleipner data
- **Prof. Qianlai Zhuang** (committee co-chair) and **Prof. Wen-wen Tung** (committee member)
- **Equinor** and **CO₂ DataShare** for making seismic data publicly available
- **Li & Li (2021)** for the original methodology
- **PyTorch community** for excellent documentation

---

## Contact

Questions? Want to collaborate? Reach out!

- **Email:** d6shptl@hotmail.com  
- **GitHub:** [@drshtrkp](https://github.com/drshtrkp)  
- **LinkedIn:** [linkedin.com/in/drshtrkp](https://www.linkedin.com/in/drshtrkp/)

---

**Last Updated:** November 2026  

*If you found this useful or interesting, feel free to ⭐ the repo!*
