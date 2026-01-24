# CO₂-NN-Sleipner: Deep Learning for Carbon Storage Monitoring

**Darshita Patel | EAPS-GDSP Student | Purdue University | May 2025**

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

Since 1996, Norway has been injecting ~1 million tons of CO₂ per year into underground rock formations beneath the North Sea at the Sleipner field (18.5 million tonnes total by 2020). This is the **world's first commercial-scale offshore CO₂ storage project**. 

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
- **Injection rate:** ~1 million tonnes CO₂/year (18.5 million total by 2020)

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
- Epochs: 200 (~3 hours on TITAN RTX 24GB)
- Normalization: Each input cube normalized separately (zero mean, unit std)

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
- **Training time:** ~3 hours (200 epochs on TITAN RTX 24GB GPU)
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

**Example: 2010 Prediction vs. Human Label**

| Input (Baseline vs Monitor) | Human-Labeled Mask | NN Prediction |
|------------------------------|-------------------|---------------|
| ![Seismic difference](placeholder) | ![Ground truth](placeholder) | ![Model output](placeholder) |

The model captures the main plume boundaries and even gets subtle features like:
- **Feathering edges** where CO₂ concentration tapers off
- **Layer-by-layer distribution** (CO₂ spreads in thin sand layers stacked vertically)
- **Chimney structures** where CO₂ migrates upward through fractures

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

The current model outputs binary predictions (CO₂ vs. no-CO₂). In production, I'd modify it to output:
- **Probability estimates** (0-100% confidence)
- **Uncertainty maps** (use dropout-based ensembles or Bayesian networks)

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
- Receptive field: 200m × 200m × 64ms at bottleneck
- Volume coverage: 1.6km × 1.6km × 512ms per patch
- Full data volume: 3.35km × 7.0km × 2,000ms

Encoder path:
- 4 downsampling blocks
- Each block: Conv3D (3×3×3) → BatchNorm → ReLU → MaxPool (2×2×2)
- Feature maps: 32 → 64 → 128 → 256

Bottleneck:
- 256 feature maps (half of original U-Net for efficiency)

Decoder path:
- 4 upsampling blocks
- Each block: Trilinear upsampling (scale=2) → Conv3D → Skip connection → Conv3D
- Feature maps: 256 → 128 → 64 → 32

Output:
- Conv3D (1×1×1) → Sigmoid
- Shape: [1, 64, 64, 64] (probability map, 0=no CO₂, 1=CO₂)
- Total parameters: ~31M

Key modifications from standard U-Net:
1. Added batch normalization before each ReLU (faster convergence)
2. Halved bottleneck channels (reduced memory/compute)
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
- **Hardware:** NVIDIA TITAN RTX (24GB VRAM)
- **Training time:** ~3 hours
- **Framework:** PyTorch 1.7.1

**Inference strategy:**
- Patch sampling: 3×5×6 = 90 overlapping patches per volume
- Overlap: ~40% in 3D
- Reconstruction: Weighted summation with 3D Hanning window
- Weighting: Full confidence in central 60%, tapers to zero at edges
- Runtime: Few seconds per volume

### Tools Used

- **PyTorch** for model implementation
- **NumPy** for data manipulation
- **Matplotlib/Seaborn** for visualization
- **Jupyter** for exploratory analysis

---

## Repository Structure

```
CO2-NN-Sleipner/
│
├── 2D/                                      # 2D U-Net implementation
│   ├── .ipynb_checkpoints/
│   ├── __pycache__/
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
│   ├── .ipynb_checkpoints/
│   ├── __pycache__/
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
├── EXTRA_FILES/                             # Helper scripts and utilities
│
├── NNpred3D/figures/                        # Output visualizations and results
│   ├── fig2_Mr_0.png                        # Layer predictions over time
│   ├── fig2_R0_0.png
│   ├── fig2_Rt_0.png
│   ├── fig5_Inloss.png                      # Training metrics
│   ├── fig6_R0_0.png                        # Comparison figures
│   └── fig6_R0_1.png
│
├── .gitignore
└── README.md
```

**Key directories:**
- **2D/** - 2D U-Net for slice-by-slice interpretation (faster training, lower quality)
- **3DD/** - 3D U-Net for volumetric interpretation (main approach, production-ready)
- **NNpred3D/figures/** - Paper-ready visualizations of results

**Note:** Seismic data files not included in repo (download from [CO₂ DataShare](https://co2datashare.org/dataset/sleipner-4d-seismic-dataset)). Large .dat files stored with Git LFS.

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
git clone https://github.com/drshtakp/CO2-NN-Sleipner.git
cd CO2-NN-Sleipner
pip install -r requirements.txt  # PyTorch, NumPy, Matplotlib, etc.
```

**Requirements:**
- Python 3.8+
- PyTorch 1.7.1+
- NVIDIA GPU with 24GB+ VRAM (for 3D U-Net)
- Git LFS (for large data files)

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

Training takes ~3 hours on TITAN RTX (24GB) for 200 epochs.

### 6. Run Inference

```bash
python CO2_identify.py --model checkpoints/best_model.pth --vintage 2010
```

Or use the testing notebook:
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

4. **Computational cost:** 3D U-Net needs 24GB GPU. Smaller GPUs would need patch-based inference.

5. **No uncertainty quantification:** Current model is deterministic. Production version should have confidence intervals.

---

## Future Work

If I continue this project (or someone else picks it up):

### Short-term
- [ ] Add dropout-based uncertainty estimates
- [ ] Test on Ketzin/In Salah datasets (different geology)
- [ ] Implement attention mechanisms (focus on plume edges)
- [ ] Real-time inference pipeline (process new surveys automatically)

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

**Key contributions from the original paper:**
1. First application of 3D U-Net to real (not synthetic) CCS monitoring data
2. Demonstrated temporal generalization (train on 2010, test on 1999-2008)
3. Proved robustness to processing mismatch through multi-vintage training
4. Showed NN learns both amplitude difference AND structural similarity
5. Compared 2D vs. 3D U-Nets under sparse labeling scenarios

**My implementation:**
- Reproduced the core 3D U-Net architecture and training methodology
- Verified the temporal consistency results across multiple vintages
- Explored the model's interpretation mechanism through controlled tests
- Documented the workflow for future CCS monitoring applications

---

## Acknowledgments

- **Prof. Elita Li** (Purdue EAPS) for research guidance and access to Sleipner data
- **Equinor** and **CO₂ DataShare** for making seismic data publicly available
- **Li & Li (2021)** for the original methodology
- **PyTorch community** for excellent documentation

---

## Contact

Questions? Want to collaborate? Reach out!

- **Email:** drshtrkp@outlook.com  
- **GitHub:** [@drshtakp](https://github.com/drshtakp)  
- **LinkedIn:** [linkedin.com/in/drashti-patel](https://www.linkedin.com/in/drshtakp/)

---

**Last Updated:** January 2026  

*If you found this useful or interesting, feel free to ⭐ the repo!*
