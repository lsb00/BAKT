# BAMTNFS

**Adaptive Knowledge Transfer and Bidirectional Search for High-Dimensional Evolutionary Multitask Feature Selection**

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Status](https://img.shields.io/badge/Status-Reproducible-green)
![Paper](https://img.shields.io/badge/Manuscript-Knowledge--Based%20Systems-orange)
![License](https://img.shields.io/badge/License-MIT-green)

This repository contains the official implementation of **BAMTNFS** (a **B**idirectional **A**daptive **M**ultitask tra**N**sfer method for neighborhood-based **F**eature **S**election), a particle-swarm-based evolutionary multitasking framework for high-dimensional feature selection.

- **Authors:** Songbin Lan, Fei Yu*, Hongrun Wu*, Zhenglong Xiang, Xuewen Xia (*corresponding authors)
- **Affiliation:** Minnan Normal University; Nanjing University of Information Science and Technology
- **Manuscript submitted to:** *Knowledge-Based Systems*

---

## 1. Overview

Existing evolutionary multitasking feature-selection methods usually determine knowledge transfer with coarse task-level similarity measures and fixed transfer parameters, which can induce **negative transfer** when related tasks share only a subset of informative features, and they often lose population diversity at later iterations.

BAMTNFS addresses these problems with four components:

1. **SU-ordered relevance neighborhoods** — features are ranked by symmetric uncertainty (SU) against the class label and partitioned into ordered neighborhoods, providing structured guidance instead of unstructured grouping.
2. **Two complementary tasks** — Task 1 performs binary bidirectional exploration; Task 2 performs probability-guided sampling via a continuous feature-weight vector.
3. **Two-stage distribution-driven adaptive transfer** — transfer intensity is derived from fitness-weighted task centers and per-particle deviations, **with no additional transfer-control parameter**.
4. **Adaptive bidirectional search** — particles are reallocated between the *few-feature* and *many-feature* directions by a three-criterion rule (average improvement → average fitness → population balance).

![Framework of BAMTNFS](Fig/framework.png)

## 2. Highlights

- **97.45% mean test accuracy** on 17 high-dimensional datasets, with the highest or tied-highest accuracy on **16 of 17** datasets.
- **Only 31.56 features selected on average**, producing the smallest subset on **13 of 17** datasets (the runner-up DEMTFS selects 61.78).
- Parameter-free transfer: no manually tuned transfer probability or weight is required.
- Dataset-level Wilcoxon signed-rank tests favor BAMTNFS over all seven competitors for both accuracy and subset size; 13 of 14 comparisons remain significant after Bonferroni correction.
- Ablation studies confirm the complementary contributions of adaptive transfer, global feature weighting, SU-ordered neighborhoods, and bidirectional search.

![SU-ordered domain partition](Fig/domain_partition.png)

## 3. Method Workflow

1. Discretize continuous features and compute the symmetric uncertainty
   $\mathrm{SU}(X,Y)=2I(X;Y)/(H(X)+H(Y))$ for every feature.
2. Sort features by decreasing SU and partition them into contiguous neighborhoods of size $\tau$.
3. Initialize two co-evolving tasks:
   - **Task 1 (binary):** $N$ particles labeled as *few-feature* or *many-feature*; positions are updated with a PSO velocity and refined within SU neighborhoods.
   - **Task 2 (continuous):** Bernoulli masks are sampled from feature-selection probabilities governed by a global feature-weight vector.
4. At each generation, compute the two-stage adaptive transfer:
   - **Stage 1:** directional discrepancy between a task center and the joint center;
   - **Stage 2:** per-particle deviation, combined with Stage 1 into a continuous transfer weight.
5. Evaluate every subset with the wrapper objective
   $\Phi = \lambda\,\epsilon + (1-\lambda)\,\dfrac{|S|}{D}$,
   where $\epsilon$ is the macro-averaged false-discovery rate estimated by 5-fold cross-validation with a 1-nearest-neighbor classifier.
6. Every $W$ generations, move one particle between the two directions using the average-improvement → average-fitness → population-balance rule.
7. Return the global best subset across both tasks and report test accuracy.

## 4. Datasets

We use 17 publicly available high-dimensional benchmark datasets (50–210 instances, 2,000–22,283 features, 2–11 classes), downloaded from the [DEMTFS repository](https://github.com/dzhenya-pso/DEMTFS).

| # | Dataset | Features | Instances | Classes |
|---:|---|---:|---:|---:|
| 1 | colon | 2,000 | 62 | 2 |
| 2 | warpPIE10P | 2,420 | 210 | 10 |
| 3 | lung | 3,312 | 203 | 5 |
| 4 | Lymphoma1 | 4,026 | 96 | 9 |
| 5 | GLIOMA | 4,434 | 50 | 4 |
| 6 | Nci | 5,244 | 61 | 8 |
| 7 | DLBCL | 5,469 | 77 | 2 |
| 8 | Brain_Tumor1 | 5,920 | 90 | 5 |
| 9 | Carcinom | 9,182 | 174 | 11 |
| 10 | Adenocarcinoma | 9,868 | 76 | 2 |
| 11 | pixraw10P | 10,000 | 100 | 10 |
| 12 | orlraws10P | 10,304 | 100 | 10 |
| 13 | Prostate_Tumor | 10,509 | 102 | 2 |
| 14 | CLL-SUB-111 | 11,340 | 111 | 3 |
| 15 | Lung_Cancer | 12,600 | 203 | 5 |
| 16 | SMK-CAN-187 | 19,993 | 187 | 2 |
| 17 | GLI-85 | 22,283 | 85 | 2 |

## 5. Repository Structure

> Please adjust the code-side names below to match the actual layout of your source code.

```
BAMTNFS/
├── src/                  # BAMTNFS source code (Python)
│   ├── main.py           # entry point: run all datasets / algorithms
│   ├── bamtnfs.py        # core algorithm (two tasks, transfer, bidirectional search)
│   ├── su_neighborhood.py# symmetric-uncertainty computation and domain partition
│   ├── fitness.py        # 1-NN wrapper fitness (5-fold CV)
│   └── utils.py          # data loading, statistics, logging
├── datasets/             # datasets (downloaded from the DEMTFS link above)
├── results/              # output tables, convergence curves, statistics
├── Fig/                  # framework and result figures
├── requirements.txt      # dependencies
└── README.md
```

## 6. Requirements

- Python 3.9 or newer
- NumPy, SciPy, pandas
- scikit-learn (1-NN classifier, data preprocessing, statistical tests)

Install dependencies:

```bash
pip install -r requirements.txt
```

> `requirements.txt` content (create it if it does not exist):
>
> ```text
> numpy>=1.24
> scipy>=1.10
> pandas>=2.0
> scikit-learn>=1.3
> matplotlib>=3.7
> ```

## 7. Usage

```bash
# 1. Download the 17 datasets from https://github.com/dzhenya-pso/DEMTFS
#    and place them in the datasets/ folder.
# 2. Run BAMTNFS (the script name below is an example; replace with your actual entry point)
python src/main.py
```

Default settings (identical to the paper):

- Generations: **100**; total population: **20** (10 particles per task)
- Neighborhood size: **τ = 3**; fitness weight: **λ = 0.9**
- Inertia weight: **w = 0.9 → 0.4**; acceleration coefficients: **c1 = c2 = c3 = 1.49445**
- Classifier: **1-nearest neighbor**, 5-fold cross-validation on the training set
- Split: features min–max normalized to [0,1], **7:3 train/test split**
- **20 independent runs** per dataset; report mean ± standard deviation

## 8. Comparison Methods

BAMTNFS is compared against seven state-of-the-art algorithms:

| Method | Year | Type |
|---|---:|---|
| PSO-EMT | 2022 | Evolutionary multitask PSO |
| MTPSO | 2022 | Evolutionary multitask PSO |
| SFE-PSO | 2023 | Single-task PSO (selection/deselection) |
| MEL | 2024 | Multi-task evolutionary learning |
| SAR-MTPSO | 2025 | Self-adjusting representation multitask PSO |
| DEMTFS | 2026 | Dynamic evolutionary multitasking framework |
| EMT-ALV | 2026 | Adaptive length-variation evolutionary multitasking |

## 9. Main Results

| Metric | BAMTNFS | Note |
|---|---:|---|
| Mean test accuracy | **97.45%** | Best average rank 1.06 (paper avg-rank table); 100% on six datasets |
| Mean selected features | **31.56** | Smallest subset on 13/17 datasets |
| Mean runtime | 265.49 s | Sixth among eight methods; DEMTFS is fastest (22.90 s) |
| Accuracy best/tied-best | **16 / 17** datasets | |
| Subset-size best | **13 / 17** datasets | |

See the manuscript for convergence curves, accuracy/feature-count heatmaps, box plots, runtime comparison, ablation studies, and Wilcoxon + Bonferroni statistical tests.

## 10. Citation

If you find this code or the paper helpful, please cite:

```bibtex
@article{lan2026bamtnfs,
  title   = {Adaptive Knowledge Transfer and Bidirectional Search for High-Dimensional Evolutionary Multitask Feature Selection},
  author  = {Lan, Songbin and Yu, Fei and Wu, Hongrun and Xiang, Zhenglong and Xia, Xuewen},
  journal = {Knowledge-Based Systems},
  year    = {2026},
  note    = {Manuscript under review}
  % volume, number, pages, doi will be updated after publication
}
```

## 11. Contact

- **First author:** Songbin Lan
- **Corresponding authors:** Fei Yu (yufei@whu.edu.cn), Hongrun Wu (wuhongrun@whu.edu.cn)
- Issues and questions are welcome through the [Issues](https://github.com/lsb00/BAMTNFS/issues) page.

## 12. Acknowledgements

This work was supported by the National Natural Science Foundation of China (62106092), the Natural Science Foundation of Fujian Province (2024J01822, 2025J01981, 2025J01980), and the Natural Science Foundation of Zhangzhou City (ZZ2024J28).

## 13. License

This project is released under the [MIT License](LICENSE).
