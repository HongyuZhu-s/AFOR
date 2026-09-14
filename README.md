# AFOR

## Adaptive Forgetting for Nonstationary Optimization

AFOR is a tensor-wise adaptive optimizer for robust cross-subject EEG
decoding. It replaces the fixed second-moment decay coefficient used by
Adam and AdamW with a dynamic coefficient estimated online from local
gradient statistics.

<p align="center">
  <a href="https://github.com/HongyuZhu-s/AFOR/blob/main/Fig/Fig1.png">
    <img
      src="https://github.com/HongyuZhu-s/AFOR/blob/main/Fig/Fig1.png?raw=1"
      alt="Overview of the AFOR optimizer"
      width="100%"
    />
  </a>
</p>

<p align="center">
  <a href="https://github.com/HongyuZhu-s/AFOR/blob/main/Fig/Fig1.png">
    Open the AFOR overview figure as a PDF
  </a>
</p>

## Introduction

Electroencephalography (EEG) provides non-invasive, millisecond-scale
monitoring of brain activity and supports applications such as emotion
recognition, motor imagery, and sleep staging. Although within-subject
decoding has achieved considerable progress, cross-subject generalization
remains a central challenge in practical applications.

EEG decoders are typically trained with Adam or AdamW under a fixed
second-moment decay coefficient. However, cross-subject learning involves low
signal-to-noise ratios, subject variability, and gradient nonstationarity. A
fixed coefficient assumes that gradient statistics are homogeneous across
layers and training stages, which can limit the adaptability of the model and
degrade generalization.

AFOR addresses this limitation by converting the fixed second-moment decay
coefficient into a dynamic, tensor-wise coefficient estimated online from
local gradient state. AFOR combines a Residual-Alignment Signal Scorer (RASS)
with an Adaptive Forgetting Controller (AFC). RASS summarizes local gradient
residuals and directional agreement into a signal-quality score. AFC maps this
score through self-referential normalization to a bounded per-step decay
coefficient. A cumulative-product initialization correction maintains the
normalization of the second-moment estimate under time-varying decay.

Under a strict cross-subject protocol on three EEG benchmarks, the paper
reports the best average performance among the compared optimizers and mean
test-accuracy improvements over Adam of 3.00%, 2.07%, and 4.38% across the
three benchmarks.

## Method

AFOR contains two coupled components:

1. **Residual-Alignment Signal Scorer (RASS)**
   - Tracks gradient residuals at fast and slow time scales.
   - Measures the directional agreement between the current gradient and the
     previous momentum direction.
   - Combines momentum magnitude and direction consistency into a tensor-wise
     signal-to-noise score.

2. **Adaptive Forgetting Controller (AFC)**
   - Maintains an online mean and variance of the signal score for every
     parameter tensor.
   - Converts the resulting Z-score to a bounded second-moment coefficient.
   - Uses a warm-up gate during the first 100 optimizer steps.
   - Uses a cumulative product of the time-varying coefficients for
     initialization correction.

In dense-gradient settings, AFOR keeps the same first-moment update and
parameter-update structure as Adam-family optimizers while adapting the
second-moment memory independently for each parameter tensor.

## Datasets

The repository does not redistribute datasets, and follow the original
dataset license and usage terms.

[[DEAP](https://www.eecs.qmul.ac.uk/mmv/datasets/deap/)]
[[BCI Competition IV 2a](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2012.00055/full)]
[[ISRUC-Sleep](https://www.sciencedirect.com/science/article/abs/pii/S0169260715002734)]


### Paper Benchmarks

| Dataset | Task | Subjects | EEG information |
| --- | --- | ---: | --- | --- |
| DEAP | Emotion recognition | 32 | 32 channels, 128 Hz; binary valence or arousal |
| BCI Competition IV 2a | Motor imagery | 9 | 22 channels, 250 Hz; four classes |
| ISRUC-Sleep SG-I | Sleep staging | 100 | Six EEG channels, 200 Hz; five sleep stages |

### Preprocessing Used by the Paper

- **DEAP:** baseline subtraction, 0.3-50 Hz fifth-order Butterworth
  band-pass filtering, and non-overlapping 1-second windows.
- **BCI IV 2a:** 0.5-3.5 seconds after cue onset, 0.3-50 Hz band-pass
  filtering, and overlapping 250-sample windows with a 125-sample stride.
- **ISRUC:** 30-second epochs segmented into overlapping 600-sample windows
  with a 300-sample stride; no band-pass filtering.

## Installation

### Install the Optimizer Package

Once the package is available on PyPI:

```bash
python -m pip install afor-optimizer
```

Then use AFOR in a PyTorch training loop:

```python
import torch
from AFOR import afor

model = torch.nn.Linear(10, 2)
optimizer = afor(
    model.parameters(),
    lr=1e-3,
    betas=(0.9, 0.999),
    beta2_min=0.99,
    dir_weight=1.0,
    eps=1e-8,
    weight_decay=1e-4,
)

inputs = torch.randn(16, 10)
targets = torch.randint(0, 2, (16,))
loss = torch.nn.functional.cross_entropy(model(inputs), targets)
loss.backward()
optimizer.step()
optimizer.zero_grad()
```

### Install

Activate the virtual environment:

```bash
# Linux or macOS
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1
```

Install the required packages:

```bash
python -m pip install --upgrade pip
python -m pip install torch numpy scipy scikit-learn tqdm matplotlib
```

The repository requires Python 3.10 or later. Install the PyTorch build that
matches your CUDA version when using an NVIDIA GPU.



## Reproducibility

The default random seed is `2024`. The training scripts use subject-wise
cross-validation and early stopping. For a reproducible run, keep the
dataset preprocessing, seed, learning rate, batch size, number of epochs,
and fold configuration fixed.

## Citation



## License

This project is released under the MIT License. See
[`LICENSE`](LICENSE) for the full text.
