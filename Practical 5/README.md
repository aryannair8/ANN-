# CNN for Candlestick Chart Trend Classification

**Practical 5** — a TensorFlow/Keras Convolutional Neural Network (CNN) that classifies candlestick chart images as **Uptrend** or **Downtrend**, using real historical stock price data (Apple Inc., AAPL). This is the standard "two-class image classification" exercise — the same structure as the classic Cats vs Dogs CNN tutorial — adapted to the quant finance domain by using candlestick chart images as the two classes instead of animal photos.

## Overview

Rather than feeding a network hand-engineered numeric indicators (as in Practical 2), this practical treats a rolling window of daily price action as an **image** — a rendered candlestick chart — and trains a CNN to visually recognize whether that chart's own trend is upward or downward. This is the same kind of pattern recognition a technical analyst performs by eye, and it is the foundational building block behind any chart-pattern-recognition system.

**Scope note:** this notebook classifies the trend that is already visible within each chart image. It does not claim to forecast future, not-yet-visible price movement — that is a substantially harder problem, discussed explicitly in the notebook's Limitations section along with why chart-shape-only forecasting tends to perform close to random.

## Model Architecture

```
Input (64×64×3) → Conv2D(8) → ReLU → MaxPool → Conv2D(16) → ReLU → MaxPool → Flatten → Dense(16) → ReLU → Dropout(0.3) → Dense(1) → Sigmoid
```

| Layer | Type | Details |
|---|---|---|
| Input | — | 64 × 64 RGB candlestick chart image |
| Conv Block 1 | Conv2D + MaxPooling2D | 8 filters, 3×3, ReLU |
| Conv Block 2 | Conv2D + MaxPooling2D | 16 filters, 3×3, ReLU |
| Flatten | — | — |
| Dense | Fully connected | 16 units, ReLU |
| Dropout | Regularization | rate 0.3 |
| Output | Dense | 1 unit, Sigmoid → P(Uptrend) |

The network is kept deliberately small: with roughly 480 labeled chart images, a large CNN would simply memorize the training set rather than learn the general visual concept of "uptrend" vs "downtrend."

## Dataset

- **Source:** real daily OHLCV data for AAPL (Apple Inc.), Feb 2015 – Feb 2017, pulled from a public dataset hosted on GitHub (`plotly/datasets/finance-charts-apple.csv`).
- **Image generation:** each 20-trading-day rolling window is rendered as a candlestick chart image using `mplfinance` (no axes or labels — a pure visual pattern, like a cropped chart screenshot).
- **Label:** `Uptrend` if the closing price on the last day of the window is higher than on the first day; `Downtrend` otherwise.
- **Split:** the timeline is cut into contiguous blocks, and whole blocks are randomly assigned to train / validation / test. This avoids leaking near-duplicate overlapping windows across splits while still mixing both trend regimes into every split.

| Split | Images |
|---|---|
| Train | ~340 |
| Validation | ~46 |
| Test | ~100 |

## What's in the Notebook

1. Introduction and objective
2. Why a CNN instead of a plain feedforward network (visual pattern recognition vs numeric indicators)
3. Loading real AAPL OHLCV data
4. Candlestick chart image generation from rolling windows, with labeling logic
5. Sample chart image previews by class
6. Block-based train/validation/test split (leakage-aware)
7. Loading images into tensors
8. CNN model definition (`Sequential` Conv2D/MaxPooling/Dense stack)
9. Compilation (binary cross-entropy loss, Adam optimizer)
10. Training loop with early stopping
11. Training accuracy/loss curves
12. Evaluation on held-out test data (accuracy, confusion matrix, classification report)
13. Sample predictions with confidence scores, randomly sampled across the test set
14. Explicit discussion of what the model has and has not demonstrated
15. Limitations
16. Better alternatives (CNN+LSTM hybrids, Vision Transformers, transfer learning, reinforcement learning)
17. Summary comparison with Practical 2

## Results

On the held-out test set (block-based split, unseen chart images):

| Metric | Value |
|---|---|
| Test Accuracy | ~87–90% |
| Test Loss (BCE) | ~0.27–0.29 |

Both classes (Uptrend and Downtrend) are predicted with balanced precision and recall — training and validation curves show the model converging without collapsing to a single predicted class. Exact numbers vary slightly by random seed and are printed when the notebook runs.

## Requirements

- Python 3.9+
- `tensorflow`
- `mplfinance`
- `pandas`, `numpy`
- `scikit-learn`
- `matplotlib`
- `jupyter` / `notebook` (or JupyterLab / VS Code) to run the `.ipynb`

## Setup

```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install tensorflow mplfinance pandas numpy scikit-learn matplotlib jupyter
```

## Running

```bash
jupyter notebook Practical_5_CNN_Candlestick_Classification.ipynb
```

Or run it end-to-end from the command line:

```bash
jupyter nbconvert --to notebook --execute --inplace Practical_5_CNN_Candlestick_Classification.ipynb
```

The notebook downloads the AAPL price data from a public GitHub-hosted CSV at runtime, so an internet connection is required on first run. Rendered candlestick images are cached to a local `candlestick_images/` folder so re-running the notebook does not regenerate images that already exist.

## Repository Structure

```
.
├── Practical_1_Perceptron.ipynb                      # single-layer Perceptron baseline
├── Practical_2_Feedforward_NN_Quant_Finance.ipynb     # feedforward NN on numeric indicators
├── Practical_5_CNN_Candlestick_Classification.ipynb   # this notebook
├── candlestick_images/                                # generated chart images (created on first run)
└── README.md
```

## Limitations

- Classifies the trend already visible within a chart image — does not, by itself, predict future price movement (a much harder problem; early tests attempting to forecast direction several days ahead performed close to random on this same pipeline)
- Small dataset — a single ticker (AAPL) over ~2 years, limiting the variety of market regimes seen
- Overlapping rolling windows are correlated with their neighbors, reducing the amount of truly independent training signal
- Chart-image classification discards useful numeric context (volume, broader market conditions, macro events)
- Financial markets are noisy and non-stationary, which is a core reason forecasting future returns is fundamentally harder than recognizing a past, already-visible trend

## Next Steps

- Extend to multiple tickers and longer historical spans for more regime diversity
- Add Batch Normalization and stronger regularization for a deeper CNN
- Combine with an LSTM/ConvLSTM to incorporate temporal sequence information across many charts
- Explore transfer learning from a pretrained image model (e.g. ResNet, EfficientNet)
- Compare against classical technical-pattern-recognition rules as a baseline

## License

Educational use. No investment advice.
