# Feedforward Neural Network for Financial Market Direction Prediction

**Practical 3** — a PyTorch feedforward neural network (multi-layer perceptron) that predicts whether a stock's next-day return will be positive or negative, using synthetic technical-indicator data. This practical builds directly on **Practical 1** (a single-layer Perceptron) and demonstrates why stacking hidden layers with non-linear activations lets a network learn patterns a plain Perceptron cannot.

## Overview

Financial markets rarely respond to indicators in a simple, linear way — momentum, overbought/oversold conditions, and volatility tend to interact with each other. A single-layer Perceptron can only draw a straight-line decision boundary between "positive return" and "negative return" days, so it misses these interaction effects.

This notebook implements a small feedforward neural network that:

- Takes five daily technical indicators as input
- Passes them through two hidden layers with ReLU activations
- Outputs a probability of a positive next-day return via a sigmoid output layer
- Trains with backpropagation (BCE loss + Adam optimizer) instead of the manual weight-update rule used in Practical 1
- Is benchmarked directly against the Practical 1 Perceptron on the same data, to make the improvement concrete rather than assumed


## Model Architecture

```
Input (5 features) → Linear(5→16) → ReLU → Linear(16→8) → ReLU → Linear(8→1) → Sigmoid
```

| Layer | Type | Size | Activation |
|---|---|---|---|
| Input | — | 5 | — |
| Hidden 1 | Linear | 16 | ReLU |
| Hidden 2 | Linear | 8 | ReLU |
| Output | Linear | 1 | Sigmoid |

## Features and Label

| Feature | Description |
|---|---|
| Daily Return (%) | Today's percentage price change |
| RSI | Relative Strength Index (momentum, 0–100) |
| Volume Change (%) | Change in trading volume vs. recent average |
| MACD | Moving Average Convergence Divergence signal |
| Volatility (%) | Rolling standard deviation of recent returns |

**Label:** `1` = tomorrow's return was positive, `0` = tomorrow's return was negative.

## What's in the Notebook

1. Introduction and objective
2. Why move from a Perceptron to a feedforward network (non-linearity, interaction effects)
3. Synthetic dataset generation (800 samples, non-linear labeling rule)
4. Train/test split and feature standardization
5. `FeedforwardNN` model definition (PyTorch `nn.Module`)
6. Loss function and optimizer setup
7. Training loop with per-epoch loss logging
8. Training loss curve plot
9. Evaluation on held-out test data (accuracy, confusion matrix, classification report)
10. Sample predictions with confidence scores
11. Head-to-head comparison against the Practical 1 Perceptron
12. Limitations of this approach for real-world finance
13. Better alternatives (Random Forest, XGBoost, LSTM, Transformers, Reinforcement Learning)
14. Summary comparison table

## Results

On the synthetic held-out test set:

| Model | Test Accuracy |
|---|---|
| Perceptron (Practical 1) | ~77% |
| Feedforward Neural Network (Practical 2) | ~84% |

Exact numbers vary slightly by random seed and are printed when the notebook runs.

## Requirements

- Python 3.9+
- `torch`
- `numpy`
- `scikit-learn`
- `matplotlib`
- `jupyter` / `notebook` (or JupyterLab / VS Code) to run the `.ipynb`

## Setup

```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install torch numpy scikit-learn matplotlib jupyter
```

## Running

```bash
jupyter notebook Practical_2_Feedforward_NN_Quant_Finance.ipynb
```

Or run it end-to-end from the command line:

```bash
jupyter nbconvert --to notebook --execute --inplace Practical_2_Feedforward_NN_Quant_Finance.ipynb
```



## Limitations

- Treats each trading day independently — no memory of past sequences
- Synthetic data only; not fitted to real market data
- Prone to overfitting if the network is scaled up without regularization
- A "black box" relative to simpler, more interpretable models like logistic regression

## Next Steps

- Swap the synthetic dataset for real historical price/indicator data
- Add Dropout / Batch Normalization / L2 regularization
- Extend to sequence models (LSTM, GRU, Transformers) to capture temporal dependencies
- Compare against gradient-boosted trees (XGBoost, LightGBM) as a non-neural baseline

