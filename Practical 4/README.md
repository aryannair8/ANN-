# Keras MLP for Multiclass Classification — Trading Signal (Buy / Hold / Sell)

**Practical 4** — a small Multi-Layer Perceptron (MLP) built with Keras that classifies a trading signal into three classes (**Buy**, **Hold**, **Sell**) from two simple technical features. Kept intentionally basic and easy to explain — the same spirit as a toy AND-gate example, just adapted to a finance domain and made genuinely multiclass.

## Overview

This practical is a minimal, explainable example of multiclass classification with Keras. Instead of predicting a single binary outcome, the network outputs a probability distribution across three trading actions based on two indicators:

- **Daily Return (%)** — how much the price moved today
- **RSI (Relative Strength Index)** — a momentum indicator (0–100)

The model is a small `Sequential` MLP with a softmax output layer, trained on a tiny hand-picked dataset of 9 examples (3 per class) so the logic stays easy to trace end to end.

## Model Architecture

```
Input (2 features) → Dense(8, ReLU) → Dense(4, ReLU) → Dense(3, Softmax)
```

| Layer | Type | Size | Activation |
|---|---|---|---|
| Input | — | 2 | — |
| Hidden 1 | Dense | 8 | ReLU |
| Hidden 2 | Dense | 4 | ReLU |
| Output | Dense | 3 | Softmax |

Softmax outputs three probabilities that sum to 1 — one per class — instead of the single 0–1 probability a sigmoid output would give for binary classification.

## Features and Labels

| Feature | Description |
|---|---|
| Daily Return (%) | Today's percentage price change |
| RSI | Momentum indicator (0–100); high = overbought, low = oversold |

| Label | Meaning |
|---|---|
| 0 — Sell | Falling price, overbought conditions |
| 1 — Hold | No strong signal either way |
| 2 — Buy | Rising price, oversold / recovering conditions |

Labels are one-hot encoded (e.g. class 2 → `[0, 0, 1]`) for use with `categorical_crossentropy`.

## What's in the Notebook

1. Imports and a small, hand-picked dataset (9 examples, 3 per class)
2. Feature scaling (standardization) — explained inline, since Daily Return and RSI are on very different numeric scales
3. MLP model definition (`Sequential` with Dense/ReLU/Softmax)
4. Model compilation (categorical cross-entropy loss, Adam optimizer)
5. Training loop
6. Predictions with per-class probabilities and the predicted label
7. A short discussion of MLP applications, including image recognition and finance-specific caveats (low signal-to-noise ratio, non-stationarity, overfitting risk)

## Results

With feature scaling and a fixed random seed, the model reaches 100% training accuracy on the toy dataset, with confident, well-separated probabilities for each class. Exact probabilities are printed when the notebook runs.

## Requirements

- Python 3.9+
- `tensorflow`
- `numpy`
- `jupyter` / `notebook` (or JupyterLab / VS Code) to run the `.ipynb`

## Setup

```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install tensorflow numpy jupyter
```

## Running

```bash
jupyter notebook Practical_4_Keras_MLP_Multiclass.ipynb
```

Or run it end-to-end from the command line:

```bash
jupyter nbconvert --to notebook --execute --inplace Practical_4_Keras_MLP_Multiclass.ipynb
```

## Repository Structure

```
.
├── Practical_1_Perceptron.ipynb                       # single-layer Perceptron baseline
├── Practical_2_Feedforward_NN_Quant_Finance.ipynb      # feedforward NN on numeric indicators
├── Practical_4_Keras_MLP_Multiclass.ipynb              # this notebook
├── Practical_5_CNN_Candlestick_Classification.ipynb    # CNN on candlestick chart images
└── README.md
```

## Why Feature Scaling Matters Here

Daily Return (roughly −3 to +3) and RSI (roughly 0–100) are on very different numeric scales. Without scaling, RSI dominates the network's early learning simply because its raw numbers are larger — not because it is actually more informative. Standardizing both features to mean 0 and standard deviation 1 lets the network weigh them fairly, and is what makes this tiny dataset learnable in the first place.

## Limitations

- Toy dataset (9 examples) — meant for explaining the mechanics of a Keras MLP, not for real trading decisions
- Financial data has a low signal-to-noise ratio and market relationships change over time (non-stationarity)
- A plain MLP can easily memorize noise rather than learn a generalizable pattern, especially on small datasets
- No train/test split — this notebook is a worked example of the training mechanics, not a generalization benchmark

## Next Steps

- Add a proper train/validation/test split and evaluate on held-out data
- Use real historical indicator data instead of hand-picked examples (see Practical 2)
- Add more features (MACD, volatility, volume) and compare against simpler baselines like logistic regression
- Explore class probabilities as a confidence signal rather than only the arg-max prediction

## License

Educational use. No investment advice.
