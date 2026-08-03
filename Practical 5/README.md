# Implementing a Multi-Layer Perceptron (MLP) using TensorFlow

## What is Happening and Why

### 1. Why use a Multi-Layer Perceptron (MLP)?

**What it is:**
A Multi-Layer Perceptron (MLP) is one of the simplest forms of an Artificial Neural Network. It consists of an input layer, one or more hidden layers, and an output layer connected through neurons.

**Why it is used:**
An MLP learns patterns from input data by adjusting its weights during training. It is commonly used for classification and regression problems involving structured or tabular data.

---

### 2. Why use ReLU in the hidden layers?

**What it is:**
ReLU (Rectified Linear Unit) is an activation function defined as:

```python
f(x) = max(0, x)
```

**Why it is used:**
Without an activation function, stacking multiple Dense layers would behave like a single linear layer. ReLU introduces non-linearity, allowing the network to learn complex relationships while also training faster than older activation functions like Sigmoid or Tanh.

---

### 3. Why use Sigmoid in the output layer?

**What it is:**
Sigmoid converts the final output into a probability between 0 and 1.

**Why it is used:**
Since the AND gate has only two possible outputs (0 or 1), Sigmoid is the appropriate activation function for binary classification.

---

### 4. Why use the Adam Optimizer?

**What it is:**
Adam is an optimization algorithm used to update the weights of the neural network during training.

**Why it is used:**
Instead of using a fixed learning rate, Adam automatically adapts the learning rate for every parameter, making training faster and more stable. It is one of the most commonly used optimizers in deep learning.

---

### 5. Why use Binary Cross-Entropy Loss?

**What it is:**
Binary Cross-Entropy measures how different the predicted probability is from the actual class.

**Why it is used:**
Since this project performs binary classification (0 or 1), Binary Cross-Entropy is more suitable than Mean Squared Error because it penalizes incorrect confident predictions more effectively.

---

### 6. Why use the AND Gate Dataset?

**What it is:**
The AND gate is one of the simplest logical datasets used in machine learning.

| Input | Output |
| ----- | ------ |
| 0,0   | 0      |
| 0,1   | 0      |
| 1,0   | 0      |
| 1,1   | 1      |

**Why it is used:**
The dataset is extremely small, making it easy to understand how an MLP learns without being distracted by large datasets.

---

### 7. How is MLP used in Image Recognition?

**What it is:**
Images are represented as numerical pixel values before being passed into the network.

**How it works:**
A grayscale image of size **28 × 28** contains **784 pixels**. These pixels are flattened into a one-dimensional vector and fed into the MLP. The hidden layers learn patterns from these pixel values, while the output layer predicts the image class.

**Limitation:**
Flattening removes the spatial relationship between neighbouring pixels. Because of this, MLPs struggle with complex image recognition tasks, and Convolutional Neural Networks (CNNs) are generally preferred.

---

### 8. How is MLP used in Finance?

**What it is:**
MLPs can learn relationships between financial indicators and future outcomes.

**Applications:**

* Credit risk prediction
* Loan default prediction
* Fraud detection
* Stock movement prediction using engineered features

**Limitation:**
Financial markets are noisy and constantly changing. A basic MLP can memorize historical patterns instead of learning meaningful relationships, causing overfitting. For this reason, feature engineering, regularization, and proper validation techniques are essential.

---

## Simplified Output Explanation

| Input    | Confidence | Explanation                                                                 |
| -------- | :--------: | --------------------------------------------------------------------------- |
| `[0, 0]` |     Low    | Both inputs are 0, so the model predicts Class 0.                           |
| `[0, 1]` |     Low    | Only one input is active, which does not satisfy the AND condition.         |
| `[1, 0]` |     Low    | Again, only one input is active, so the prediction remains Class 0.         |
| `[1, 1]` |    High    | Both inputs are active, satisfying the AND condition, resulting in Class 1. |
