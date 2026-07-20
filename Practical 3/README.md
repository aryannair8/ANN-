# Implementing a Feedforward Neural Network using TensorFlow or PyTorch

## What is Happening and Why

### 1. Why use PyTorch instead of TensorFlow?

**What it is:** PyTorch is a highly popular AI library created by Meta (Facebook).

**Why it is used:** While TensorFlow acts like a factory assembly line where everything is built ahead of time, PyTorch feels like writing standard Python code. It is highly flexible and explicit, making it the favorite tool for AI researchers building modern LLMs.

### 2. Why write an explicit class structure?

**What it is:**
```python
class FeedforwardNN(nn.Module):
```

**Why it is used:** PyTorch forces you to design your AI network explicitly. You define the physical components inside `__init__` (the layers and filters), and you explicitly detail the flow of data inside the `forward` function. This blueprint structure gives you total control over how information flows through the network.

### 3. Why are there three lines (`zero_grad`, `backward`, `step`) in the loop?

**What they are:** This is the core engine of PyTorch learning.

- **`optimizer.zero_grad()`** — Clears out old math notes from the previous round so they don't mess up the new feedback.
- **`loss.backward()`** — A highly advanced mathematical process called *Backpropagation*. It calculates the exact amount of blame every single weight and bias dial deserves for the final mistake.
- **`optimizer.step()`** — Takes that blame list and physically twists every single dial across the entire network in the correct direction.

### 4. Why use `with torch.no_grad():` at the end?

**What it is:** This turns off PyTorch's internal "learning and tracking mode."

**Why it is used:** When testing the model, you don't want it to keep updating its dials or wasting computer memory tracking mistakes. This command sets the AI into a strict "read-only" evaluation mode.

## Simplified Output Explanation

| Input       | Confidence | Explanation |
|-------------|:----------:|-------------|
| `[0.0, 0.0]` | 20.82% | The AI sees no input activity, keeping its confidence very low and well below the 50% threshold (Class 0). |
| `[0.0, 1.0]` | 23.71% | Only one input is on, which fails to convince the panel; the score remains low (Class 0). |
| `[1.0, 0.0]` | 31.24% | Again, a single active input is not enough to pass the strict rule, staying below 50% (Class 0). |
| `[1.0, 1.0]` | 73.44% | Both inputs fire simultaneously, causing the AI's confidence to surge way past 50% (Class 1). |
</content>
</invoke>
