# Implementing a single neuron model manually

## 1. Why use TensorFlow instead of the previous simple code?

**What it is:** TensorFlow is a high-powered AI framework created by Google.

**Why it is used:** In the last code, you manually coded a single neuron using raw math. TensorFlow handles all that complicated math behind the scenes automatically. It allows you to build massive networks with multiple layers and millions of neurons using just a few simple building blocks.

## 2. Why do we have a "Hidden Layer" with 4 neurons?

**What it is:** `Dense(4, activation='relu')`

**Why it is used:** Instead of a single neuron making a lonely decision, we created a hidden committee of 4 neurons. Each of these 4 neurons focuses on a different pattern in the data. `relu` is an activation switch that acts like a human filter — if an impression is negative, it instantly zeroes it out, allowing the network to focus purely on positive features.

## 3. Why does the final layer use sigmoid?

**What it is:** `Dense(1, activation='sigmoid')`

**Why it is used:** The hidden committee passes their opinions to a final single head judge. The sigmoid function acts like a confidence slider. Instead of returning a harsh, instant binary switch (0 or 1), it outputs a smooth percentage like 0.8523 (85.2% confident it is a 1). We then use 0.5 (50%) as our human threshold to turn that percentage into a final Pass or Fail verdict.

## 4. Why do we use adam and binary_crossentropy?

**What they are:** The optimizer (`adam`) is the AI's internal mathematical personal trainer. The loss (`binary_crossentropy`) is the grading scale.

**Why they are used:** `binary_crossentropy` calculates the exact mathematical penalty for how far the AI's percentage guess was from the truth. `adam` is a smart, adaptive algorithm that uses that penalty to instantly twist all the weights and biases across the entire network in the right direction, working much faster than our basic manual learning loop.

## The Big Picture: What is actually happening?

This code shows the exact evolutionary jump from a single neuron to a Deep Neural Network.

When the model starts, the entire panel of judges is completely blind. Their internal weights are completely randomized. When you run the "BEFORE" test, they guess randomly, often giving terrible raw scores like 0.612 (Pass) for inputs that should obviously be 0 (Fail).

Then, the workshop begins (`model.fit`). Because the network is more complex than a single neuron, looping through the data only 10 times is like sending a human to a 10-minute training seminar — it is a great start, but not nearly enough time to perfectly align all 4 hidden judges and the head judge.

While the model will actively tweak its dials and begin moving its percentages closer to the real answers, it usually requires around 500 to 1000 epochs of practice to get this specific multi-layered network to output perfect 100% accurate results.

## OUTPUT

### Before Training (Blind Guessing)

- The AI guesses randomly: It gives every input a lazy score around 50% to 59%.
- It fails completely: Because every guess is above 50%, it marks everything as a 1.

### During Training (The Short Workshop)

- The AI tries to learn: The "loss" score decreases from 0.7128 to 0.7082, showing it is actively tweaking its dials.
- It runs out of time: Because it only gets 10 rounds of practice, the accuracy stays stuck at 50%. It is like ending a study session after only 2 minutes.

### After Training (The Unfinished Result)

- It made small progress: It successfully dragged the score for `[0 0]` down to 0.4975, correctly labeling it a 0.
- It remains confused: It ran out of time before it could drag the other incorrect answers (`[0 1]` and `[1 0]`) down below 50%.
