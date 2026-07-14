# Perceptron Learning Practical 1

## 1. Why do we have Weights and Biases?

**What they are:** They are the internal dials of the AI. More accurately, they act as the AI's "memory" and "judgment filters."

**Why they are used:** The computer possesses zero concept of human abstractions like "AND," "logic," or "fairness." It strictly processes numerical equations.

- **The Weights** — These dials function as importance filters for each piece of incoming data. If `weight_1` becomes a high positive number, the AI learns that `input_1` is incredibly important to the final answer. If a weight is near zero, the AI learns to ignore that input entirely.
- **The Bias** — This dial represents the AI's inherent threshold — its baseline tendency to say "yes" or "no" before it even looks at the data. Think of it as a constant background pressure that shifts the final math up or down.
- **The Collaboration** — By automatically twisting the weights and bias dials simultaneously, the computer reshapes a generic mathematical formula until its outputs perfectly mirror your real-world answers.

## 2. Why do we calculate the Error?

**What it is:**
```
error = true_answer - ai_guess
```

**Why it is used:** This is the AI's feedback loop and moral compass. Without this calculation, the computer would never know if its current dial settings are good or bad. It serves as an explicit directional guide:

| Error | Meaning |
|---|---|
| `0` | Perfect match. The AI requires no changes, leaving the dials exactly as they are. |
| `+1` | The true answer was 1 but the AI guessed 0. This tells the AI: "Your combined math score was way too low. You need to turn your dials **UP** to be more generous next time." |
| `-1` | The true answer was 0 but the AI guessed 1. This tells the AI: "Your math score was far too high. You were too reckless. Turn your dials **DOWN** to be stricter next time." |

## 3. Why multiply by the inputs (`in_1` and `in_2`) during learning?

**What it is:**
```
self.weight_1 += learning_rate * error * in_1
```

**Why it is used:** This enforces fairness and accountability in the learning process. It ensures the AI only modifies the specific parts of its brain that actually contributed to the mistake.

Imagine the AI makes a bad guess on the input `[0, 1]`:

1. Because `in_1` was exactly `0`, that first input had a grand total of zero impact on the final bad guess. It was completely silent on stage.
2. By multiplying the update formula by `in_1` (which is `0`), the whole mathematical adjustment for `weight_1` instantly collapses to zero (`learning_rate * error * 0 = 0`).
3. Consequently, `weight_1` stays perfectly protected and unchanged, while `weight_2` (associated with the active input `1`) takes the full heat of the correction.

## 4. The Big Picture: What is actually happening?

The AI begins in a state of absolute, blank-slate ignorance.

- **The Blind Guess** — It looks at the first pair of inputs (like `[0,0]`), runs its completely blank equation, and guesses randomly.
- **The Course Correction** — It checks its cheat sheet, calculates the error, and nudges its dials a tiny fraction of a millimeter (controlled by the cautious `learning_rate` of `0.1`).
- **The Iteration** — It moves to the next example, repeats the process, and tweaks the dials again.
- **The Repetition (Epochs)** — Running through the 4 examples just once is not enough. The AI might fix its dials for `[1,1]` only to accidentally ruin its performance for `[0,1]`. Therefore, we force it to repeat the entire 4-step loop 10 times over (`rounds=10`).


## OUTPUT
- `[0 0] -> 0`: Both inputs are missing, so the AI stays safely at zero.
- `[0 1] -> 0`: Only one input is on, which is not enough to trigger a pass.
- `[1 0] -> 0`: Again, one single input fails to meet the strict threshold.
- `[1 1] -> 1`: Both inputs are active, successfully forcing the AI to fire a one.