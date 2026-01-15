# Manual Forward Propagation and Naive Optimization Strategies  
*A Step-by-Step Neural Network Example*

This document demonstrates **how a simple neural network works from input to loss**, and **why naive random optimization strategies fail**, using **small numbers that can be calculated by hand**.

---

## Network Architecture

x → Dense → ReLU → Dense → Softmax → Cross-Entropy


---

## 1. Input and Ground Truth

### Input vector

$$
x =
\begin{bmatrix}
1 \\
2
\end{bmatrix}
$$

### Ground-truth label

- Correct class: **Class 0**

One-hot encoding:

$$
y =
\begin{bmatrix}
1 & 0
\end{bmatrix}
$$

---

## 2. Dense Layer 1

### Weights and bias

$$
W^{(1)} =
\begin{bmatrix}
1 & -1 \\
0 & 1
\end{bmatrix}
\quad
b^{(1)} =
\begin{bmatrix}
0 & 0
\end{bmatrix}
$$

### Forward computation

$$
z^{(1)} = x^T W^{(1)} + b^{(1)}
$$

$$
=
\begin{bmatrix}
1 & 2
\end{bmatrix}
\begin{bmatrix}
1 & -1 \\
0 & 1
\end{bmatrix}
=
\begin{bmatrix}
1 & 1
\end{bmatrix}
$$

---

## 3. ReLU Activation

$$
a^{(1)} = \max(0, z^{(1)})
$$

$$
=
\begin{bmatrix}
1 & 1
\end{bmatrix}
$$

Since all values are positive, ReLU does not change them.

---

## 4. Dense Layer 2

### Weights and bias

$$
W^{(2)} =
\begin{bmatrix}
1 & 0 \\
0 & 1
\end{bmatrix}
\quad
b^{(2)} =
\begin{bmatrix}
0 & 0
\end{bmatrix}
$$

### Forward computation

$$
z^{(2)} = a^{(1)} W^{(2)}
=
\begin{bmatrix}
1 & 1
\end{bmatrix}
$$

---

## 5. Softmax Activation

### Softmax formula

$$
\text{softmax}(z_i) =
\frac{e^{z_i}}{\sum_j e^{z_j}}
$$

### Calculation

$$
e^1 = 2.718
$$

$$
\text{sum} = 2.718 + 2.718 = 5.436
$$

$$
\hat{y} =
\begin{bmatrix}
0.5 & 0.5
\end{bmatrix}
$$

---

## 6. Cross-Entropy Loss

### Formula

$$
L = -\log(\hat{y}_{\text{true class}})
$$

### Calculation

$$
L = -\log(0.5) = 0.693
$$

📌 The loss is relatively high, meaning the model is uncertain.

---

## 7. Naive Optimization Strategies

### Strategy 1: Randomly Reset Weights

Randomly replace the second dense layer weights:

$$
W^{(2)} =
\begin{bmatrix}
-1 & 2 \\
1 & 0
\end{bmatrix}
$$

#### Forward result

$$
z^{(2)} = [1, 1]
\begin{bmatrix}
-1 & 2 \\
1 & 0
\end{bmatrix}
=
\begin{bmatrix}
0 & 2
\end{bmatrix}
$$

Softmax:

$$
\hat{y} =
\begin{bmatrix}
0.12 & 0.88
\end{bmatrix}
$$

Loss:

$$
L = -\log(0.12) = 2.12
$$

❌ Loss increased significantly  
➡ Random guessing has **no direction**

---

### Strategy 2: Randomly Perturb Weights

Add small noise to the weights:

$$
W^{(2)} =
\begin{bmatrix}
1.1 & -0.1 \\
0.1 & 0.9
\end{bmatrix}
$$

#### Forward result

$$
z^{(2)} =
[1, 1]
\begin{bmatrix}
1.1 & -0.1 \\
0.1 & 0.9
\end{bmatrix}
=
\begin{bmatrix}
1.2 & 0.8
\end{bmatrix}
$$

Softmax:

$$
\hat{y} =
\begin{bmatrix}
0.6 & 0.4
\end{bmatrix}
$$

Loss:

$$
L = -\log(0.6) = 0.51
$$

✔ Loss decreased  
❗ But this improvement is **not guaranteed**

➡ This is a **random walk**, not learning.

---

## 8. Why Gradient Descent Is Necessary

Gradient Descent:
- Computes **how each weight affects the loss**
- Updates weights in the **direction that reduces loss**
- Guarantees improvement for small enough steps

📌 Learning requires **direction**, not luck.

---

## 9. Loss Comparison Summary

| Method | Loss |
|------|------|
| Initial model | 0.693 |
| Random reset | 2.12 |
| Random perturbation | 0.51 (unstable) |
| Gradient descent | Consistently decreases |

---

## 10. Key Teaching Message

> Neural networks do not learn by chance.  
> They learn by knowing **which parameters to increase or decrease**.

---

## Next Steps

- Manually derive gradients for this example
- Implement backpropagation step-by-step
- Compare gradient descent with random strategies in code

---

**Author:**  
Neural Network Teaching Example
