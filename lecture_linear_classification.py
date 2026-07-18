import math
import random
from dataclasses import dataclass

from execute_util import link, image, text, code_cell, iframe
from lecture_util import article_link, youtube_link
from references_linclass import (
    rosenblatt_1958, novikoff_1962, minsky_papert_1969,
    cox_1958, cortes_vapnik_1995,
    fisher_1936, hoerl_kennard_1970, crammer_2006, bishop_2006,
)


def main():
    welcome()
    what_is_classification()
    linear_classification_overview()
    lda()
    logistic_regression()
    perceptron()
    linear_svm()
    evaluation_metrics()


############################################################

def welcome():
    text("## Linear Classification Techniques")
    text("**Presenters**: AJ Barcelona, A De Guzman, J Gambito")

############################################################

def what_is_classification():
    text("## Recap: Supervised vs. Unsupervised Learning")
    image("images/ai_ml_dl.png", width=300)
    text("## Supervised Learning: Two broad categories")
    text("## Classification vs. Regression")
    text("**Regression** predicts a *continuous* value — e.g., predicting Alex Eala's score next match.")
    text("**Classification** predicts a *discrete class label* — e.g., spam or not-spam.")
    text("### Binary Classification")
    text("**Two classes:** output is 0 or 1 (Yes/No, Pass/Fail).")
    text("**Examples:** email spam filter, bank loan approval, medical risk flagging.")

    text("### Multiclass Classification")
    text("**More than two classes:** news categories, book subjects, student streams.")
    text("A problem can be binary *or* multiclass depending on how we define the classes.")
    text("Sentiment analysis is binary ('positive'/'negative') but becomes multiclass if we add 'neutral'.")

    text("### How classification works — five steps")
    steps: list[str] = []
    steps.append("Data Collection: gather labeled examples.")        # @inspect steps
    steps.append("Feature Extraction: identify attributes that distinguish classes.")  # @inspect steps
    steps.append("Model Training: learn patterns connecting features to labels.")      # @inspect steps
    steps.append("Model Evaluation: test on unseen data to measure accuracy.")        # @inspect steps
    steps.append("Prediction: classify new inputs using learned patterns.")            # @inspect steps


############################################################

def linear_classification_overview():
    text("## What is Linear Classification?")
    text("A linear classifier summarizes training data into a fixed set of **weights** $w$ and a **bias** $b$.")
    text("It computes a score as a dot product:")
    text("$$\\text{score}(x) = w \\cdot x + b$$")
    text("- $x$ = feature vector")
    text("- $w$ = weights (importance of each feature)")
    text("- $b$ = bias (baseline offset)")

    text("### Decision rule")
    text("If $\\text{score}(x) >= 75$ → **Class A**.")
    text("If $\\text{score}(x) <  75$ → **Class B**.")
    text("The boundary where score = 0 is called the **Decision Boundary** (a hyperplane).")

    text("### Linear vs. Non-linear Classifiers")
    text("**Linear** classifiers draw a straight boundary — fast, interpretable, great for large datasets:")
    linear_examples: list[str] = [
        "LDA",
        "Logistic Regression",
        "Perceptron",
        "Linear SVM",
        "SGD Classifier",
        "Ridge Classifier",
        "Passive-Aggressive Classifier",
        "Linear Discriminant Functions",
    ]  # @inspect linear_examples

    text("**Non-linear** classifiers draw curved boundaries — more powerful but slower and harder to interpret:")
    nonlinear_examples: list[str] = [
        "K-Nearest Neighbours",
        "Kernel SVM",
        "Decision Trees",
        "Random Forests",
        "Multi-layer Neural Networks",
    ]  # @inspect nonlinear_examples
    text("")

############################################################
# 1. LDA

def lda():
    text("## 1. Linear Discriminant Analysis (LDA)")
    link(fisher_1936)
    text("LDA is a statistical classification technique that works by **projection**.")
    text("It collapses high-dimensional data onto a single line, choosing the angle that maximizes class separation.")
    image("images/pca_lda.webp", width=600)
    image("images/lda_diagram_2.webp", width=400)
    text("### Introduction")
    text("To deal with classification problems with 2 or more classes, most Machine Learning (ML) algorithms work the same way.")
    text("Usually, they apply some kind of transformation to the input data with the effect of **reducing the original input dimensions to a new (smaller) one**")
    text("The goal is to project the data to a new space.")
    text("Then, once projected, they try to classify the data points by finding a linear separation.")

    text("Suppose we want to classify the red and blue circles correctly. It is clear that with a simple linear model we will not get a good result.")
    image("images/linearly-inseperable-data.png", width=300)
    text("What if we could transform the data so that we could draw a line that separates the 2 classes?")
    image("images/feature_transformation.png", width=600)

    text("### What is dimensionality and what is dimensionality reduction?")
    image("images/dimensionality_reduction.webp", width=600)
    image("images/dimensionality_reduction_2.webp", width=600)
    text("### Geometric Intuition")
    text("Example: Houses on the market with price, age, size, distance to public transport and number of rooms")
    iframe("lda-explorable/index.html", height=2000)
    text("### The Mathematical Concept — Fisher's Criterion")
    text("Maximize the ratio of **between-class scatter** to **within-class scatter**:")
    text("$$J(w) = \\frac{w^T S_B w}{w^T S_W w}$$")
    text("- $S_B$: Between-Class Scatter (how far apart the group means are).")
    text("- $S_W$: Within-Class Scatter (how spread out each group is internally).")
    text("Optimal projection direction: $w = S_W^{-1}(m_1 - m_2)$.")

    text("### Live Demo — LDA Projection")
    code_cell("""
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
X0 = np.random.randn(30, 2) + [2, 2]   # Class 0
X1 = np.random.randn(30, 2) + [6, 5]   # Class 1
X = np.vstack([X0, X1])

# Compute LDA direction: w = Sw^{-1} (m1 - m0)
m0, m1 = X0.mean(axis=0), X1.mean(axis=0)
Sw = (X0 - m0).T @ (X0 - m0) + (X1 - m1).T @ (X1 - m1)
w = np.linalg.inv(Sw) @ (m1 - m0)
w = w / np.linalg.norm(w)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

# Left: 2D scatter + LDA axis
ax1.scatter(*X0.T, color='steelblue', label='Class 0', alpha=0.7)
ax1.scatter(*X1.T, color='tomato',    label='Class 1', alpha=0.7)
cx, cy = X.mean(axis=0)
t = np.linspace(-5, 5, 100)
ax1.plot(cx + t*w[0], cy + t*w[1], 'k--', lw=1.5, label='LDA axis')
ax1.set_title('Original 2D Data + LDA Direction')
ax1.legend(); ax1.set_aspect('equal')

# Right: 1D projected distributions
proj0, proj1 = X0 @ w, X1 @ w
ax2.hist(proj0, bins=12, alpha=0.6, color='steelblue', label='Class 0')
ax2.hist(proj1, bins=12, alpha=0.6, color='tomato',    label='Class 1')
boundary = (proj0.mean() + proj1.mean()) / 2
ax2.axvline(boundary, color='black', linestyle='--', label=f'Boundary ≈ {boundary:.2f}')
ax2.set_title('1D Projected Distributions')
ax2.legend()

plt.tight_layout()
plt.show()
""")


############################################################
# 2. Logistic Regression

def sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-z))


def logistic_regression():
    text("## 2. Logistic Regression")
    link(cox_1958), link(bishop_2006)
    text("Despite its name, Logistic Regression is a **classifier**.")
    text("Instead of a blunt yes/no answer, it outputs a **probability** between 0% and 100%.")

    text("### The Analogy: The Weather Forecaster")
    text("A Perceptron-style robot says: 'It will rain. Yes or No.'")
    text("A Logistic Regression robot behaves like a meteorologist: 'There is an **82% chance** of rain today.'")
    text("You then apply a threshold — if chance > 50%, pack an umbrella.")

    text("### The Mathematical Concept — the Sigmoid Function")
    text("$$\\sigma(z) = \\frac{1}{1 + e^{-z}}$$")
    text("- $z$ is any real-valued score.")
    text("- Output is always in $(0, 1)$ — a valid probability.")
    text("- When $z = 0$: $\\sigma(0) = 0.5$ (50% confidence).")
    text("- Large positive $z$ → probability near 1.0.")
    text("- Large negative $z$ → probability near 0.0.")

    s_zero  = sigmoid(0.0)    # @inspect s_zero
    s_pos   = sigmoid(3.0)    # @inspect s_pos
    s_neg   = sigmoid(-3.0)   # @inspect s_neg


############################################################
# 3. Perceptron

@dataclass
class PerceptronParams:
    weights: list[float]
    bias: float


def dot(a: list[float], b: list[float]) -> float:
    return sum(ai * bi for ai, bi in zip(a, b))


def perceptron():
    text("## 3. The Perceptron")
    link(rosenblatt_1958)

    text("### Definition")
    text("The Perceptron is the simplest type of artificial neural network and one of the earliest algorithms designed for **supervised binary classification** (sorting data into exactly two classes).")
    text("It was introduced by Frank Rosenblatt in **1958**.")
    text("*\"The Perceptron looks at a set of numbers describing something, multiplies each number by how important it is, adds them up, and decides which of two classes the thing belongs to.\"*")
    image("images/perceptron.png", width=600)
    text("### The Perceptron Formula")
    text("The entire decision-making process can be written as a single mathematical formula:")
    text("$$y = f(w_1 x_1 + w_2 x_2 + \\cdots + w_n x_n + b)$$")
    text("This is often written more compactly using summation notation:")
    text("$$y = f\\!\\left(\\sum_{i=1}^{n} w_i x_i + b\\right)$$")

    text("### Breakdown of the Formula")
    text("| Symbol | Name |\n|---|---|\n| $x_1, x_2, \\ldots, x_n$ | Inputs / Features |\n| $w_1, w_2, \\ldots, w_n$ | Weights |\n| $b$ | Bias |\n| $\\sum (w_i \\cdot x_i)$ | Weighted Sum / Net Input |\n| $f(\\,)$ | Activation Function |\n| $y$ | Output |")

    text("### The Step (Activation) Function")
    text("The classic Perceptron activation function is the **step function**:")
    text("$$f(z) = \\begin{cases} 1 & \\text{if } z \\geq 0 \\\\ 0 & \\text{if } z < 0 \\end{cases}$$")

    text("### How the Perceptron Separates the Classes")
    text("**The Decision Boundary:** The equation $\\sum w_i x_i + b = 0$ is literally the equation of a **line** (2 features) or a **flat plane** (more features). This line is called the **decision boundary**.")
    text("**Separating the Class Fields:** Before computing, identify which columns are *features* (inputs) and which is the *target* (the class label you check against).")

    text("### Example 1: AND Logic Gate")
    text("Given weights (already trained): $w_1 = 1$, $w_2 = 1$, bias $b = -1.5$")
    and_data: list[tuple[int, int, int]] = [
        (0, 0, 0),
        (0, 1, 0),
        (1, 0, 0),
        (1, 1, 1),
    ]  # @inspect and_data
    image("images/perceptron_example1_1.png", width=600)
    text("**Step 1:** Write the formula for this problem:")
    text("$$z = (w_1 \\cdot x_1) + (w_2 \\cdot x_2) + b$$")
    w1_and = 1.0   # @inspect w1_and
    w2_and = 1.0   # @inspect w2_and
    b_and  = -1.5  # @inspect b_and
    text("$$z = (1)(x_1) + (1)(x_2) + (-1.5)$$")

    text("**Step 2:** Substitute each row's feature values and apply the step function.")
    image("images/perceptron_example1_2.png", width=600)

    text("**Step 3:** Separate the classes.")
    text("The decision boundary is $x_1 + x_2 - 1.5 = 0$, or rewritten as $x_2 = 1.5 - x_1$.")
    text("Only the point $(1, 1)$ lands above this line, correctly isolating the single Class 1 point from the three Class 0 points.")

    image("images/perceptron_example1_3.png", width=600)
    text("### Example 2: Training the Perceptron")
    link(novikoff_1962)
    text("The perceptron is **guaranteed to converge** when data is linearly separable.")
    text("Starting point: $w_1 = 0$, $w_2 = 0$, $b = 0$, learning rate $\\eta = 1$")
    text("$$z = (w_1 \\cdot x_1) + (w_2 \\cdot x_2) + b$$")
    text("**Learning rule:**")
    text("$$w_{i,\\text{new}} = w_{i,\\text{old}} + \\eta(t - y) \\cdot x_i$$")
    text("$$b_{\\text{new}} = b_{\\text{old}} + \\eta(t - y)$$")
    image("images/perceptron_example2.png", width=600)

    text("### Live Demo — Perceptron Training")
    code_cell("""
import numpy as np
import matplotlib.pyplot as plt

# AND gate dataset
X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)
y = np.array([0, 0, 0, 1])

# Train perceptron from scratch
w = np.array([0.0, 0.0])
b = 0.0
eta = 1.0
epoch_errors = []

for epoch in range(20):
    errors = 0
    for xi, ti in zip(X, y):
        yi = 1 if (w @ xi + b) >= 0 else 0
        if yi != ti:
            w += eta * (ti - yi) * xi
            b += eta * (ti - yi)
            errors += 1
    epoch_errors.append(errors)
    if errors == 0:
        break

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

# Left: learned decision boundary
colors = ['steelblue' if c == 0 else 'tomato' for c in y]
ax1.scatter(X[:,0], X[:,1], c=colors, s=150, zorder=3, edgecolors='black')
for xi, lbl in zip(X, ['(0,0)→0','(0,1)→0','(1,0)→0','(1,1)→1']):
    ax1.annotate(lbl, xi, textcoords='offset points', xytext=(8, 5), fontsize=9)
if abs(w[1]) > 1e-6:
    x1v = np.linspace(-0.5, 1.5, 100)
    ax1.plot(x1v, (-w[0]*x1v - b) / w[1], 'k--', lw=1.5,
             label=f'w=[{w[0]:.1f}, {w[1]:.1f}], b={b:.1f}')
ax1.set_xlim(-0.5, 1.5); ax1.set_ylim(-0.5, 1.5)
ax1.set_title('AND Gate — Learned Decision Boundary')
ax1.legend(fontsize=8); ax1.set_aspect('equal'); ax1.grid(True, alpha=0.3)

# Right: misclassifications per epoch
ax2.bar(range(1, len(epoch_errors)+1), epoch_errors, color='steelblue', alpha=0.7)
ax2.set_xlabel('Epoch'); ax2.set_ylabel('Misclassifications')
ax2.set_title(f'Training Errors per Epoch')
ax2.set_xticks(range(1, len(epoch_errors)+1))

plt.tight_layout()
plt.show()
print(f"Converged in {len(epoch_errors)} epoch(s)")
print(f"Final: w1={w[0]:.1f}, w2={w[1]:.1f}, bias={b:.1f}")
""")

    text("### The XOR Problem: Perceptron's Hard Limit")
    link(minsky_papert_1969)
    text("The perceptron only works when data is **linearly separable**.")
    text("XOR cannot be solved by any single straight line:")
    xor_data: list[tuple[list[float], int]] = [
        ([0.0, 0.0], 0),
        ([1.0, 1.0], 0),
        ([0.0, 1.0], 1),
        ([1.0, 0.0], 1),
    ]  # @inspect xor_data
    text("No hyperplane can separate these four points — this limitation motivated multi-layer networks.")


############################################################
# 4. Linear SVM

def hinge_loss(score: float, y: int) -> float:
    return max(0.0, 1.0 - y * score)


def linear_svm():
    text("## 4. Linear Support Vector Machine (Linear SVM)")
    link(cortes_vapnik_1995)

    text("### Definition")
    text("A Linear Support Vector Machine is a supervised machine learning algorithm used for **binary classification**.")
    text("It finds the optimal separating **hyperplane** (a straight line in 2D, a flat plane in 3D) that divides two classes while **maximizing the margin** — the empty buffer zone between the boundary and the nearest data points of each class.")
    text("*Linear SVM draws the single straight line that is as far away as possible from the nearest points of both classes, and those nearest points are the **support vectors** that pin the line in place.*")
    image("images/lsvm.png", width=600)
    text("### The Linear SVM Formula")
    text("Just like the Perceptron, Linear SVM uses a weighted sum plus bias — the **decision function**:")
    text("$$f(x) = w \\cdot x + b = w_1 x_1 + w_2 x_2 + \\cdots + w_n x_n + b$$")
    text("The predicted class uses the **sign function** instead of a step function:")
    text("$$y = \\text{sign}(w \\cdot x + b) = \\begin{cases} +1 & \\text{if } w \\cdot x + b \\geq 0 \\\\ -1 & \\text{if } w \\cdot x + b < 0 \\end{cases}$$")

    text("### Breakdown of the Formula")
    text("| Symbol | Name |\n|---|---|\n| $x = (x_1, \\ldots, x_n)$ | Feature vector |\n| $w = (w_1, \\ldots, w_n)$ | Weight vector |\n| $b$ | Bias / Intercept |\n| $w \\cdot x + b$ | Decision function |\n| $\\text{sign}(\\,)$ | Sign function |\n| $y_i$ | True class label |\n| $\\|w\\|$ | Norm (magnitude) of $w$ |\n| Support vectors | — |")

    text("### The Margin")
    text("SVM does not just draw one line — it conceptually draws **three parallel lines**: the decision boundary in the middle, and two margin boundaries on either side touching the nearest points of each class:")
    text("- **Positive margin boundary:** $w \\cdot x + b = +1$")
    text("- **Negative margin boundary:** $w \\cdot x + b = -1$")
    text("The distance between these two margin boundaries is the **margin**, calculated directly from the length of $w$:")
    text("$$\\text{Margin width} = \\frac{2}{\\|w\\|}$$")

    text("### How Linear SVM Separates the Classes")
    text("**The Decision Boundary and the Margin 'Street':** The equation $w \\cdot x + b = 0$ is the decision boundary, exactly as in the Perceptron. But SVM adds two more parallel lines ($w \\cdot x + b = 1$ and $w \\cdot x + b = -1$) that mark the edges of the widest possible empty street between the classes.")
    text("**Support Vectors — The Points That Actually Matter:** Once trained, most points are irrelevant to the final boundary. Only the points where $y_i(w \\cdot x_i + b) = 1$ — sitting exactly on the margin boundaries — are the **support vectors**.")
    text("**Separating the Class Fields:** Identify which columns are *features* ($x_1, x_2, \\ldots$) and which is the *target label* $y \\in \\{-1, +1\\}$.")

    text("### Example 1: Classifying Points with a Trained Linear SVM")
    text("Given trained weights: $w = (1,\\, 1)$, bias $b = -5$")
    svm_points: list[tuple[str, float, float, int]] = [
        ("P1", 0.0, 1.0, -1),
        ("P2", 1.0, 3.0, -1),
        ("P3", 4.0, 2.0, +1),
        ("P4", 5.0, 4.0, +1),
    ]  # @inspect svm_points

    text("**Step 1:** Write the decision function for this problem:")
    text("$$f(x) = (1)(x_1) + (1)(x_2) + (-5) = x_1 + x_2 - 5$$")
    w1_svm = 1.0   # @inspect w1_svm
    w2_svm = 1.0   # @inspect w2_svm
    b_svm  = -5.0  # @inspect b_svm
    image("images/svm_1_1.png", width=600)
    text("**Step 2:** Substitute each point's feature values and apply the sign function.")
    f_p1 = w1_svm * svm_points[0][1] + w2_svm * svm_points[0][2] + b_svm  # @inspect f_p1
    y_p1 = 1 if f_p1 >= 0 else -1  # @inspect y_p1
    f_p2 = w1_svm * svm_points[1][1] + w2_svm * svm_points[1][2] + b_svm  # @inspect f_p2
    y_p2 = 1 if f_p2 >= 0 else -1  # @inspect y_p2
    f_p3 = w1_svm * svm_points[2][1] + w2_svm * svm_points[2][2] + b_svm  # @inspect f_p3
    y_p3 = 1 if f_p3 >= 0 else -1  # @inspect y_p3
    f_p4 = w1_svm * svm_points[3][1] + w2_svm * svm_points[3][2] + b_svm  # @inspect f_p4
    y_p4 = 1 if f_p4 >= 0 else -1  # @inspect y_p4

    image("images/svm_1_23.png", width=600)
    text("**Step 3:** Separate the classes.")
    text("The decision boundary is $x_1 + x_2 - 5 = 0$, or rewritten as $x_2 = 5 - x_1$.")
    text("The **sign** of $f(x)$ determines the label: $f(x) < 0 \\Rightarrow -1$, $f(x) > 0 \\Rightarrow +1$.")

    text("### Example 2: Identifying the Support Vectors")
    text("**Step 1:** For every point compute the **functional margin** $y_i \\times f(x_i)$.")
    text("A value of exactly **1** means the point sits precisely on its margin boundary.")
    fm_p1 = svm_points[0][3] * f_p1  # @inspect fm_p1
    fm_p2 = svm_points[1][3] * f_p2  # @inspect fm_p2
    fm_p3 = svm_points[2][3] * f_p3  # @inspect fm_p3
    fm_p4 = svm_points[3][3] * f_p4  # @inspect fm_p4
    image("images/svm_2_1.png", width=600)
    text("**Step 2:** Interpret — P2 ($y \\times f = 1$) and P3 ($y \\times f = 1$) are the **support vectors**. P1 and P4 are far inside the safe zone.")

    text("### Example 3: Computing the Margin Width")
    text("**Step 1:** Compute the norm of the weight vector:")
    text("$$\\|w\\| = \\sqrt{w_1^2 + w_2^2} = \\sqrt{1^2 + 1^2} = \\sqrt{2} \\approx 1.414$$")
    norm_w = math.sqrt(w1_svm**2 + w2_svm**2)  # @inspect norm_w
    image("images/svm_3_1.png", width=600)

    text("**Step 2:** Apply the margin formula:")
    text("$$\\text{Margin width} = \\frac{2}{\\|w\\|} = \\frac{2}{\\sqrt{2}} \\approx 1.414$$")
    margin_width = 2.0 / norm_w  # @inspect margin_width

    text("**Step 3:** Write the two margin boundary equations using $b = -5$:")
    text("- **Positive margin boundary:** $x_1 + x_2 - 5 = +1 \\Rightarrow x_1 + x_2 = 6$")
    text("- **Negative margin boundary:** $x_1 + x_2 - 5 = -1 \\Rightarrow x_1 + x_2 = 4$")
    text("**Step 4:** Interpret — P2 lies on $x_1 + x_2 = 4$ and P3 lies on $x_1 + x_2 = 6$. Moving any other point leaves the boundary unchanged; moving a support vector shifts it.")
    text("### Live Demo — Linear SVM")
    code_cell("""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Data & parameters (edit these!) ─────────────────────────
points = [("P1", 0.0, 1.0, -1),
          ("P2", 1.0, 3.0, -1),
          ("P3", 4.0, 2.0, +1),
          ("P4", 5.0, 4.0, +1)]
w1, w2, b = 1.0, 1.0, -5.0
# ─────────────────────────────────────────────────────────────

norm_w = np.sqrt(w1**2 + w2**2)
margin = 2.0 / norm_w

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

# ── Left: scatter + decision boundary + margins ──────────────
clr = {-1: 'steelblue', +1: 'tomato'}
x1v = np.linspace(-1, 7, 300)

ax1.plot(x1v, (-w1*x1v - b    ) / w2, 'k-',  lw=2,   label='Decision boundary (f=0)')
ax1.plot(x1v, (-w1*x1v - b + 1) / w2, 'r--', lw=1.5, label='Positive margin (f=+1)')
ax1.plot(x1v, (-w1*x1v - b - 1) / w2, 'b--', lw=1.5, label='Negative margin (f=−1)')
ax1.fill_between(x1v,
                 (-w1*x1v - b - 1) / w2,
                 (-w1*x1v - b + 1) / w2,
                 alpha=0.08, color='gray')

for name, x1c, x2c, label in points:
    f  = w1*x1c + w2*x2c + b
    fm = label * f
    is_sv = abs(fm - 1.0) < 1e-6
    ax1.scatter(x1c, x2c, c=clr[label], s=180, zorder=4,
                edgecolors='gold', linewidths=3.5 if is_sv else 0)
    ax1.annotate(f'{name}  f={f:.1f}', (x1c, x2c),
                 textcoords='offset points', xytext=(8, 5), fontsize=9)

ax1.set_xlim(-1, 7); ax1.set_ylim(-1, 7)
ax1.set_title(f'SVM: w=[{w1},{w2}], b={b}  |  margin={margin:.3f}')
ax1.set_xlabel('x₁'); ax1.set_ylabel('x₂'); ax1.grid(True, alpha=0.3)
blue_p = mpatches.Patch(color='steelblue', label='Class −1')
red_p  = mpatches.Patch(color='tomato',    label='Class +1')
gold_p = mpatches.Patch(color='gold',      label='Support Vector (ring)')
ax1.legend(handles=[blue_p, red_p, gold_p], fontsize=8, loc='upper left')

# ── Right: functional-margin bar chart ───────────────────────
names, fms, bclrs = [], [], []
for name, x1c, x2c, label in points:
    f  = w1*x1c + w2*x2c + b
    fm = label * f
    names.append(name)
    fms.append(fm)
    bclrs.append('gold' if abs(fm - 1.0) < 1e-6 else 'steelblue')

bars = ax2.bar(names, fms, color=bclrs, edgecolor='black', alpha=0.85)
ax2.axhline(1.0, color='crimson', ls='--', lw=1.5, label='Margin threshold (=1)')
for bar, fm in zip(bars, fms):
    ax2.text(bar.get_x() + bar.get_width()/2, fm + 0.05, f'{fm:.1f}',
             ha='center', fontsize=11, fontweight='bold')
ax2.set_ylabel('y · f(x)'); ax2.set_title('Functional Margins  (gold = support vector)')
ax2.legend(fontsize=9); ax2.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()

# ── Console summary ───────────────────────────────────────────
print(f"||w|| = {norm_w:.4f}   |   Margin = 2/||w|| = {margin:.4f}")
print()
print(f"{'Point':>5} | {'f(x)':>6} | {'y':>3} | {'y*f(x)':>7} | Support Vector?")
print("-" * 46)
for name, x1c, x2c, label in points:
    f  = w1*x1c + w2*x2c + b
    fm = label * f
    sv = "YES" if abs(fm - 1.0) < 1e-6 else "no"
    print(f"  {name:>3}  | {f:>6.1f} | {label:>+3} | {fm:>7.1f} | {sv}")
""")


############################################################
# Evaluation Metrics

def evaluation_metrics():
    text("## Evaluation Metrics")
    text("**Accuracy alone can mislead.** If 99% of emails are safe, always predicting 'safe' gives 99% accuracy — but catches zero spam.")

    text("### Confusion Matrix")
    text("| | Predicted −1 (No) | Predicted +1 (Yes) |\n|---|---|---|\n| **Actual −1** | TN (True Negative) | FP (False Positive) |\n| **Actual +1** | FN (False Negative) | TP (True Positive) |")
    
    text("### Live Demo — Confusion Matrix & Metrics")
    code_cell("""
import numpy as np
import matplotlib.pyplot as plt

# ── Edit these to experiment ─────────────────────────────────
y_true = [1, 0, 1, 1, 0, 0, 1, 0, 1, 0]
y_pred = [1, 0, 1, 0, 0, 1, 1, 0, 0, 0]
# ─────────────────────────────────────────────────────────────

yt = np.array(y_true)
yp = np.array(y_pred)

TP = int(((yt == 1) & (yp == 1)).sum())
TN = int(((yt == 0) & (yp == 0)).sum())
FP = int(((yt == 0) & (yp == 1)).sum())
FN = int(((yt == 1) & (yp == 0)).sum())

prec = TP / (TP + FP) if (TP + FP) > 0 else 0.0
rec  = TP / (TP + FN) if (TP + FN) > 0 else 0.0
f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
acc  = (TP + TN) / len(yt)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

# ── Left: confusion matrix heatmap ───────────────────────────
cm_arr = np.array([[TN, FP], [FN, TP]])
ax1.imshow(cm_arr, cmap='Blues', vmin=0, vmax=max(cm_arr.max(), 1))
ax1.set_xticks([0, 1])
ax1.set_yticks([0, 1])
ax1.set_xticklabels(['Predicted 0 (Neg)', 'Predicted 1 (Pos)'], fontsize=10)
ax1.set_yticklabels(['Actual 0 (Neg)', 'Actual 1 (Pos)'], fontsize=10)
ax1.set_xlabel('Predicted Label', fontsize=11)
ax1.set_ylabel('Actual Label', fontsize=11)
cell_names = [['TN', 'FP'], ['FN', 'TP']]
thresh = max(cm_arr.max() / 2.0, 1)
for i in range(2):
    for j in range(2):
        txt_color = 'white' if cm_arr[i, j] >= thresh else 'black'
        ax1.text(j, i, f'{cell_names[i][j]}\\n{cm_arr[i, j]}',
                 ha='center', va='center', fontsize=14,
                 fontweight='bold', color=txt_color)
ax1.set_title('Confusion Matrix', fontsize=13, fontweight='bold')

# ── Right: metrics bar chart ──────────────────────────────────
bar_labels  = ['Precision', 'Recall', 'F1', 'Accuracy']
bar_values  = [prec, rec, f1, acc]
bar_colors  = ['steelblue', 'tomato', 'seagreen', 'mediumpurple']
bars = ax2.bar(bar_labels, bar_values, color=bar_colors, alpha=0.85, edgecolor='black')
for bar, val in zip(bars, bar_values):
    ax2.text(bar.get_x() + bar.get_width() / 2, val + 0.02, f'{val:.2f}',
             ha='center', fontsize=11, fontweight='bold')
ax2.set_ylim(0, 1.15)
ax2.set_title('Evaluation Metrics', fontsize=13, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()

print(f"TP={TP}  TN={TN}  FP={FP}  FN={FN}")
print(f"Precision : {prec:.4f}")
print(f"Recall    : {rec:.4f}")
print(f"F1        : {f1:.4f}")
print(f"Accuracy  : {acc:.4f}")
""")
    text("### Precision, Recall, F1")
    text("**Precision**: of all predicted positives, how many were truly positive?")
    text("$$\\text{Precision} = \\frac{TP}{TP + FP}$$")
    text("**Recall**: of all true positives, how many did we catch?")
    text("$$\\text{Recall} = \\frac{TP}{TP + FN}$$")
    text("**F1**: harmonic mean of precision and recall.")
    text("$$F_1 = 2 \\cdot \\frac{\\text{Precision} \\times \\text{Recall}}{\\text{Precision} + \\text{Recall}}$$")
    text("### When to use each metric")
    text("- **Precision** matters when false positives are costly (spam filter — don't block real emails).")
    text("- **Recall** matters when false negatives are costly (disease screening — don't miss sick patients).")
    text("- **F1** balances both; useful with imbalanced class distributions.")
    text("- **Accuracy** works when classes are balanced and all errors equally bad.")

    

    text("### Algorithm Quick Reference — Covered in Depth")
    text("| Algorithm | Analogy | Output | Best For |\n|---|---|---|---|\n| Perceptron | Strict Club Bouncer | Hard Binary | Simple binary tasks |\n| Logistic Regression | Weather Forecaster | Probability | When confidence matters |\n| Linear SVM | Highway Construction | Margin Distance | Maximizing safety and generalization |\n| Fisher's LDA | Flashlights & Shadows | 1D Projection | Statistical group separation |")

    text("### Other Linear Classifiers (Overview)")
    text("| Algorithm | Analogy | Key Idea |\n|---|---|---|\n| Ridge Classifier | Cautious Investor | L2 penalty keeps weights small and balanced |\n| Passive-Aggressive | Driving Instructor | Silent when correct; aggressive correction on error |\n| SGD Classifier | Blindfolded Hiker | Optimization framework — any loss, one sample at a time |\n| Discriminant Functions | Video Game Scoring | One score formula per class; pick the highest |")


if __name__ == "__main__":
    main()
