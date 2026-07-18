import math
import random
from dataclasses import dataclass

from execute_util import link, image, text
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
    perceptron()
    logistic_regression()
    linear_svm()
    lda()
    evaluation_metrics()
    text("Next time: Neural networks — stacking linear classifiers with non-linearities.")


############################################################

def welcome():
    text("## Linear Classification Techniques")
    text("**Presenters**: AJ Barcelona, A De Guzman, J Gambito")

############################################################

def what_is_classification():
    text("### Recap:Supervised vs. Unsupervised Learning")
    text("### Supervised Learning: Two broad categories")
    text("## Classification vs. Regression")
    text("**Regression** predicts a *continuous* value — e.g., predicting Alex Eala's score next match.")
    text("**Classification** predicts a *discrete class label* — e.g., spam or not-spam.")

    text("### Binary Classification")
    text("Two classes: output is 0 or 1 (Yes/No, Pass/Fail).")
    text("Examples: email spam filter, bank loan approval, medical risk flagging.")
    text("Common binary algorithms: Logistic Regression, k-NN, Decision Trees, SVM, Naïve Bayes.")

    text("### Multiclass Classification")
    text("More than two classes: news categories, book subjects, student streams.")
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
    text("If $\\text{score}(x) > 0$ → **Class A**.")
    text("If $\\text{score}(x) < 0$ → **Class B**.")
    text("The boundary where score = 0 is called the **Decision Boundary** (a hyperplane).")

    text("### Linear vs. Non-linear Classifiers")
    text("**Linear** classifiers draw a straight boundary — fast, interpretable, great for large datasets:")
    linear_examples: list[str] = [
        "Logistic Regression",
        "Linear SVM",
        "Perceptron",
        "SGD Classifier",
        "LDA",
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


############################################################
# 1. Perceptron

@dataclass
class PerceptronParams:
    weights: list[float]
    bias: float


def dot(a: list[float], b: list[float]) -> float:
    return sum(ai * bi for ai, bi in zip(a, b))


def perceptron():
    text("## 1. The Perceptron")
    link(rosenblatt_1958)
    text("The Perceptron is the ancestor of modern deep learning, created by Frank Rosenblatt in **1958**.")
    text("It is a virtual 'neuron': takes inputs, multiplies each by a weight, sums them, and makes a hard yes/no decision.")
    text("It learns by **trial and error** — when it makes a mistake, it nudges its parameters.")

    text("### The Analogy: The Strict Club Bouncer")
    text("Imagine a bouncer with a mental checklist:")
    text("1. Nice shoes? (value $x_1$)")
    text("2. Polite attitude? (value $x_2$)")
    text("The bouncer values politeness more than shoes, so $w_2 > w_1$.")
    text("If the total score clears a baseline (bias $b$), you get in.")
    text("If the bouncer wrongly rejects a polite person, they **nudge their mental rules** — that nudge is the learning rule.")

    text("### The Mathematical Concept")
    text("Compute score $z$:")
    text("$$z = w_1 x_1 + w_2 x_2 + b$$")
    text("Apply the Step Function:")
    text("- If $z \\geq 0$ → predict $+1$ (Pass)")
    text("- If $z < 0$ → predict $-1$ (Fail)")
    text("If prediction $\\hat{y}$ ≠ actual $y$, update:")
    text("$$w_{\\text{new}} = w_{\\text{old}} + \\eta(y - \\hat{y})x, \\qquad b_{\\text{new}} = b_{\\text{old}} + \\eta(y - \\hat{y})$$")
    text("where $\\eta$ (eta) is the **learning rate**.")

    text("### Worked Example: Student Pass/Fail")
    text("Dataset — predict Pass (+1) or Fail (−1) from hours studied ($x_1$):")
    dataset: list[tuple[str, float, int]] = [
        ("S1", 8.0, +1),
        ("S2", 2.0, -1),
    ]  # @inspect dataset

    text("Initialize: $w_1 = -0.5$, $b = 0.5$, $\\eta = 0.1$.")
    w1 = -0.5  # @inspect w1
    b  = 0.5   # @inspect b
    eta = 0.1  # @inspect eta

    text("**Step 1**: Test on S1 ($x_1 = 8.0$, $y = +1$).")
    x1_s1 = 8.0  # @inspect x1_s1
    z_s1 = w1 * x1_s1 + b  # @inspect z_s1
    # z_s1 = -3.5 → predict -1 (wrong — actual is +1)
    y_hat_s1 = 1 if z_s1 >= 0 else -1  # @inspect y_hat_s1

    text("Prediction is wrong → update weights.")
    y_s1 = +1
    update = eta * (y_s1 - y_hat_s1)  # @inspect update
    w1_new = w1 + update * x1_s1      # @inspect w1_new
    b_new  = b  + update               # @inspect b_new

    text("Verify with updated weights:")
    z_check = w1_new * x1_s1 + b_new  # @inspect z_check
    # z_check = 9.5 → predict +1 (correct!)

    text("The model flipped its weight from negative to positive, learning that 'studying more = pass'.")

    text("### Training on a linearly separable dataset")
    link(novikoff_1962)
    text("The perceptron is **guaranteed to converge** when data is linearly separable.")

    def perceptron_predict(params: PerceptronParams, x: list[float]) -> int:
        score = dot(params.weights, x) + params.bias  # @inspect score
        return 1 if score >= 0 else -1

    def perceptron_update(params: PerceptronParams, x: list[float], y: int) -> PerceptronParams:
        y_hat = perceptron_predict(params, x)  # @inspect y_hat
        if y_hat == y:
            return params
        new_weights = [w + eta * (y - y_hat) * xi for w, xi in zip(params.weights, x)]  # @inspect new_weights
        new_bias = params.bias + eta * (y - y_hat)                                        # @inspect new_bias
        return PerceptronParams(weights=new_weights, bias=new_bias)

    train_data: list[tuple[list[float], int]] = [
        ([1.0, 2.0], -1),
        ([2.0, 1.0], -1),
        ([1.5, 1.0], -1),
        ([7.0, 8.0], +1),
        ([8.0, 7.0], +1),
        ([6.0, 9.0], +1),
    ]  # @inspect train_data

    d = len(train_data[0][0])  # @inspect d
    params = PerceptronParams(weights=[0.0] * d, bias=0.0)  # @inspect params

    num_epochs = 20  # @inspect num_epochs
    converged_epoch = num_epochs
    for epoch in range(num_epochs):
        errors = 0
        for x, y in train_data:
            params = perceptron_update(params, x, y)
            if perceptron_predict(params, x) != y:
                errors += 1
        if errors == 0:
            converged_epoch = epoch + 1  # @inspect converged_epoch
            break

    final_weights = params.weights  # @inspect final_weights
    final_bias    = params.bias     # @inspect final_bias

    text("### The XOR problem: perceptron's hard limit")
    link(minsky_papert_1969)
    text("The perceptron only works when data is **linearly separable**.")
    text("XOR cannot be solved by any single straight line:")
    xor_data: list[tuple[list[float], int]] = [
        ([0.0, 0.0], -1),
        ([1.0, 1.0], -1),
        ([0.0, 1.0], +1),
        ([1.0, 0.0], +1),
    ]  # @inspect xor_data
    text("No hyperplane can separate these four points — this limitation motivated multi-layer networks.")


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
# 3. Linear SVM

def hinge_loss(score: float, y: int) -> float:
    return max(0.0, 1.0 - y * score)


def linear_svm():
    text("## 3. Linear Support Vector Machine (Linear SVM)")
    link(cortes_vapnik_1995)
    text("Both the Perceptron and Logistic Regression find *a* separating hyperplane.")
    text("An SVM finds the **widest possible road** between two classes — the maximum-margin hyperplane.")

    text("### The Analogy: Building a Grand Highway")
    text("You are building a highway between two towns: Red Town and Blue Town.")
    text("You don't want a narrow road hugging the outer houses — a child might wander onto the asphalt.")
    text("Instead, you pave as much empty land as possible between the towns.")
    text("The outermost houses sitting directly on the highway edge are the **Support Vectors**.")
    text("They define the limits of your highway's **Margin** (width).")

    text("### The Mathematical Concept")
    text("Minimize $\\frac{1}{2}\\|w\\|^2$ subject to every point lying on the correct side:")
    text("$$y_i(w^T x_i + b) \\geq 1 \\quad \\forall\\, i$$")
    text("A data point where $y_i(w^T x_i + b) = 1$ sits exactly on the margin edge — it is a **Support Vector**.")


############################################################
# 4. LDA

def lda():
    text("## 4. Linear Discriminant Analysis (LDA)")
    link(fisher_1936)
    text("LDA is a statistical classification technique that works by **projection**.")
    text("It collapses high-dimensional data onto a single line, choosing the angle that maximizes class separation.")
    image("images/pca_lda.webp", width=600)
    text("### Introduction")
    text("To deal with classification problems with 2 or more classes, most Machine Learning (ML) algorithms work the same way.")
    text("Usually, they apply some kind of transformation to the input data with the effect of **reducing the original input dimensions to a new (smaller) one**")
    text("The goal is to project the data to a new space.")
    text("Then, once projected, they try to classify the data points by finding a linear separation.")
    
    text("Suppose we want to classify the red and blue circles correctly. It is clear that with a simple linear model we will not get a good result.")
    image("images/linearly-inseperable-data.png", width=300)
    text("What if we could transform the data so that we could draw a line that separates the 2 classes?")
    image("images/feature_transformation.png", width=600)

    text("### The Mathematical Concept — Fisher's Criterion")
    text("Maximize the ratio of **between-class scatter** to **within-class scatter**:")
    text("$$J(w) = \\frac{w^T S_B w}{w^T S_W w}$$")
    text("- $S_B$: Between-Class Scatter (how far apart the group means are).")
    text("- $S_W$: Within-Class Scatter (how spread out each group is internally).")
    text("Optimal projection direction: $w = S_W^{-1}(m_1 - m_2)$.")


############################################################
# Evaluation Metrics

def evaluation_metrics():
    text("## Evaluation Metrics")
    text("**Accuracy alone can mislead.** If 99% of emails are safe, always predicting 'safe' gives 99% accuracy — but catches zero spam.")

    text("### Confusion Matrix")
    text("| | Predicted −1 (No) | Predicted +1 (Yes) |")
    text("|---|---|---|")
    text("| **Actual −1** | TN (True Negative) | FP (False Positive) |")
    text("| **Actual +1** | FN (False Negative) | TP (True Positive) |")

    y_true = [1, 0, 1, 1, 0, 0, 1, 0, 1, 0]  # @inspect y_true
    y_pred = [1, 0, 1, 0, 0, 1, 1, 0, 0, 0]  # @inspect y_pred

    def confusion_matrix(yt: list[int], yp: list[int]) -> dict[str, int]:
        tp = sum(1 for a, b in zip(yt, yp) if a == 1 and b == 1)
        tn = sum(1 for a, b in zip(yt, yp) if a == 0 and b == 0)
        fp = sum(1 for a, b in zip(yt, yp) if a == 0 and b == 1)
        fn = sum(1 for a, b in zip(yt, yp) if a == 1 and b == 0)
        return {"TP": tp, "TN": tn, "FP": fp, "FN": fn}

    cm = confusion_matrix(y_true, y_pred)  # @inspect cm

    text("### Precision, Recall, F1")
    text("**Precision**: of all predicted positives, how many were truly positive?")
    text("$$\\text{Precision} = \\frac{TP}{TP + FP}$$")
    text("**Recall**: of all true positives, how many did we catch?")
    text("$$\\text{Recall} = \\frac{TP}{TP + FN}$$")
    text("**F1**: harmonic mean of precision and recall.")
    text("$$F_1 = 2 \\cdot \\frac{\\text{Precision} \\times \\text{Recall}}{\\text{Precision} + \\text{Recall}}$$")

    prec = cm["TP"] / (cm["TP"] + cm["FP"]) if (cm["TP"] + cm["FP"]) > 0 else 0.0  # @inspect prec
    rec  = cm["TP"] / (cm["TP"] + cm["FN"]) if (cm["TP"] + cm["FN"]) > 0 else 0.0  # @inspect rec
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0               # @inspect f1
    acc  = (cm["TP"] + cm["TN"]) / len(y_true)                                       # @inspect acc

    text("### When to use each metric")
    text("- **Precision** matters when false positives are costly (spam filter — don't block real emails).")
    text("- **Recall** matters when false negatives are costly (disease screening — don't miss sick patients).")
    text("- **F1** balances both; useful with imbalanced class distributions.")
    text("- **Accuracy** works when classes are balanced and all errors equally bad.")

    text("### Algorithm Quick Reference — Covered in Depth")
    text("| Algorithm | Analogy | Output | Best For |")
    text("|---|---|---|---|")
    text("| Perceptron | Strict Club Bouncer | Hard Binary | Simple binary tasks |")
    text("| Logistic Regression | Weather Forecaster | Probability | When confidence matters |")
    text("| Linear SVM | Highway Construction | Margin Distance | Maximizing safety and generalization |")
    text("| Fisher's LDA | Flashlights & Shadows | 1D Projection | Statistical group separation |")

    text("### Other Linear Classifiers (Overview)")
    text("| Algorithm | Analogy | Key Idea |")
    text("|---|---|---|")
    text("| Ridge Classifier | Cautious Investor | L2 penalty keeps weights small and balanced |")
    text("| Passive-Aggressive | Driving Instructor | Silent when correct; aggressive correction on error |")
    text("| SGD Classifier | Blindfolded Hiker | Optimization framework — any loss, one sample at a time |")
    text("| Discriminant Functions | Video Game Scoring | One score formula per class; pick the highest |")


if __name__ == "__main__":
    main()
