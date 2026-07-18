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

    text("### Definition")
    text("The Perceptron is the simplest type of artificial neural network and one of the earliest algorithms designed for **supervised binary classification** (sorting data into exactly two classes).")
    text("It was introduced by Frank Rosenblatt in **1958**.")
    text("*\"The Perceptron looks at a set of numbers describing something, multiplies each number by how important it is, adds them up, and decides which of two classes the thing belongs to.\"*")

    text("### The Perceptron Formula")
    text("The entire decision-making process can be written as a single mathematical formula:")
    text("$$y = f(w_1 x_1 + w_2 x_2 + \\cdots + w_n x_n + b)$$")
    text("This is often written more compactly using summation notation:")
    text("$$y = f\\!\\left(\\sum_{i=1}^{n} w_i x_i + b\\right)$$")

    text("### Breakdown of the Formula")
    text(
        "| Symbol | Name |\n"
        "|---|---|\n"
        "| $x_1, x_2, \\ldots, x_n$ | Inputs / Features |\n"
        "| $w_1, w_2, \\ldots, w_n$ | Weights |\n"
        "| $b$ | Bias |\n"
        "| $\\sum (w_i \\cdot x_i)$ | Weighted Sum / Net Input |\n"
        "| $f(\\,)$ | Activation Function |\n"
        "| $y$ | Output |"
    )

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
    z1 = w1_and * and_data[0][0] + w2_and * and_data[0][1] + b_and  # @inspect z1
    y1 = 1 if z1 >= 0 else 0  # @inspect y1
    z2 = w1_and * and_data[1][0] + w2_and * and_data[1][1] + b_and  # @inspect z2
    y2 = 1 if z2 >= 0 else 0  # @inspect y2
    z3 = w1_and * and_data[2][0] + w2_and * and_data[2][1] + b_and  # @inspect z3
    y3 = 1 if z3 >= 0 else 0  # @inspect y3
    z4 = w1_and * and_data[3][0] + w2_and * and_data[3][1] + b_and  # @inspect z4
    y4 = 1 if z4 >= 0 else 0  # @inspect y4
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

    def perceptron_predict(params: PerceptronParams, x: list[float]) -> int:
        score = dot(params.weights, x) + params.bias  # @inspect score
        return 1 if score >= 0 else 0

    def perceptron_update(params: PerceptronParams, x: list[float], t: int) -> PerceptronParams:
        y_hat = perceptron_predict(params, x)  # @inspect y_hat
        if y_hat == t:
            return params
        new_weights = [w + eta * (t - y_hat) * xi for w, xi in zip(params.weights, x)]  # @inspect new_weights
        new_bias = params.bias + eta * (t - y_hat)                                        # @inspect new_bias
        return PerceptronParams(weights=new_weights, bias=new_bias)

    def run_perceptron_training(train_data, params, num_epochs):
        converged_epoch = num_epochs
        for epoch in range(num_epochs):
            errors = 0
            for x, t in train_data:
                params = perceptron_update(params, x, t)
                if perceptron_predict(params, x) != t:
                    errors += 1
            if errors == 0:
                converged_epoch = epoch + 1
                break
        return params, converged_epoch

    train_data: list[tuple[list[float], int]] = [
        ([0.0, 0.0], 0),
        ([0.0, 1.0], 0),
        ([1.0, 0.0], 0),
        ([1.0, 1.0], 1),
    ]  # @inspect train_data

    eta = 1.0  # @inspect eta
    params = PerceptronParams(weights=[0.0, 0.0], bias=0.0)  # @inspect params
    num_epochs = 20  # @inspect num_epochs

    params, converged_epoch = run_perceptron_training(train_data, params, num_epochs)  # @stepover @inspect converged_epoch

    final_weights = params.weights  # @inspect final_weights
    final_bias    = params.bias     # @inspect final_bias

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
