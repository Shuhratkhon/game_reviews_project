import os
import re
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc)

os.makedirs("charts", exist_ok=True)


# ---------- 1. data + preprocessing ----------
df = pd.read_csv("supercell_reviews_dataset.csv")
print("Raw reviews:", len(df))

df = df.dropna(subset=["review_text", "rating"])
df = df.drop_duplicates(subset="review_text")

# 3-star reviews are mixed, so we leave them out
# 1-2 stars = negative (0), 4-5 stars = positive (1)
df = df[df["rating"] != 3].copy()
df["label"] = (df["rating"] >= 4).astype(int)


def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", " ", text)        # links
    text = re.sub(r"[^a-z' ]", " ", text)       # keep only letters
    text = re.sub(r"\s+", " ", text)
    return text.strip()


df["clean_text"] = df["review_text"].apply(clean_text)

print("Reviews after cleaning:", len(df))
print("Negative:", (df["label"] == 0).sum(), " Positive:", (df["label"] == 1).sum())

X_train, X_test, y_train, y_test = train_test_split(
    df["clean_text"], df["label"], test_size=0.2, stratify=df["label"], random_state=42
)


# ---------- 2-3. models ----------
# note: no english stop words list, because it removes "not" and "no"
# and those words matter a lot for sentiment
models = {
    "Logistic Regression": (
        LogisticRegression(max_iter=1000),
        {"clf__C": [0.1, 1, 10, 100], "clf__class_weight": [None, "balanced"]}
    ),
    "Naive Bayes": (
        MultinomialNB(),
        {"clf__alpha": [0.01, 0.1, 0.5, 1.0]}
    ),
    "Linear SVM": (
        LinearSVC(),
        {"clf__C": [0.01, 0.1, 1, 10], "clf__class_weight": [None, "balanced"]}
    ),
}

tfidf_grid = {
    "tfidf__ngram_range": [(1, 1), (1, 2)],
    "tfidf__min_df": [1, 2],
    "tfidf__sublinear_tf": [True, False],
}


def get_scores(y_true, y_pred):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, average="macro"),
        "Recall": recall_score(y_true, y_pred, average="macro"),
        "F1-score": f1_score(y_true, y_pred, average="macro"),
    }


results = []
tuned_models = {}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, (clf, clf_grid) in models.items():
    print("\n" + "=" * 50)
    print(name)
    print("=" * 50)

    # default settings first
    pipe = Pipeline([("tfidf", TfidfVectorizer()), ("clf", clf)])
    start = time.time()
    pipe.fit(X_train, y_train)
    scores = get_scores(y_test, pipe.predict(X_test))
    results.append({"Model": name, "Version": "Default", **scores, "Time (s)": time.time() - start})
    print("Default F1:", round(scores["F1-score"], 4))

    # 4. hyperparameter tuning with 5-fold cross validation
    grid = GridSearchCV(pipe, {**tfidf_grid, **clf_grid}, cv=cv, scoring="f1_macro", n_jobs=-1)
    start = time.time()
    grid.fit(X_train, y_train)
    scores = get_scores(y_test, grid.predict(X_test))
    results.append({"Model": name, "Version": "Tuned", **scores, "Time (s)": time.time() - start})

    tuned_models[name] = grid.best_estimator_
    print("Best params:", grid.best_params_)
    print("Best CV F1:", round(grid.best_score_, 4))
    print("Tuned test F1:", round(scores["F1-score"], 4))


# ---------- 5. evaluation ----------
results = pd.DataFrame(results)
print("\nRESULTS (test set, macro average)\n")
print(results.round(4).to_string(index=False))
results.round(4).to_csv("model_results.csv", index=False)
print("\nSaved: model_results.csv")

tuned = results[results["Version"] == "Tuned"]
best_name = tuned.loc[tuned["F1-score"].idxmax(), "Model"]
best_model = tuned_models[best_name]
print("\nBest model:", best_name)


# ---------- 6. charts ----------
BG = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
BLUE = "#2a78d6"
RED = "#e34948"
model_colors = dict(zip(models.keys(), ["#2a78d6", "#eb6834", "#1baf7a"]))

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "font.family": "sans-serif", "font.sans-serif": ["Segoe UI", "DejaVu Sans"],
    "font.size": 10, "text.color": INK,
    "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK_2,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.axisbelow": True,
    "grid.color": "#e1e0d9", "grid.linewidth": 0.8,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelcolor": INK_2, "ytick.labelcolor": INK_2,
    "legend.frameon": False,
})


def titles(fig, title, subtitle):
    fig.text(0.015, 0.975, title, fontsize=15, fontweight="bold", va="top")
    fig.text(0.015, 0.925, subtitle, fontsize=10, color=INK_2, va="top")


# chart 1: default vs tuned
fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharey=True)
x = np.arange(len(models))
w = 0.36

for ax, metric in zip(axes, ["Accuracy", "F1-score"]):
    default_vals = results[results["Version"] == "Default"][metric].values
    tuned_vals = results[results["Version"] == "Tuned"][metric].values

    b1 = ax.bar(x - w / 2, default_vals, w, label="Default settings", color="#c3c2b7", edgecolor=BG, linewidth=2)
    b2 = ax.bar(x + w / 2, tuned_vals, w, label="After tuning", color=BLUE, edgecolor=BG, linewidth=2)
    ax.bar_label(b1, fmt="%.3f", padding=3, fontsize=9, color=MUTED)
    ax.bar_label(b2, fmt="%.3f", padding=3, fontsize=9, color=INK, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(models.keys())
    ax.set_ylim(0, 1.05)
    ax.set_title(metric, loc="left", fontsize=12, fontweight="bold")
    ax.grid(axis="x", visible=False)
    ax.tick_params(axis="x", length=0)

axes[0].set_ylabel("Score on test set")
fig.legend(*axes[0].get_legend_handles_labels(), loc="upper right", ncol=2, bbox_to_anchor=(0.99, 0.975))
titles(fig, "Hyperparameter tuning improves every model",
       "Test-set scores (20% of reviews), macro average over negative and positive classes.")
fig.tight_layout(rect=[0, 0, 1, 0.9])
fig.savefig("charts/model_comparison.png", dpi=300)
plt.close(fig)
print("Saved: charts/model_comparison.png")

# chart 2: confusion matrix of the best model
cm = confusion_matrix(y_test, best_model.predict(X_test))
cm_pct = cm / cm.sum(axis=1, keepdims=True) * 100
blues = LinearSegmentedColormap.from_list("blues", ["#eef4fc", "#86b6ef", "#2a78d6", "#104281"])

fig, ax = plt.subplots(figsize=(7, 6.5))
ax.imshow(cm_pct, cmap=blues, vmin=0, vmax=100)
names = ["Negative", "Positive"]

for i in range(2):
    for j in range(2):
        color = "white" if cm_pct[i, j] > 55 else INK
        ax.text(j, i - 0.07, str(cm[i, j]), ha="center", va="center", fontsize=22, fontweight="bold", color=color)
        ax.text(j, i + 0.15, f"{cm_pct[i, j]:.0f}% of real {names[i].lower()}", ha="center", va="center",
                fontsize=9, color=color)

ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(names)
ax.set_yticklabels(names)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.grid(False)
for side in ["left", "bottom"]:
    ax.spines[side].set_visible(False)
ax.tick_params(length=0)

titles(fig, f"Confusion matrix - {best_name}", "Tuned model on the test set. Diagonal = correct predictions.")
fig.tight_layout(rect=[0, 0, 1, 0.88])
fig.savefig("charts/confusion_matrix.png", dpi=300)
plt.close(fig)
print("Saved: charts/confusion_matrix.png")

# chart 3: ROC curves of tuned models
fig, ax = plt.subplots(figsize=(7.5, 7))
ax.plot([0, 1], [0, 1], color="#c3c2b7", linewidth=1)
ax.text(0.62, 0.55, "random guess", color=MUTED, fontsize=9, rotation=45)

for name, model in tuned_models.items():
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)[:, 1]
    else:
        y_score = model.decision_function(X_test)
    fpr, tpr, _ = roc_curve(y_test, y_score)
    ax.plot(fpr, tpr, linewidth=2, color=model_colors[name], label=f"{name}  (AUC {auc(fpr, tpr):.3f})")

ax.set_xlim(0, 1)
ax.set_ylim(0, 1.01)
ax.set_aspect("equal")
ax.set_xlabel("False positive rate")
ax.set_ylabel("True positive rate")
ax.legend(loc="lower right")
titles(fig, "ROC curves - tuned models", "Closer to the top-left corner is better. Logistic Regression and Linear SVM almost overlap.")
fig.tight_layout(rect=[0, 0, 1, 0.88])
fig.savefig("charts/roc_curves.png", dpi=300)
plt.close(fig)
print("Saved: charts/roc_curves.png")

# chart 4: words that push a review to negative / positive (logistic regression)
lr = tuned_models["Logistic Regression"]
words = lr.named_steps["tfidf"].get_feature_names_out()
coefs = lr.named_steps["clf"].coef_[0]
order = np.argsort(coefs)
top = np.concatenate([order[:15], order[-15:]])

fig, ax = plt.subplots(figsize=(10, 9.5))
ax.barh(words[top], coefs[top], height=0.7, color=[RED if c < 0 else BLUE for c in coefs[top]])
ax.axvline(0, color=MUTED, linewidth=1)
ax.set_xlabel("Logistic Regression weight")
ax.grid(axis="y", visible=False)
ax.tick_params(axis="y", length=0)
ax.legend(handles=[Patch(color=RED, label="pushes toward negative"), Patch(color=BLUE, label="pushes toward positive")],
          loc="lower right")
titles(fig, "Words the model relies on most", "The 15 strongest negative and positive words or word pairs.")
fig.tight_layout(rect=[0, 0, 1, 0.92])
fig.savefig("charts/top_words.png", dpi=300)
plt.close(fig)
print("Saved: charts/top_words.png")


# ---------- 8. live demo ----------
def predict_review(text):
    label = best_model.predict([clean_text(text)])[0]
    result = "POSITIVE" if label == 1 else "NEGATIVE"

    if hasattr(best_model, "predict_proba"):
        prob = best_model.predict_proba([clean_text(text)])[0][label]
        result += f" ({prob * 100:.1f}% sure)"
    return result


print("\n" + "=" * 50)
print("LIVE DEMO - model:", best_name)
print("=" * 50)

examples = [
    "Love this game, the new update is amazing and the graphics look great",
    "Too many bugs and the game keeps crashing after the update. Waste of money",
    "Pay to win, matchmaking is unfair and support never answers",
]
for e in examples:
    print(f"\n> {e}\n  {predict_review(e)}")

print("\nType your own review (press Enter on an empty line to quit)")
while True:
    try:
        text = input("\nReview: ").strip()
    except EOFError:
        break
    if not text:
        break
    print(" ", predict_review(text))
