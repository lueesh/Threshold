import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
from ipywidgets import interact, FloatSlider, IntSlider
from scipy.interpolate import interp1d
from scipy.stats import norm

def interactive_roc_with_distributions(mu_neg=15, sigma_neg=4, mu_pos=25, sigma_pos=3, n=500, threshold=20):
    np.random.seed(42)

    # Simulate data
    neg = np.random.normal(mu_neg, sigma_neg, n)
    pos = np.random.normal(mu_pos, sigma_pos, n)
    scores = np.concatenate([neg, pos])
    labels = np.concatenate([np.zeros(n), np.ones(n)])

    # ROC and AUC
    fpr, tpr, _ = roc_curve(labels, scores)
    auc = roc_auc_score(labels, scores)

    # Remove duplicates for safe interpolation
    fpr_unique, indices = np.unique(fpr, return_index=True)
    tpr_unique = tpr[indices]

    # Smooth ROC curve
    interp_fpr = np.linspace(0, 1, 200)
    interp_func = interp1d(fpr_unique, tpr_unique, kind='linear', fill_value="extrapolate")
    interp_tpr = interp_func(interp_fpr)

    # Current threshold classification
    tp = np.sum(pos >= threshold)
    fn = np.sum(pos < threshold)
    fp = np.sum(neg >= threshold)
    tn = np.sum(neg < threshold)

    tpr_val = tp / (tp + fn) if (tp + fn) > 0 else 0
    fpr_val = fp / (fp + tn) if (fp + tn) > 0 else 0

    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # ROC curve
    axes[0].plot(interp_fpr, interp_tpr, label=f'ROC-Kurve (AUC = {auc:.2f})', color='navy')
    axes[0].plot([0, 1], [0, 1], 'k--', alpha=0.5)
    axes[0].scatter(fpr_val, tpr_val, color='red', label=f'Schwellenwert = {threshold:.1f}')
    axes[0].set_xlim(0, 1)
    axes[0].set_ylim(0, 1)
    axes[0].set_xlabel("False Positive Rate (1 - Spezifität)")
    axes[0].set_ylabel("True Positive Rate (Sensitivität)")
    axes[0].set_title("ROC-Kurve")
    axes[0].legend(loc='lower right')
    axes[0].grid(False)

    # Distributions
    x = np.linspace(0, 40, 1000)
    y_neg = norm.pdf(x, mu_neg, sigma_neg)
    y_pos = norm.pdf(x, mu_pos, sigma_pos)

    axes[1].plot(x, y_neg, label='Gesunde', color='#1f77b4')
    axes[1].plot(x, y_pos, label='Depressive', color='#ff7f0e', linestyle='--')
    axes[1].axvline(threshold, color='black', linestyle='--', label=f'Schwellenwert = {threshold:.1f}')
    axes[1].set_xlim(0, 40)
    axes[1].set_ylim(0, max(y_neg.max(), y_pos.max()) * 1.1)
    axes[1].set_xlabel("Testwert")
    axes[1].set_ylabel("Dichte")
    axes[1].set_title("Verteilung der Scores")
    axes[1].legend(loc='upper right')
    axes[1].grid(False)

    plt.tight_layout()
    plt.show()

    # Print metrics
    print(f"Sensitivität (TPR): {tpr_val:.2f}")
    print(f"1 - Spezifität (FPR): {fpr_val:.2f}")
    print(f"TP: {tp}, FN: {fn}, FP: {fp}, TN: {tn}")

# Interaktive Steuerung
interact(
    interactive_roc_with_distributions,
    mu_neg=FloatSlider(value=15, min=5, max=25, step=0.5, description="μ negativ"),
    sigma_neg=FloatSlider(value=4, min=1, max=10, step=0.5, description="σ negativ"),
    mu_pos=FloatSlider(value=25, min=5, max=35, step=0.5, description="μ positiv"),
    sigma_pos=FloatSlider(value=3, min=1, max=10, step=0.5, description="σ positiv"),
    n=IntSlider(value=500, min=100, max=2000, step=100, description="n"),
    threshold=FloatSlider(value=20, min=5, max=35, step=0.1, description="Schwellenwert")
)
