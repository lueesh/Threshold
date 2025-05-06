import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
from scipy.interpolate import interp1d
from scipy.stats import norm

# Streamlit UI
st.title("Interaktive ROC-Kurve mit Schwellenwert und Verteilungen")

# Sliders
mu_neg = st.slider("Mittelwert Gesunde", 5.0, 25.0, 15.0, step=0.5)
sigma_neg = st.slider("Standardabweichung Gesunde", 1.0, 10.0, 4.0, step=0.5)
mu_pos = st.slider("Mittelwert Depressive", 5.0, 35.0, 25.0, step=0.5)
sigma_pos = st.slider("Standardabweichung Depressive", 1.0, 10.0, 3.0, step=0.5)
threshold = st.slider("Schwellenwert", 5.0, 35.0, 20.0, step=0.1)

# Fixed sample size
n = 2000

# Simulate data
np.random.seed(42)
neg = np.random.normal(mu_neg, sigma_neg, n)
pos = np.random.normal(mu_pos, sigma_pos, n)
scores = np.concatenate([neg, pos])
labels = np.concatenate([np.zeros(n), np.ones(n)])

# ROC + AUC
fpr, tpr, _ = roc_curve(labels, scores)
auc = roc_auc_score(labels, scores)

# Smooth ROC curve
fpr_unique, indices = np.unique(fpr, return_index=True)
tpr_unique = tpr[indices]
interp_fpr = np.linspace(0, 1, 200)
interp_func = interp1d(fpr_unique, tpr_unique, kind='linear', fill_value="extrapolate")
interp_tpr = interp_func(interp_fpr)

# Threshold classification
tp = np.sum(pos >= threshold)
fn = np.sum(pos < threshold)
fp = np.sum(neg >= threshold)
tn = np.sum(neg < threshold)

tpr_val = tp / (tp + fn) if (tp + fn) > 0 else 0
fpr_val = fp / (fp + tn) if (fp + tn) > 0 else 0

# Plot
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# ROC-Kurve
axes[0].plot(interp_fpr, interp_tpr, label=f'AUC = {auc:.2f}', color='navy')
axes[0].plot([0, 1], [0, 1], 'k--', alpha=0.5)
axes[0].scatter(fpr_val, tpr_val, color='red', label=f'Schwellenwert = {threshold:.1f}')
axes[0].set_xlim(0, 1)
axes[0].set_ylim(0, 1)
axes[0].set_xlabel("FPR (1 - Spezifität)")
axes[0].set_ylabel("TPR (Sensitivität)")
axes[0].set_title("ROC-Kurve")
axes[0].legend(loc='lower right')
axes[0].grid(False)

# Verteilung
x = np.linspace(0, 40, 1000)
y_neg = norm.pdf(x, mu_neg, sigma_neg)
y_pos = norm.pdf(x, mu_pos, sigma_pos)
axes[1].plot(x, y_neg, label='Gesunde', color='#1f77b4')
axes[1].plot(x, y_pos, label='Depressive', color='#ff7f0e', linestyle='--')
axes[1].axvline(threshold, color='black', linestyle='--', label=f'Schwellenwert = {threshold:.1f}')
axes[1].set_xlim(0, 40)
axes[1].set_ylim(0, max(y_neg.max(), y_pos.max()) * 1.1)
axes[1].set_xlabel("Testwert")
axes[1].set_ylabel("Häufigkeit")
axes[1].set_yticklabels([])                # ← remove numbers
axes[1].set_title("Verteilung der Scores")
axes[1].legend(loc='upper right')
axes[1].grid(False)

st.pyplot(fig)

# Show metrics
st.markdown(f"**Sensitivität (TPR):** {tpr_val:.2f}")
st.markdown(f"**1 - Spezifität (FPR):** {fpr_val:.2f}")
st.markdown(f"**TP:** {tp}, **FN:** {fn}, **FP:** {fp}, **TN:** {tn}")
