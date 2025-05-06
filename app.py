import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Distribution parameters
mu_gesund, sigma_gesund = 15, 4
mu_depressiv, sigma_depressiv = 25, 3

x = np.linspace(0, 40, 1000)
y_gesund = norm.pdf(x, mu_gesund, sigma_gesund)
y_depressiv = norm.pdf(x, mu_depressiv, sigma_depressiv)

st.title("Klassifikation bei unterschiedlichem Schwellenwert")
schwelle = st.slider("Wähle den Schwellenwert", 10.0, 30.0, 20.0, 0.5)

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(x, y_gesund, label='Gesunde', color='#1f77b4')
ax.plot(x, y_depressiv, label='Depressive', color='#ff7f0e', linestyle='--')

ax.fill_between(x, y_gesund, where=(x >= schwelle), facecolor='none', edgecolor='blue', hatch='///', linewidth=0.0, label='Falsch Positiv (FP)')
ax.fill_between(x, y_depressiv, where=(x <= schwelle), facecolor='none', edgecolor='#e63602', hatch='\\\\', linewidth=0.0, label='Falsch Negativ (FN)')
ax.fill_between(x, y_gesund, where=(x < schwelle), color='#78daf4', alpha=0.5, label='Richtig Negativ (RN)')
ax.fill_between(x, y_depressiv, where=(x >= schwelle), color='orange', alpha=0.5, label='Richtig Positiv (RP)')

ax.axvline(schwelle, color='black', linestyle='--', linewidth=1.5, label=f'Schwellenwert = {schwelle:.1f}')
ax.set_xlim(0, 40)
ax.set_ylim(0, max(y_gesund.max(), y_depressiv.max()) * 1.1)
ax.set_xlabel("Testwert")
ax.set_ylabel("Häufigkeit")
ax.set_yticklabels([])
ax.legend(loc='upper left')

st.pyplot(fig)
