import matplotlib.pyplot as plt


COLORS = {
    "GLM":        "#6C8EAD",
    "RegGLM":     "#3A7CA5",   # proper regularised GLM (sklearn)
    "AGLM-Lin":   "#A8C0D6",   # AGLM without basis expansion
    "AGLM-Lvar":  "#1A3C6E",   # full AGLM with L-variable basis
    "GAM":        "#5C8A5C",
    "GBM":        "#C0392B",
    "CatBoost":   "#8E44AD",
    "DerivLasso": "#E67E22",
}

plt.rcParams.update({
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "grid.linestyle":    "--",
})
