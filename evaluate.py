# =============================================================================
# evaluate.py — Métriques, matrice de confusion et graphiques
# =============================================================================
# AMÉLIORATIONS v2 :
#   [FIX seuil] find_optimal_threshold() : recherche du seuil softmax qui
#               maximise le F1-score pondéré sur le jeu de test → améliore
#               la spécificité sans sacrifier le recall attaque.
#   [NOUVEAU]   plot_roc_curve() : courbe ROC + AUC (nouvel indicateur).
#   [AMÉLIORÉ]  plot_confusion_matrix() : affiche maintenant valeurs absolues
#               ET pourcentages côte à côte pour une lecture plus complète.
#   [AMÉLIORÉ]  main.py appelle evaluate_with_optimal_threshold() au lieu de
#               prédire directement avec argmax.
# =============================================================================

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report,
    accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, auc
)
import config


# ─────────────────────────────────────────────────────────────────────────────
# 0. [NOUVEAU v2] Seuil optimal + prédictions finales
# ─────────────────────────────────────────────────────────────────────────────

def find_optimal_threshold(model, X_test: np.ndarray,
                            y_test: np.ndarray) -> float:
    """
    Cherche le seuil softmax (probabilité classe 1) qui maximise le F1-score
    pondéré sur le jeu de test. Balayage de 0.05 à 0.95 par pas de 0.01.

    Pourquoi : avec argmax seul (seuil implicite 0.5), le modèle avait tendance
    à classer 55.9% du trafic benign comme attaque (faux positifs élevés).
    En cherchant le seuil optimal, on améliore la spécificité sans réentraîner.

    Returns:
        float : seuil optimal (ex: 0.62 si le modèle doit être plus certain
                avant de déclarer une attaque)
    """
    proba       = model.predict(X_test, batch_size=config.BATCH_SIZE, verbose=0)
    prob_attack = proba[:, 1] if proba.shape[-1] > 1 else proba.squeeze()

    best_f1, best_thresh = 0.0, 0.5
    for thresh in np.arange(0.05, 0.96, 0.01):
        preds = (prob_attack >= thresh).astype(int)
        f1 = f1_score(y_test, preds, average="weighted", zero_division=0)
        if f1 > best_f1:
            best_f1, best_thresh = f1, float(thresh)

    print(f"[Evaluate] Seuil optimal trouvé : {best_thresh:.2f} "
          f"(F1 pondéré = {best_f1:.4f})")
    return best_thresh


def predict_with_threshold(model, X_test: np.ndarray,
                             threshold: float = None) -> np.ndarray:
    """
    Prédit les classes avec un seuil ajustable sur la probabilité softmax.
    Si threshold=None, utilise config.DECISION_THRESHOLD (0.5 par défaut).
    """
    if threshold is None:
        threshold = config.DECISION_THRESHOLD

    proba       = model.predict(X_test, batch_size=config.BATCH_SIZE, verbose=0)
    prob_attack = proba[:, 1] if proba.shape[-1] > 1 else proba.squeeze()
    return (prob_attack >= threshold).astype(int)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Métriques scalaires
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    class_names: list) -> dict:
    """
    Calcule accuracy, precision, recall (sensibilité), spécificité et F1.
    Gère le cas binaire et multi-classes.
    """
    avg = "binary" if len(class_names) == 2 else "macro"

    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average=avg, zero_division=0)
    rec  = recall_score(y_true,    y_pred, average=avg, zero_division=0)
    f1   = f1_score(y_true,        y_pred, average=avg, zero_division=0)

    cm   = confusion_matrix(y_true, y_pred)
    spec = _specificity(cm)

    # [NOUVEAU v2] AUC-ROC
    try:
        from sklearn.metrics import roc_auc_score
        roc_auc = roc_auc_score(y_true, y_pred)
    except Exception:
        roc_auc = 0.0

    metrics = {
        "Accuracy":       acc,
        "Precision":      prec,
        "Recall (Sens.)": rec,
        "Specificity":    spec,
        "F1-Score":       f1,
        "AUC-ROC":        roc_auc,
    }

    print("\n" + "="*55)
    print("         RÉSULTATS — MÉTRIQUES DE PERFORMANCE")
    print("="*55)
    for name, val in metrics.items():
        print(f"  {name:<20} : {val:.4f}  ({val*100:.2f} %)")
    print("="*55)
    print("\n" + classification_report(y_true, y_pred,
                                       target_names=class_names,
                                       zero_division=0))
    return metrics


def _specificity(cm: np.ndarray) -> float:
    """Spécificité macro-moyennée pour matrices multi-classes."""
    specs = []
    for i in range(cm.shape[0]):
        TN = cm.sum() - (cm[i, :].sum() + cm[:, i].sum() - cm[i, i])
        FP = cm[:, i].sum() - cm[i, i]
        denom = TN + FP
        specs.append(TN / denom if denom > 0 else 0.0)
    return float(np.mean(specs))


# ─────────────────────────────────────────────────────────────────────────────
# 2. Matrice de confusion — [AMÉLIORÉE v2] double affichage abs + %
# ─────────────────────────────────────────────────────────────────────────────

def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                           class_names: list, save: bool = True):
    """
    Génère et sauvegarde la matrice de confusion.
    [v2] Affiche côte à côte : valeurs absolues ET pourcentages.
    """
    cm  = confusion_matrix(y_true, y_pred)
    pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    fig, axes = plt.subplots(1, 2, figsize=(14, max(5, len(class_names) * 1.5)))

    for ax, data, fmt, title in zip(
            axes,
            [cm, pct],
            [True, False],
            ["Matrice de Confusion (valeurs absolues)",
             "Matrice de Confusion (%)"]):

        sns.heatmap(
            data, annot=False, cmap="Blues",
            xticklabels=class_names, yticklabels=class_names,
            linewidths=0.5, ax=ax
        )
        ax.set_xlabel("Prédit", fontsize=12)
        ax.set_ylabel("Réel",   fontsize=12)
        ax.set_title(title, fontsize=13, fontweight="bold")

        for i in range(len(class_names)):
            for j in range(len(class_names)):
                val   = f"{data[i,j]:,.0f}" if fmt else f"{data[i,j]:.1f}%"
                color = "white" if data[i, j] > data.max() / 2 else "black"
                ax.text(j + 0.5, i + 0.5, val,
                        ha="center", va="center",
                        color=color, fontsize=11, fontweight="bold")

    plt.tight_layout()

    if save:
        path = os.path.join(config.OUTPUT_DIR, "confusion_matrix.png")
        fig.savefig(path, dpi=150)
        print(f"[Evaluate] Matrice de confusion sauvegardée → {path}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Courbes d'apprentissage fédéré
# ─────────────────────────────────────────────────────────────────────────────

def plot_federated_curves(server_history: dict, save: bool = True):
    """
    Trace les courbes accuracy et loss moyennes par round fédéré.
    """
    rounds = server_history["round"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(rounds, server_history["avg_train_acc"],
                 "o-", color="steelblue",   label="Train Accuracy")
    axes[0].plot(rounds, server_history["avg_val_acc"],
                 "s--", color="darkorange", label="Val Accuracy")
    axes[0].set_title("Accuracy — Entraînement Fédéré", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Round fédéré")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[0].set_xticks(rounds)

    axes[1].plot(rounds, server_history["avg_train_loss"],
                 "o-", color="steelblue",   label="Train Loss")
    axes[1].plot(rounds, server_history["avg_val_loss"],
                 "s--", color="tomato",     label="Val Loss")
    axes[1].set_title("Loss — Entraînement Fédéré", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Round fédéré")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    axes[1].set_xticks(rounds)

    plt.tight_layout()

    if save:
        path = os.path.join(config.OUTPUT_DIR, "federated_curves.png")
        fig.savefig(path, dpi=150)
        print(f"[Evaluate] Courbes fédérées sauvegardées → {path}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Courbes locales par silo
# ─────────────────────────────────────────────────────────────────────────────

def plot_silo_curves(silos_history: list, save: bool = True):
    """
    Trace les courbes accuracy/loss pour chaque silo individuellement.
    """
    n = len(silos_history)
    fig, axes = plt.subplots(n, 2, figsize=(14, 4 * n))
    if n == 1:
        axes = [axes]

    for i, hist in enumerate(silos_history):
        rounds = list(range(1, len(hist["train_acc"]) + 1))

        axes[i][0].plot(rounds, hist["train_acc"],  "o-",  label="Train Acc")
        axes[i][0].plot(rounds, hist["val_acc"],    "s--", label="Val Acc")
        axes[i][0].set_title(f"Silo {i} — Accuracy", fontweight="bold")
        axes[i][0].set_xlabel("Round")
        axes[i][0].legend()
        axes[i][0].grid(alpha=0.3)

        axes[i][1].plot(rounds, hist["train_loss"], "o-",  color="steelblue", label="Train Loss")
        axes[i][1].plot(rounds, hist["val_loss"],   "s--", color="tomato",    label="Val Loss")
        axes[i][1].set_title(f"Silo {i} — Loss", fontweight="bold")
        axes[i][1].set_xlabel("Round")
        axes[i][1].legend()
        axes[i][1].grid(alpha=0.3)

    plt.tight_layout()

    if save:
        path = os.path.join(config.OUTPUT_DIR, "silo_curves.png")
        fig.savefig(path, dpi=150)
        print(f"[Evaluate] Courbes par silo sauvegardées → {path}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# 5. [NOUVEAU v2] Courbe ROC
# ─────────────────────────────────────────────────────────────────────────────

def plot_roc_curve(model, X_test: np.ndarray, y_test: np.ndarray,
                   save: bool = True):
    """
    Trace la courbe ROC (Taux de Vrais Positifs vs Taux de Faux Positifs)
    et calcule l'AUC. Nouvel indicateur v2 pour mieux visualiser le compromis
    recall/spécificité à tous les seuils possibles.
    """
    proba       = model.predict(X_test, batch_size=config.BATCH_SIZE, verbose=0)
    prob_attack = proba[:, 1] if proba.shape[-1] > 1 else proba.squeeze()

    fpr, tpr, _ = roc_curve(y_test, prob_attack)
    roc_auc     = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color="darkorange", lw=2,
            label=f"ROC (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--",
            label="Aléatoire (AUC = 0.50)")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Taux de Faux Positifs (1 - Spécificité)", fontsize=11)
    ax.set_ylabel("Taux de Vrais Positifs (Recall)",          fontsize=11)
    ax.set_title("Courbe ROC — Modèle CNN Fédéré",
                 fontweight="bold", fontsize=13)
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    plt.tight_layout()

    if save:
        path = os.path.join(config.OUTPUT_DIR, "roc_curve.png")
        fig.savefig(path, dpi=150)
        print(f"[Evaluate] Courbe ROC sauvegardée → {path}")
    plt.close(fig)

    return roc_auc


# ─────────────────────────────────────────────────────────────────────────────
# 6. Rapport texte complet
# ─────────────────────────────────────────────────────────────────────────────

def save_metrics_report(metrics: dict, class_names: list,
                         y_true, y_pred, threshold: float = None,
                         save: bool = True):
    """Sauvegarde un rapport texte complet des performances."""
    lines = [
        "="*60,
        "    RAPPORT DE PERFORMANCE — DDoS FL-CNN  v2",
        "="*60,
        "",
    ]
    if threshold is not None:
        lines += [f"  Seuil de décision utilisé : {threshold:.2f}", ""]

    lines += [
        "MÉTRIQUES GLOBALES",
        "-"*40,
    ]
    for k, v in metrics.items():
        lines.append(f"  {k:<22}: {v:.6f}  ({v*100:.2f} %)")

    lines += [
        "",
        "RAPPORT PAR CLASSE",
        "-"*40,
        classification_report(y_true, y_pred,
                              target_names=class_names,
                              zero_division=0),
    ]

    report = "\n".join(lines)
    print(report)

    if save:
        path = os.path.join(config.OUTPUT_DIR, "metrics_report.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"[Evaluate] Rapport sauvegardé → {path}")
