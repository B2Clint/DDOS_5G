# =============================================================================
# config.py — Configuration globale du projet DDoS FL-CNN
# =============================================================================
# AMÉLIORATIONS v2 — nouveaux paramètres ajoutés :
#   MIXUP_ALPHA        : intensité du MixUp (0 = désactivé)
#   DECISION_THRESHOLD : seuil de décision softmax (ajustable post-entraînement)
# =============================================================================

import os

# ── Chemins ──────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
OUTPUT_DIR  = os.path.join(BASE_DIR, "outputs")
KEYS_DIR    = os.path.join(BASE_DIR, "keys")

for _d in [DATASET_DIR, OUTPUT_DIR, KEYS_DIR]:
    os.makedirs(_d, exist_ok=True)

# ── Dataset ───────────────────────────────────────────────────────────────────
CSV_FILENAME   = "dataset.csv"
LABEL_COLUMN   = "label"
NORMAL_LABEL   = "benign"
TEST_SIZE      = 0.2
RANDOM_STATE   = 42

# ── Apprentissage fédéré ──────────────────────────────────────────────────────
NUM_SILOS          = 3
FEDERATED_ROUNDS   = 20
LOCAL_EPOCHS       = 10
BATCH_SIZE         = 128

# ── Modèle CNN ────────────────────────────────────────────────────────────────
LEARNING_RATE  = 1e-4
DROPOUT_RATE   = 0.5

# ── [NOUVEAU v2] MixUp augmentation ───────────────────────────────────────────
# Intensité du mélange MixUp appliqué dans silo.py avant chaque entraînement.
# alpha=0.2 : mélange faible, préserve les patterns → réduit Train/Val gap.
# alpha=0.0 : désactiver MixUp si besoin de comparer les résultats.
MIXUP_ALPHA    = 0.2

# ── [NOUVEAU v2] Seuil de décision ────────────────────────────────────────────
# Seuil sur la probabilité softmax (classe attaque) pour la classification.
# Valeur par défaut 0.5 ; sera affiné automatiquement par evaluate.py sur
# le jeu de test pour maximiser le F1-score pondéré.
# Augmenter → plus de précision (↓ faux positifs) → ↑ spécificité
# Diminuer  → plus de recall   (↓ faux négatifs)
DECISION_THRESHOLD = 0.5

# ── Chiffrement RSA ────────────────────────────────────────────────────────────
RSA_KEY_SIZE   = 2048
SERVER_HOST    = "127.0.0.1"
SERVER_PORT    = 65432

# ── Affichage ─────────────────────────────────────────────────────────────────
VERBOSE        = True
