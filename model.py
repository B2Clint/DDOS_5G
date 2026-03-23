# =============================================================================
# model.py — Architecture CNN 1D pour la détection d'attaques DDoS
# =============================================================================
# AMÉLIORATIONS v2 :
#   [FIX 3] SpatialDropout1D ajouté après chaque bloc Conv1D
#           Le Dropout standard désactivait des neurones individuels.
#           SpatialDropout1D désactive des feature maps entières → régularise
#           mieux les données séquentielles et réduit la corrélation entre
#           filtres → écart Train/Val réduit sans dégrader les performances.
# =============================================================================
"""
Architecture :
    Input (n_features, 1)
    ├── Conv1D(64, k=3, ReLU) → BatchNorm → SpatialDropout1D(0.1) → MaxPool(2)
    ├── Conv1D(128, k=3, ReLU) → BatchNorm → SpatialDropout1D(0.15) → MaxPool(2)
    ├── Conv1D(256, k=3, ReLU) → BatchNorm → SpatialDropout1D(0.2) → GlobalAvgPool
    ├── Dense(512, ReLU, L2) → Dropout(0.5)
    ├── Dense(256, ReLU, L2) → Dropout(0.4)
    ├── Dense(128, ReLU, L2) → Dropout(0.2)
    └── Dense(n_classes, Softmax)
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
import config


def build_cnn_model(n_features: int, n_classes: int) -> keras.Model:
    """
    Construit et compile le modèle CNN 1D.

    Args:
        n_features : nombre de features d'entrée
        n_classes  : nombre de classes de sortie

    Returns:
        model compilé (non entraîné)
    """
    inputs = keras.Input(shape=(n_features, 1), name="input_features")

    # ── Bloc 1 ────────────────────────────────────────────────────────────────
    x = layers.Conv1D(64, kernel_size=3, padding="same", activation="relu",
                      name="conv1")(inputs)
    x = layers.BatchNormalization(name="bn1")(x)
    # [FIX 3] SpatialDropout1D : annule des feature maps entières
    # Taux faible (0.1) sur le 1er bloc pour préserver les features bas-niveau
    x = layers.SpatialDropout1D(0.1, name="sdrop1")(x)
    x = layers.MaxPooling1D(pool_size=2, name="pool1")(x)

    # ── Bloc 2 ────────────────────────────────────────────────────────────────
    x = layers.Conv1D(128, kernel_size=3, padding="same", activation="relu",
                      name="conv2")(x)
    x = layers.BatchNormalization(name="bn2")(x)
    # Taux moyen (0.15) sur le 2e bloc
    x = layers.SpatialDropout1D(0.15, name="sdrop2")(x)
    x = layers.MaxPooling1D(pool_size=2, name="pool2")(x)

    # ── Bloc 3 ────────────────────────────────────────────────────────────────
    x = layers.Conv1D(256, kernel_size=3, padding="same", activation="relu",
                      name="conv3")(x)
    x = layers.BatchNormalization(name="bn3")(x)
    # Taux plus élevé (0.2) sur le 3e bloc (features haut-niveau plus redondantes)
    x = layers.SpatialDropout1D(0.2, name="sdrop3")(x)
    x = layers.GlobalAveragePooling1D(name="gap")(x)

    # ── Couches Dense avec L2 + Dropout progressif ────────────────────────────
    x = layers.Dense(512, activation="relu",
                     kernel_regularizer=regularizers.l2(0.001),
                     name="dense1")(x)
    x = layers.Dropout(0.5, name="drop1")(x)

    x = layers.Dense(256, activation="relu",
                     kernel_regularizer=regularizers.l2(0.001),
                     name="dense2")(x)
    x = layers.Dropout(0.4, name="drop2")(x)

    x = layers.Dense(128, activation="relu",
                     kernel_regularizer=regularizers.l2(0.001),
                     name="dense3")(x)
    x = layers.Dropout(0.2, name="drop3")(x)

    # ── Sortie ────────────────────────────────────────────────────────────────
    loss_fn = "sparse_categorical_crossentropy"
    outputs = layers.Dense(n_classes, activation="softmax", name="output")(x)

    model = keras.Model(inputs, outputs, name="DDoS_CNN")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.LEARNING_RATE),
        loss=loss_fn,
        metrics=["accuracy"]   # métriques avancées calculées dans evaluate.py
    )

    return model


def get_weights(model: keras.Model) -> list:
    """Extrait les poids du modèle sous forme de liste de numpy arrays."""
    return model.get_weights()


def set_weights(model: keras.Model, weights: list):
    """Injecte une liste de numpy arrays dans le modèle."""
    model.set_weights(weights)


def fedavg_aggregate(weights_list: list) -> list:
    """
    Agrégation FedAvg : moyenne pondérée équitable des poids de tous les silos.

    Args:
        weights_list : liste de listes de numpy arrays (un par silo)

    Returns:
        Liste de numpy arrays agrégés
    """
    aggregated = []
    for layer_weights in zip(*weights_list):
        avg = np.mean(np.array(layer_weights), axis=0)
        aggregated.append(avg)
    return aggregated


def print_model_summary(n_features: int, n_classes: int):
    """Affiche le résumé du modèle CNN."""
    model = build_cnn_model(n_features, n_classes)
    model.summary()
    return model
