# =============================================================================
# silo.py — Client fédéré (Silo) avec chiffrement RSA
# =============================================================================
# AMÉLIORATIONS v2 :
#   [FIX 3a] MixUp augmentation locale (alpha=0.2)
#            Mélange linéaire de paires d'échantillons → frontières de décision
#            plus douces → moins de surapprentissage → écart Train/Val réduit.
#   [FIX 3b] Gradient clipping (clipnorm=1.0)
#            Évite les explosions de gradient qui creusent l'écart Train/Val.
#   [FIX 3c] Seuil de décision optimal calculé par silo sur val set
#            Affiché dans les logs pour information (utilisé dans evaluate.py).
# =============================================================================
"""
Rôle d'un silo :
  1. Récupérer les poids globaux du serveur (ou les poids initiaux)
  2. Entraîner localement le CNN sur sa partition de données (+ MixUp)
  3. Chiffrer les poids mis à jour avec la clé publique du serveur
  4. Envoyer au serveur via TCP et attendre les poids agrégés
  5. Déchiffrer les poids reçus avec sa propre clé privée
"""

import pickle
import socket
import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score
import config
import crypto_utils as cu
from model import build_cnn_model, set_weights, get_weights


class FederatedSilo:
    """Client fédéré (silo) avec chiffrement RSA hybride."""

    def __init__(
        self,
        silo_id:          int,
        X_train:          np.ndarray,
        y_train:          np.ndarray,
        X_val:            np.ndarray,
        y_val:            np.ndarray,
        n_features:       int,
        n_classes:        int,
        server_public_key,
        silo_private_key,
    ):
        self.silo_id           = silo_id
        self.X_train           = X_train
        self.y_train           = y_train
        self.X_val             = X_val
        self.y_val             = y_val
        self.n_features        = n_features
        self.n_classes         = n_classes
        self.server_public_key = server_public_key
        self.silo_private_key  = silo_private_key

        # Construire le modèle local
        self.model = build_cnn_model(n_features, n_classes)

        # Historique local complet (tous rounds)
        self.local_history = {
            "train_loss": [], "train_acc": [],
            "val_loss":   [], "val_acc":   []
        }

    # ─────────────────────────────────────────────────────────────────────────
    # [FIX 3a] MixUp augmentation
    # ─────────────────────────────────────────────────────────────────────────

    def _mixup_batch(self, X: np.ndarray, y: np.ndarray,
                     alpha: float = 0.2) -> tuple:
        """
        MixUp : mélange linéaire de paires d'échantillons aléatoires.
        x̃ = λ·xᵢ + (1-λ)·xⱼ  |  ỹ = λ·yᵢ + (1-λ)·yⱼ

        Effet : le modèle apprend des frontières de décision plus douces
                → moins d'overfitting → écart Train/Val se resserre.
        alpha=0.2 : mélange faible, préserve les patterns de chaque classe.
        """
        lam  = np.random.beta(alpha, alpha)
        perm = np.random.permutation(len(X))
        X_mix = (lam * X + (1 - lam) * X[perm]).astype(np.float32)
        # Labels entiers → MixUp en float (sparse_categorical ne supporte pas
        # les labels mixtes, on garde le label dominant)
        y_mix = np.where(lam >= 0.5, y, y[perm])
        return X_mix, y_mix

    # ─────────────────────────────────────────────────────────────────────────
    # [FIX 3c] Seuil optimal
    # ─────────────────────────────────────────────────────────────────────────

    def _find_best_threshold(self) -> float:
        """
        Cherche le seuil softmax (classe 1) qui maximise le F1-score pondéré
        sur la validation. Balayage 0.10 → 0.90 par pas de 0.05.
        Affiché dans les logs uniquement (le seuil global est calculé dans
        evaluate.py sur le jeu de test complet).
        """
        proba = self.model.predict(self.X_val,
                                   batch_size=config.BATCH_SIZE, verbose=0)
        # proba shape : (N, n_classes) → probabilité classe 1
        prob_attack = proba[:, 1] if proba.shape[-1] > 1 else proba.squeeze()

        best_f1, best_thresh = 0.0, 0.5
        for thresh in np.arange(0.10, 0.91, 0.05):
            preds = (prob_attack >= thresh).astype(int)
            f1 = f1_score(self.y_val, preds, average="weighted", zero_division=0)
            if f1 > best_f1:
                best_f1, best_thresh = f1, float(thresh)
        return best_thresh

    # ─────────────────────────────────────────────────────────────────────────
    # Entraînement local
    # ─────────────────────────────────────────────────────────────────────────

    def local_train(self, global_weights: list, fed_round: int) -> dict:
        """
        Met à jour les poids du modèle local avec `global_weights`,
        entraîne sur les données du silo, retourne les métriques.
        """
        # 1. Injecter les poids globaux
        set_weights(self.model, global_weights)

        # 2. [FIX 3a] Appliquer MixUp sur les données d'entraînement
        X_aug, y_aug = self._mixup_batch(
            self.X_train, self.y_train,
            alpha=config.MIXUP_ALPHA
        )

        # 3. [FIX 3b] Recompiler avec gradient clipping
        self.model.compile(
            optimizer=keras.optimizers.Adam(
                learning_rate=config.LEARNING_RATE,
                clipnorm=1.0           # ← évite les explosions de gradient
            ),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )

        # 4. Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=2,
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.3,
                patience=1,
                min_lr=1e-6,
                verbose=1
            )
        ]

        # 5. Class weights (calculés localement sur la partition du silo)
        classes = np.unique(self.y_train)
        weights = compute_class_weight(
            class_weight="balanced",
            classes=classes,
            y=self.y_train
        )
        class_weight_dict = dict(zip(classes, weights))
        print(f"[Silo {self.silo_id}] Class weights : {class_weight_dict}")

        # 6. Entraînement sur données augmentées MixUp
        hist = self.model.fit(
            X_aug, y_aug,
            epochs=config.LOCAL_EPOCHS,
            batch_size=config.BATCH_SIZE,
            validation_data=(self.X_val, self.y_val),
            class_weight=class_weight_dict,
            callbacks=callbacks,
            verbose=1 if config.VERBOSE else 0
        )

        # 7. Métriques du dernier epoch
        metrics = {
            "train_loss": float(hist.history["loss"][-1]),
            "train_acc":  float(hist.history["accuracy"][-1]),
            "val_loss":   float(hist.history["val_loss"][-1]),
            "val_acc":    float(hist.history["val_accuracy"][-1]),
        }

        # Enregistrement dans l'historique local
        for k in ["train_loss", "train_acc", "val_loss", "val_acc"]:
            self.local_history[k].append(metrics[k])

        # 8. [FIX 3c] Calculer et logger le seuil optimal local
        best_thresh = self._find_best_threshold()
        print(
            f"[Silo {self.silo_id}] Round {fed_round+1} — "
            f"Loss={metrics['train_loss']:.4f}  Acc={metrics['train_acc']:.4f}  "
            f"Val_Acc={metrics['val_acc']:.4f}  "
            f"Seuil optimal local={best_thresh:.2f}"
        )

        return metrics

    # ─────────────────────────────────────────────────────────────────────────
    # Communication chiffrée avec le serveur
    # ─────────────────────────────────────────────────────────────────────────

    def send_and_receive(self, metrics: dict) -> list:
        """
        Envoie les poids locaux chiffrés au serveur et reçoit les poids agrégés.

        Returns:
            Nouvelle liste de poids globaux agrégés.
        """
        weights = get_weights(self.model)

        # 1. Chiffrer les poids avec la clé publique du serveur
        encrypted_weights = cu.encrypt_weights(weights, self.server_public_key)

        # 2. Préparer le paquet complet
        packet = pickle.dumps({
            "silo_id":           self.silo_id,
            "encrypted_weights": encrypted_weights,
            "metrics":           metrics
        })

        # 3. Connexion TCP au serveur
        port    = config.SERVER_PORT
        retries = 10
        for attempt in range(retries):
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.settimeout(600)
                    s.connect((config.SERVER_HOST, port))
                    cu.send_blob(s, packet)
                    print(f"[Silo {self.silo_id}] Poids envoyés au serveur.")

                    # 4. Attendre les poids agrégés chiffrés
                    encrypted_global = cu.recv_blob(s)
                break
            except ConnectionRefusedError:
                if attempt < retries - 1:
                    time.sleep(0.3)
                else:
                    raise

        # 5. Déchiffrer avec la clé privée du silo
        new_global_weights = cu.decrypt_weights(encrypted_global,
                                                 self.silo_private_key)
        print(f"[Silo {self.silo_id}] Poids globaux reçus et déchiffrés.")

        return new_global_weights

    # ─────────────────────────────────────────────────────────────────────────
    # Prédiction finale (après tous les rounds)
    # ─────────────────────────────────────────────────────────────────────────

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Retourne les classes prédites pour X."""
        probs = self.model.predict(X, verbose=0)
        return np.argmax(probs, axis=1)
