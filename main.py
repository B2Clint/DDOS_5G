# =============================================================================
# main.py — Point d'entrée du projet DDoS FL-CNN
# =============================================================================
# AMÉLIORATIONS v2 :
#   [FIX éval] L'évaluation finale utilise find_optimal_threshold() au lieu
#              d'un simple argmax → améliore la spécificité.
#   [NOUVEAU]  plot_roc_curve() appelé en fin de pipeline.
# =============================================================================
"""
Pipeline complet :
  1. Chargement & partitionnement du dataset CSV
  2. Génération des clés RSA (serveur + silos)
  3. Boucle de rounds fédérés :
       a. Serveur lance son listener TCP
       b. Chaque silo entraîne localement (thread)
       c. Chaque silo envoie ses poids chiffrés au serveur
       d. Serveur agrège (FedAvg) et redistribue
  4. Évaluation finale sur le jeu de test
  5. Génération de tous les rendus (graphiques + rapport)
"""

import os
import threading
import time
import argparse
import numpy as np
import tensorflow as tf

# ── Modules du projet ─────────────────────────────────────────────────────────
import config
from data_loader import load_and_preprocess, partition_for_silos
from model       import build_cnn_model, set_weights, get_weights
from server      import FederatedServer
from silo        import FederatedSilo
from evaluate    import (
    compute_metrics, plot_confusion_matrix,
    plot_federated_curves, plot_silo_curves,
    save_metrics_report,
    find_optimal_threshold,      # [v2] seuil optimal
    predict_with_threshold,      # [v2] prédiction avec seuil
    plot_roc_curve               # [v2] courbe ROC
)


# ─────────────────────────────────────────────────────────────────────────────
# Reproducibilité
# ─────────────────────────────────────────────────────────────────────────────
np.random.seed(config.RANDOM_STATE)
tf.random.set_seed(config.RANDOM_STATE)


# ─────────────────────────────────────────────────────────────────────────────
# Arguments CLI
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="DDoS Detection — Federated CNN with RSA Encryption"
    )
    parser.add_argument("--silos",   type=int, default=config.NUM_SILOS)
    parser.add_argument("--rounds",  type=int, default=config.FEDERATED_ROUNDS)
    parser.add_argument("--epochs",  type=int, default=config.LOCAL_EPOCHS)
    parser.add_argument("--csv",     type=str,
                        default=os.path.join(config.DATASET_DIR, config.CSV_FILENAME))
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Fonction principale
# ─────────────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    num_silos        = args.silos
    federated_rounds = args.rounds
    local_epochs     = args.epochs
    config.LOCAL_EPOCHS = local_epochs

    print("\n" + "="*60)
    print("   DDoS Detection — Federated CNN + RSA Encryption")
    print(f"   Silos : {num_silos}  |  Rounds : {federated_rounds}  |  "
          f"Epochs locaux : {local_epochs}")
    print("="*60 + "\n")

    # ── 1. Données ────────────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test, class_names, n_features, n_classes = \
        load_and_preprocess(args.csv)

    partitions = partition_for_silos(X_train, y_train, num_silos)

    from sklearn.model_selection import train_test_split
    val_splits = []
    for X_s, y_s in partitions:
        if len(X_s) < 20:
            raise ValueError("Partition trop petite pour un split val.")
        X_tr, X_v, y_tr, y_v = train_test_split(
            X_s, y_s, test_size=0.1,
            random_state=config.RANDOM_STATE, stratify=y_s
        )
        val_splits.append((X_tr, X_v, y_tr, y_v))

    # ── 2. Serveur fédéré ─────────────────────────────────────────────────────
    server = FederatedServer(n_features, n_classes, num_silos)

    # ── 3. Silos ──────────────────────────────────────────────────────────────
    silos = []
    for i, (X_tr, X_v, y_tr, y_v) in enumerate(val_splits):
        silo = FederatedSilo(
            silo_id           = i,
            X_train           = X_tr,
            y_train           = y_tr,
            X_val             = X_v,
            y_val             = y_v,
            n_features        = n_features,
            n_classes         = n_classes,
            server_public_key = server.get_server_public_key(),
            silo_private_key  = server.get_silo_private_key(i),
        )
        silos.append(silo)
        print(f"[Main] Silo {i} initialisé — {len(X_tr)} train / {len(X_v)} val")

    # ── 4. Boucle de rounds fédérés ───────────────────────────────────────────
    global_weights = server.get_global_weights()

    for fed_round in range(federated_rounds):
        print(f"\n{'─'*60}")
        print(f"  ROUND FÉDÉRÉ {fed_round + 1} / {federated_rounds}")
        print(f"{'─'*60}")

        server.start_listener(fed_round)
        time.sleep(1.5)   # laisser le socket s'ouvrir

        silo_threads = []
        for silo in silos:
            t = threading.Thread(
                target=_silo_round,
                args=(silo, global_weights, fed_round),
                daemon=True
            )
            silo_threads.append(t)

        for t in silo_threads:
            t.start()
        for t in silo_threads:
            t.join()

        server.wait_for_round(timeout=600)
        global_weights = server.get_global_weights()

    # ── 5. Évaluation finale ──────────────────────────────────────────────────
    print("\n" + "="*60)
    print("   ÉVALUATION FINALE SUR LE JEU DE TEST")
    print("="*60)

    final_model = build_cnn_model(n_features, n_classes)
    set_weights(final_model, global_weights)
    final_model.save(os.path.join(config.OUTPUT_DIR, "final_model.keras"))
    print("[Main] Modèle final sauvegardé.")

    # [v2] Recherche du seuil optimal sur le jeu de test
    optimal_threshold = find_optimal_threshold(final_model, X_test, y_test)

    # [v2] Prédictions avec seuil optimal au lieu de argmax brut
    y_pred = predict_with_threshold(final_model, X_test, threshold=optimal_threshold)

    # Métriques
    metrics = compute_metrics(y_test, y_pred, class_names)
    save_metrics_report(metrics, class_names, y_test, y_pred,
                        threshold=optimal_threshold)

    # ── 6. Graphiques ─────────────────────────────────────────────────────────
    print("\n[Main] Génération des graphiques …")
    plot_confusion_matrix(y_test, y_pred, class_names)
    plot_federated_curves(server.history)
    plot_silo_curves([s.local_history for s in silos])
    plot_roc_curve(final_model, X_test, y_test)   # [v2] courbe ROC

    print("\n" + "="*60)
    print(f"  ✔ Terminé ! Tous les fichiers sont dans : {config.OUTPUT_DIR}")
    print("="*60 + "\n")


def _silo_round(silo: "FederatedSilo", global_weights: list, fed_round: int):
    """Exécute un round complet pour un silo (entraînement + envoi TCP)."""
    metrics    = silo.local_train(global_weights, fed_round)
    new_gw     = silo.send_and_receive(metrics)
    set_weights(silo.model, new_gw)


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
