# =============================================================================
# server.py — Serveur fédéré central
# =============================================================================
"""
Rôle du serveur :
  • Génère la paire RSA du serveur + une paire par silo
  • Écoute sur SERVER_PORT (TCP) les poids chiffrés de chaque silo
  • Déchiffre, agrège (FedAvg), rechiffre pour chaque silo et renvoie
  • Journalise les métriques globales par round
"""

import socket
import threading
import time
import numpy as np
import config
import crypto_utils as cu
from model import fedavg_aggregate, build_cnn_model, set_weights


class FederatedServer:
    """Serveur d'apprentissage fédéré avec chiffrement RSA."""

    def __init__(self, n_features: int, n_classes: int, num_silos: int = None):
        self.n_features  = n_features
        self.n_classes   = n_classes
        self.num_silos   = num_silos or config.NUM_SILOS

        # Génération des clés RSA
        print("[Server] Génération des clés RSA …")
        self.server_priv, self.server_pub = cu.generate_rsa_keypair("server")
        self.silo_pub_keys = {}
        for i in range(self.num_silos):
            _, pub = cu.generate_rsa_keypair(f"silo_{i}")
            self.silo_pub_keys[i] = pub
            print(f"[Server] Clés générées pour Silo {i}")

        # Modèle global initial
        global_model = build_cnn_model(n_features, n_classes)
        self.global_weights = global_model.get_weights()

        # Stockage temporaire des poids reçus
        self._received_weights = {}
        self._lock  = threading.Lock()
        self._ready = threading.Event()

        # Historique des métriques pour les graphiques
        self.history = {
            "round": [],
            "avg_train_loss": [],
            "avg_train_acc":  [],
            "avg_val_loss":   [],
            "avg_val_acc":    []
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Démarrage du serveur TCP
    # ─────────────────────────────────────────────────────────────────────────

    def start_listener(self, fed_round: int):
        """
        Lance un thread TCP qui attend les poids de tous les silos pour
        le round `fed_round`.
        """
        self._received_weights  = {}
        self._metrics_received  = {}
        self._ready.clear()

        t = threading.Thread(
            target=self._listen_loop,
            args=(fed_round,),
            daemon=True
        )
        t.start()
        return t

    def _listen_loop(self, fed_round: int):
        """Boucle d'écoute TCP — reçoit les poids de chaque silo."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as srv:
            srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            srv.bind((config.SERVER_HOST, config.SERVER_PORT))
            srv.listen(self.num_silos)
            print(f"[Server] Round {fed_round+1} — En attente sur port {config.SERVER_PORT} …")

            connections = 0
            while connections < self.num_silos:
                conn, addr = srv.accept()
                t = threading.Thread(
                    target=self._handle_silo,
                    args=(conn, fed_round),
                    daemon=True
                )
                t.start()
                connections += 1

        # Attendre que tous les silos aient été traités
        while len(self._received_weights) < self.num_silos:
            time.sleep(0.05)

        self._aggregate_and_notify(fed_round)

    def _handle_silo(self, conn, fed_round: int):
        """Reçoit et déchiffre les données d'un silo."""
        try:
            with conn:
                # 1. Recevoir le paquet (silo_id + poids chiffrés + métriques)
                blob = cu.recv_blob(conn)
                import pickle
                packet = pickle.loads(blob)

                silo_id  = packet["silo_id"]
                payload  = packet["encrypted_weights"]
                metrics  = packet["metrics"]

                # 2. Déchiffrer avec la clé privée du serveur
                weights = cu.decrypt_weights(payload, self.server_priv)

                with self._lock:
                    self._received_weights[silo_id] = weights
                    self._metrics_received[silo_id] = metrics
                    print(f"[Server] Poids reçus du Silo {silo_id} "
                          f"(train_acc={metrics['train_acc']:.4f})")

                # 3. Attendre que tous les silos aient envoyé
                while len(self._received_weights) < self.num_silos:
                    time.sleep(0.05)

                # 4. Renvoyer les poids globaux agrégés chiffrés pour ce silo
                encrypted_global = cu.encrypt_weights(
                    self.global_weights,
                    self.silo_pub_keys[silo_id]
                )
                import struct
                cu.send_blob(conn, encrypted_global)

        except Exception as e:
            print(f"[Server] Erreur avec un silo : {e}")

    def _aggregate_and_notify(self, fed_round: int):
        """Agrège les poids (FedAvg) et met à jour le modèle global."""
        all_weights = list(self._received_weights.values())
        self.global_weights = fedavg_aggregate(all_weights)

        # Calcul des métriques moyennes
        metrics_list = list(self._metrics_received.values())
        avg_tl  = np.mean([m["train_loss"] for m in metrics_list])
        avg_ta  = np.mean([m["train_acc"]  for m in metrics_list])
        avg_vl  = np.mean([m.get("val_loss", 0) for m in metrics_list])
        avg_va  = np.mean([m.get("val_acc",  0) for m in metrics_list])

        self.history["round"].append(fed_round + 1)
        self.history["avg_train_loss"].append(avg_tl)
        self.history["avg_train_acc"].append(avg_ta)
        self.history["avg_val_loss"].append(avg_vl)
        self.history["avg_val_acc"].append(avg_va)

        print(f"\n[Server] ✔ Round {fed_round+1} agrégé — "
              f"Acc moy={avg_ta:.4f}  Loss moy={avg_tl:.4f}\n")
        self._ready.set()

    # ─────────────────────────────────────────────────────────────────────────
    # API publique
    # ─────────────────────────────────────────────────────────────────────────

    def wait_for_round(self, timeout: float = 300.0):
        """Bloque jusqu'à la fin d'un round ou timeout."""
        self._ready.wait(timeout=timeout)

    def get_global_weights(self):
        return self.global_weights

    def get_server_public_key(self):
        return self.server_pub

    def get_silo_private_key(self, silo_id: int):
        return cu.load_private_key(f"silo_{silo_id}")
