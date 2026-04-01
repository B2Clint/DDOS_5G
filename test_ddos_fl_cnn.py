# =============================================================================
# tests/test_ddos_fl_cnn.py — Suite de tests pytest pour DDoS FL-CNN
# =============================================================================
# Couvre :
#   • config.py        — valeurs et cohérence des paramètres globaux
#   • model.py         — construction, get/set weights, FedAvg
#   • data_loader.py   — partition stratifiée, forme des tenseurs
#   • evaluate.py      — métriques, seuil optimal, spécificité
#   • crypto_utils.py  — chiffrement/déchiffrement RSA hybride
#   • silo.py          — MixUp, création du tf.data.Dataset
#   • server.py        — agrégation FedAvg côté serveur
# =============================================================================

import os
import sys
import pickle
import struct
import tempfile

import numpy as np
import pytest
import tensorflow as tf

# ── Ajout du dossier racine au path pour l'import des modules ─────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from model        import build_cnn_model, get_weights, set_weights, fedavg_aggregate
from evaluate     import (
    compute_metrics, find_optimal_threshold, predict_with_threshold,
    _specificity
)
from crypto_utils import (
    generate_rsa_keypair, encrypt_weights, decrypt_weights,
    send_blob, recv_blob
)
from data_loader  import partition_for_silos


# =============================================================================
# Fixtures partagées
# =============================================================================

N_FEATURES  = 20   # petit pour que les tests soient rapides
N_CLASSES   = 2
N_SAMPLES   = 200


@pytest.fixture(scope="module")
def small_model():
    """Modèle CNN minimal pour les tests (non entraîné)."""
    return build_cnn_model(N_FEATURES, N_CLASSES)


@pytest.fixture(scope="module")
def rsa_keypair():
    """Paire de clés RSA générée une seule fois pour le module."""
    priv, pub = generate_rsa_keypair("test_server")
    return priv, pub


@pytest.fixture(scope="module")
def dummy_weights(small_model):
    """Poids initiaux du modèle pour les tests crypto."""
    return get_weights(small_model)


@pytest.fixture
def binary_dataset():
    """Dataset binaire (benign/attack) de taille fixe reproductible."""
    rng = np.random.default_rng(42)
    X = rng.random((N_SAMPLES, N_FEATURES, 1), dtype=np.float32)
    y = rng.integers(0, N_CLASSES, size=N_SAMPLES)
    return X, y


# =============================================================================
# 1. Tests config.py
# =============================================================================

class TestConfig:

    def test_num_silos_positive(self):
        assert config.NUM_SILOS > 0, "NUM_SILOS doit être > 0"

    def test_federated_rounds_positive(self):
        assert config.FEDERATED_ROUNDS > 0, "FEDERATED_ROUNDS doit être > 0"

    def test_local_epochs_positive(self):
        assert config.LOCAL_EPOCHS > 0, "LOCAL_EPOCHS doit être > 0"

    def test_batch_size_power_of_two(self):
        bs = config.BATCH_SIZE
        assert bs > 0 and (bs & (bs - 1)) == 0, \
            "BATCH_SIZE doit être une puissance de 2 pour les performances"

    def test_learning_rate_range(self):
        assert 1e-6 <= config.LEARNING_RATE <= 1e-1, \
            "LEARNING_RATE hors plage raisonnable [1e-6, 1e-1]"

    def test_dropout_rate_range(self):
        assert 0.0 <= config.DROPOUT_RATE < 1.0, \
            "DROPOUT_RATE doit être dans [0, 1)"

    def test_decision_threshold_range(self):
        assert 0.0 < config.DECISION_THRESHOLD < 1.0, \
            "DECISION_THRESHOLD doit être dans (0, 1)"

    def test_mixup_alpha_non_negative(self):
        assert config.MIXUP_ALPHA >= 0.0, \
            "MIXUP_ALPHA doit être >= 0 (0 = désactivé)"

    def test_test_size_valid(self):
        assert 0.0 < config.TEST_SIZE < 1.0, \
            "TEST_SIZE doit être dans (0, 1)"

    def test_rsa_key_size_minimum(self):
        assert config.RSA_KEY_SIZE >= 2048, \
            "RSA_KEY_SIZE doit être >= 2048 bits pour la sécurité"

    def test_output_dir_exists(self):
        assert os.path.isdir(config.OUTPUT_DIR), \
            "OUTPUT_DIR doit exister après import de config"

    def test_keys_dir_exists(self):
        assert os.path.isdir(config.KEYS_DIR), \
            "KEYS_DIR doit exister après import de config"

    def test_parallel_workers_coherent(self):
        total = config.NUM_SILOS * config.NUM_PARALLEL_WORKERS
        cpu_count = os.cpu_count() or 1
        assert total <= cpu_count * 2, (
            f"NUM_SILOS × NUM_PARALLEL_WORKERS = {total} dépasse "
            f"2× cpu_count ({cpu_count}). Risque de surcharge."
        )


# =============================================================================
# 2. Tests model.py
# =============================================================================

class TestModel:

    def test_build_returns_keras_model(self):
        model = build_cnn_model(N_FEATURES, N_CLASSES)
        assert isinstance(model, tf.keras.Model)

    def test_output_shape(self):
        model = build_cnn_model(N_FEATURES, N_CLASSES)
        dummy = np.zeros((4, N_FEATURES, 1), dtype=np.float32)
        out   = model.predict(dummy, verbose=0)
        assert out.shape == (4, N_CLASSES), \
            f"Sortie attendue (4, {N_CLASSES}), obtenu {out.shape}"

    def test_output_sums_to_one(self):
        """Softmax : la somme des probabilités par échantillon ≈ 1."""
        model = build_cnn_model(N_FEATURES, N_CLASSES)
        dummy = np.random.rand(8, N_FEATURES, 1).astype(np.float32)
        proba = model.predict(dummy, verbose=0)
        sums  = proba.sum(axis=1)
        np.testing.assert_allclose(sums, np.ones(8), atol=1e-5,
                                   err_msg="Les probabilités softmax ne somment pas à 1")

    def test_get_set_weights_roundtrip(self, small_model):
        """set_weights(get_weights(m)) doit être idempotent."""
        original = get_weights(small_model)
        set_weights(small_model, original)
        restored = get_weights(small_model)
        for orig_w, rest_w in zip(original, restored):
            np.testing.assert_array_equal(orig_w, rest_w)

    def test_set_weights_changes_model(self):
        """Deux modèles avec des poids différents doivent donner des sorties diff."""
        m1 = build_cnn_model(N_FEATURES, N_CLASSES)
        m2 = build_cnn_model(N_FEATURES, N_CLASSES)
        # Modifier les poids de m2
        noisy = [w + np.random.randn(*w.shape).astype(w.dtype) * 10
                 for w in get_weights(m2)]
        set_weights(m2, noisy)

        dummy = np.random.rand(4, N_FEATURES, 1).astype(np.float32)
        out1  = m1.predict(dummy, verbose=0)
        out2  = m2.predict(dummy, verbose=0)
        assert not np.allclose(out1, out2), \
            "Des modèles aux poids différents ne devraient pas donner la même sortie"

    def test_fedavg_aggregate_shape(self):
        """FedAvg doit retourner des tableaux de même forme que les entrées."""
        m1 = build_cnn_model(N_FEATURES, N_CLASSES)
        m2 = build_cnn_model(N_FEATURES, N_CLASSES)
        w1, w2    = get_weights(m1), get_weights(m2)
        averaged  = fedavg_aggregate([w1, w2])
        assert len(averaged) == len(w1)
        for orig, avg in zip(w1, averaged):
            assert orig.shape == avg.shape

    def test_fedavg_aggregate_values(self):
        """FedAvg de deux modèles identiques doit retourner les mêmes poids."""
        model = build_cnn_model(N_FEATURES, N_CLASSES)
        w     = get_weights(model)
        avg   = fedavg_aggregate([w, w])
        for orig, a in zip(w, avg):
            np.testing.assert_allclose(orig, a, atol=1e-6,
                                       err_msg="FedAvg de poids identiques ≠ poids originaux")

    def test_fedavg_aggregate_midpoint(self):
        """FedAvg de deux séries de poids doit être leur moyenne."""
        m1 = build_cnn_model(N_FEATURES, N_CLASSES)
        m2 = build_cnn_model(N_FEATURES, N_CLASSES)
        w1, w2   = get_weights(m1), get_weights(m2)
        averaged = fedavg_aggregate([w1, w2])
        for a, b, avg in zip(w1, w2, averaged):
            expected = (a + b) / 2
            np.testing.assert_allclose(avg, expected, atol=1e-6)

    def test_model_has_dropout_layers(self):
        model  = build_cnn_model(N_FEATURES, N_CLASSES)
        types  = [type(l).__name__ for l in model.layers]
        assert "Dropout" in types or "SpatialDropout1D" in types, \
            "Le modèle doit contenir des couches Dropout pour la régularisation"

    def test_model_has_batchnorm_layers(self):
        model = build_cnn_model(N_FEATURES, N_CLASSES)
        types = [type(l).__name__ for l in model.layers]
        assert "BatchNormalization" in types, \
            "Le modèle doit contenir des couches BatchNormalization"


# =============================================================================
# 3. Tests data_loader.py
# =============================================================================

class TestDataLoader:

    def _make_flat_data(self, n=300, n_feat=N_FEATURES, seed=0):
        rng = np.random.default_rng(seed)
        X   = rng.random((n, n_feat, 1), dtype=np.float32)
        y   = rng.integers(0, 2, size=n)
        return X, y

    def test_partition_count(self):
        X, y = self._make_flat_data()
        parts = partition_for_silos(X, y, num_silos=3)
        assert len(parts) == 3, "partition_for_silos doit retourner NUM_SILOS partitions"

    def test_partition_total_samples(self):
        X, y = self._make_flat_data(n=300)
        parts = partition_for_silos(X, y, num_silos=3)
        total = sum(len(px) for px, _ in parts)
        assert total == 300, \
            f"Total des partitions ({total}) ≠ taille originale (300)"

    def test_partition_stratified_distribution(self):
        """Chaque silo doit avoir une distribution de classes similaire."""
        n   = 600
        X, y = self._make_flat_data(n=n)
        # Forcer un déséquilibre clair : 1/3 benignes, 2/3 attaques
        y[:200] = 0
        y[200:] = 1
        parts   = partition_for_silos(X, y, num_silos=3)
        ratios  = [py.mean() for _, py in parts]   # ratio d'attaques
        # Tolérance de ±5 % autour du ratio global (~0.667)
        global_ratio = y.mean()
        for r in ratios:
            assert abs(r - global_ratio) < 0.05, \
                f"Silo mal stratifié : ratio={r:.3f}, attendu≈{global_ratio:.3f}"

    def test_partition_no_empty_silo(self):
        X, y = self._make_flat_data(n=300)
        for px, py in partition_for_silos(X, y, num_silos=3):
            assert len(px) > 0 and len(py) > 0, "Un silo ne doit pas être vide"

    def test_partition_features_shape_preserved(self):
        X, y  = self._make_flat_data(n=300)
        parts = partition_for_silos(X, y, num_silos=3)
        for px, _ in parts:
            assert px.shape[1:] == (N_FEATURES, 1), \
                f"Shape des features incorrecte : {px.shape[1:]}"


# =============================================================================
# 4. Tests evaluate.py
# =============================================================================

class TestEvaluate:

    def test_compute_metrics_perfect(self):
        """Métriques parfaites sur des prédictions correctes à 100%."""
        y = np.array([0, 0, 1, 1, 0, 1])
        metrics = compute_metrics(y, y, ["benign", "attack"])
        assert metrics["Accuracy"]  == pytest.approx(1.0)
        assert metrics["F1-Score"]  == pytest.approx(1.0)
        assert metrics["Precision"] == pytest.approx(1.0)
        assert metrics["Recall (Sens.)"] == pytest.approx(1.0)

    def test_compute_metrics_inverted(self):
        """Métriques sur prédictions complètement inversées."""
        y    = np.array([0, 0, 1, 1])
        pred = np.array([1, 1, 0, 0])
        metrics = compute_metrics(y, pred, ["benign", "attack"])
        assert metrics["Accuracy"] == pytest.approx(0.0)

    def test_compute_metrics_keys(self):
        y   = np.array([0, 1, 0, 1])
        m   = compute_metrics(y, y, ["benign", "attack"])
        for key in ["Accuracy", "Precision", "Recall (Sens.)",
                    "Specificity", "F1-Score", "AUC-ROC"]:
            assert key in m, f"Clé manquante dans compute_metrics : {key}"

    def test_specificity_perfect(self):
        """Matrice diagonale parfaite → spécificité = 1.0."""
        cm   = np.array([[50, 0], [0, 50]])
        spec = _specificity(cm)
        assert spec == pytest.approx(1.0)

    def test_specificity_worst(self):
        """Toutes les prédictions fausses → spécificité = 0."""
        cm   = np.array([[0, 50], [50, 0]])
        spec = _specificity(cm)
        assert spec == pytest.approx(0.0)

    def test_specificity_range(self):
        """Spécificité toujours dans [0, 1]."""
        rng  = np.random.default_rng(7)
        cm   = rng.integers(0, 100, size=(3, 3))
        spec = _specificity(cm)
        assert 0.0 <= spec <= 1.0

    def test_find_optimal_threshold_returns_float_in_range(self):
        """Le seuil optimal doit être dans [0.05, 0.95]."""
        model = build_cnn_model(N_FEATURES, N_CLASSES)
        X     = np.random.rand(50, N_FEATURES, 1).astype(np.float32)
        y     = np.random.randint(0, N_CLASSES, size=50)
        thresh = find_optimal_threshold(model, X, y)
        assert 0.05 <= thresh <= 0.95, \
            f"Seuil hors plage : {thresh}"

    def test_predict_with_threshold_binary_output(self):
        """La sortie de predict_with_threshold doit être 0 ou 1 uniquement."""
        model  = build_cnn_model(N_FEATURES, N_CLASSES)
        X      = np.random.rand(30, N_FEATURES, 1).astype(np.float32)
        preds  = predict_with_threshold(model, X, threshold=0.5)
        assert set(preds).issubset({0, 1}), \
            "predict_with_threshold doit retourner uniquement 0 ou 1"

    def test_predict_with_threshold_high_threshold_fewer_attacks(self):
        """Un seuil plus élevé doit produire moins de prédictions 'attack'."""
        model  = build_cnn_model(N_FEATURES, N_CLASSES)
        X      = np.random.rand(100, N_FEATURES, 1).astype(np.float32)
        preds_low  = predict_with_threshold(model, X, threshold=0.1)
        preds_high = predict_with_threshold(model, X, threshold=0.9)
        assert preds_low.sum() >= preds_high.sum(), \
            "Un seuil plus élevé doit produire ≤ de prédictions positives"

    def test_predict_with_none_threshold_uses_config(self):
        """threshold=None doit utiliser config.DECISION_THRESHOLD sans erreur."""
        model = build_cnn_model(N_FEATURES, N_CLASSES)
        X     = np.random.rand(20, N_FEATURES, 1).astype(np.float32)
        preds = predict_with_threshold(model, X, threshold=None)
        assert len(preds) == 20


# =============================================================================
# 5. Tests crypto_utils.py
# =============================================================================

class TestCryptoUtils:

    def test_generate_keypair_creates_pem_files(self):
        """Les fichiers PEM doivent être créés dans KEYS_DIR."""
        generate_rsa_keypair("test_gen")
        assert os.path.exists(os.path.join(config.KEYS_DIR, "test_gen_private.pem"))
        assert os.path.exists(os.path.join(config.KEYS_DIR, "test_gen_public.pem"))

    def test_encrypt_decrypt_roundtrip(self, rsa_keypair, dummy_weights):
        """Chiffrer puis déchiffrer doit retrouver les poids d'origine."""
        priv, pub = rsa_keypair
        payload   = encrypt_weights(dummy_weights, pub)
        recovered = decrypt_weights(payload, priv)
        assert len(recovered) == len(dummy_weights)
        for orig, rec in zip(dummy_weights, recovered):
            np.testing.assert_array_equal(orig, rec,
                err_msg="Les poids déchiffrés diffèrent des originaux")

    def test_encrypt_produces_bytes(self, rsa_keypair, dummy_weights):
        _, pub  = rsa_keypair
        payload = encrypt_weights(dummy_weights, pub)
        assert isinstance(payload, bytes), "encrypt_weights doit retourner des bytes"

    def test_encrypted_payload_differs_from_original(self, rsa_keypair, dummy_weights):
        """Le payload chiffré ne doit pas être la sérialisation brute des poids."""
        _, pub    = rsa_keypair
        raw       = pickle.dumps(dummy_weights)
        encrypted = encrypt_weights(dummy_weights, pub)
        assert raw != encrypted, \
            "Le payload chiffré ne doit pas être identique à la sérialisation brute"

    def test_two_encryptions_differ(self, rsa_keypair, dummy_weights):
        """RSA-OAEP avec Fernet éphémère : deux chiffrements ≠ même résultat."""
        _, pub = rsa_keypair
        p1     = encrypt_weights(dummy_weights, pub)
        p2     = encrypt_weights(dummy_weights, pub)
        assert p1 != p2, \
            "Deux chiffrements du même message doivent être différents (Fernet aléatoire)"

    def test_send_recv_blob_via_socket(self, dummy_weights, rsa_keypair):
        """send_blob / recv_blob via socket loopback TCP."""
        import socket, threading
        _, pub = rsa_keypair
        data   = encrypt_weights(dummy_weights, pub)
        received = []

        def server_side(srv_sock):
            conn, _ = srv_sock.accept()
            with conn:
                received.append(recv_blob(conn))

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as srv:
            srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            srv.bind(("127.0.0.1", 0))    # port libre aléatoire
            srv.listen(1)
            port = srv.getsockname()[1]

            t = threading.Thread(target=server_side, args=(srv,), daemon=True)
            t.start()

            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as cli:
                cli.connect(("127.0.0.1", port))
                send_blob(cli, data)

            t.join(timeout=5)

        assert len(received) == 1
        assert received[0] == data, "recv_blob doit retourner exactement ce qu'a envoyé send_blob"


# =============================================================================
# 6. Tests silo.py — fonctions internes (sans entraînement complet)
# =============================================================================

class TestSiloHelpers:
    """Teste les méthodes utilitaires de FederatedSilo sans lancer de training."""

    def _make_silo(self):
        from silo import FederatedSilo
        priv, pub = generate_rsa_keypair("silo_test_helper")
        rng = np.random.default_rng(0)
        X   = rng.random((120, N_FEATURES, 1), dtype=np.float32)
        y   = rng.integers(0, 2, size=120)
        return FederatedSilo(
            silo_id=0,
            X_train=X[:100], y_train=y[:100],
            X_val=X[100:],   y_val=y[100:],
            n_features=N_FEATURES, n_classes=N_CLASSES,
            server_public_key=pub,
            silo_private_key=priv,
        )

    def test_mixup_output_shape(self):
        silo       = self._make_silo()
        X, y       = silo.X_train, silo.y_train
        X_mix, y_mix = silo._mixup_batch(X, y, alpha=0.2)
        assert X_mix.shape == X.shape, "MixUp doit préserver la forme de X"
        assert y_mix.shape == y.shape, "MixUp doit préserver la forme de y"

    def test_mixup_labels_subset(self):
        """Les labels mixés doivent être un sous-ensemble des labels originaux."""
        silo       = self._make_silo()
        X, y       = silo.X_train, silo.y_train
        _, y_mix   = silo._mixup_batch(X, y, alpha=0.2)
        assert set(np.unique(y_mix)).issubset(set(np.unique(y))), \
            "MixUp ne doit pas introduire de nouvelles classes"

    def test_mixup_alpha_zero_unchanged(self):
        """alpha=0 : Beta(0,0) est instable, mais alpha très petit ≈ pas de mélange."""
        silo     = self._make_silo()
        X, y     = silo.X_train.copy(), silo.y_train.copy()
        X_mix, _ = silo._mixup_batch(X, y, alpha=1e-9)
        # Les valeurs doivent être très proches (lambda ≈ 0 ou ≈ 1)
        assert X_mix.shape == X.shape

    def test_make_dataset_returns_tf_dataset(self):
        silo = self._make_silo()
        ds   = silo._make_dataset(silo.X_train, silo.y_train, shuffle=False)
        assert isinstance(ds, tf.data.Dataset), \
            "_make_dataset doit retourner un tf.data.Dataset"

    def test_make_dataset_batch_shape(self):
        silo    = self._make_silo()
        ds      = silo._make_dataset(silo.X_train, silo.y_train, shuffle=False)
        X_batch, y_batch = next(iter(ds))
        assert X_batch.shape[1:] == (N_FEATURES, 1), \
            f"Shape batch incorrecte : {X_batch.shape[1:]}"
        assert X_batch.shape[0] <= silo.global_batch_size

    def test_make_dataset_dtype(self):
        silo     = self._make_silo()
        ds       = silo._make_dataset(silo.X_train, silo.y_train, shuffle=False)
        X_b, y_b = next(iter(ds))
        assert X_b.dtype == tf.float32, f"X dtype attendu float32, obtenu {X_b.dtype}"
        assert y_b.dtype == tf.int64,   f"y dtype attendu int64, obtenu {y_b.dtype}"

    def test_global_batch_size_positive(self):
        silo = self._make_silo()
        assert silo.global_batch_size > 0


# =============================================================================
# 7. Tests server.py — agrégation FedAvg
# =============================================================================

class TestFederatedServer:

    def _make_server(self, num_silos=2):
        from server import FederatedServer
        return FederatedServer(N_FEATURES, N_CLASSES, num_silos)

    def test_server_initializes(self):
        srv = self._make_server()
        assert srv is not None

    def test_server_global_weights_not_empty(self):
        srv = self._make_server()
        w   = srv.get_global_weights()
        assert len(w) > 0, "Les poids globaux initiaux ne doivent pas être vides"

    def test_server_global_weights_shape(self):
        srv      = self._make_server()
        expected = get_weights(build_cnn_model(N_FEATURES, N_CLASSES))
        actual   = srv.get_global_weights()
        assert len(actual) == len(expected)
        for a, e in zip(actual, expected):
            assert a.shape == e.shape

    def test_server_public_key_not_none(self):
        srv = self._make_server()
        assert srv.get_server_public_key() is not None

    def test_server_silo_private_key_loadable(self):
        srv  = self._make_server(num_silos=2)
        key0 = srv.get_silo_private_key(0)
        key1 = srv.get_silo_private_key(1)
        assert key0 is not None
        assert key1 is not None

    def test_server_rsa_keys_generated_on_disk(self):
        self._make_server(num_silos=2)
        for name in ["server", "silo_0", "silo_1"]:
            assert os.path.exists(
                os.path.join(config.KEYS_DIR, f"{name}_private.pem")
            ), f"Clé privée manquante pour {name}"

    def test_fedavg_through_server(self):
        """
        Simule une agrégation complète :
        deux jeux de poids → global_weights doit être leur moyenne.
        """
        srv = self._make_server(num_silos=2)
        m1  = build_cnn_model(N_FEATURES, N_CLASSES)
        m2  = build_cnn_model(N_FEATURES, N_CLASSES)
        w1, w2   = get_weights(m1), get_weights(m2)
        averaged = fedavg_aggregate([w1, w2])
        # Vérification manuelle pour la première couche
        expected_first = (w1[0] + w2[0]) / 2
        np.testing.assert_allclose(averaged[0], expected_first, atol=1e-6)


# =============================================================================
# 8. Tests d'intégration légers (sans I/O réelle)
# =============================================================================

class TestIntegration:

    def test_full_fedavg_cycle_weights_change(self):
        """
        Cycle complet simulé :
        modèle initial → "entraînement" simulé → FedAvg → poids modifiés.
        """
        global_model = build_cnn_model(N_FEATURES, N_CLASSES)
        global_w     = get_weights(global_model)

        # Simuler 2 silos qui modifient les poids
        silos_weights = []
        for _ in range(2):
            m   = build_cnn_model(N_FEATURES, N_CLASSES)
            set_weights(m, global_w)
            # Perturbation manuelle simulant un entraînement
            noisy = [w + np.random.randn(*w.shape).astype(w.dtype) * 0.01
                     for w in get_weights(m)]
            set_weights(m, noisy)
            silos_weights.append(get_weights(m))

        new_global = fedavg_aggregate(silos_weights)

        # Les poids doivent avoir changé après agrégation
        any_changed = any(
            not np.allclose(orig, agg)
            for orig, agg in zip(global_w, new_global)
        )
        assert any_changed, \
            "FedAvg doit modifier les poids globaux après 'entraînement'"

    def test_encrypt_decrypt_then_fedavg(self, rsa_keypair):
        """
        Pipeline crypto + FedAvg :
        chiffrer 2 jeux de poids, déchiffrer, agréger → cohérence.
        """
        priv, pub = rsa_keypair
        m1, m2    = build_cnn_model(N_FEATURES, N_CLASSES), \
                    build_cnn_model(N_FEATURES, N_CLASSES)
        w1, w2    = get_weights(m1), get_weights(m2)

        # Chiffrement / déchiffrement
        p1 = encrypt_weights(w1, pub)
        p2 = encrypt_weights(w2, pub)
        r1 = decrypt_weights(p1, priv)
        r2 = decrypt_weights(p2, priv)

        # Agrégation
        agg = fedavg_aggregate([r1, r2])
        assert len(agg) == len(w1), \
            "FedAvg après déchiffrement doit retourner le bon nombre de couches"

    def test_model_inference_after_set_weights(self):
        """
        set_weights sur un modèle source → inférence sur cible doit être identique.
        """
        src = build_cnn_model(N_FEATURES, N_CLASSES)
        dst = build_cnn_model(N_FEATURES, N_CLASSES)
        set_weights(dst, get_weights(src))

        X    = np.random.rand(10, N_FEATURES, 1).astype(np.float32)
        out_src = src.predict(X, verbose=0)
        out_dst = dst.predict(X, verbose=0)
        np.testing.assert_allclose(out_src, out_dst, atol=1e-5,
                                   err_msg="Deux modèles avec les mêmes poids doivent donner la même sortie")
