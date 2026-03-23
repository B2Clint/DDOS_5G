# =============================================================================
# data_loader.py — Chargement, prétraitement et partitionnement du CSV
# =============================================================================
# AMÉLIORATIONS v2 :
#   [FIX 1] partition_for_silos → partitionnement STRATIFIÉ
#           Cause du Silo 0 plafonné à 68% : le découpage np.array_split
#           séquentiel concentrait par hasard les benignes dans certains silos.
#           Chaque silo reçoit maintenant exactement la même distribution.
#   [FIX 2] SMOTE + RandomUnderSampler combinés (au lieu de SMOTE seul)
#           SMOTE pur sur-génère des benignes synthétiques peu réalistes →
#           le modèle les confond avec des attaques → faux positifs élevés.
#           Combinaison : ratio benign/attaque → ~1:2 au lieu de 1:8.
# =============================================================================

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
import config


def load_and_preprocess(csv_path: str = 'dataset.csv'):
    """
    Charge le CSV, encode les labels, normalise les features.

    Retourne :
        X_train, X_test  : np.ndarray float32  (n_samples, n_features, 1)
        y_train, y_test  : np.ndarray int       (classes encodées)
        class_names      : list[str]            (noms des classes)
        n_features       : int
        n_classes        : int
    """
    if csv_path is None:
        csv_path = os.path.join(config.DATASET_DIR, config.CSV_FILENAME)

    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Dataset introuvable : {csv_path}\n"
            f"Place ton fichier CSV dans le dossier dataset/ sous le nom '{config.CSV_FILENAME}'."
        )

    print(f"[DataLoader] Chargement de {csv_path} …")
    df = pd.read_csv(csv_path, low_memory=False)
    print(f"[DataLoader] Dimensions brutes : {df.shape}")

    # ── Nettoyage ─────────────────────────────────────────────────────────────
    drop_cols = [c for c in df.columns if c.lower() in
                 {"flow id", "source ip", "destination ip", "timestamp",
                  "src ip", "dst ip", "src port", "dst port",
                  "index", "slice"}]
    if drop_cols:
        df.drop(columns=drop_cols, inplace=True)
        print(f"[DataLoader] Colonnes supprimées : {drop_cols}")

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    # ── Séparation features / label ───────────────────────────────────────────
    label_col = config.LABEL_COLUMN
    if label_col not in df.columns:
        matches = [c for c in df.columns if c.lower() == label_col.lower()]
        if matches:
            label_col = matches[0]
        else:
            raise ValueError(
                f"Colonne label '{config.LABEL_COLUMN}' introuvable.\n"
                f"Colonnes disponibles : {list(df.columns)}"
            )

    y_raw = df[label_col].astype(str).values
    X_raw = df.drop(columns=[label_col])
    X_raw = X_raw.select_dtypes(include=[np.number])
    print(f"[DataLoader] Features numériques retenues : {X_raw.shape[1]}")

    # ── Encodage des labels ────────────────────────────────────────────────────
    le = LabelEncoder()
    y  = le.fit_transform(y_raw).astype(np.int64)
    class_names = list(le.classes_)
    n_classes   = len(class_names)
    print(f"[DataLoader] Classes détectées ({n_classes}) : {class_names}")

    # ── Features ──────────────────────────────────────────────────────────────
    X = X_raw.values.astype(np.float32)

    # ── Train / Test split — STRATIFIÉ ────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y
    )

    # ── Reshape pour CNN 1D : (n_samples, n_features, 1) ─────────────────────
    n_features = X_train.shape[1]
    X_test = X_test.reshape(-1, n_features, 1)

    # ── [FIX 2] Rééquilibrage combiné SMOTE + RandomUnderSampler ─────────────
    # SMOTE pur (ancienne version) : créait autant de benignes que d'attaques
    # (ratio 1:1) → modèle perdait la notion du déséquilibre réel → beaucoup
    # de faux positifs sur trafic benign → spécificité dégradée.
    #
    # Nouvelle stratégie :
    #   • SMOTE(sampling_strategy=0.3) → benign atteint 30% du total
    #   • RandomUnderSampler(0.5)      → attaques réduites, ratio final ~1:2
    print("[DataLoader] Rééquilibrage combiné SMOTE + UnderSampler en cours …")
    print(f"[DataLoader] Distribution avant : benign={(y_train==0).sum()} "
          f"/ attaque={(y_train!=0).sum()}")

    over  = SMOTE(
        sampling_strategy=0.3,
        random_state=config.RANDOM_STATE,
        k_neighbors=5
    )
    under = RandomUnderSampler(
        sampling_strategy=0.5,
        random_state=config.RANDOM_STATE
    )
    pipeline_resample = ImbPipeline([("over", over), ("under", under)])

    X_train_flat, y_train = pipeline_resample.fit_resample(X_train, y_train)
    X_train = X_train_flat.reshape(-1, n_features, 1)

    print(f"[DataLoader] Après rééquilibrage : {X_train.shape[0]} échantillons "
          f"(benign={(y_train==0).sum()} / attaque={(y_train!=0).sum()})")
    print(f"[DataLoader] Train : {X_train.shape}  Test : {X_test.shape}")

    return X_train, X_test, y_train, y_test, class_names, n_features, n_classes


def partition_for_silos(X_train, y_train, num_silos: int):
    """
    Partitionne (X_train, y_train) en `num_silos` sous-ensembles.

    [FIX 1] Partitionnement STRATIFIÉ au lieu de np.array_split séquentiel.
    Chaque silo reçoit la même distribution de classes (benign/attaque).
    C'était la cause principale du Silo 0 plafonné à 68% Val Accuracy.

    Retourne : liste de (X_silo, y_silo)
    """
    remaining_X = X_train.copy()
    remaining_y = y_train.copy()
    partitions  = []

    for i in range(num_silos - 1):
        fraction = 1.0 / (num_silos - i)
        silo_X, remaining_X, silo_y, remaining_y = train_test_split(
            remaining_X, remaining_y,
            test_size=1.0 - fraction,
            random_state=config.RANDOM_STATE + i,
            stratify=remaining_y          # ← correctif clé
        )
        partitions.append((silo_X, silo_y))
        print(f"[DataLoader] Silo {i} : {len(silo_X)} échantillons "
              f"(benign={(silo_y==0).sum()} / attaque={(silo_y!=0).sum()})")

    partitions.append((remaining_X, remaining_y))
    print(f"[DataLoader] Silo {num_silos-1} : {len(remaining_X)} échantillons "
          f"(benign={(remaining_y==0).sum()} / attaque={(remaining_y!=0).sum()})")

    return partitions
