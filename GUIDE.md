# =============================================================================
#  GUIDE D'INSTALLATION ET D'EXÉCUTION — DDoS FL-CNN
#  Détection d'attaques DDoS par CNN + Apprentissage Fédéré + RSA
# =============================================================================

## STRUCTURE DU PROJET
```
DDoS_FL_CNN/
├── main.py          ← Point d'entrée (lancer ce fichier)
├── config.py        ← Tous les paramètres ajustables
├── server.py        ← Serveur fédéré (agrégation FedAvg + RSA)
├── silo.py          ← Client fédéré (entraînement local + chiffrement)
├── model.py         ← Architecture CNN 1D
├── data_loader.py   ← Chargement & partitionnement CSV
├── crypto_utils.py  ← Utilitaires RSA hybride + sockets TCP
├── evaluate.py      ← Métriques, graphiques, rapport
├── requirements.txt ← Dépendances Python
├── dataset/         ← ⚠ PLACER VOTRE CSV ICI
├── outputs/         ← Fichiers générés automatiquement
└── keys/            ← Clés RSA générées automatiquement
```

---

## ÉTAPE 1 — PRÉREQUIS SYSTÈME

- Python 3.9, 3.10 ou 3.11  (⚠ TensorFlow ne supporte pas encore Python 3.12)
- PyCharm (Community ou Professional)
- pip à jour

Vérifier votre version Python :
```
python --version
```

---

## ÉTAPE 2 — CRÉATION DE L'ENVIRONNEMENT VIRTUEL (dans PyCharm)

1. Ouvrir PyCharm → File → Open → sélectionner le dossier DDoS_FL_CNN/
2. Aller dans File → Settings → Project → Python Interpreter
3. Cliquer sur "Add Interpreter" → "Add Local Interpreter"
4. Choisir "Virtualenv Environment" → New environment
5. Sélectionner Python 3.10 (ou 3.9 / 3.11) comme base
6. Valider → OK

---

## ÉTAPE 3 — INSTALLATION DES DÉPENDANCES

Dans le terminal PyCharm (onglet "Terminal" en bas) :

```bash
pip install -r requirements.txt
```

Si TensorFlow pose problème sous Windows :
```bash
pip install tensorflow-cpu>=2.12.0
```

Pour GPU (NVIDIA + CUDA 11.8) :
```bash
pip install tensorflow[and-cuda]>=2.12.0
```

---

## ÉTAPE 4 — PRÉPARATION DU DATASET

1. Placer votre fichier CSV dans le dossier : dataset/
2. Renommer-le en : dataset.csv
   (ou modifier CSV_FILENAME dans config.py)

3. Ouvrir config.py et adapter ces deux paramètres :

```python
LABEL_COLUMN = "label"      # ← nom exact de votre colonne cible
NORMAL_LABEL = "BENIGN"     # ← valeur correspondant au trafic normal
```

⚠ Vérifier le nom exact de la colonne label dans votre CSV :
```python
import pandas as pd
df = pd.read_csv("dataset/dataset.csv", nrows=5)
print(df.columns.tolist())
```

---

## ÉTAPE 5 — CONFIGURATION DU PROJET (config.py)

Paramètres clés à ajuster selon vos besoins :

| Paramètre         | Défaut  | Description                              |
|-------------------|---------|------------------------------------------|
| NUM_SILOS         | 3       | Nombre de silos (clients fédérés)        |
| FEDERATED_ROUNDS  | 10      | Rounds de communication fédéré           |
| LOCAL_EPOCHS      | 5       | Epochs d'entraînement local par round    |
| BATCH_SIZE        | 64      | Taille des mini-lots                     |
| LEARNING_RATE     | 0.001   | Taux d'apprentissage Adam                |
| TEST_SIZE         | 0.2     | Fraction du dataset réservée aux tests   |
| RSA_KEY_SIZE      | 2048    | Taille des clés RSA (bits)               |
| SERVER_PORT       | 9000    | Port TCP du serveur fédéré               |

---

## ÉTAPE 6 — LANCEMENT DU PROJET

### Option A — Via PyCharm (recommandé)
1. Ouvrir main.py dans l'éditeur
2. Clic droit → "Run 'main'"
   ou appuyer sur Shift+F10

### Option B — Via le terminal PyCharm

Lancement avec les paramètres par défaut (config.py) :
```bash
python main.py
```

Lancement avec paramètres personnalisés (sans modifier config.py) :
```bash
# 5 silos, 15 rounds, 3 epochs locaux
python main.py --silos 5 --rounds 15 --epochs 3

# CSV dans un autre dossier
python main.py --csv /chemin/vers/mon_fichier.csv

# Toutes les options ensemble
python main.py --silos 4 --rounds 20 --epochs 5 --csv dataset/mon_dataset.csv
```

---

## ÉTAPE 7 — SUIVI DE L'EXÉCUTION

L'exécution affiche en temps réel :
```
============================================================
   DDoS Detection — Federated CNN + RSA Encryption
   Silos : 3  |  Rounds : 10  |  Epochs locaux : 5
============================================================

[DataLoader] Chargement de dataset/dataset.csv …
[DataLoader] Dimensions brutes : (100000, 79)
[DataLoader] Features numériques retenues : 76
[DataLoader] Classes détectées (2) : ['BENIGN', 'DDoS']
[DataLoader] Train : (80000, 76, 1)  Test : (20000, 76, 1)
[DataLoader] Silo 0 : 26666 échantillons
[DataLoader] Silo 1 : 26667 échantillons
[DataLoader] Silo 2 : 26667 échantillons

[Server] Génération des clés RSA …
[Server] Clés générées pour Silo 0, 1, 2

────────────────────────────────────────────────────────────
  ROUND FÉDÉRÉ 1 / 10
────────────────────────────────────────────────────────────
[Server] Round 1 — En attente sur port 9000 …
[Silo 0] Round 1 — Loss=0.4123  Acc=0.8756  Val_Acc=0.8812
[Silo 1] Round 1 — Loss=0.3987  Acc=0.8901  Val_Acc=0.8934
[Silo 2] Round 1 — Loss=0.4210  Acc=0.8623  Val_Acc=0.8701
[Server] Poids reçus du Silo 0 (train_acc=0.8756)
...
[Server] ✔ Round 1 agrégé — Acc moy=0.8760  Loss moy=0.4107
```

---

## ÉTAPE 8 — FICHIERS GÉNÉRÉS (dossier outputs/)

À la fin de l'exécution, le dossier outputs/ contient :

| Fichier                  | Description                                     |
|--------------------------|-------------------------------------------------|
| federated_curves.png     | Courbes accuracy & loss par round fédéré        |
| silo_curves.png          | Courbes accuracy & loss par silo                |
| confusion_matrix.png     | Matrice de confusion (en pourcentage)           |
| metrics_report.txt       | Rapport texte : accuracy, précision, F1, etc.  |
| final_model.keras        | Modèle entraîné sauvegardé (rechargeable)       |
| keys/                    | Clés RSA générées (server + silos)              |

---

## DÉPANNAGE FRÉQUENT

### ❌ "Dataset introuvable"
→ Vérifier que votre CSV est dans dataset/ et s'appelle dataset.csv
→ Ou utiliser : python main.py --csv /chemin/complet/fichier.csv

### ❌ "Colonne label introuvable"
→ Ouvrir config.py → modifier LABEL_COLUMN avec le nom exact de votre colonne

### ❌ "Port already in use" (OSError: [Errno 98])
→ Changer SERVER_PORT dans config.py (ex: 9100)
→ Ou attendre quelques secondes et relancer

### ❌ "ConnectionRefusedError"
→ Le serveur n'a pas démarré assez vite → augmenter le délai dans main.py :
   time.sleep(0.3) → time.sleep(1.0)

### ❌ Manque de RAM / lenteur excessive
→ Réduire BATCH_SIZE (ex: 32) dans config.py
→ Réduire NUM_SILOS ou FEDERATED_ROUNDS
→ Utiliser un sous-ensemble du CSV :
```python
df = pd.read_csv(..., nrows=50000)
```

### ❌ TensorFlow non installé correctement
```bash
pip uninstall tensorflow tensorflow-cpu tensorflow-gpu
pip install tensorflow==2.13.0
```

---

## RECHARGEMENT DU MODÈLE FINAL (après entraînement)

```python
import tensorflow as tf
import numpy as np

# Charger le modèle sauvegardé
model = tf.keras.models.load_model("outputs/final_model.keras")

# Prédire sur de nouvelles données (shape: n_samples x n_features x 1)
X_new = np.random.rand(10, 76, 1).astype("float32")  # exemple
predictions = model.predict(X_new)
classes = np.argmax(predictions, axis=1)
print(classes)
```

---

## RÉSUMÉ DE L'ARCHITECTURE TECHNIQUE

```
                        ┌─────────────────────────────────┐
                        │        SERVEUR FÉDÉRÉ           │
                        │  • Clé RSA privée (déchiffrement)│
                        │  • Agrégation FedAvg             │
                        │  • Distribution clés publiques   │
                        └────────────┬────────────────────┘
                                     │ TCP (127.0.0.1:9000)
               ┌─────────────────────┼─────────────────────┐
               │                     │                      │
        ┌──────▼──────┐       ┌──────▼──────┐      ┌──────▼──────┐
        │   SILO 0    │       │   SILO 1    │      │   SILO N    │
        │  CNN local  │       │  CNN local  │      │  CNN local  │
        │  Données 1/N│       │  Données 1/N│      │  Données 1/N│
        │  Clé RSA    │       │  Clé RSA    │      │  Clé RSA    │
        │  privée     │       │  privée     │      │  privée     │
        └─────────────┘       └─────────────┘      └─────────────┘

Chiffrement : RSA-OAEP (2048 bits) + AES-128 Fernet (hybride)
Agrégation  : FedAvg (moyenne des poids de tous les silos)
Modèle CNN  : Conv1D × 3 + BatchNorm + Dense × 2 + Softmax
```
