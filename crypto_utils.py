# =============================================================================
# crypto_utils.py — Chiffrement asymétrique RSA des poids du modèle
# =============================================================================
"""
Flux RSA dans l'apprentissage fédéré :
  1. Le SERVEUR génère une paire de clés RSA (pub / priv).
  2. Le SERVEUR diffuse sa clé PUBLIQUE à tous les silos.
  3. Chaque SILO chiffre ses poids avec la clé publique du serveur, puis envoie
     le message chiffré via socket TCP.
  4. Le SERVEUR déchiffre avec sa clé PRIVÉE, agrège (FedAvg), rechiffre le
     modèle global avec la clé publique du silo concerné et renvoie.
  5. Chaque SILO déchiffre le modèle global reçu avec sa propre clé privée.
"""

import os
import pickle
import struct
import numpy as np
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.backends import default_backend
from cryptography.fernet import Fernet
import config


# ─────────────────────────────────────────────────────────────────────────────
# Génération de clés RSA
# ─────────────────────────────────────────────────────────────────────────────

def generate_rsa_keypair(name: str):
    """
    Génère une paire RSA et la sauvegarde dans keys/.
    Retourne (private_key, public_key).
    """
    private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=config.RSA_KEY_SIZE,
        backend=default_backend()
    )
    public_key = private_key.public_key()

    # Sérialisation PEM
    priv_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.TraditionalOpenSSL,
        encryption_algorithm=serialization.NoEncryption()
    )
    pub_pem = public_key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo
    )

    with open(os.path.join(config.KEYS_DIR, f"{name}_private.pem"), "wb") as f:
        f.write(priv_pem)
    with open(os.path.join(config.KEYS_DIR, f"{name}_public.pem"), "wb") as f:
        f.write(pub_pem)

    return private_key, public_key


def load_public_key(name: str):
    """Charge une clé publique PEM depuis keys/."""
    path = os.path.join(config.KEYS_DIR, f"{name}_public.pem")
    with open(path, "rb") as f:
        return serialization.load_pem_public_key(f.read(), backend=default_backend())


def load_private_key(name: str):
    """Charge une clé privée PEM depuis keys/."""
    path = os.path.join(config.KEYS_DIR, f"{name}_private.pem")
    with open(path, "rb") as f:
        return serialization.load_pem_private_key(f.read(), password=None, backend=default_backend())


# ─────────────────────────────────────────────────────────────────────────────
# Chiffrement hybride RSA + Fernet (symétrique)
# ─────────────────────────────────────────────────────────────────────────────
# RSA seul ne peut pas chiffrer de gros blobs (poids du CNN).
# On utilise donc un schéma hybride :
#   • On génère une clé symétrique Fernet (AES-128-CBC + HMAC)
#   • On chiffre les poids avec Fernet
#   • On chiffre la clé Fernet avec RSA-OAEP
# Le destinataire fait l'inverse.

def encrypt_weights(weights: list, recipient_public_key) -> bytes:
    """
    Chiffre une liste de tableaux numpy (poids du modèle).
    Retourne un blob binaire à envoyer sur le réseau.
    """
    # 1. Sérialiser les poids en bytes
    raw = pickle.dumps(weights)

    # 2. Générer une clé symétrique Fernet éphémère
    sym_key = Fernet.generate_key()
    fernet  = Fernet(sym_key)

    # 3. Chiffrer les poids avec Fernet
    encrypted_weights = fernet.encrypt(raw)

    # 4. Chiffrer la clé sym avec RSA-OAEP
    encrypted_sym_key = recipient_public_key.encrypt(
        sym_key,
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None
        )
    )

    # 5. Packager : [4 octets longueur clé chiffrée][clé chiffrée][poids chiffrés]
    payload = struct.pack(">I", len(encrypted_sym_key)) + encrypted_sym_key + encrypted_weights
    return payload


def decrypt_weights(payload: bytes, recipient_private_key) -> list:
    """
    Déchiffre un blob produit par encrypt_weights().
    Retourne la liste de tableaux numpy originale.
    """
    # 1. Extraire la longueur de la clé chiffrée
    key_len = struct.unpack(">I", payload[:4])[0]
    encrypted_sym_key = payload[4: 4 + key_len]
    encrypted_weights = payload[4 + key_len:]

    # 2. Déchiffrer la clé symétrique avec RSA
    sym_key = recipient_private_key.decrypt(
        encrypted_sym_key,
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None
        )
    )

    # 3. Déchiffrer les poids avec Fernet
    fernet = Fernet(sym_key)
    raw    = fernet.decrypt(encrypted_weights)

    # 4. Désérialiser
    return pickle.loads(raw)


# ─────────────────────────────────────────────────────────────────────────────
# Envoi / Réception de blobs via socket TCP
# ─────────────────────────────────────────────────────────────────────────────

def send_blob(sock, data: bytes):
    """Envoie un blob binaire préfixé par sa taille (8 octets big-endian)."""
    length = struct.pack(">Q", len(data))
    sock.sendall(length + data)


def recv_blob(sock) -> bytes:
    """Reçoit un blob envoyé par send_blob()."""
    raw_len = _recv_exact(sock, 8)
    length  = struct.unpack(">Q", raw_len)[0]
    return _recv_exact(sock, length)


def _recv_exact(sock, n: int) -> bytes:
    """Lit exactement n octets depuis le socket."""
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("Socket fermé prématurément.")
        buf += chunk
    return buf
