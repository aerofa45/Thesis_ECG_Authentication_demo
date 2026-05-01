import os
import math
import json
import zipfile
import warnings
from dataclasses import dataclass
from pathlib import Path

import gradio as gr
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import butter, filtfilt, find_peaks, resample

warnings.filterwarnings("ignore")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
APP_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_DIR_NAME = os.environ.get("MODEL_DIR", "ecg_demo_saved_models")
MODEL_DIR = str(APP_ROOT / DEFAULT_MODEL_DIR_NAME)



# =========================
# Artifact discovery / unzip
# =========================

def prepare_artifacts():
    """Find or unzip the model artifact folder.

    Supported layouts:
    1. ./ecg_demo_saved_models/...
    2. ./ecg_demo_saved_models.zip
    3. any root-level .zip containing ecg_demo_saved_models/ or model files
    """
    global MODEL_DIR

    direct = APP_ROOT / DEFAULT_MODEL_DIR_NAME
    if direct.exists() and direct.is_dir():
        MODEL_DIR = str(direct)
        return MODEL_DIR

    preferred_zip = APP_ROOT / f"{DEFAULT_MODEL_DIR_NAME}.zip"
    zip_candidates = []
    if preferred_zip.exists():
        zip_candidates.append(preferred_zip)
    zip_candidates.extend([p for p in APP_ROOT.glob("*.zip") if p != preferred_zip])

    for zpath in zip_candidates:
        try:
            extract_dir = APP_ROOT / "_extracted_artifacts"
            extract_dir.mkdir(exist_ok=True)
            with zipfile.ZipFile(zpath, "r") as zf:
                zf.extractall(extract_dir)

            possible = extract_dir / DEFAULT_MODEL_DIR_NAME
            if possible.exists() and possible.is_dir():
                MODEL_DIR = str(possible)
                return MODEL_DIR

            # Some zips contain the model files directly.
            model_like = list(extract_dir.glob("*.pt")) + list(extract_dir.glob("*.joblib"))
            if model_like:
                MODEL_DIR = str(extract_dir)
                return MODEL_DIR
        except Exception as exc:
            print(f"Could not unzip {zpath}: {exc}")

    # Keep the conventional folder name even if missing; app will show missing models.
    MODEL_DIR = str(direct)
    return MODEL_DIR


MODEL_DIR = prepare_artifacts()
print("Using model artifact folder:", MODEL_DIR)
print("Device:", DEVICE)



# =========================
# Configuration loading
# =========================

@dataclass
class DemoCFG:
    target_fs: int = 250
    segment_seconds: float = 2.0
    embedding_dim: int = 64
    bandpass_low: float = 0.5
    bandpass_high: float = 40.0
    normalize_per_segment: bool = True


cfg = DemoCFG()

config_path = os.path.join(MODEL_DIR, "config.json")
if os.path.exists(config_path):
    try:
        with open(config_path, "r") as f:
            saved_cfg = json.load(f)
        cfg.target_fs = int(saved_cfg.get("target_fs", cfg.target_fs))
        cfg.segment_seconds = float(saved_cfg.get("segment_seconds", cfg.segment_seconds))
        cfg.embedding_dim = int(saved_cfg.get("embedding_dim", cfg.embedding_dim))
        cfg.bandpass_low = float(saved_cfg.get("bandpass_low", cfg.bandpass_low))
        cfg.bandpass_high = float(saved_cfg.get("bandpass_high", cfg.bandpass_high))
        cfg.normalize_per_segment = bool(saved_cfg.get("normalize_per_segment", cfg.normalize_per_segment))
    except Exception as exc:
        print("Could not read config.json:", exc)

SEG_LEN = int(cfg.target_fs * cfg.segment_seconds)
print("Expected segment length:", SEG_LEN)


# =========================
# 4. Model definitions
# =========================

class ConvEncoder(nn.Module):
    def __init__(self, emb_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 16, 7, padding=3),
            nn.BatchNorm1d(16), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, 5, padding=2),
            nn.BatchNorm1d(32), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 5, padding=2),
            nn.BatchNorm1d(64), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Linear(64, emb_dim)

    def forward(self, x):
        z = self.net(x).squeeze(-1)
        z = self.fc(z)
        z = F.normalize(z, dim=-1)
        return z


class SimpleVerifier(nn.Module):
    def __init__(self, emb_dim=64):
        super().__init__()
        self.encoder = ConvEncoder(emb_dim=emb_dim)
        self.head = nn.Linear(1, 1)

    def forward(self, x1, x2):
        z1 = self.encoder(x1)
        z2 = self.encoder(x2)
        cos = F.cosine_similarity(z1, z2).unsqueeze(-1)
        logit = self.head(cos)
        return logit.squeeze(-1)

    def predict_similarity(self, x1, x2):
        return torch.sigmoid(self.forward(x1, x2))


class SiameseCNNBiLSTMEncoder(nn.Module):
    def __init__(self, emb_dim=64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 16, 7, padding=3), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, 5, padding=2), nn.ReLU(),
            nn.MaxPool1d(2),
        )
        self.lstm = nn.LSTM(
            input_size=32,
            hidden_size=32,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.fc = nn.Linear(64, emb_dim)

    def forward(self, x):
        h = self.conv(x)
        h = h.transpose(1, 2)
        out, _ = self.lstm(h)
        z = out[:, -1, :]
        z = self.fc(z)
        z = F.normalize(z, dim=-1)
        return z


class SiameseVerifier(nn.Module):
    def __init__(self, emb_dim=64):
        super().__init__()
        self.encoder = SiameseCNNBiLSTMEncoder(emb_dim=emb_dim)
        self.classifier = nn.Sequential(
            nn.Linear(emb_dim * 4, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
        )

    def forward(self, x1, x2):
        z1 = self.encoder(x1)
        z2 = self.encoder(x2)
        feat = torch.cat([z1, z2, torch.abs(z1 - z2), z1 * z2], dim=-1)
        return self.classifier(feat).squeeze(-1)

    def predict_similarity(self, x1, x2):
        return torch.sigmoid(self.forward(x1, x2))


class PINNEncoder(nn.Module):
    def __init__(self, emb_dim=64, seq_len=None):
        super().__init__()
        if seq_len is None:
            seq_len = SEG_LEN
        self.seq_len = seq_len
        self.conv = nn.Sequential(
            nn.Conv1d(1, 16, 9, padding=4), nn.BatchNorm1d(16), nn.ReLU(),
            nn.Conv1d(16, 32, 7, padding=3), nn.BatchNorm1d(32), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 5, padding=2), nn.BatchNorm1d(64), nn.ReLU(),
            nn.MaxPool1d(2),
        )
        self.emb = nn.Linear(64, emb_dim)
        self.recon = nn.Sequential(
            nn.Linear(emb_dim, 128),
            nn.ReLU(),
            nn.Linear(128, seq_len),
        )

    def forward(self, x):
        h = self.conv(x)
        pooled = h.mean(dim=-1)
        z = F.normalize(self.emb(pooled), dim=-1)
        recon = self.recon(z)
        return z, recon


class ProposedPINNVerifier(nn.Module):
    def __init__(self, emb_dim=64, alpha=1.0, beta=0.2, gamma=0.1, delta=0.05):
        super().__init__()
        self.encoder = PINNEncoder(emb_dim=emb_dim, seq_len=SEG_LEN)
        self.classifier = nn.Sequential(
            nn.Linear(emb_dim * 4, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta

    def forward(self, x1, x2):
        z1, r1 = self.encoder(x1)
        z2, r2 = self.encoder(x2)
        feat = torch.cat([z1, z2, torch.abs(z1 - z2), z1 * z2], dim=-1)
        logits = self.classifier(feat).squeeze(-1)
        return logits, z1, z2, r1, r2

    def predict_similarity(self, x1, x2):
        logits, *_ = self.forward(x1, x2)
        return torch.sigmoid(logits)


class HDCEncoder:
    def __init__(self, signal_len, chunk_size=16, hv_dim=2000, seed=42):
        self.signal_len = signal_len
        self.chunk_size = chunk_size
        self.hv_dim = hv_dim
        self.rng = np.random.default_rng(seed)
        self.num_chunks = signal_len // chunk_size
        if self.num_chunks < 1:
            raise ValueError("chunk_size too large for signal length")
        self.pos_hv = self.rng.choice([-1.0, 1.0], size=(self.num_chunks, hv_dim)).astype(np.float32)
        self.local_feat_dim = 8
        self.proj = self.rng.normal(0, 1, size=(self.local_feat_dim, hv_dim)).astype(np.float32)

    def _local_features(self, chunk):
        chunk = chunk.astype(np.float32)
        d1 = np.diff(chunk, prepend=chunk[0])
        feats = np.array([
            chunk.mean(), chunk.std(), chunk.min(), chunk.max(), np.ptp(chunk),
            np.mean(chunk ** 2), d1.mean(), d1.std(),
        ], dtype=np.float32)
        return feats

    def encode_one(self, signal_1d):
        x = np.asarray(signal_1d, dtype=np.float32)
        usable = self.num_chunks * self.chunk_size
        x = x[:usable]
        chunks = x.reshape(self.num_chunks, self.chunk_size)
        hv_sum = np.zeros(self.hv_dim, dtype=np.float32)
        for i in range(self.num_chunks):
            feats = self._local_features(chunks[i])
            feat_hv = feats @ self.proj
            feat_hv = np.where(feat_hv >= 0, 1.0, -1.0).astype(np.float32)
            hv_sum += feat_hv * self.pos_hv[i]
        hv = np.where(hv_sum >= 0, 1.0, -1.0).astype(np.float32)
        return hv / (np.linalg.norm(hv) + 1e-8)

    def similarity(self, sig1, sig2):
        h1 = self.encode_one(sig1)
        h2 = self.encode_one(sig2)
        return float(np.dot(h1, h2))


class HDCEncoderVectorized:
    def __init__(self, signal_len, chunk_size=16, hv_dim=2000, seed=42):
        self.signal_len = signal_len
        self.chunk_size = chunk_size
        self.hv_dim = hv_dim
        self.rng = np.random.default_rng(seed)
        self.num_chunks = signal_len // chunk_size
        if self.num_chunks < 1:
            raise ValueError("chunk_size too large for signal length")
        self.pos_hv = self.rng.choice([-1.0, 1.0], size=(self.num_chunks, hv_dim)).astype(np.float32)
        self.local_feat_dim = 8
        self.proj = self.rng.normal(0, 1, size=(self.local_feat_dim, hv_dim)).astype(np.float32)

    def _chunk_features(self, chunks):
        d1 = np.diff(chunks, axis=1, prepend=chunks[:, :1])
        feats = np.stack([
            chunks.mean(axis=1), chunks.std(axis=1), chunks.min(axis=1), chunks.max(axis=1),
            np.ptp(chunks, axis=1), np.mean(chunks ** 2, axis=1), d1.mean(axis=1), d1.std(axis=1),
        ], axis=1).astype(np.float32)
        return feats

    def encode_one(self, signal_1d):
        x = np.asarray(signal_1d, dtype=np.float32)
        usable = self.num_chunks * self.chunk_size
        x = x[:usable]
        chunks = x.reshape(self.num_chunks, self.chunk_size)
        feats = self._chunk_features(chunks)
        feat_hv = feats @ self.proj
        feat_hv = np.where(feat_hv >= 0, 1.0, -1.0).astype(np.float32)
        bound = feat_hv * self.pos_hv
        hv_sum = bound.sum(axis=0)
        hv = np.where(hv_sum >= 0, 1.0, -1.0).astype(np.float32)
        return hv / (np.linalg.norm(hv) + 1e-8)

    def similarity(self, sig1, sig2):
        h1 = self.encode_one(sig1)
        h2 = self.encode_one(sig2)
        return float(np.dot(h1, h2))


class LatentHDCProjector(nn.Module):
    def __init__(self, in_dim=64, hv_dim=2000):
        super().__init__()
        self.register_buffer("proj", torch.randn(in_dim, hv_dim))
        self.hv_dim = hv_dim

    def forward(self, z):
        h = z @ self.proj
        h = torch.sign(h)
        h[h == 0] = 1.0
        return F.normalize(h, dim=-1)


class PINNEncoderFHN(nn.Module):
    def __init__(self, emb_dim=64, seq_len=None):
        super().__init__()
        if seq_len is None:
            seq_len = SEG_LEN
        self.seq_len = seq_len
        self.conv = nn.Sequential(
            nn.Conv1d(1, 16, 9, padding=4), nn.BatchNorm1d(16), nn.ReLU(),
            nn.Conv1d(16, 32, 7, padding=3), nn.BatchNorm1d(32), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 5, padding=2), nn.BatchNorm1d(64), nn.ReLU(),
            nn.MaxPool1d(2),
        )
        self.emb = nn.Linear(64, emb_dim)
        self.v_head = nn.Sequential(nn.Linear(emb_dim, 128), nn.ReLU(), nn.Linear(128, seq_len))
        self.w_head = nn.Sequential(nn.Linear(emb_dim, 128), nn.ReLU(), nn.Linear(128, seq_len))
        self.I_param = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))

    def forward(self, x):
        h = self.conv(x)
        pooled = h.mean(dim=-1)
        z = F.normalize(self.emb(pooled), dim=-1)
        v = self.v_head(z)
        w = self.w_head(z)
        return z, v, w


class HybridHDCPINNVerifierFHN(nn.Module):
    def __init__(self, emb_dim=64, hv_dim=2000, alpha=1.0, beta=0.15, gamma=0.05, delta=0.05,
                 eta=0.25, zeta=0.20, fhn_eps=0.08, fhn_a=0.7, fhn_b=0.8):
        super().__init__()
        self.seq_len = SEG_LEN
        self.dt = 1.0 / cfg.target_fs
        self.encoder = PINNEncoderFHN(emb_dim=emb_dim, seq_len=self.seq_len)
        self.hdc = LatentHDCProjector(in_dim=emb_dim, hv_dim=hv_dim)
        self.classifier = nn.Sequential(nn.Linear(emb_dim * 4 + 1, 64), nn.ReLU(), nn.Linear(64, 1))
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.eta = eta
        self.zeta = zeta
        self.fhn_eps = fhn_eps
        self.fhn_a = fhn_a
        self.fhn_b = fhn_b

    def forward(self, x1, x2):
        z1, v1, w1 = self.encoder(x1)
        z2, v2, w2 = self.encoder(x2)
        h1 = self.hdc(z1)
        h2 = self.hdc(z2)
        hdc_sim = F.cosine_similarity(h1, h2).unsqueeze(-1)
        feat = torch.cat([z1, z2, torch.abs(z1 - z2), z1 * z2, hdc_sim], dim=-1)
        logits = self.classifier(feat).squeeze(-1)
        return logits, z1, z2, h1, h2, v1, w1, v2, w2

    def predict_similarity(self, x1, x2):
        logits, *_ = self.forward(x1, x2)
        return torch.sigmoid(logits)


class PINNEncoderMcSharry(nn.Module):
    def __init__(self, emb_dim=64, seq_len=None):
        super().__init__()
        if seq_len is None:
            seq_len = SEG_LEN
        self.seq_len = seq_len
        self.conv = nn.Sequential(
            nn.Conv1d(1, 16, 9, padding=4), nn.BatchNorm1d(16), nn.ReLU(),
            nn.Conv1d(16, 32, 7, padding=3), nn.BatchNorm1d(32), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 5, padding=2), nn.BatchNorm1d(64), nn.ReLU(),
            nn.MaxPool1d(2),
        )
        self.emb = nn.Linear(64, emb_dim)
        self.sig_head = nn.Sequential(nn.Linear(emb_dim, 128), nn.ReLU(), nn.Linear(128, seq_len))
        self.z0 = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

    def forward(self, x):
        h = self.conv(x)
        pooled = h.mean(dim=-1)
        z = F.normalize(self.emb(pooled), dim=-1)
        s = self.sig_head(z)
        return z, s


class HybridHDCPINNVerifierMcSharry(nn.Module):
    def __init__(self, emb_dim=64, hv_dim=2000, alpha=1.0, beta=0.15, gamma=0.05, delta=0.05,
                 eta=0.25, zeta=0.20):
        super().__init__()
        self.seq_len = SEG_LEN
        self.dt = 1.0 / cfg.target_fs
        self.encoder = PINNEncoderMcSharry(emb_dim=emb_dim, seq_len=self.seq_len)
        self.hdc = LatentHDCProjector(in_dim=emb_dim, hv_dim=hv_dim)
        self.classifier = nn.Sequential(nn.Linear(emb_dim * 4 + 1, 64), nn.ReLU(), nn.Linear(64, 1))
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.eta = eta
        self.zeta = zeta
        self.register_buffer("theta_i", torch.tensor([
            -0.70 * math.pi, -0.15 * math.pi, 0.00 * math.pi, 0.12 * math.pi, 0.55 * math.pi,
        ], dtype=torch.float32))
        self.register_buffer("a_i", torch.tensor([0.10, -0.12, 1.00, -0.25, 0.35], dtype=torch.float32))
        self.register_buffer("b_i", torch.tensor([0.22, 0.08, 0.05, 0.08, 0.25], dtype=torch.float32))

    def finite_diff(self, x):
        return (x[:, 1:] - x[:, :-1]) / self.dt

    def wrapped_angle(self, angle):
        return torch.atan2(torch.sin(angle), torch.cos(angle))

    def mcsharry_physics_loss(self, s):
        device = s.device
        _, T = s.shape
        theta = torch.linspace(-math.pi, math.pi, T - 1, device=device).unsqueeze(0)
        ds_dt = self.finite_diff(s)
        s_mid = s[:, :-1]
        rhs = torch.zeros_like(s_mid)
        for k in range(len(self.theta_i)):
            dtheta = self.wrapped_angle(theta - self.theta_i[k])
            rhs = rhs - self.a_i[k] * dtheta * torch.exp(-(dtheta ** 2) / (2.0 * self.b_i[k] ** 2))
        rhs = rhs - (s_mid - self.encoder.z0)
        return torch.mean((ds_dt - rhs) ** 2)

    def forward(self, x1, x2):
        z1, s1 = self.encoder(x1)
        z2, s2 = self.encoder(x2)
        h1 = self.hdc(z1)
        h2 = self.hdc(z2)
        hdc_sim = F.cosine_similarity(h1, h2).unsqueeze(-1)
        feat = torch.cat([z1, z2, torch.abs(z1 - z2), z1 * z2, hdc_sim], dim=-1)
        logits = self.classifier(feat).squeeze(-1)
        return logits, z1, z2, h1, h2, s1, s2

    def predict_similarity(self, x1, x2):
        logits, *_ = self.forward(x1, x2)
        return torch.sigmoid(logits)

# =========================
# 5. Load available saved models dynamically
# =========================

def first_existing(*filenames):
    for filename in filenames:
        path = os.path.join(MODEL_DIR, filename)
        if os.path.exists(path):
            return path
    return None


def load_state_dict_flexible(model, path):
    payload = torch.load(path, map_location=DEVICE)
    if isinstance(payload, dict) and "state_dict" in payload:
        state = payload["state_dict"]
        extra = payload.get("extra_info", {})
    else:
        state = payload
        extra = {}
    model.load_state_dict(state, strict=True)
    model.to(DEVICE)
    model.eval()
    return model, extra


MODELS = {}
MODEL_META = []

def register_model(name, kind, obj, path=None, notes="", threshold_key=None, score_type="probability"):
    MODELS[name] = {
        "kind": kind,
        "object": obj,
        "path": path,
        "notes": notes,
        "threshold_key": threshold_key,
        "score_type": score_type,
    }
    MODEL_META.append({
        "Model": name,
        "Type": kind,
        "Status": "Loaded" if obj is not None else "Unavailable",
        "Path": path or "",
        "Notes": notes,
    })


def register_error(name, kind, path, err):
    MODELS[name] = {"kind": kind, "object": None, "path": path, "notes": str(err), "threshold_key": None, "score_type": "unknown"}
    MODEL_META.append({"Model": name, "Type": kind, "Status": f"FAILED: {type(err).__name__}", "Path": path or "", "Notes": str(err)})

# CNN baseline
path = first_existing("demo2_cnn.pt")
if path:
    try:
        model, extra = load_state_dict_flexible(SimpleVerifier(emb_dim=cfg.embedding_dim), path)
        register_model("CNN baseline", "torch", model, path, "Lightweight 1D CNN verifier", "demo2_test")
    except Exception as e:
        register_error("CNN baseline", "torch", path, e)

# Siamese CNN-BiLSTM
path = first_existing("demo3_siamese_cnn_bilstm.pt")
if path:
    try:
        model, extra = load_state_dict_flexible(SiameseVerifier(emb_dim=cfg.embedding_dim), path)
        register_model("Siamese CNN-BiLSTM", "torch", model, path, "Sequence-aware Siamese baseline", "demo3_test")
    except Exception as e:
        register_error("Siamese CNN-BiLSTM", "torch", path, e)

# PINN-only
path = first_existing("demo4_pinn_only.pt")
if path:
    try:
        model, extra = load_state_dict_flexible(ProposedPINNVerifier(emb_dim=cfg.embedding_dim, alpha=1.0, beta=0.15, gamma=0.05, delta=0.05), path)
        register_model("PINN-only structured model", "torch", model, path, "Reconstruction + smoothness + morphology, no HDC branch", "demo4_pinn_test")
    except Exception as e:
        register_error("PINN-only structured model", "torch", path, e)

# FHN hybrid
path = first_existing("demo5_hdc_pinn_fhn.pt")
if path:
    try:
        model, extra = load_state_dict_flexible(HybridHDCPINNVerifierFHN(emb_dim=cfg.embedding_dim, hv_dim=2000, eta=0.25, zeta=0.20), path)
        register_model("FHN HDC-PINN hybrid", "torch", model, path, "HDC + explicit FitzHugh-Nagumo physics residual", "demo5_test")
    except Exception as e:
        register_error("FHN HDC-PINN hybrid", "torch", path, e)

# FHN no-physics if saved
path = first_existing("fhn_hybrid_no_physics.pt", "demo5b_fhn_no_physics.pt")
if path:
    try:
        model, extra = load_state_dict_flexible(HybridHDCPINNVerifierFHN(emb_dim=cfg.embedding_dim, hv_dim=2000, eta=0.25, zeta=0.0), path)
        register_model("FHN hybrid no physics", "torch", model, path, "FHN-family hybrid with zeta=0", "ablation_no_phys_test")
    except Exception as e:
        register_error("FHN hybrid no physics", "torch", path, e)

# Full McSharry
path = first_existing("demo6_full_mcsharry.pt", "demo6_hdc_pinn_mcsharry.pt")
if path:
    try:
        model, extra = load_state_dict_flexible(HybridHDCPINNVerifierMcSharry(emb_dim=cfg.embedding_dim, hv_dim=2000, eta=0.25, zeta=0.20), path)
        register_model("Full McSharry hybrid", "torch", model, path, "McSharry-family HDC-PINN with explicit physics", "demo6_test")
    except Exception as e:
        register_error("Full McSharry hybrid", "torch", path, e)

# McSharry no physics
path = first_existing("demo6b_mcsharry_no_physics.pt", "demo6b_mcsharry_no_physics_best.pt")
if path:
    try:
        model, extra = load_state_dict_flexible(HybridHDCPINNVerifierMcSharry(emb_dim=cfg.embedding_dim, hv_dim=2000, eta=0.25, zeta=0.0), path)
        register_model("McSharry hybrid no physics", "torch", model, path, "Matched McSharry ablation with zeta=0", "mcsharry_no_phys_test")
    except Exception as e:
        register_error("McSharry hybrid no physics", "torch", path, e)

# McSharry no HDC loss
path = first_existing("demo6c_mcsharry_no_hdc_loss.pt")
if path:
    try:
        model, extra = load_state_dict_flexible(HybridHDCPINNVerifierMcSharry(emb_dim=cfg.embedding_dim, hv_dim=2000, eta=0.0, zeta=0.20), path)
        register_model("McSharry hybrid no HDC loss", "torch", model, path, "Matched McSharry ablation with eta=0", "mcsharry_no_hdc_test")
    except Exception as e:
        register_error("McSharry hybrid no HDC loss", "torch", path, e)

# SVM baseline
path = first_existing("demo1_handcrafted_svm.joblib")
if path:
    try:
        svm_clf = joblib.load(path)
        register_model("Handcrafted SVM baseline", "sklearn", svm_clf, path, "Handcrafted pair features + SVM", "demo1_test")
    except Exception as e:
        register_error("Handcrafted SVM baseline", "sklearn", path, e)

# HDC baselines
path = first_existing("demo4_hdc_encoder.joblib", "demo4_hdc_encoder_naive.joblib")
if path:
    try:
        hdc = joblib.load(path)
        register_model("HDC baseline", "hdc", hdc, path, "Standalone HDC encoder; score shown as mapped cosine", "demo4_test", score_type="cosine_mapped")
    except Exception as e:
        register_error("HDC baseline", "hdc", path, e)
else:
    # fallback if no HDC joblib exists
    hdc = HDCEncoder(signal_len=SEG_LEN, chunk_size=16, hv_dim=2000, seed=42)
    register_model("HDC baseline", "hdc", hdc, None, "Fallback HDC encoder recreated from config", "demo4_test", score_type="cosine_mapped")

path = first_existing("demo4b_hdc_encoder_vectorized.joblib")
if path:
    try:
        hdc_vec = joblib.load(path)
        register_model("Vectorized HDC baseline", "hdc", hdc_vec, path, "Vectorized HDC encoder", "demo4_vec_test", score_type="cosine_mapped")
    except Exception as e:
        register_error("Vectorized HDC baseline", "hdc", path, e)

load_table = pd.DataFrame(MODEL_META)
print("Loaded model names:", list(MODELS.keys()))

# =========================
# 6. Load saved thresholds and result tables, if present
# =========================

def load_json_if_exists(filename):
    path = os.path.join(MODEL_DIR, filename)
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {}

thresholds_json = load_json_if_exists("thresholds_and_metrics.json")

# Also load CSV result tables for display.
RESULT_TABLES = {}
for filename in [
    "main_results.csv",
    "timing_results.csv",
    "fhn_ablation_results.csv",
    "mcsharry_ablation_results.csv",
    "hdc_fair_results.csv",
    "hdc_fair_timing.csv",
    "robustness_results.csv",
    "short_window_results.csv",
    "ptb_results.csv",
    "subject_split_table.csv",
]:
    path = os.path.join(MODEL_DIR, filename)
    if os.path.exists(path):
        try:
            RESULT_TABLES[filename] = pd.read_csv(path)
        except Exception as e:
            print("Could not read", filename, e)

print("Loaded threshold keys:", list(thresholds_json.keys()))
print("Loaded result tables:", list(RESULT_TABLES.keys()))

# Map displayed model names to likely metric keys.
DEFAULT_THRESHOLD_MAP = {
    "Handcrafted SVM baseline": "demo1_test",
    "CNN baseline": "demo2_test",
    "Siamese CNN-BiLSTM": "demo3_test",
    "PINN-only structured model": "demo4_pinn_test",
    "HDC baseline": "demo4_test",
    "Vectorized HDC baseline": "demo4_vec_test",
    "FHN HDC-PINN hybrid": "demo5_test",
    "FHN hybrid no physics": "ablation_no_phys_test",
    "Full McSharry hybrid": "demo6_test",
    "McSharry hybrid no physics": "mcsharry_no_phys_test",
    "McSharry hybrid no HDC loss": "mcsharry_no_hdc_test",
}


def get_saved_threshold(model_name, fallback=0.5):
    key = DEFAULT_THRESHOLD_MAP.get(model_name)
    if key and key in thresholds_json:
        thr = thresholds_json[key].get("Threshold", None)
        if thr is not None and np.isfinite(thr):
            # HDC raw thresholds may be cosine [-1,1]. The app score maps HDC to [0,1].
            if MODELS.get(model_name, {}).get("kind") == "hdc" and -1.0 <= float(thr) <= 1.0:
                return float((float(thr) + 1.0) / 2.0)
            return float(thr)
    return float(fallback)

threshold_preview = pd.DataFrame([
    {"Model": name, "SavedThresholdUsedByApp": get_saved_threshold(name), "Type": info["kind"]}
    for name, info in MODELS.items()
])
display(threshold_preview)

# =========================
# 7. ECG loading and preprocessing
# =========================

def load_ecg_file(path):
    path = str(path)
    if path.endswith(".npy"):
        x = np.load(path)
        x = np.asarray(x).squeeze()
        if x.ndim > 1:
            x = x[:, 0]
        return x.astype(np.float32)

    # Try CSV with numeric columns.
    try:
        df = pd.read_csv(path)
        num = df.select_dtypes(include=[np.number])
        if num.shape[1] > 0:
            return num.iloc[:, 0].to_numpy(dtype=np.float32)
    except Exception:
        pass

    # Try plain text or comma-delimited numeric file.
    try:
        x = np.loadtxt(path, delimiter=",")
    except Exception:
        x = np.loadtxt(path)
    x = np.asarray(x).squeeze()
    if x.ndim > 1:
        x = x[:, 0]
    return x.astype(np.float32)


def butter_bandpass_filter_demo(x, lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    if low <= 0 or high >= 1 or low >= high:
        return x
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, x).astype(np.float32)


def preprocess_ecg(x, input_fs=250, apply_bandpass=False):
    x = np.asarray(x, dtype=np.float32).copy()
    x = np.nan_to_num(x)
    input_fs = int(input_fs)

    if apply_bandpass and len(x) > 12:
        try:
            x = butter_bandpass_filter_demo(x, cfg.bandpass_low, cfg.bandpass_high, input_fs)
        except Exception:
            pass

    if input_fs != cfg.target_fs and len(x) > 2:
        new_len = max(int(round(len(x) * cfg.target_fs / input_fs)), 2)
        x = resample(x, new_len).astype(np.float32)

    if len(x) >= SEG_LEN:
        start = (len(x) - SEG_LEN) // 2
        x = x[start:start + SEG_LEN]
    else:
        x = np.pad(x, (0, SEG_LEN - len(x)), mode="constant")

    if cfg.normalize_per_segment:
        x = (x - x.mean()) / (x.std() + 1e-8)
    return x.astype(np.float32)


def extract_single_features(x, fs):
    x = np.asarray(x, dtype=np.float32)
    dx = np.diff(x, prepend=x[0])
    peaks, _ = find_peaks(x, distance=max(1, int(0.25 * fs)))
    rr = np.diff(peaks) / fs if len(peaks) > 1 else np.array([0.0], dtype=np.float32)
    feats = [
        x.mean(), x.std(), x.min(), x.max(), np.ptp(x),
        np.mean(x ** 2), np.sum(np.abs(dx)), dx.mean(), dx.std(),
        len(peaks), rr.mean() if len(rr) else 0.0, rr.std() if len(rr) else 0.0,
    ]
    return np.asarray(feats, dtype=np.float32)


def pair_features(seg1, seg2, fs):
    f1 = extract_single_features(seg1, fs)
    f2 = extract_single_features(seg2, fs)
    corr = np.corrcoef(seg1, seg2)[0, 1] if np.std(seg1) > 1e-8 and np.std(seg2) > 1e-8 else 0.0
    dist_l2 = np.linalg.norm(seg1 - seg2)
    dist_l1 = np.abs(seg1 - seg2).mean()
    return np.concatenate([f1, f2, np.abs(f1 - f2), [corr, dist_l2, dist_l1]]).astype(np.float32)


def three_feature_checks(x1, x2):
    corr = float(np.corrcoef(x1, x2)[0, 1]) if np.std(x1) > 1e-8 and np.std(x2) > 1e-8 else 0.0
    mad = float(np.mean(np.abs(x1 - x2)))
    rms1 = float(np.sqrt(np.mean(x1 ** 2)))
    rms2 = float(np.sqrt(np.mean(x2 ** 2)))
    rms_ratio = float(min(rms1, rms2) / (max(rms1, rms2) + 1e-8))
    return pd.DataFrame([
        {"Feature check": "Waveform correlation", "Value": round(corr, 4), "Interpretation": "Higher means more similar shape"},
        {"Feature check": "Mean absolute difference", "Value": round(mad, 4), "Interpretation": "Lower means closer amplitudes"},
        {"Feature check": "RMS energy ratio", "Value": round(rms_ratio, 4), "Interpretation": "Closer to 1 means similar signal energy"},
    ])


def make_signal_plot(x1, x2):
    t = np.arange(len(x1)) / cfg.target_fs
    fig = plt.figure(figsize=(10, 3.5))
    plt.plot(t, x1, label="Segment 1", linewidth=1.5)
    plt.plot(t, x2, label="Segment 2", linewidth=1.2, alpha=0.8)
    plt.xlabel("Time (s)")
    plt.ylabel("Normalized amplitude")
    plt.title("Preprocessed ECG segments used by the models")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    return fig

# =========================
# 8. Prediction helpers
# =========================

MODEL_DISPLAY_ORDER = [
    "Handcrafted SVM baseline",
    "CNN baseline",
    "Siamese CNN-BiLSTM",
    "HDC baseline",
    "Vectorized HDC baseline",
    "PINN-only structured model",
    "FHN HDC-PINN hybrid",
    "FHN hybrid no physics",
    "Full McSharry hybrid",
    "McSharry hybrid no physics",
    "McSharry hybrid no HDC loss",
]


def predict_one_model(model_name, info, x1, x2, t1, t2, use_saved_thresholds=True, manual_threshold=0.5):
    obj = info.get("object")
    kind = info.get("kind")
    if obj is None:
        return {
            "Model": model_name,
            "Score": None,
            "Threshold": None,
            "Decision": "Unavailable",
            "RawScore": None,
            "Notes": info.get("notes", "Model did not load"),
        }

    threshold = get_saved_threshold(model_name, fallback=manual_threshold) if use_saved_thresholds else float(manual_threshold)

    try:
        if kind == "torch":
            with torch.no_grad():
                score = float(obj.predict_similarity(t1, t2).detach().cpu().item())
            raw_score = score
            notes = info.get("notes", "Torch model")

        elif kind == "hdc":
            raw_cosine = float(obj.similarity(x1, x2))
            score = (raw_cosine + 1.0) / 2.0
            raw_score = raw_cosine
            notes = info.get("notes", "HDC model") + f"; raw cosine={raw_cosine:.4f}"

        elif kind == "sklearn":
            feats = pair_features(x1, x2, cfg.target_fs).reshape(1, -1)
            if hasattr(obj, "predict_proba"):
                score = float(obj.predict_proba(feats)[0, 1])
            elif hasattr(obj, "decision_function"):
                raw = float(obj.decision_function(feats)[0])
                score = 1.0 / (1.0 + np.exp(-raw))
            else:
                score = float(obj.predict(feats)[0])
            raw_score = score
            notes = info.get("notes", "Sklearn model")

        else:
            raise ValueError(f"Unknown model kind: {kind}")

        decision = "MATCH / Same person" if score >= threshold else "NON-MATCH / Different person"
        return {
            "Model": model_name,
            "Score": round(float(score), 4),
            "Threshold": round(float(threshold), 4),
            "Decision": decision,
            "RawScore": round(float(raw_score), 4),
            "Notes": notes,
        }
    except Exception as e:
        return {
            "Model": model_name,
            "Score": None,
            "Threshold": round(float(threshold), 4),
            "Decision": "FAILED",
            "RawScore": None,
            "Notes": f"{type(e).__name__}: {e}",
        }


def predict_all_models(file1_path, file2_path, input_fs=250, threshold=0.5, use_saved_thresholds=True, apply_bandpass=False):
    if file1_path is None or file2_path is None:
        empty_results = pd.DataFrame(columns=["Model", "Score", "Threshold", "Decision", "RawScore", "Notes"])
        empty_features = pd.DataFrame(columns=["Feature check", "Value", "Interpretation"])
        return empty_results, empty_features, None

    raw1 = load_ecg_file(file1_path)
    raw2 = load_ecg_file(file2_path)

    x1 = preprocess_ecg(raw1, int(input_fs), apply_bandpass=apply_bandpass)
    x2 = preprocess_ecg(raw2, int(input_fs), apply_bandpass=apply_bandpass)

    t1 = torch.tensor(x1).float().unsqueeze(0).unsqueeze(0).to(DEVICE)
    t2 = torch.tensor(x2).float().unsqueeze(0).unsqueeze(0).to(DEVICE)

    rows = []
    for model_name in MODEL_DISPLAY_ORDER:
        if model_name in MODELS:
            rows.append(predict_one_model(model_name, MODELS[model_name], x1, x2, t1, t2, use_saved_thresholds, threshold))

    # Include any extra loaded models not in the display order.
    for model_name, info in MODELS.items():
        if model_name not in MODEL_DISPLAY_ORDER:
            rows.append(predict_one_model(model_name, info, x1, x2, t1, t2, use_saved_thresholds, threshold))

    results = pd.DataFrame(rows)
    features = three_feature_checks(x1, x2)
    fig = make_signal_plot(x1, x2)
    return results, features, fig

# =========================
# 10. Prepare result table previews for the app
# =========================

def result_table_options():
    return list(RESULT_TABLES.keys())


def show_result_table(name):
    if not name or name not in RESULT_TABLES:
        return pd.DataFrame({"Message": ["No result table selected or table unavailable."]})
    df = RESULT_TABLES[name].copy()
    return df

available_tables = result_table_options()
print("Available tables for app:", available_tables)


# =========================
# Demo examples
# =========================
def create_synthetic_examples():
    t = np.linspace(0, cfg.segment_seconds, SEG_LEN)
    same1 = np.sin(2 * np.pi * 3 * t) + 0.2 * np.sin(2 * np.pi * 9 * t)
    rng = np.random.default_rng(42)
    same2 = same1 + 0.05 * rng.normal(size=SEG_LEN)
    diff1 = np.sin(2 * np.pi * 5 * t + 0.8) + 0.4 * np.sin(2 * np.pi * 11 * t)
    np.savetxt(APP_ROOT / "synthetic_same_1.txt", same1)
    np.savetxt(APP_ROOT / "synthetic_same_2.txt", same2)
    np.savetxt(APP_ROOT / "synthetic_diff_1.txt", diff1)

create_synthetic_examples()



# =========================
# Gradio UI
# =========================

with gr.Blocks(title="ECG Biometrics Multi-Model Demo") as demo:
    gr.Markdown("""
    # ECG Biometric Verification Demo

    Upload two ECG segments and compare every supported saved model found in the artifact bundle.

    This demo supports the thesis artifact zip/folder and automatically loads available SVM, CNN, Siamese, HDC, PINN-only, FHN, and McSharry-family models.
    """)

    with gr.Accordion("Loaded artifact summary", open=True):
        gr.Dataframe(value=load_table, label="Loaded models", wrap=True)
        if available_tables:
            table_choice = gr.Dropdown(choices=available_tables, value=available_tables[0], label="Saved result table")
            table_out = gr.Dataframe(label="Saved result table preview", wrap=True)
            table_choice.change(fn=show_result_table, inputs=table_choice, outputs=table_out)
        else:
            gr.Markdown("No result CSV tables were found in the artifact folder.")

    with gr.Row():
        file1 = gr.File(label="Upload ECG segment 1 (.txt, .csv, .npy)", type="filepath")
        file2 = gr.File(label="Upload ECG segment 2 (.txt, .csv, .npy)", type="filepath")

    with gr.Row():
        fs = gr.Number(value=cfg.target_fs, label="Input sampling rate (Hz)")
        thresh = gr.Slider(0.0, 1.0, value=0.5, step=0.01, label="Manual threshold")

    with gr.Row():
        use_saved = gr.Checkbox(value=True, label="Use saved model-specific thresholds when available")
        bandpass = gr.Checkbox(value=False, label="Apply bandpass filter before demo inference")

    run_btn = gr.Button("Compare all loaded models", variant="primary")

    gr.Markdown("## Model comparison")
    results_out = gr.Dataframe(label="Scores and decisions", wrap=True)

    gr.Markdown("## Three ECG feature checks")
    feature_out = gr.Dataframe(label="Interpretable signal checks", wrap=True)

    gr.Markdown("## Preprocessed ECG segments")
    plot_out = gr.Plot(label="Model input signals")

    run_btn.click(
        fn=predict_all_models,
        inputs=[file1, file2, fs, thresh, use_saved, bandpass],
        outputs=[results_out, feature_out, plot_out],
    )

    gr.Examples(
        examples=[
            ["synthetic_same_1.txt", "synthetic_same_2.txt", cfg.target_fs, 0.5, True, False],
            ["synthetic_same_1.txt", "synthetic_diff_1.txt", cfg.target_fs, 0.5, True, False],
        ],
        inputs=[file1, file2, fs, thresh, use_saved, bandpass],
    )

    gr.Markdown("""
    **Demo notes**

    - Torch models output sigmoid scores in `[0, 1]`.
    - HDC models produce cosine similarity internally; this app maps it with `(cosine + 1) / 2` for a comparable display score.
    - Saved model-specific thresholds come from `thresholds_and_metrics.json` when that file exists. Otherwise the manual slider is used.
    - Thesis metrics should be reported from the saved result CSV files, not from synthetic demo signals.
    """)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)), share=False)
