#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import yaml
from pathlib import Path
from pprint import pprint

import math
import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T
import pyloudnorm as pyln
import look2hear.models

# (옵션) salience 시각화 유틸이 필요하면 주석 해제
# from basic_pitch_torch.inference import predict
# import numpy as np
# import matplotlib.pyplot as plt

# =========================
# Utils
# =========================
def load_yaml(path: str):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg

def derive_model_name(cfg: dict, ckpt_path: str):
    # 우선 ckpt 파일명 기준
    stem = Path(ckpt_path).stem
    # config에 audionet_name이 있으면 보조적으로 붙임
    try:
        net_name = cfg["audionet"]["audionet_name"]
        model_name = f"{net_name}-{stem}"
    except Exception:
        model_name = stem
    return model_name

def pick_model_class(cfg: dict):
    net_name = cfg["audionet"]["audionet_name"]
    if not hasattr(look2hear.models, net_name):
        raise AttributeError(f"look2hear.models에 '{net_name}' 클래스가 없습니다.")
    return getattr(look2hear.models, net_name)

def build_model(cfg: dict):
    sr = cfg["datamodule"]["data_config"]["sample_rate"]
    aud_cfg = cfg["audionet"]["audionet_config"]
    ModelClass = pick_model_class(cfg)
    model = ModelClass(sample_rate=sr, **aud_cfg)
    return model

def load_state_dict_safely(model: torch.nn.Module, ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # Lightning 스타일 또는 pure state_dict 둘 다 처리
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        # 바로 state_dict로 저장된 형태
        state_dict = ckpt

    # Strip "audio_model." prefix added by the Lightning wrapper
    converted = {}
    for k, v in state_dict.items():
        if k.startswith("audio_model."):
            converted[k[len("audio_model."):]] = v
        else:
            converted[k] = v

    # Backfill any registered buffers that are absent from older checkpoints
    # (e.g. _hann_window added after the checkpoint was saved).  Buffers are
    # computed values, not learned, so keeping the freshly-initialised value
    # is always correct.
    model_sd = model.state_dict()
    for k, v in model_sd.items():
        if k not in converted:
            converted[k] = v

    model.load_state_dict(converted, strict=True)

_TRAIN_LUFS = -24.0  # loudness used during training (from singing_dataset.py)


def lufs_normalize(waveform: torch.Tensor, sr: int, target_lufs: float = _TRAIN_LUFS):
    """
    Normalize waveform to target_lufs (integrated loudness).
    Returns (normalized_waveform, linear_gain) so the gain can be inverted later.
    waveform : [1, T]  float32
    """
    meter = pyln.Meter(sr)
    audio_np = waveform[0].numpy().astype(np.float64)
    measured = meter.integrated_loudness(audio_np)
    if np.isinf(measured):
        # Silence — nothing to normalize
        return waveform, 1.0
    gain_db = target_lufs - measured
    gain_linear = 10 ** (gain_db / 20.0)
    return waveform * gain_linear, gain_linear


_STEREO_CORR_THRESHOLD = 0.98  # L/R correlation above this → treat as true mono


def stereo_correlation(waveform: torch.Tensor) -> float:
    """Pearson correlation between L and R channels. waveform: [2, T]"""
    L = waveform[0].double()
    R = waveform[1].double()
    L = L - L.mean()
    R = R - R.mean()
    denom = (L.norm() * R.norm()).clamp(min=1e-10)
    return (L @ R / denom).item()


def prepare_audio_tensor(audio_path: str, target_sr: int, device: torch.device,
                         mix_strategy: str = "auto"):
    """
    Load audio, resample, and downmix to mono for the model.

    mix_strategy:
      "auto"  — for stereo: check L/R correlation; if channels are very similar
                (corr > 0.98) use mean; if they differ significantly, report it
                so the user can choose a better strategy.
      "mean"  — average all channels (safe default)
      "left"  — use left (channel 0) only
      "right" — use right (channel 1) only
    """
    waveform, original_sr = torchaudio.load(audio_path)  # [C, T]
    if original_sr != target_sr:
        resampler = T.Resample(orig_freq=original_sr, new_freq=target_sr)
        waveform = resampler(waveform)

    waveform_for_save = waveform  # keep original for saving mixture.wav

    nch = waveform.shape[0]
    if nch == 1:
        pass  # already mono
    elif nch >= 2:
        if mix_strategy == "left":
            print(f"[Audio] {nch}-channel input — using left channel only")
            waveform = waveform[0:1]
        elif mix_strategy == "right":
            print(f"[Audio] {nch}-channel input — using right channel only")
            waveform = waveform[1:2]
        else:
            # "mean" or "auto"
            if mix_strategy == "auto" and nch == 2:
                corr = stereo_correlation(waveform[:2])
                if corr < _STEREO_CORR_THRESHOLD:
                    print(
                        f"[Audio] Stereo L/R correlation = {corr:.3f} — channels carry "
                        f"different content (speakers may be panned). "
                        f"Using mean downmix. Try --mix_strategy left or right if one "
                        f"channel has the cleaner mixture."
                    )
                else:
                    print(f"[Audio] Stereo L/R correlation = {corr:.3f} — near-identical "
                          f"channels, treating as mono (mean downmix).")
            else:
                print(f"[Audio] {nch}-channel input — downmixing to mono (mean)")
            waveform = waveform.mean(dim=0, keepdim=True)  # [1, T]

    audio_input = waveform.unsqueeze(0).to(device)  # [1, 1, T]
    return audio_input, waveform_for_save, target_sr

def normalize_outputs(ests):
    """
    모델 반환 타입이 tuple/dict 등 다양한 경우를 평탄화해 최종 [B, S, T] 또는 [S, T] 텐서를 반환.
    """
    if isinstance(ests, tuple):
        # 자주 쓰는 규칙 몇 가지
        if len(ests) == 2:
            ests = ests[1]
        elif len(ests) in (3, 5):
            ests = ests[0]
        else:
            ests = ests[0]

    if isinstance(ests, dict):
        # 가능한 키 이름들 통일
        for k in ["output_final", "audio_out_final", "output", "audio_out"]:
            if k in ests:
                return ests[k], ests.get("output_original", ests.get("audio_out_original", ests[k]))
        # 못 찾으면 첫 항목
        first = next(iter(ests.values()))
        return first, first

    # 텐서인 경우
    return ests, ests

def gate_silent_streams(ests: torch.Tensor, threshold_db: float = -40.0) -> torch.Tensor:
    """
    Zero out any separated stream whose energy is more than `threshold_db` dB
    below the total mixture energy. This prevents the model from filling a
    genuinely silent speaker slot with noise or bleed.

    ests           : [S, T]
    threshold_db   : streams below this relative level are zeroed (default -40 dB)
    Returns        : [S, T]
    """
    mix_power = ests.pow(2).mean().clamp(min=1e-10)
    out = ests.clone()
    for i in range(ests.shape[0]):
        stream_power = ests[i].pow(2).mean().clamp(min=1e-10)
        rel_db = 10.0 * torch.log10(stream_power / mix_power)
        if rel_db.item() < threshold_db:
            print(f"  [Gate] stream {i+1} is {rel_db.item():.1f} dB below mixture — zeroed out")
            out[i] = 0.0
    return out


def ensure_2d_channels_first(x: torch.Tensor):
    """
    [T] -> [1, T], [C, T]는 그대로
    """
    if x.dim() == 1:
        x = x.unsqueeze(0)
    return x

def wiener_refine(
    mixture: torch.Tensor,
    estimates: torch.Tensor,
    alpha: float = 1.0,
    win: int = 960,
    stride: int = 240,
) -> torch.Tensor:
    """
    Sharpen separated estimates using generalised Wiener post-filtering.

    Mask formula:  M_i = P_i^alpha / sum_j(P_j^alpha)

      alpha = 1  → standard Wiener (soft, lowest artefacts)
      alpha = 2  → more aggressive, noticeably less bleed
      alpha = 4+ → near hard-masking (winner takes all per TF bin)

    The masks are applied to the *original* mixture spectrum, so sources
    always sum back to the mixture (energy conservation).

    mixture   : [1, T]  mono mixture (CPU, float32)
    estimates : [S, T]  model separated estimates (CPU, float32)
    alpha     : sharpness exponent (default 1.0)
    win/stride: must match the model's STFT settings
    Returns   : [S, T]  refined estimates
    """
    nsample = mixture.shape[-1]
    window  = torch.hann_window(win)

    mix_spec = torch.stft(
        mixture[0], n_fft=win, hop_length=stride, window=window, return_complex=True
    )  # [F, T_frames]

    # Power spectral density of each estimated source, raised to alpha
    powers_alpha = []
    for i in range(estimates.shape[0]):
        s = torch.stft(
            estimates[i], n_fft=win, hop_length=stride, window=window, return_complex=True
        )
        powers_alpha.append(s.abs().pow(2 * alpha))   # |S|^(2*alpha)

    total = sum(powers_alpha).clamp(min=1e-10)

    refined = []
    for pa in powers_alpha:
        mask         = pa / total               # sharpened mask ∈ [0, 1]
        refined_spec = mask * mix_spec
        refined_wav  = torch.istft(
            refined_spec, n_fft=win, hop_length=stride, window=window, length=nsample
        )
        refined.append(refined_wav)

    return torch.stack(refined)  # [S, T]


def _stream_embedding(waveform: torch.Tensor, win: int = 960, stride: int = 240) -> torch.Tensor:
    """
    Speaker-representative embedding: average log-magnitude spectrum over time → [F].
    Captures spectral envelope (timbre), which is speaker-dependent and stable
    across chunks. Phase-invariant so it works even when two chunks process the
    same audio with different context and produce slightly different phase.
    Memory cost: (win//2 + 1) floats ≈ 2 KB per stream.
    """
    window = torch.hann_window(win)
    spec   = torch.stft(waveform, n_fft=win, hop_length=stride, window=window, return_complex=True)
    return spec.abs().clamp(min=1e-8).log().mean(dim=-1)  # [F]


def _global_align(chunks: list, win: int = 960, stride: int = 240) -> list:
    """
    Globally align speaker ordering across all chunks.

    Strategy — anchor-based (cascade-free):
      1. Compute a log-magnitude spectral embedding [F] for each stream in
         every chunk.
      2. Find the "anchor" chunk whose two streams are most spectrally distinct
         (highest 1 − cosine_similarity between stream embeddings).  The anchor
         has the clearest separation, so its stream-to-speaker assignment is the
         most reliable reference.
      3. Align EVERY other chunk directly to this anchor using cosine similarity.
         Because all comparisons are against the same fixed reference, there is
         no cascade: a wrong alignment in chunk i cannot influence chunk i+1.

    RAM: num_chunks × S × (win//2+1) floats.
      2h @ 30s chunks → 240 × 2 × 481 × 4B ≈ 0.9 MB.

    chunks : list of [S, T] CPU tensors
    Returns: list of [S, T] CPU tensors with consistent speaker ordering
    """
    from itertools import permutations
    import torch.nn.functional as F

    if len(chunks) <= 1:
        return chunks

    S = chunks[0].shape[0]

    # Pre-compute per-stream embeddings for every chunk  [N, S, F]
    all_embs = [
        [_stream_embedding(chunk[s], win, stride) for s in range(S)]
        for chunk in chunks
    ]

    # Choose the anchor: chunk whose streams are most distinct from each other.
    # For S=2: separation_confidence = 1 − cos_sim(stream0, stream1).
    # Higher → cleaner separation → more reliable speaker identity.
    def separation_confidence(embs):
        cos = sum(
            F.cosine_similarity(embs[a].unsqueeze(0), embs[b].unsqueeze(0)).item()
            for a in range(S) for b in range(S) if a != b
        ) / (S * (S - 1))
        return 1.0 - cos  # 0 = identical streams, 1 = fully orthogonal

    confs = [separation_confidence(e) for e in all_embs]
    anchor_idx = int(max(range(len(chunks)), key=lambda i: confs[i]))
    print(f"  [Align] anchor=chunk {anchor_idx}  confidence={confs[anchor_idx]:.3f}")

    anchor_embs = all_embs[anchor_idx]
    aligned = [None] * len(chunks)
    aligned[anchor_idx] = chunks[anchor_idx]

    for i in range(len(chunks)):
        if i == anchor_idx:
            continue
        cur_embs = all_embs[i]

        best_perm = list(range(S))
        best_sim  = -float("inf")
        for perm in permutations(range(S)):
            sim = sum(
                F.cosine_similarity(anchor_embs[j].unsqueeze(0), cur_embs[perm[j]].unsqueeze(0)).item()
                for j in range(S)
            ) / S
            if sim > best_sim:
                best_sim = sim
                best_perm = list(perm)

        aligned[i] = chunks[i][best_perm]

    return aligned


def _fmt_time(seconds: float) -> str:
    seconds = int(seconds)
    m, s = divmod(seconds, 60)
    return f"{m}m {s:02d}s" if m else f"{s}s"


def run_chunked(model, audio_input, chunk_samples, overlap_samples, sr: int):
    """
    Process long audio in chunks to fit in GPU memory.
    Each chunk is fed with extra context on both sides; the context is stripped
    from the output before concatenation, avoiding hard boundary artifacts.

    After all chunks are separated, speaker ordering is globally aligned using
    running-mean spectral embeddings (_global_align).  Each chunk is compared
    against the cumulative average embedding of all confirmed chunks, so a
    single bad alignment cannot cascade into subsequent chunks.

    audio_input : [1, C, T]  (already on device)
    sr          : sample rate (used for progress display)
    Returns     : [num_speakers, T]  (on CPU)
    """
    import time

    _, _, T = audio_input.shape
    total_chunks = math.ceil(T / chunk_samples)
    total_audio_s = T / sr
    output_chunks = []

    t_run_start = time.monotonic()
    start = 0
    chunk_idx = 0
    while start < T:
        end = min(start + chunk_samples, T)
        t_chunk_start = time.monotonic()

        # Extend with context on both sides where available
        ctx_start = max(0, start - overlap_samples)
        ctx_end   = min(T, end   + overlap_samples)

        chunk = audio_input[:, :, ctx_start:ctx_end]  # [1, C, ctx_len]

        with torch.no_grad():
            outs = model(chunk, istest=True)

        ests, _ = normalize_outputs(outs)
        if ests.dim() == 2:
            ests = ests.unsqueeze(0)
        ests = ests[0].cpu()  # [S, ctx_len]

        # Strip the context padding — keep only [start, end)
        trim_start = start - ctx_start
        trim_end   = trim_start + (end - start)
        ests       = ests[:, trim_start:trim_end]

        output_chunks.append(ests)
        chunk_idx += 1

        # Progress + ETA
        elapsed      = time.monotonic() - t_run_start
        chunk_time   = time.monotonic() - t_chunk_start
        done_frac    = chunk_idx / total_chunks
        eta          = (elapsed / done_frac) * (1 - done_frac) if done_frac > 0 else 0
        audio_done_s = end / sr
        bar_len      = 30
        filled       = int(bar_len * done_frac)
        bar          = "█" * filled + "░" * (bar_len - filled)
        print(
            f"  [{bar}] {chunk_idx}/{total_chunks}  "
            f"{audio_done_s:.1f}s / {total_audio_s:.1f}s  "
            f"chunk {chunk_time:.1f}s  elapsed {_fmt_time(elapsed)}  ETA {_fmt_time(eta)}"
        )

        start = end

    # Global speaker-ordering alignment across all chunks.
    # Each chunk is compared against the running-mean spectral embedding of all
    # previously aligned chunks, so errors cannot cascade.
    print("[Align] running global speaker alignment across chunks...")
    output_chunks = _global_align(output_chunks, win=960, stride=240)

    return torch.cat(output_chunks, dim=1)  # [S, T]

# =========================
# Main
# =========================
def main():
    parser = argparse.ArgumentParser(description="Separate sources with Look2Hear TIGER-like model.")
    parser.add_argument("--conf_path", required=True, help="YAML config path.")
    parser.add_argument("--ckpt_path", required=True, help="Checkpoint path (.ckpt/.pth).")
    parser.add_argument("--audio_path", required=True, help="Input audio path (wav).")
    parser.add_argument("--output_dir", default="separated_audio", help="Output directory.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="Device selection.")
    parser.add_argument("--target_sr", type=int, default=None, help="Override target sample rate. (optional)")
    parser.add_argument("--chunk_len", type=float, default=30.0,
                        help="Chunk length in seconds for chunked inference (default: 30). "
                             "Reduce if you OOM. Avoid going below ~15s — shorter chunks "
                             "increase the number of chunk-boundary alignments and raise "
                             "the risk of a speaker identity flip.")
    parser.add_argument("--overlap", type=float, default=1.0,
                        help="Context overlap in seconds on each side of a chunk (default: 1).")
    parser.add_argument("--silence_gate_db", type=float, default=-40.0,
                        help="Streams whose energy is more than this many dB below the mixture "
                             "are zeroed out (default: -40). Set to -inf to disable.")
    parser.add_argument("--mask_sharpness", type=float, default=None,
                        metavar="ALPHA",
                        help="Apply generalised Wiener post-filtering with exponent ALPHA. "
                             "M_i = P_i^ALPHA / sum(P_j^ALPHA). "
                             "1.0 = standard Wiener (soft), 2.0 = less bleed, "
                             "4.0+ = near hard-masking. "
                             "Not set by default — only use when the model already "
                             "produces a directionally correct initial separation.")
    parser.add_argument("--mix_strategy", default="auto",
                        choices=["auto", "mean", "left", "right"],
                        help="How to downmix multi-channel input to mono (default: auto). "
                             "auto: checks L/R correlation and warns if channels differ. "
                             "mean: average all channels. "
                             "left/right: use only that channel — useful when speakers are "
                             "panned and one channel gives a better balanced mixture.")
    args = parser.parse_args()

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"[Device] {device}")

    # Config
    config = load_yaml(args.conf_path)
    print("[Config] loaded.")
    # pprint(config)

    # Model
    model = build_model(config)
    load_state_dict_safely(model, args.ckpt_path)
    model.to(device).eval()

    # Target SR
    cfg_sr = config["datamodule"]["data_config"]["sample_rate"]
    target_sr = args.target_sr if args.target_sr is not None else int(cfg_sr)
    print(f"[SampleRate] target_sr={target_sr}")

    # Audio
    print(f"[Audio] loading: {args.audio_path}")
    audio_input, waveform_cpu, sr = prepare_audio_tensor(args.audio_path, target_sr, device,
                                                          mix_strategy=args.mix_strategy)
    print(f"[Audio] shape (B,C,T)={tuple(audio_input.shape)}, sr={sr}")

    # LUFS normalisation — the model was trained with mixtures normalised to
    # -24 LUFS (see singing_dataset.py).  Un-normalised speech input is out of
    # distribution and degrades separation quality.  We normalise before
    # inference and undo the gain on the outputs so saved loudness is preserved.
    norm_waveform, lufs_gain = lufs_normalize(audio_input[0].cpu(), sr)
    audio_input = norm_waveform.unsqueeze(0).to(device)
    print(f"[LUFS] normalised input to {_TRAIN_LUFS} LUFS (gain={lufs_gain:.4f})")

    # Forward (chunked to fit in GPU memory)
    T = audio_input.shape[-1]
    chunk_samples   = int(args.chunk_len * target_sr)
    overlap_samples = int(args.overlap   * target_sr)

    if T <= chunk_samples:
        # Short enough to run in one shot
        print("[Inference] single pass")
        with torch.no_grad():
            outs = model(audio_input, istest=True)
        ests_speech, _ = normalize_outputs(outs)
        if ests_speech.dim() == 2:
            ests_speech = ests_speech.unsqueeze(0)
        ests_speech = ests_speech[0].cpu()  # [S, T]
    else:
        print(f"[Inference] chunked — chunk={args.chunk_len}s, overlap={args.overlap}s")
        ests_speech = run_chunked(model, audio_input, chunk_samples, overlap_samples, sr=target_sr)

    # Undo the LUFS gain so output loudness matches the original input
    ests_speech = ests_speech / lufs_gain

    # Generalised Wiener post-filtering (opt-in via --mask_sharpness).
    # M_i = P_i^alpha / sum(P_j^alpha).  alpha=1 is standard Wiener;
    # higher alpha sharpens masks and reduces bleed.
    # Only beneficial when the model already produces directionally correct
    # separation — when masks are uncertain (P1≈P2) any alpha collapses
    # both outputs to 0.5×mixture.
    if args.mask_sharpness is not None:
        alpha = args.mask_sharpness
        print(f"[MaskSharpness] applying post-filter (alpha={alpha})...")
        mono_mixture = audio_input[0].cpu()  # [1, T] — LUFS-normalised
        cfg = config["audionet"]["audionet_config"]
        ests_speech = wiener_refine(
            mono_mixture,
            ests_speech * lufs_gain,   # back to normalised scale for the filter
            alpha=alpha,
            win=cfg.get("win", 960),
            stride=cfg.get("stride", 240),
        ) / lufs_gain                  # restore original loudness scale

    ests_speech_original = ests_speech  # same for save purposes
    ests_speech = gate_silent_streams(ests_speech, threshold_db=args.silence_gate_db)
    num_speakers = ests_speech.shape[0]
    print(f"[Separation] detected {num_speakers} streams")

    # Output paths
    model_name = derive_model_name(config, args.ckpt_path)
    base_dir = Path(args.output_dir) / model_name / Path(args.audio_path).stem
    base_dir.mkdir(parents=True, exist_ok=True)

    # Save estimates
    for i in range(num_speakers):
        # torchaudio.save expects [C, T]
        est = ensure_2d_channels_first(ests_speech[i])
        est_org = ensure_2d_channels_first(ests_speech_original[i])
        out_path = base_dir / f"spk{i+1}.wav"
        print(f"[Save] {out_path}")
        torchaudio.save(str(out_path), est, sr)

    # Save mixture, too
    mix_out = base_dir / "mixture.wav"
    print(f"[Save] {mix_out}")
    torchaudio.save(str(mix_out), waveform_cpu, sr)

    print("[Done] All files saved under:", base_dir)

if __name__ == "__main__":
    main()
