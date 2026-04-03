#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import yaml
from pathlib import Path
from pprint import pprint

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


def prepare_audio_tensor(audio_path: str, target_sr: int, device: torch.device):
    waveform, original_sr = torchaudio.load(audio_path)  # [C, T]
    if original_sr != target_sr:
        resampler = T.Resample(orig_freq=original_sr, new_freq=target_sr)
        waveform = resampler(waveform)
    # Downmix to mono. The model was trained on mono and its internal energy-
    # correction formula assumes nch=1; passing stereo produces garbage output.
    # We preserve the original stereo waveform for saving the mixture file.
    waveform_for_save = waveform
    if waveform.shape[0] > 1:
        print(f"[Audio] {waveform.shape[0]}-channel input — downmixing to mono")
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

def _align_to_previous(
    prev: torch.Tensor,
    cur: torch.Tensor,
    compare_samples: int = 24000,
    min_confidence: float = 0.01,
) -> torch.Tensor:
    """
    Ensure speaker ordering in `cur` is consistent with `prev` by comparing
    the tail of `prev` against the head of `cur` for every permutation.
    Returns `cur` with speakers reordered if needed.

    prev, cur         : [S, T]  on CPU
    compare_samples   : how many boundary samples to compare (default: 24000 = 1s at 24kHz)
    min_confidence    : minimum difference in similarity scores to commit to a flip.
                        If both orderings look equally bad (e.g. silence at boundary)
                        we keep the current ordering rather than flipping randomly.
    """
    n = min(compare_samples, prev.shape[-1], cur.shape[-1])
    ref = prev[:, -n:]   # [S, n]
    cand = cur[:, :n]    # [S, n]
    S = prev.shape[0]

    from itertools import permutations
    scores = {}
    for perm in permutations(range(S)):
        scores[perm] = sum(
            torch.nn.functional.cosine_similarity(ref[i].unsqueeze(0), cand[perm[i]].unsqueeze(0)).item()
            for i in range(S)
        ) / S

    best_perm = max(scores, key=scores.__getitem__)
    identity  = tuple(range(S))
    # Only reorder if the best permutation is meaningfully better than identity
    if best_perm != identity and (scores[best_perm] - scores[identity]) < min_confidence:
        return cur  # not confident enough — keep current ordering
    return cur[list(best_perm)]


def run_chunked(model, audio_input, chunk_samples, overlap_samples):
    """
    Process long audio in chunks to fit in GPU memory.
    Each chunk is fed with extra context on both sides; the context is stripped
    from the output before concatenation, avoiding hard boundary artifacts.

    After separation, each chunk's speaker ordering is aligned to the previous
    chunk using cosine similarity so that speaker identities do not flip between
    chunks.

    audio_input : [1, C, T]  (already on device)
    Returns     : [num_speakers, T]  (on CPU)
    """
    _, _, T = audio_input.shape
    output_chunks = []

    start = 0
    chunk_idx = 0
    while start < T:
        end = min(start + chunk_samples, T)

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

        # Strip the context padding to get only [start, end)
        trim_start = start - ctx_start
        trim_end   = trim_start + (end - start)
        ests = ests[:, trim_start:trim_end]

        # Align speaker ordering to the previous chunk to prevent identity flips.
        # Use the full overlap window as the comparison region — much more
        # reliable than a short boundary snippet, especially when there is
        # silence or a rapid speech transition right at the boundary.
        if output_chunks:
            ests = _align_to_previous(output_chunks[-1], ests, compare_samples=overlap_samples)

        output_chunks.append(ests)
        chunk_idx += 1
        print(f"  chunk {chunk_idx}: samples [{start}, {end}) / {T}")

        start = end

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
    audio_input, waveform_cpu, sr = prepare_audio_tensor(args.audio_path, target_sr, device)
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
        ests_speech = run_chunked(model, audio_input, chunk_samples, overlap_samples)

    # Undo the LUFS gain so output loudness matches the original input
    ests_speech = ests_speech / lufs_gain

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
