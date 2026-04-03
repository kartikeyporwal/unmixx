# [ICASSP 2026] Official code for [UNMIXX: Untangling Highly Correlated Singing Voices Mixtures](https://www.arxiv.org/abs/2601.12802)

**UNMIXX** a novel framework for the multiple singing voices separation (MSVS). While similar to speech separation, MSVS presents unique challenges, namely data scarcity and highly correlated nature of singing voice. To address these issues, we propose  three key components: (1) a musically informed mixing strategy to construct highly correlated training mixtures, (2) a reverse attention that drives the two outputs apart using cross attention and (3) a magnitude penalty loss penalizing energy erroneously assigned to the other output. Experiments show that UNMIXX achieves substantial improvements, with more than ~2.2 dB SDRi gains on MedleyVox evaluation set over prior method. Audio samples are available on our [demo page](https://unmixx.github.io/).


---

## Quickstart

### 1) Data Preprocessing
We use a total of **400 hours** of singing datasets for training.  
Download the datasets and follow the preprocessing steps in [MedleyVox preprocessing steps](https://github.com/jeonchangbin49/MedleyVox/tree/main/svs/preprocess).

- [Children’s song dataset (CSD)](https://zenodo.org/record/4785016#.Y2-r2y_kFqs) — 4.9 hours  
- [NUS](https://drive.google.com/drive/folders/12pP9uUl0HTVANU3IPLnumTJiRjPtVUMx) — 1.9 hours  
- [VocalSet](https://zenodo.org/record/1193957#.Y2-tmC_kFqs) — 8.8 hours  
- [Jsut-song](https://sites.google.com/site/shinnosuketakamichi/publication/jsut-song) — 0.4 hours  
- [Jvs_music](https://sites.google.com/site/shinnosuketakamichi/research-topics/jvs_music) — 2.3 hours  
- [Musdb-hq (train subset)](https://sigsep.github.io/datasets/musdb.html) — 2.0 hours  
  - Single singing regions extracted using [musdb-lyrics extension](https://zenodo.org/record/3989267#.Y2-wBOzP1qs)  
- [OpenSinger](https://github.com/Multi-Singer/Multi-Singer.github.io) — 51.9 hours  
- [K_multisinger](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=465) — 169.6 hours  
- [K_multitimbre](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=473) — 150.8 hours  



### 2) Training

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python audio_train.py --conf_dir configs/unmixx.yml
```

---

### 3) Inference
```bash
python inference.py \
  --conf_path ckpt/conf.yml \
  --ckpt_path ckpt/best.ckpt \
  --audio_path sample_music/free_mixture.wav \
  --output_dir separated_audio
```

Outputs will be saved to `separated_audio/`.

#### Inference options

| Flag | Default | Description |
|---|---|---|
| `--chunk_len` | `30.0` | Chunk length in seconds. Reduce if OOM; avoid going below ~15s. |
| `--overlap` | `1.0` | Context overlap in seconds on each side of a chunk. |
| `--mix_strategy` | `auto` | How to downmix multi-channel input: `auto`, `mean`, `left`, `right`. `auto` measures L/R correlation and warns when channels carry different content. |
| `--silence_gate_db` | `-40.0` | Zero out streams more than N dB below the mixture (e.g. absent speaker). Set to `-inf` to disable. |
| `--wiener` | off | Enable Wiener post-filtering. Sharpens good separations; do not use when separation quality is already poor. |
| `--device` | `auto` | `auto`, `cpu`, or `cuda`. |

> **Note:** This model separates exactly **2 speakers/singers**. It was trained exclusively on singing voice mixtures.

---

## Changes & Optimizations (branch: `optimizations`)

### Performance
- **Hann window cached as buffer** — `torch.hann_window()` was recreated on every STFT/iSTFT call. Now registered once as a model buffer.
- **Skip redundant STFT at inference** — magnitude/phase outputs are only needed during training loss computation. Skipped when `istest=True`.
- **Zero accumulator cleanup** — removed spurious `requires_grad=True` on a temporary accumulator tensor in `UConvBlock`.

### Inference correctness
- **Stereo input downmixed to mono** — the model's energy-correction formula assumes single-channel input; stereo produced garbled output. Multi-channel audio is now averaged to mono before separation.
- **L/R correlation check (`--mix_strategy auto`)** — for stereo inputs, reports inter-channel correlation so users know whether to try `--mix_strategy left` or `right` for better results.
- **LUFS normalisation** — input is normalised to −24 LUFS before inference (matching training conditions) and the gain is inverted on the outputs.
- **Checkpoint loading robustness** — `load_state_dict_safely` now backfills buffers absent from older checkpoints (e.g. `_hann_window`) and unconditionally strips the `audio_model.` Lightning prefix.
- **`torch.load` FutureWarning suppressed** — `weights_only=False` set explicitly.

### Long-audio / chunked inference
- **Speaker identity alignment via overlap region** — chunks share a 1-second overlap that both process independently. Alignment compares this identical audio in the **spectral magnitude domain** (phase-invariant) rather than raw waveforms at the boundary, preventing speaker identity flips for 10+ minute files.
- **Silence gating** — streams more than `--silence_gate_db` dB below the mixture are zeroed out, preventing noise fill when one speaker is absent.
- **Progress bar + ETA** — chunked inference prints a live progress bar with per-chunk timing and estimated time remaining.

### Optional post-processing
- **Wiener post-filtering (`--wiener`)** — uses the model's initial power estimates to build sharper Wiener masks applied to the original mixture STFT. Reduces residual bleed when the initial separation is already directionally correct. Disabled by default because it collapses outputs when separation quality is low.

---

#### Implementation references:
- TIGER: https://github.com/JusperLee/TIGER
- MedleyVox: https://github.com/jeonchangbin49/MedleyVox/tree/main
