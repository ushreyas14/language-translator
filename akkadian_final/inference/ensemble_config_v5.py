# ============================================================================
# ENSEMBLE CONFIG CHANGES FOR 3-MODEL (V5 INTEGRATED)
# ============================================================================
# Copy this EnsembleConfig into your akkadian-35 notebook, replacing the old one.
#
# Changes vs the "fast" version:
#   - extra_model_paths: now includes your V5 merged model
#   - num_beams: 1 → 4 (moderate beam search)
#   - num_beam_cands: 1 → 2
#   - num_sample_cands: 0 → 1
#   - batch_size: 8 → 4
#   - max_new_tokens: 256 → 384
#   - agreement_bonus: 0.035 → 0.05
#   - use_adaptive_beams: False → True
#
# Total candidates per sample: 3 models × (2 beam + 1 sample) = 9 candidates
# ============================================================================

@dataclass
class EnsembleConfig:
    test_data_path: str = "/kaggle/input/competitions/deep-past-initiative-machine-translation/test.csv"
    output_dir: str = "/kaggle/working/"
    model_a_path: str = "/kaggle/input/datasets/jeenilmakwana/akkadian-33/byt5-akkadian-optimized-34x"
    model_b_path: str = "/kaggle/input/models/mattiaangeli/byt5-akkadian-mbr-v2/pytorch/default/1"

    # ---- YOUR V5 MODEL: update this path after uploading to Kaggle ----
    extra_model_paths: Tuple[str, ...] = (
        "/kaggle/input/datasets/ushreyas14/akkadian-v5-merged/akkadian_v5_merged",
    )

    max_input_length: int = 512
    max_new_tokens: int = 384          # restored from 256
    batch_size: int = 4                # was 8 (need room for beam search)
    num_workers: int = 2
    num_buckets: int = 6

    # ---- 3-MODEL ENSEMBLE: moderate beam search ----
    num_beam_cands: int = 2            # was 1 — 2 beam candidates per model
    num_beams: int = 4                 # was 1 — 4-beam search
    length_penalty: float = 1.3
    early_stopping: bool = True
    repetition_penalty: float = 1.2

    # ---- 1 stochastic sample per model ----
    num_sample_cands: int = 1          # was 0 — adds diversity to MBR pool
    mbr_top_p: float = 0.92
    mbr_temperature: float = 0.75
    mbr_pool_cap: int = 36

    # ---- Stronger consensus with 3 models ----
    agreement_bonus: float = 0.05      # was 0.035
    use_competition_utility: bool = True

    use_mixed_precision: bool = True
    use_better_transformer: bool = True
    use_bucket_batching: bool = True
    use_adaptive_beams: bool = True    # was False — re-enabled
    checkpoint_freq: int = 200
    seed: int = 42

    def __post_init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        if not torch.cuda.is_available():
            self.use_mixed_precision = False
            self.use_better_transformer = False
        self.use_bf16_amp = bool(
            self.use_mixed_precision
            and self.device.type == "cuda"
            and cuda_bf16_supported()
        )
        if self.num_beams < self.num_beam_cands:
            raise ValueError("num_beams must be >= num_beam_cands")
