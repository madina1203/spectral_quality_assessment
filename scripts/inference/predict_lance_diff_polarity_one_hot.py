import os
import sys
import argparse
import logging
import numpy as np
import torch
import lance
import pandas as pd
import lightning.pytorch as pl
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import multiprocessing as mp

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
try:
    from src.transformers.model_bce_loss_one_hot import SimpleSpectraTransformer
except ImportError:
    print("ERROR: Could not import SimpleSpectraTransformer.")
    print(
        "Please run this script from your project's root directory or ensure 'src' is in your PYTHONPATH."
    )
    sys.exit(1)


# --- Logging Setup ---
RANK = os.environ.get("LOCAL_RANK", os.environ.get("SLURM_PROCID", "0"))
logging.basicConfig(
    level=logging.INFO,
    format=f"[Rank {RANK}] %(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
script_logger = logging.getLogger(__name__)


class LanceIndexDataset(Dataset):
    """
    Dataset that filters indices based ONLY on OHE Polarity.
    """

    def __init__(self, lance_dataset_path: str, polarity: int = None, rank: str = "0"):
        super().__init__()
        self.lance_dataset_path = lance_dataset_path
        if not os.path.exists(lance_dataset_path):
            raise FileNotFoundError(f"Lance dataset not found at: {lance_dataset_path}")

        ds = lance.dataset(self.lance_dataset_path)

        # Removed 'dataset_id' from columns to reduce I/O overhead
        scanner = ds.scanner(columns=["instrument_settings"])
        batch_reader = scanner.to_batches()

        valid_indices = []
        current_offset = 0

        # OHE Mapping: Index 14: Negative [1, 0], Index 15: Positive [0, 1]
        POLARITY_NEG_IDX = 14
        POLARITY_POS_IDX = 15

        script_logger.info(
            f"Rank {rank}: Filtering for Polarity: {polarity} (Dataset ID filtering disabled)"
        )

        for batch in batch_reader:
            df = batch.to_pandas()

            if polarity is not None:
                if polarity == 1:  # Positive
                    mask = df["instrument_settings"].apply(lambda x: x[POLARITY_POS_IDX] > 0.5)
                else:  # Negative
                    mask = df["instrument_settings"].apply(lambda x: x[POLARITY_NEG_IDX] > 0.5)

                matched_indices = df.index[mask].tolist()
            else:
                # If no polarity filter is provided, take all indices in this batch
                matched_indices = df.index.tolist()

            valid_indices.extend([idx + current_offset for idx in matched_indices])
            current_offset += len(df)

        self.filtered_indices = valid_indices
        self.length = len(self.filtered_indices)
        script_logger.info(f"LanceIndexDataset initialized. Found {self.length} matching samples.")

    def __len__(self):
        return self.length

    def __getitem__(self, idx: int) -> int:
        return self.filtered_indices[idx]


class LanceBatchCollator:
    def __init__(self, lance_dataset_path: str, rank: str = "0"):
        self.lance_dataset_path = lance_dataset_path
        self.ds = None
        self.rank = rank
        self.log_count = 0

    def __call__(self, batch_indices: list[int]) -> dict:
        if self.ds is None:
            self.ds = lance.dataset(self.lance_dataset_path)

        data_batch = self.ds.take(batch_indices).to_pydict()
        batch_size = len(batch_indices)

        mz_arrays = [torch.tensor(arr, dtype=torch.float32) for arr in data_batch["mz_array"]]
        intensity_arrays = [
            torch.tensor(arr, dtype=torch.float32) for arr in data_batch["intensity_array"]
        ]
        instrument_settings = torch.tensor(data_batch["instrument_settings"], dtype=torch.float32)
        labels = torch.tensor(data_batch["label"], dtype=torch.float32).view(-1, 1)
        precursor_mz = torch.tensor(data_batch["precursor_mz"], dtype=torch.float32).view(-1, 1)

        max_len = max(mz.shape[0] for mz in mz_arrays)
        mz_padded = torch.zeros(batch_size, max_len)
        intensity_padded = torch.zeros(batch_size, max_len)

        for i in range(batch_size):
            length = mz_arrays[i].shape[0]
            mz_padded[i, :length] = mz_arrays[i]
            intensity_padded[i, :length] = intensity_arrays[i]

        batch_dict = {
            "mz": mz_padded,
            "intensity": intensity_padded,
            "instrument_settings": instrument_settings,
            "labels": labels,
            "precursor_mz": precursor_mz,
            "original_index": torch.tensor(batch_indices, dtype=torch.long),
        }
        return batch_dict


def run_predictions(args):
    # 1. Load Model
    from src.transformers.model_bce_loss_one_hot import SimpleSpectraTransformer

    model = SimpleSpectraTransformer.load_from_checkpoint(args.checkpoint_path)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Pass them to the dataset
    dataset = LanceIndexDataset(args.lance_path, polarity=args.polarity, rank=RANK)
    # 2. Filtered Dataset

    collator = LanceBatchCollator(args.lance_path, rank=RANK)
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, collate_fn=collator, num_workers=args.num_workers
    )

    # 3. Predict
    results = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Predicting"):
            inputs = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()
            }
            logits = model(inputs)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()

            for i in range(len(probs)):
                results.append(
                    {
                        "original_index": batch["original_index"][i].item(),
                        "probability": probs[i],
                        "label": batch["labels"][i].item(),
                        "precursor_mz": batch["precursor_mz"][i].item(),
                    }
                )

    # 4. Save Results
    pd.DataFrame(results).to_csv(args.output_csv, index=False)
    script_logger.info(f"Saved results to {args.output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--lance_path", type=str, required=True)
    parser.add_argument("--output_csv", type=str, required=True)
    parser.add_argument("--polarity", type=int, choices=[0, 1], help="0 for Neg, 1 for Pos")
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--num_workers", type=int, default=8)
    args = parser.parse_args()

    mp.set_start_method("spawn", force=True)
    run_predictions(args)
