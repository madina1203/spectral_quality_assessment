"""
Training script for spectral quality assessment using Non-Negative PU (Positive-Unlabeled) loss.

This script implements training with:
- Distributed Data Parallel (DDP) support for multi-GPU training
- Lance dataset format for efficient data loading

- Per-dataset loss tracking and logging
- Early stopping based on validation recall
"""

import os
import datetime
import logging

# Enable MPS fallback for Mac
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import sys
import multiprocessing as mp

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
import torch.nn as nn
import lightning.pytorch as pl
from torch.utils.data import DataLoader, Subset, ConcatDataset
import numpy as np
import argparse
import lance
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import Callback, EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    auc,
)
from sklearn.mixture import GaussianMixture
from scipy.stats import rankdata, norm
import matplotlib.pyplot as plt
from src.transformers.model_nn_pu_loss_detach_diff_polarity import SimpleSpectraTransformer
from lightning.pytorch.strategies import DDPStrategy
from torch.utils.data import Dataset
from typing import Dict, Any
import pandas as pd

# Initialize logging with rank information for distributed training
# Falls back to "0" if not in a DDP context
RANK = os.environ.get("LOCAL_RANK", os.environ.get("SLURM_PROCID", "0"))
logging.basicConfig(
    level=logging.INFO,
    format=f"[Rank {RANK}] %(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
script_logger = logging.getLogger(__name__)

SEED = 1
pl.seed_everything(SEED, workers=True)


class LanceIndexDataset(Dataset):
    """
    PyTorch Dataset that returns indices for efficient distributed data loading.
    
    This dataset only stores indices; actual data loading is deferred to the
    collate function to enable efficient batched reads from Lance format.
    
    Args:
        lance_dataset_path: Path to the Lance dataset directory
        rank: Process rank for logging purposes (default: "0")
    """

    def __init__(self, lance_dataset_path: str, rank: str = "0"):
        super().__init__()
        self.lance_dataset_path = lance_dataset_path

        # We only need to open the dataset once to get the total length
        if not os.path.exists(lance_dataset_path):
            script_logger.error(f"Dataset not found at: {lance_dataset_path}")
            raise FileNotFoundError(f"Lance dataset not found at: {lance_dataset_path}")

        ds = lance.dataset(self.lance_dataset_path)
        self.length = ds.count_rows()

        script_logger.info(
            f"LanceIndexDataset (rank {rank}) initialized for {self.lance_dataset_path}. Length: {self.length}"
        )

    def __len__(self):
        return self.length

    def __getitem__(self, idx: int) -> int:
        # Return the index itself, not the data
        return idx


class LanceBatchCollator:
    """
    Collate function that receives a batch of indices, fetches them from Lance
    in a single batched call, and pads the sequences.

    Conditionally includes 'dataset_id' if it exists in the Lance table schema.
    """

    def __init__(self, lance_dataset_path: str, rank: str = "0"):
        self.lance_dataset_path = lance_dataset_path
        self.ds = None  # We'll open this once per worker
        self.rank = rank
        self.log_count = 0

    def __call__(self, batch_indices: list[int]) -> dict:
        # 1. Open the dataset (if not already open in this worker process)
        if self.ds is None:
            self.ds = lance.dataset(self.lance_dataset_path)
            script_logger.info(f"Collate (rank {self.rank}) opened dataset handle.")

        # Log the first batch of indices to confirm DDP sampling
        if self.log_count < 1:
            script_logger.info(
                f"Collate (rank {self.rank}) fetching indices (first 10): {batch_indices[:10]}"
            )
            self.log_count += 1

        # 2. Fetch all data in ONE batched call
        data_batch = self.ds.take(batch_indices).to_pydict()

        # 3. Get batch size
        batch_size = len(batch_indices)

        # 4. Convert and pad (same logic as your old collate_fn)
        mz_arrays = [torch.tensor(arr, dtype=torch.float32) for arr in data_batch["mz_array"]]
        intensity_arrays = [
            torch.tensor(arr, dtype=torch.float32) for arr in data_batch["intensity_array"]
        ]

        # These fields are already batched lists from to_pydict()
        instrument_settings = torch.tensor(data_batch["instrument_settings"], dtype=torch.float32)
        labels = torch.tensor(data_batch["label"], dtype=torch.float32).view(-1, 1)
        precursor_mz = torch.tensor(data_batch["precursor_mz"], dtype=torch.float32).view(-1, 1)

        # 5. Pad
        max_len = max(mz.shape[0] for mz in mz_arrays)
        mz_padded = torch.zeros(batch_size, max_len)
        intensity_padded = torch.zeros(batch_size, max_len)

        for i in range(batch_size):
            length = mz_arrays[i].shape[0]
            mz_padded[i, :length] = mz_arrays[i]
            intensity_padded[i, :length] = intensity_arrays[i]

        # Conditionally add dataset_id if present in the dataset schema
        dataset_ids = data_batch.get("dataset_id", None)

        batch_dict = {
            "mz": mz_padded,
            "intensity": intensity_padded,
            "instrument_settings": instrument_settings,
            "labels": labels,
            "precursor_mz": precursor_mz,
            "original_idx": torch.tensor(batch_indices, dtype=torch.long),
        }

        if dataset_ids is not None:
            # dataset_ids will be a list of strings, just pass it along
            batch_dict["dataset_id"] = dataset_ids

        return batch_dict


class ImprovedPyTorchSklearnWrapper:
    """
    Wrapper class that provides a sklearn-like interface for PyTorch Lightning models.
    
    Handles model initialization, device placement, data loading, and training
    with distributed training support via PyTorch Lightning.
    
    Args:
        model: PyTorch Lightning model instance
        train_dataset: Training dataset
        val_dataset: Validation dataset
        d_model: Model dimension (default: 64)
        n_layers: Number of transformer layers (default: 2)
        dropout: Dropout rate (default: 0.3)
        linear_lr: Learning rate for linear layers (default: 0.0004755751039)
        encoder_lr: Learning rate for encoder layers (default: 0.0004755751039)
        batch_size: Batch size per GPU (default: 64)
        epochs: Number of training epochs (default: 5)
        device: Target device (None for auto-detection)
        num_workers: Number of DataLoader workers per GPU (default: 4)
        instrument_embedding_dim: Dimension of instrument embedding (default: 32)
        force_cpu: Force CPU usage even if GPU is available (default: False)
        logger: PyTorch Lightning logger instance (default: None)
    """
    
    def __init__(
        self,
        model,
        train_dataset,
        val_dataset,
        d_model=64,
        n_layers=2,
        dropout=0.3,
        linear_lr=0.0004755751039,
        encoder_lr=0.0004755751039,
        batch_size=64,
        epochs=5,
        device=None,
        num_workers=4,
        instrument_embedding_dim=32,
        force_cpu=False,
        logger=None,
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.d_model = d_model
        self.n_layers = n_layers
        self.dropout = dropout
        self.linear_lr = linear_lr
        self.encoder_lr = encoder_lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.num_workers = num_workers
        self.instrument_embedding_dim = instrument_embedding_dim
        self.force_cpu = force_cpu
        self.logger = logger  # This is the CSVLogger

        if force_cpu:
            self.device = torch.device("cpu")
        elif device is None:
            self.device = torch.device(
                "cuda"
                if torch.cuda.is_available()
                else "mps" if torch.backends.mps.is_available() else "cpu"
            )
        else:
            self.device = torch.device(device)

        script_logger.info(f"Using device: {self.device}")
        self.model = self.model.to(self.device)

        if hasattr(self.model, "spectrum_encoder"):
            self.model.spectrum_encoder = self.model.spectrum_encoder.to(self.device)
            if hasattr(self.model.spectrum_encoder, "peak_encoder"):
                self.model.spectrum_encoder.peak_encoder = (
                    self.model.spectrum_encoder.peak_encoder.to(self.device)
                )
                if hasattr(self.model.spectrum_encoder.peak_encoder, "mz_encoder"):
                    self.model.spectrum_encoder.peak_encoder.mz_encoder = (
                        self.model.spectrum_encoder.peak_encoder.mz_encoder.to(self.device)
                    )

        self.is_fitted_ = False
        self.classes_ = np.array([-1.0, 1.0])

    def predict_proba(self, indices):
        """
        Predict class probabilities for given indices.
        
        Args:
            indices: Array of dataset indices to predict
            
        Returns:
            Array of shape (n_samples, 2) with probabilities for negative and positive classes
        """
        self.model.eval()
        test_dataset = Subset(self.val_dataset, indices)
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            persistent_workers=True,
            pin_memory=True,
        )
        all_probs = []
        with torch.no_grad():
            for batch in test_loader:
                batch = {
                    k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }
                self.model = self.model.to(self.device)
                logits = self.model(batch)
                probs = torch.sigmoid(logits)
                all_probs.append(probs.cpu().numpy())
        probs_numpy = np.vstack(all_probs)
        return np.column_stack([1 - probs_numpy, probs_numpy])

    def fit(
        self,
        train_indices,
        val_indices=None,
        callbacks=None,
        train_lance_path=None,
        val_lance_path=None,
    ):
        """
        Train the model using PyTorch Lightning.
        
        Args:
            train_indices: Array of training dataset indices
            val_indices: Array of validation dataset indices (optional)
            callbacks: List of PyTorch Lightning callbacks (optional)
            train_lance_path: Path to training Lance dataset
            val_lance_path: Path to validation Lance dataset
            
        Returns:
            Tuple of (self, trainer) for chaining
        """
        script_logger.info(f"Wrapper.fit() called.")

        # Get rank for the collator
        rank = os.environ.get("LOCAL_RANK", os.environ.get("SLURM_PROCID", "0"))

        # --- Create the collators ---
        train_collator = LanceBatchCollator(train_lance_path, rank=rank)
        val_collator = LanceBatchCollator(val_lance_path, rank=rank)

        script_logger.info("Creating Training DataLoader...")
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=train_collator,
            persistent_workers=True,
            pin_memory=True,
            shuffle=True,
        )

        val_loader = None
        if val_indices is not None:
            script_logger.info("Creating Validation DataLoader...")

            val_loader = DataLoader(
                self.val_dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                collate_fn=val_collator,
                persistent_workers=True,
                pin_memory=True,
                shuffle=False,
            )
        if callbacks is None:
            callbacks = []

        early_stop_callback = EarlyStopping(
            monitor="val_recall",  # Changed from "val_loss"
            patience=7,
            verbose=True,
            mode="max",  # Changed from "min" because we want high recall
        )
        callbacks.append(early_stop_callback)

        # ModelCheckpoint saves the best model based on validation recall
        model_checkpoint_callback = ModelCheckpoint(
            monitor="val_recall",
            save_top_k=1,
            mode="max",
            dirpath=self.logger.log_dir,
            filename="best_model-{epoch:02d}-{val_recall:.4f}",
        )
        callbacks.append(model_checkpoint_callback)
        script_logger.info("Initializing pl.Trainer...")
        trainer = pl.Trainer(
            max_epochs=self.epochs,
            logger=self.logger,
            callbacks=callbacks,
            enable_progress_bar=True,
            accelerator="gpu",
            devices=2,
            strategy=DDPStrategy(gradient_as_bucket_view=True, static_graph=True),
            precision="16-mixed",
        )

        script_logger.info("Trainer.fit() starting.")
        trainer.fit(self.model, train_dataloaders=train_loader, val_dataloaders=val_loader)
        self.is_fitted_ = True
        script_logger.info("Trainer.fit() finished.")
        return self, trainer

    def predict(self, indices, threshold=0.5):
        """
        Predict binary class labels for given indices.
        
        Args:
            indices: Array of dataset indices to predict
            threshold: Probability threshold for positive class (default: 0.5)
            
        Returns:
            Array of predicted labels (-1.0 for negative, 1.0 for positive)
        """
        probs = self.predict_proba(indices)[:, 1]
        return np.array([1.0 if p > threshold else -1.0 for p in probs])


class FixedHoldoutPUCallback(Callback):
    """
    PyTorch Lightning callback for computing PU (Positive-Unlabeled) learning metrics.
    
    Tracks per-dataset losses, computes GMM-based AUROC, and generates validation
    probability distributions. Supports distributed training with proper data gathering.
    
    Args:
        wrapper: ImprovedPyTorchSklearnWrapper instance
        train_indices: Training dataset indices
        val_indices: Validation dataset indices
        train_dataset: Training dataset
        val_dataset: Validation dataset
    """
    
    def __init__(self, wrapper, train_indices, val_indices, train_dataset, val_dataset):
        super().__init__()
        self.wrapper = wrapper
        self.train_indices = train_indices
        self.val_indices = val_indices
        self.val_dataset = val_dataset
        self.validation_step_outputs = []
        self.training_step_outputs = []
        self.batch_stats = []


    # def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
    #     """Collect training batch outputs for per-dataset loss calculation."""
    #     if isinstance(outputs, dict) and 'logits' in outputs and 'labels' in outputs:
    #
    #         logits = outputs['logits'].detach()
    #         labels = outputs['labels'].detach()
    #
    #         total_size = labels.shape[0]
    #         if total_size > 0:
    #             pos_count = (labels > 0).sum().item()
    #             pos_percent = (pos_count / total_size) * 100
    #             self.batch_stats.append(pos_percent)
    #
    #         # Check for dataset_id and get per-sample loss
    #         dataset_ids = batch.get('dataset_id', None)
    #         script_logger.info(
    #             f"[CALLBACK TRAINING_BATCH_END] Batch {batch_idx}: dataset_ids present = {dataset_ids is not None}")
    #         if dataset_ids is not None and batch_idx < 3:  # Only log first 3 batches
    #             script_logger.info(
    #                 f"[CALLBACK TRAINING_BATCH_END] Type: {type(dataset_ids)}, Length: {len(dataset_ids)}, First 3: {dataset_ids[:3]}")
    #
    #         output_dict = {
    #             'logits': logits,
    #             'labels': labels
    #         }
    #
    #         if dataset_ids is not None:
    #             try:
    #                 criterion = pl_module.loss_fn
    #                 original_reduction = criterion.reduction
    #                 criterion.reduction = 'none'
    #                 per_sample_loss = criterion(logits, labels.to(logits.dtype))
    #                 criterion.reduction = original_reduction
    #
    #                 output_dict['per_sample_loss'] = per_sample_loss.detach()
    #                 output_dict['dataset_id'] = dataset_ids
    #
    #             except AttributeError:
    #                 script_logger.warning(
    #                     "Callback could not find 'pl_module.bce_loss'. "
    #                     "Per-dataset training loss logging will be skipped."
    #                 )
    #             except Exception as e:
    #                 script_logger.warning(
    #                     f"Callback error calculating per-sample loss: {e}. "
    #                     "Per-dataset loss logging will be skipped."
    #                 )
    #
    #         self.training_step_outputs.append(output_dict)
    #
    #     else:
    #         script_logger.warning(
    #             "Training step output did not contain expected 'logits' and 'labels' keys.")

    def on_train_epoch_end(self, trainer, pl_module):
        """Calculate and log per-dataset training loss."""
        script_logger.info(
            f"Training epoch end. "
            f"trainer.global_rank: {getattr(trainer, 'global_rank', 'N/A')}"
        )

        local_batch_stats = self.batch_stats
        if trainer.world_size > 1:
            import torch.distributed as dist

            gathered_batch_stats_list = [None] * trainer.world_size
            dist.all_gather_object(gathered_batch_stats_list, local_batch_stats)
        else:
            gathered_batch_stats_list = [local_batch_stats]

        # Clear for next epoch
        self.batch_stats.clear()

        if trainer.global_rank == 0:
            final_batch_stats = [item for sublist in gathered_batch_stats_list for item in sublist]

            if final_batch_stats:
                try:
                    avg_pos_perc = np.mean(final_batch_stats)
                    std_pos_perc = np.std(final_batch_stats)
                    min_pos_perc = np.min(final_batch_stats)
                    max_pos_perc = np.max(final_batch_stats)

                    script_logger.info("--- Training Batch Sampler Stats (Epoch END) ---")
                    script_logger.info(
                        f"Total batches processed (all ranks): {len(final_batch_stats)}"
                    )
                    script_logger.info(f"Positive samples per batch (%):")
                    script_logger.info(f"  Avg: {avg_pos_perc:.2f}%")
                    script_logger.info(f"  Std: {std_pos_perc:.2f}%")
                    script_logger.info(f"  Min: {min_pos_perc:.2f}%")
                    script_logger.info(f"  Max: {max_pos_perc:.2f}%")

                    # Log to CSVLogger
                    if trainer.logger is not None:
                        trainer.logger.log_metrics(
                            {
                                "sampler_avg_pos_perc": float(avg_pos_perc),
                                "sampler_std_pos_perc": float(std_pos_perc),
                            },
                            step=trainer.current_epoch,
                        )

                    # Create a histogram visualization
                    log_dir = trainer.logger.log_dir if trainer.logger else "."
                    os.makedirs(log_dir, exist_ok=True)  # Ensure dir exists
                    plt.figure(figsize=(10, 6))
                    plt.hist(
                        final_batch_stats, bins=30, edgecolor="black", alpha=0.7, range=(0, 100)
                    )
                    plt.title(
                        f"Distribution of Positive Samples per Batch (Epoch {trainer.current_epoch})"
                    )
                    plt.xlabel("Percentage of Positive Samples in Batch")
                    plt.ylabel("Number of Batches")
                    plt.axvline(
                        avg_pos_perc,
                        color="red",
                        linestyle="dashed",
                        linewidth=2,
                        label=f"Avg: {avg_pos_perc:.2f}%",
                    )
                    plt.axvline(
                        50, color="blue", linestyle="dotted", linewidth=2, label="50% (Ideal)"
                    )
                    plt.legend()
                    plt.tight_layout()
                    hist_path = os.path.join(
                        log_dir, f"sampler_dist_epoch_{trainer.current_epoch}.png"
                    )
                    plt.savefig(hist_path)
                    plt.close("all")  # Clean up memory
                    script_logger.info(f"Sampler distribution histogram saved to {hist_path}")

                except Exception as e:
                    script_logger.error(f"Error calculating sampler distribution: {e}")
                    import traceback

                    script_logger.error(traceback.format_exc())
            else:
                script_logger.warning("No batch sampler stats collected.")

        if not self.training_step_outputs:
            script_logger.info(
                "No training outputs collected. Skipping per-dataset training loss."
            )
            self.training_step_outputs.clear()
            return

        # Gather training losses and dataset IDs
        all_per_sample_loss = torch.cat(
            [out["per_sample_loss"] for out in self.training_step_outputs]
        )
        all_dataset_ids = []
        for out in self.training_step_outputs:
            all_dataset_ids.extend(out["dataset_id"])

        # Gather from all GPUs
        gathered_per_sample_loss = pl_module.all_gather(all_per_sample_loss)

        # Gather dataset_ids using torch.distributed
        if trainer.world_size > 1:
            import torch.distributed as dist

            gathered_dataset_ids_list = [None] * trainer.world_size
            dist.all_gather_object(gathered_dataset_ids_list, all_dataset_ids)
        else:
            gathered_dataset_ids_list = [all_dataset_ids]

        self.training_step_outputs.clear()

        if trainer.global_rank != 0:
            script_logger.info("Not rank 0. Skipping training metric calculation.")
            return

        # From this point on, we are only on the main process (rank 0)
        script_logger.info("Rank 0 processing gathered training results.")

        final_per_sample_loss = torch.cat([g for g in gathered_per_sample_loss]).cpu().numpy()

        # Flatten the list of lists of strings
        final_dataset_ids = []
        for sublist in gathered_dataset_ids_list:
            if isinstance(sublist, list):
                final_dataset_ids.extend(sublist)
            else:
                script_logger.warning(f"Unexpected type in gathered_dataset_ids: {type(sublist)}")

        script_logger.info(
            f"Total training samples collected: {len(final_per_sample_loss)}"
        )
        script_logger.info(
            f"Total training dataset_ids collected: {len(final_dataset_ids)}"
        )

        # Calculate per-dataset training loss
        if len(final_dataset_ids) > 0:
            try:
                import pandas as pd

                df = pd.DataFrame(
                    {"loss": final_per_sample_loss.squeeze(), "dataset_id": final_dataset_ids}
                )
                per_dataset_loss = df.groupby("dataset_id")["loss"].mean()

                script_logger.info("[CALLBACK] Per-Dataset Training Loss:")
                metrics_to_log = {}
                for dataset_id, avg_loss in per_dataset_loss.items():
                    safe_metric_name = f"train_loss_{str(dataset_id).replace('.', '_')}"
                    script_logger.info(f"  {safe_metric_name}: {avg_loss:.4f}")
                    metrics_to_log[safe_metric_name] = avg_loss

                if trainer.logger is not None:
                    trainer.logger.log_metrics(metrics_to_log, step=trainer.current_epoch)

            except ImportError:
                script_logger.warning("Pandas not installed. Cannot log per-dataset training loss.")
            except Exception as e:
                script_logger.error(f"[CALLBACK] Error calculating per-dataset training loss: {e}")
                import traceback

                script_logger.error(traceback.format_exc())
        else:
            script_logger.info(
                "[CALLBACK] No dataset_ids found in training data. Skipping per-dataset training loss."
            )

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        if isinstance(outputs, dict) and "logits" in outputs and "labels" in outputs:

            logits = outputs["logits"].detach()
            labels = outputs["labels"].detach()

            # Check for dataset_id and get per-sample loss
            dataset_ids = batch.get("dataset_id", None)

            script_logger.info(
                f"[CALLBACK BATCH_END] Batch {batch_idx}: dataset_ids present = {dataset_ids is not None}"
            )
            if dataset_ids is not None and batch_idx < 3:  # Only log first 3 batches
                script_logger.info(
                    f"[CALLBACK BATCH_END] Type: {type(dataset_ids)}, Length: {len(dataset_ids)}, First 3: {dataset_ids[:3]}"
                )

            output_dict = {"logits": logits, "labels": labels}

            if dataset_ids is not None:
                try:
                    criterion = pl_module.loss_fn
                    original_reduction = criterion.reduction
                    criterion.reduction = "none"
                    per_sample_loss = criterion(logits, labels.to(logits.dtype))
                    criterion.reduction = original_reduction

                    output_dict["per_sample_loss"] = per_sample_loss.detach()
                    output_dict["dataset_id"] = dataset_ids

                except AttributeError:
                    script_logger.warning(
                        "Callback could not find 'pl_module.bce_loss'. "
                        "Per-dataset loss logging will be skipped."
                    )
                except Exception as e:
                    script_logger.warning(
                        f"Callback error calculating per-sample loss: {e}. "
                        "Per-dataset loss logging will be skipped."
                    )

            self.validation_step_outputs.append(output_dict)

        else:
            script_logger.warning(
                "Validation step output did not contain expected 'logits' and 'labels' keys."
            )

    def on_validation_epoch_end(self, trainer, pl_module):

        script_logger.info(
            f"[CALLBACK] Validation epoch end. "
            f"trainer.global_rank: {getattr(trainer, 'global_rank', 'N/A')}, "
            f"trainer.world_size: {getattr(trainer, 'world_size', 'N/A')}"
        )

        if not self.validation_step_outputs:
            all_logits = torch.empty(0, 1, device=pl_module.device)
            all_labels = torch.empty(0, 1, device=pl_module.device)
            all_per_sample_loss = torch.empty(0, 1, device=pl_module.device)
            all_dataset_ids = []
        else:
            all_logits = torch.cat([out["logits"] for out in self.validation_step_outputs])
            all_labels = torch.cat([out["labels"] for out in self.validation_step_outputs])

            # Gather only from batches that had per-sample loss
            all_per_sample_loss = torch.cat(
                [
                    out["per_sample_loss"]
                    for out in self.validation_step_outputs
                    if "per_sample_loss" in out
                ]
            )
            # Gather dataset_ids (list of strings)
            all_dataset_ids = []
            for out in self.validation_step_outputs:
                if "dataset_id" in out:
                    all_dataset_ids.extend(out["dataset_id"])

        # Gather tensors from all GPUs
        gathered_logits = pl_module.all_gather(all_logits)
        gathered_labels = pl_module.all_gather(all_labels)
        gathered_per_sample_loss = pl_module.all_gather(all_per_sample_loss)

        # Use torch.distributed for gathering lists of strings (not supported by all_gather)
        if trainer.world_size > 1:
            import torch.distributed as dist

            gathered_dataset_ids_list = [None] * trainer.world_size
            dist.all_gather_object(gathered_dataset_ids_list, all_dataset_ids)
        else:
            gathered_dataset_ids_list = [all_dataset_ids]

        self.validation_step_outputs.clear()

        if trainer.global_rank != 0:
            script_logger.info("[CALLBACK] Not rank 0. Skipping metric calculation.")
            return

        
        script_logger.info("[CALLBACK] Rank 0 processing gathered results.")

        final_logits = torch.cat([g for g in gathered_logits]).cpu()
        final_labels = torch.cat([g for g in gathered_labels]).cpu()
        final_per_sample_loss = torch.cat([g for g in gathered_per_sample_loss]).cpu().numpy()

        # Flatten the list of lists of strings
        final_dataset_ids = []
        for sublist in gathered_dataset_ids_list:
            if isinstance(sublist, list):
                final_dataset_ids.extend(sublist)
            else:
                script_logger.warning(f"Unexpected type in gathered_dataset_ids: {type(sublist)}")

        original_val_size = len(self.val_indices)
        script_logger.info(
            f"[CALLBACK] Size of gathered logits before trimming: {len(final_logits)}"
        )
        script_logger.info(f"[CALLBACK] Original validation set size: {original_val_size}")
        script_logger.info(
            f"[CALLBACK] Size of gathered dataset_ids before trimming: {len(final_dataset_ids)}"
        )

        # Trim all gathered data to match the original validation set size
        final_logits = final_logits[:original_val_size]
        final_labels = final_labels[:original_val_size]
        final_per_sample_loss = final_per_sample_loss[:original_val_size]
        final_dataset_ids = final_dataset_ids[:original_val_size]

        if final_logits.numel() == 0:
            script_logger.warning(
                "Warning (Rank 0): No validation logits collected. Skipping PU metrics for this epoch."
            )
            if trainer.logger:
                trainer.logger.log_metrics(
                    {"pu_val_auroc_gmm": 0.5, "pu_val_f1": 0.0}, step=trainer.current_epoch
                )
            return

        # --- Per-Dataset Loss Calculation (on Rank 0) ---
        script_logger.info(
            f"[CALLBACK] Length of final_dataset_ids after trimming: {len(final_dataset_ids)}"
        )
        script_logger.info(
            f"[CALLBACK] First 10 dataset_ids: {final_dataset_ids[:10] if len(final_dataset_ids) > 0 else 'EMPTY'}"
        )

        if len(final_dataset_ids) > 0:
            try:
                df = pd.DataFrame(
                    {"loss": final_per_sample_loss.squeeze(), "dataset_id": final_dataset_ids}
                )
                per_dataset_loss = df.groupby("dataset_id")["loss"].mean()

                script_logger.info("[CALLBACK] Per-Dataset Validation Loss:")
                metrics_to_log = {}
                for dataset_id, avg_loss in per_dataset_loss.items():
                    safe_metric_name = f"val_loss_{str(dataset_id).replace('.', '_')}"
                    script_logger.info(f"  {safe_metric_name}: {avg_loss:.4f}")
                    metrics_to_log[safe_metric_name] = avg_loss

                if trainer.logger is not None:
                    trainer.logger.log_metrics(metrics_to_log, step=trainer.current_epoch)

            except ImportError:
                script_logger.warning("Pandas not installed. Cannot log per-dataset loss.")
            except Exception as e:
                script_logger.error(f"[CALLBACK] Error calculating per-dataset loss: {e}")
                import traceback

                script_logger.error(traceback.format_exc())
        else:
            script_logger.info(
                "[CALLBACK] No dataset_ids found in validation data. Skipping per-dataset loss."
            )

        val_probs_s1 = final_logits.sigmoid().numpy().squeeze()
        val_true_labels = final_labels.numpy().squeeze()

        log_dir = trainer.logger.log_dir if trainer.logger else None
        if log_dir is not None:
            epoch = trainer.current_epoch
            # Use sequential indices since we no longer have a combined dataset
            indices = np.arange(len(val_probs_s1))

            if len(indices) == len(val_probs_s1):
                df_data = {
                    "index": indices, 
                    "probability": val_probs_s1,
                    "true_labels": val_true_labels,
                }
                if len(final_dataset_ids) == len(indices):
                    df_data["dataset_id"] = final_dataset_ids
                else:
                    script_logger.warning(
                        f"[CALLBACK] Mismatch between indices ({len(indices)}) and dataset_ids ({len(final_dataset_ids)}). "
                        "ID column will not be saved in prob CSV."
                    )

                df = pd.DataFrame(df_data)
                csv_path = os.path.join(log_dir, f"val_probs_epoch_no_pu_{epoch}.csv")
                df.to_csv(csv_path, index=False)
                script_logger.info(
                    f"[CALLBACK] ✅ Saved validation probabilities (with dataset_id) to {csv_path}"
                )
            else:
                script_logger.error(
                    f"Length mismatch: Indices: {len(indices)}, Probs: {len(val_probs_s1)}"
                )

        # --- PU Metric Calculation ---
        val_labeled_pos_indices = np.where(val_true_labels == 1)[0]

        additional_pu_metrics = calculate_pu_metrics(
            probabilities=val_probs_s1,
            true_labels=val_true_labels,
            labeled_pos_indices=val_labeled_pos_indices,
            epoch=trainer.current_epoch,
            log_dir=log_dir,
        )

        metrics = {
            "pu_val_auroc_gmm": additional_pu_metrics["auroc_gmm"],
            "val_area_under_percentile_ranks": additional_pu_metrics[
                "val_area_under_percentile_ranks"
            ],
        }

        if trainer.logger is not None:
            trainer.logger.log_metrics(metrics, step=trainer.current_epoch)

        script_logger.info(f"\nEpoch {trainer.current_epoch} PU Metrics:")
        for name, value in metrics.items():
            script_logger.info(f"  {name}: {value:.4f}")


def plot_probability_distribution(probabilities, gmm, epoch, log_dir):
    """
    Plot the probability score distribution with fitted GMM components.
   
    """
    try:
        # ... (plotting logic) ...
        plt.savefig(
            os.path.join(log_dir, f"probability_distribution_epoch_{epoch}.png"),
            dpi=300,
            bbox_inches="tight",
        )
        script_logger.info(f"Plot saved for epoch {epoch}")
    except Exception as e:
        script_logger.error(f"Error creating plot for epoch {epoch}: {e}")
    finally:
        plt.close("all")
        import gc

        gc.collect()


def calculate_pu_metrics(probabilities, true_labels, labeled_pos_indices, epoch=None, log_dir=None):
    script_logger.info("\n--- calculate_pu_metrics (v2 - Theoretical GMM AUROC) --- DEBUG LOGS ---")
    script_logger.info(f"Input probabilities (first 10): {probabilities[:10]}")
    script_logger.info(f"Input true_labels (first 10): {true_labels[:10]}")
    script_logger.info(f"Input labeled_pos_indices (first 10): {labeled_pos_indices[:10]}")
    script_logger.info(
        f"Total samples: {len(probabilities)}, Total true labels: {len(true_labels)}, Total labeled_pos: {len(labeled_pos_indices)}"
    )

    if len(probabilities) != len(true_labels):
        script_logger.error(
            f"ERROR: Length mismatch IN calculate_pu_metrics! probabilities ({len(probabilities)}) != true_labels ({len(true_labels)})"
        )
        raise ValueError(
            f"Length mismatch: probabilities ({len(probabilities)}) != true_labels ({len(true_labels)})"
        )

    if not isinstance(labeled_pos_indices, np.ndarray):
        labeled_pos_indices = np.array(labeled_pos_indices)

    if len(labeled_pos_indices) > 0 and np.any(labeled_pos_indices >= len(probabilities)):
        script_logger.warning(
            f"Warning (v2 script): Some labeled positive indices are out of bounds. Adjusting indices..."
        )
        original_count = len(labeled_pos_indices)
        labeled_pos_indices = labeled_pos_indices[labeled_pos_indices < len(probabilities)]
        script_logger.warning(
            f"Adjusted labeled_pos_indices count from {original_count} to {len(labeled_pos_indices)}"
        )

    auprc_val = 0.0
    auroc_gmm_val = 0.5
    fitted_gmm = None

    script_logger.info("Computing GMM-based AUROC (theoretical)")
    if len(probabilities) < 2:
        script_logger.warning(
            "Not enough samples for GMM fitting (<2). Returning default AUROC GMM."
        )
    else:
        X = probabilities.reshape(-1, 1)
        try:
            gmm = GaussianMixture(n_components=2, random_state=SEED, reg_covar=1e-6)
            gmm.fit(X)
            fitted_gmm = gmm
            means = gmm.means_.flatten()
            variances = gmm.covariances_.flatten()
            stds = np.sqrt(np.maximum(variances, 1e-12))
            weights = gmm.weights_.flatten()
            script_logger.info(f"GMM Means: {means}, Std Devs: {stds}, Weights: {weights}")
            if means[0] > means[1]:
                means, stds, weights = means[::-1], stds[::-1], weights[::-1]
                script_logger.info("GMM components swapped for order.")
            mean_neg, std_neg, mean_pos, std_pos = means[0], stds[0], means[1], stds[1]
            script_logger.info(
                f"Neg Comp: Mean={mean_neg:.4f}, Std={std_neg:.4f}. Pos Comp: Mean={mean_pos:.4f}, Std={std_pos:.4f}"
            )
            if std_neg < 1e-6 or std_pos < 1e-6:
                script_logger.warning(
                    f"Warning (v2 script): GMM std dev near zero. AUROC GMM might be unreliable. Defaulting to 0.5."
                )
                auroc_gmm_val = 0.5
            else:
                # ... (AUC calculation logic) ...
                # This part was missing from your provided code, assuming it's correct
                min_val, max_val = min(0.0, np.min(X) - 0.1), max(1.0, np.max(X) + 0.1)
                thresholds = np.linspace(min_val, max_val, num=500)
                tpr_theoretical = 1 - norm.cdf(thresholds, loc=mean_pos, scale=std_pos)
                fpr_theoretical = 1 - norm.cdf(thresholds, loc=mean_neg, scale=std_neg)
                fpr_final = np.concatenate(([0], fpr_theoretical[::-1], [1]))
                tpr_final = np.concatenate(([0], tpr_theoretical[::-1], [1]))
                unique_fpr, unique_indices = np.unique(fpr_final, return_index=True)
                fpr_final, tpr_final = unique_fpr, tpr_final[unique_indices]
                valid_pts = ~(
                    np.isnan(fpr_final)
                    | np.isinf(fpr_final)
                    | np.isnan(tpr_final)
                    | np.isinf(tpr_final)
                )
                fpr_final, tpr_final = fpr_final[valid_pts], tpr_final[valid_pts]
                sort_order = np.argsort(fpr_final)
                fpr_final, tpr_final = fpr_final[sort_order], tpr_final[sort_order]
                if len(fpr_final) > 1:
                    auroc_gmm_val = auc(fpr_final, tpr_final)
                else:
                    auroc_gmm_val = 0.5
                # ... end AUC calculation
                script_logger.info(f"Calculated Theoretical AUROC GMM: {auroc_gmm_val:.4f}")
                if epoch is not None and log_dir is not None and fitted_gmm is not None:
                    plot_probability_distribution(probabilities, fitted_gmm, epoch, log_dir)
        except Exception as e:
            script_logger.error(
                f"Error during Theoretical GMM AUROC calculation: {e}. Defaulting to 0.5."
            )
            auroc_gmm_val = 0.5

    script_logger.info("Computing percentile rank and AUPRC")
    if len(labeled_pos_indices) == 0:
        script_logger.warning("No valid labeled positive samples for AUPRC calculation.")
    else:
        # ... (AUPRC calculation logic) ...
        # This part was missing from your provided code, assuming it's correct
        if len(probabilities) > 1 and np.max(probabilities) != np.min(probabilities):
            ranks = rankdata(probabilities, method="average")
            percentile_ranks = (ranks - 1) / (len(ranks) - 1)
        else:
            percentile_ranks = (
                np.zeros_like(probabilities) if len(probabilities) > 0 else np.array([])
            )
        labeled_pos_ranks = percentile_ranks[labeled_pos_indices]
        if len(labeled_pos_ranks) > 0:
            sorted_ranks = np.sort(labeled_pos_ranks)
            n_pos = len(sorted_ranks)
            if n_pos > 1:
                x_vals = np.arange(1, n_pos + 1) / n_pos
                auprc_val = np.trapz(sorted_ranks, x=x_vals)
            elif n_pos == 1:
                auprc_val = float(sorted_ranks[0])
        # ... end AUPRC calculation
        script_logger.info(f"Calculated AUPRC: {auprc_val:.4f}")

    return {"auroc_gmm": float(auroc_gmm_val), "val_area_under_percentile_ranks": float(auprc_val)}


def main(args):
    """
    Main training function.
    
    Sets up distributed training, initializes datasets, model, and trainer,
    then runs the training loop with PU learning metrics tracking.
    
    Args:
        args: ArgumentParser namespace with training configuration
    """
    mp.set_start_method("spawn")
    rank = os.environ.get("LOCAL_RANK", os.environ.get("SLURM_PROCID", "0"))
    script_logger.info(f"--- Main function started ---")
    script_logger.info(f"  SLURM_PROCID: {os.environ.get('SLURM_PROCID')}")
    script_logger.info(f"  LOCAL_RANK: {os.environ.get('LOCAL_RANK')}")
    script_logger.info(f"  Global RANK (used for logging): {rank}")

    if torch.cuda.is_available():
        script_logger.info(f"Available GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            script_logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")

    experiment_name = "training_from_lance_nnpu_loss_11_12_more_instrument_settings"
    log_dir = os.path.join(args.log_dir, experiment_name)
    os.makedirs(log_dir, exist_ok=True)

    script_logger.info(f"CSVLogger initialized. Log dir: {log_dir}")
    csv_logger = CSVLogger(
        save_dir=args.log_dir,
        name=experiment_name,
        version=datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
        flush_logs_every_n_steps=10,
    )

    TRAIN_LANCE_PATH = os.path.join(args.lance_uri, "train_data")
    VAL_LANCE_PATH = os.path.join(args.lance_uri_val, "validation_data")

    script_logger.info(f"Loading TRAINING data from Lance store: {TRAIN_LANCE_PATH}")
    train_dataset = LanceIndexDataset(TRAIN_LANCE_PATH, rank=rank)
    script_logger.info(f"Length of training dataset: {len(train_dataset)}")

    script_logger.info(f"Loading VALIDATION data from Lance store: {VAL_LANCE_PATH}")
    val_dataset = LanceIndexDataset(VAL_LANCE_PATH, rank=rank)
    script_logger.info(f"Length of validation dataset: {len(val_dataset)}")

    train_indices = np.arange(len(train_dataset))
    val_indices = np.arange(len(val_dataset))

    script_logger.info("Initializing SimpleSpectraTransformer model...")
    model = SimpleSpectraTransformer(
        d_model=args.d_model,
        n_layers=args.n_layers,
        dropout=args.dropout,
        linear_lr=args.linear_lr,
        encoder_lr=args.encoder_lr,
        instrument_embedding_dim=args.instrument_embedding_dim,
        weight_decay=args.weight_decay,
        prior_pos=args.prior_pos,
        prior_neg=args.prior_neg,
    )

    wrapper_args = dict(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        d_model=args.d_model,
        n_layers=args.n_layers,
        dropout=args.dropout,
        linear_lr=args.linear_lr,
        encoder_lr=args.encoder_lr,
        batch_size=args.batch_size,
        epochs=args.epochs,
        num_workers=args.num_workers,
        instrument_embedding_dim=args.instrument_embedding_dim,
        logger=csv_logger,
    )

    try:
        script_logger.info("Initializing ImprovedPyTorchSklearnWrapper...")
        pytorch_model = ImprovedPyTorchSklearnWrapper(**wrapper_args, force_cpu=args.force_cpu)
    except Exception as e:
        script_logger.error(f"Error during model initialization with preferred device: {e}")
        script_logger.warning("Trying to use CPU only for all operations...")
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        model_cpu = SimpleSpectraTransformer(
            d_model=args.d_model,
            n_layers=args.n_layers,
            dropout=args.dropout,
            lr=args.lr,
            instrument_embedding_dim=args.instrument_embedding_dim,
            linear_lr=args.linear_lr,
            encoder_lr=args.encoder_lr,
            weight_decay=args.weight_decay,
            prior_pos=args.prior_pos,
            prior_neg=args.prior_neg,
        )

        pytorch_model = ImprovedPyTorchSklearnWrapper(
            **wrapper_args, model=model_cpu, force_cpu=True
        )

    script_logger.info("Calling Wrapper.fit() to start training...")
    pytorch_model.fit(
        train_indices=train_indices,
        val_indices=val_indices,
        train_lance_path=TRAIN_LANCE_PATH,
        val_lance_path=VAL_LANCE_PATH,
    )

    script_logger.info("\nTraining finished.")
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_save_path = os.path.join(log_dir, f"pu_model_v2_{timestamp}.pt")

    # Save only on Rank 0 to avoid file write conflicts
    if rank == "0":
        torch.save(pytorch_model.model.state_dict(), model_save_path)
        script_logger.info(f"\nModel saved to {model_save_path}")

        summary_path = os.path.join(log_dir, f"experiment_summary_{timestamp}.txt")
        with open(summary_path, "w") as f:
            f.write("=== Experiment Summary (GMM AUROC, Train/Val Only) ===\n\n")
            f.write("Hyperparameters:\n")
            for param, value in vars(args).items():
                f.write(f"{param}: {value}\n")
        script_logger.info(f"Experiment summary saved to {summary_path}")

    script_logger.info(f"Logs can be found in: {log_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lance_uri", type=str, required=True, help="Path to Lance database directory"
    )
    parser.add_argument(
        "--lance_uri_val", type=str, required=True, help="Path to Lance database directory"
    )

    parser.add_argument("--log_dir", type=str, default="./logs", help="Directory for logs")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size PER GPU")
    parser.add_argument(
        "--num_workers", type=int, default=8, help="Number of DataLoader workers PER GPU"
    )
    parser.add_argument("--d_model", type=int, default=64, help="Model dimension")
    parser.add_argument("--n_layers", type=int, default=2, help="Number of transformer layers")
    parser.add_argument("--dropout", type=float, default=0.1712215566511, help="Dropout rate")
    parser.add_argument(
        "--linear_lr", type=float, default=0.0004755751039, help="Learning rate of linear layers"
    )
    parser.add_argument(
        "--encoder_lr", type=float, default=0.0004755751039, help="Learning rate of encoder layer"
    )
    parser.add_argument("--epochs", type=int, default=15, help="Number of epochs")
    parser.add_argument(
        "--instrument_embedding_dim",
        type=int,
        default=16,
        help="Dimension of the instrument embedding output",
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.001, help="Weight decay for optimizer"
    )
    parser.add_argument(
        "--unlabeled_weight",
        type=float,
        default=0.9,
        help="Weight for bce loss for unlabeled samples",
    )
    parser.add_argument(
        "--force_cpu", action="store_true", help="Force using CPU instead of GPU/MPS"
    )
    parser.add_argument("--prior_pos", type=float, default=0.5)
    parser.add_argument("--prior_neg", type=float, default=0.35)
    args = parser.parse_args()
    main(args)
