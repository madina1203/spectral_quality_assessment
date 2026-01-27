#!/usr/bin/env python
"""
Lance Dataset Builder for Training set

Converts paired mzML + CSV files into Lance format for efficient model training.
Extracts MS1/MS2 scan pairs, applies  preprocessing, standardizes
instrument settings, and one-hot encodes categorical features.

Features:
    - Parallel processing of mzML/CSV file pairs
    - Global feature standardization across datasets
    - Per-dataset MS2 count capping with random sampling
    - Support for exceptional datasets requiring serial processing
"""

import os
import re
import argparse
import numpy as np
import pandas as pd
import pyarrow as pa
import lance
from pyteomics import mzml
import spectrum_utils.spectrum as sus
from typing import List, Dict, Any, Tuple
from datetime import datetime
from multiprocessing import Pool, cpu_count
import logging
import psutil
import random

random.seed(42)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def get_scan_number(id_string: str) -> int:
    """Extracts the scan number from an mzML ID string."""
    match = re.search(r"scan=(\d+)", id_string)
    return int(match.group(1)) if match else None


def get_one_hot_vector(value: float) -> List[float]:
    """
    Convert binary value to one-hot encoding.
    
    Args:
        value: Binary value (0 or 1)
        
    Returns:
        [1.0, 0.0] for 0, [0.0, 1.0] for 1, [0.0, 0.0] for invalid/NaN
    """
    try:
        val_int = int(value)
        if val_int == 0:
            return [1.0, 0.0]
        elif val_int == 1:
            return [0.0, 1.0]
        else:
            return [1.0, 0.0]
    except (ValueError, TypeError):
        return [0.0, 0.0]


def get_instrument_settings_columns() -> List[str]:
    """Return numerical instrument setting columns for standardization."""
    return [
        "MS2 Isolation Width",
        "Charge State",
        "Ion Injection Time (ms)",
        "Conversion Parameter C",
        "Energy1",
        "Orbitrap Resolution",
        "AGC Target",
        "HCD Energy(1)",
        "HCD Energy(2)",
        "HCD Energy(3)",
        "HCD Energy(4)",
        "HCD Energy(5)",
        "LM m/z-Correction (ppm)",
        "Micro Scan Count",
    ]


def load_and_preprocess_scans(mzml_file: str, max_peaks: int = 400) -> List[Dict[str, Any]]:
    """
    Load and preprocess spectra from an mzML file.
    
    Args:
        mzml_file: Path to mzML file
        max_peaks: Maximum number of peaks to retain after filtering
        
    Returns:
        List of scan dictionaries with preprocessed spectral data
    """
    scan_list = []

    if not os.path.exists(mzml_file):
        logger.error(f"mzML file not found: {mzml_file}")
        return scan_list

    try:
        with mzml.read(mzml_file) as reader:
            for spectrum in reader:
                ms_level = spectrum.get("ms level")
                id_string = spectrum.get("id")
                scan_number = get_scan_number(id_string)

                if scan_number is None or ms_level is None:
                    continue

                mz_array = spectrum.get("m/z array")
                intensity_array = spectrum.get("intensity array")
                mzml_precursor_mz = 0.0

                try:
                    if ms_level == 1:
                        mz_spectrum = sus.MsmsSpectrum(
                            identifier=str(scan_number),
                            precursor_mz=np.nan,
                            precursor_charge=np.nan,
                            mz=mz_array,
                            intensity=intensity_array,
                            retention_time=spectrum.get("scan start time", 0),
                        )
                        mz_spectrum = mz_spectrum.filter_intensity(
                            min_intensity=0.01, max_num_peaks=max_peaks
                        )
                        mz_spectrum = mz_spectrum.scale_intensity(scaling="root")
                        mz_array = mz_spectrum.mz
                        intensity_array = mz_spectrum.intensity

                    elif ms_level == 2:
                        precursors = spectrum.get("precursorList", {}).get("precursor", [])
                        if precursors:
                            selected_ions = (
                                precursors[0].get("selectedIonList", {}).get("selectedIon", [])
                            )
                            if selected_ions:
                                val = selected_ions[0].get("selected ion m/z")
                                if val is not None:
                                    mzml_precursor_mz = float(val)

                    scan_list.append(
                        {
                            "scan_number": scan_number,
                            "ms_level": ms_level,
                            "mz_array": mz_array,
                            "intensity_array": intensity_array,
                            "precursor_mz": mzml_precursor_mz,
                        }
                    )

                except Exception as e:
                    logger.warning(
                        f"Skipping scan {scan_number} in {os.path.basename(mzml_file)}: {e}"
                    )
                    continue

        scan_list.sort(key=lambda x: x["scan_number"])
        ms1_count = sum(1 for s in scan_list if s["ms_level"] == 1)
        ms2_count = sum(1 for s in scan_list if s["ms_level"] == 2)
        logger.debug(
            f"  Loaded {ms1_count} MS1 and {ms2_count} MS2 scans from {os.path.basename(mzml_file)}"
        )

    except Exception as e:
        logger.error(f"Error reading mzML file {mzml_file}: {e}")

    return scan_list


def load_ms2_data(csv_file: str) -> pd.DataFrame:
    """Loads and cleans MS2 metadata from a CSV file."""
    if not os.path.exists(csv_file):
        logger.error(f"CSV file not found: {csv_file}")
        return pd.DataFrame()

    try:
        ms2_df = pd.read_csv(csv_file)
        ms2_df["Scan"] = ms2_df["Scan"].astype(int)
        if not ms2_df.empty:
            logger.debug(f"  Loaded {len(ms2_df)} MS2 records from {os.path.basename(csv_file)}")
        return ms2_df
    except Exception as e:
        logger.error(f"Error reading CSV file {csv_file}: {e}")
        return pd.DataFrame()


def compute_stats(ms2_data: pd.DataFrame) -> Dict[int, Dict[str, float]]:
    """Compute mean and std for each numerical instrument setting column."""
    instrument_settings_cols = get_instrument_settings_columns()
    all_settings = []

    for _, row in ms2_data.iterrows():
        settings = [
            float(row[col]) for col in instrument_settings_cols if col in row and pd.notna(row[col])
        ]
        if len(settings) == len(instrument_settings_cols):
            all_settings.append(settings)

    if not all_settings:
        logger.warning("No valid instrument settings found to compute stats.")
        return {}

    all_settings = np.array(all_settings)
    feature_stats = {
        i: {"mean": np.mean(all_settings[:, i]), "std": np.std(all_settings[:, i])}
        for i in range(all_settings.shape[1])
    }
    return feature_stats


def scale_features(
    instrument_settings: List[float], feature_stats: Dict[int, Dict[str, float]]
) -> np.ndarray:
    """Apply z-score standardization to instrument settings using precomputed stats."""
    scaled_settings = []
    for i, value in enumerate(instrument_settings):
        if i in feature_stats:
            mean_val = feature_stats[i]["mean"]
            std_val = feature_stats[i]["std"]
            if np.isnan(mean_val) or np.isnan(std_val):
                scaled_value = 0.0
            else:
                scaled_value = (value - mean_val) / std_val if std_val > 0 else 0.0
            scaled_settings.append(scaled_value)
        else:
            scaled_settings.append(value)

    return np.array(scaled_settings, dtype=np.float32)


def align_and_format_data(
    scan_list: List[Dict[str, Any]],
    ms2_data: pd.DataFrame,
    feature_stats: Dict[int, Dict[str, float]],
    source_file: str,
    dataset_id: str,
    mzml_filepath: str,
) -> List[Dict[str, Any]]:
    """
    Align MS1/MS2 scan pairs and format for Lance storage.
    
    Each MS2 scan is paired with its preceding MS1 scan. Instrument settings
    are standardized (numerical) or one-hot encoded (categorical).
    
    Args:
        scan_list: List of preprocessed scans from mzML
        ms2_data: DataFrame with MS2 metadata from CSV
        feature_stats: Global mean/std for numerical feature standardization
        source_file: Base filename for provenance
        dataset_id: MassIVE dataset identifier
        mzml_filepath: Full path to source mzML file
        
    Returns:
        List of formatted data pairs ready for Lance storage
    """
    data_pairs = []
    ms2_scan_info = ms2_data.set_index("Scan").to_dict("index")

    numerical_cols = get_instrument_settings_columns()
    categorical_cols = ["Polarity", "Ionization", "Mild Trapping Mode", "Activation1"]

    current_ms1_data = None
    current_ms1_scan_number = None

    for scan in scan_list:
        scan_number = scan["scan_number"]
        ms_level = scan["ms_level"]

        if ms_level == 1:
            current_ms1_data = {
                "mz_array": scan["mz_array"],
                "intensity_array": scan["intensity_array"],
            }
            current_ms1_scan_number = scan_number

        elif ms_level == 2 and current_ms1_data is not None:
            if scan_number in ms2_scan_info:
                ms2_info = ms2_scan_info[scan_number]
                compound_name = ms2_info.get("Compound_name", "")

                raw_numerical_settings = []
                valid_row = True

                for col in numerical_cols:
                    val = ms2_info.get(col)
                    if val is None or pd.isna(val):
                        valid_row = False
                        break
                    raw_numerical_settings.append(float(val))

                for cat_col in categorical_cols:
                    if pd.isna(ms2_info.get(cat_col)):
                        valid_row = False
                        break

                if not valid_row:
                    continue

                scaled_settings = scale_features(raw_numerical_settings, feature_stats)

                polarity_ohe = get_one_hot_vector(ms2_info.get("Polarity"))
                ionization_ohe = get_one_hot_vector(ms2_info.get("Ionization"))
                trapping_ohe = get_one_hot_vector(ms2_info.get("Mild Trapping Mode"))
                activation_ohe = get_one_hot_vector(ms2_info.get("Activation1"))

                final_instrument_settings = np.concatenate(
                    [scaled_settings, polarity_ohe, ionization_ohe, trapping_ohe, activation_ohe]
                ).astype(np.float32)

                precursor_from_mzml = scan.get("precursor_mz", 0.0)
                label = ms2_info["label"]

                data_pairs.append(
                    {
                        "ms1_scan_number": current_ms1_scan_number,
                        "ms2_scan_number": scan_number,
                        "mz_array": current_ms1_data["mz_array"].tolist(),
                        "intensity_array": current_ms1_data["intensity_array"].tolist(),
                        "instrument_settings": final_instrument_settings.tolist(),
                        "precursor_mz": float(precursor_from_mzml),
                        "label": float(label),
                        "compound_name": str(compound_name),
                        "source_file": source_file,
                        "dataset_id": dataset_id,
                        "mzml_filepath": mzml_filepath,
                    }
                )

    return data_pairs


def process_single_file(
    args: Tuple[str, str, str, Dict[int, Dict[str, float]], int, int, int],
) -> List[Dict[str, Any]]:
    """Process a single mzML-CSV file pair (parallelizable worker function)."""
    mzml_file, csv_file, dataset_id, feature_stats, file_idx, total_files, max_peaks = args
    process = psutil.Process()
    file_base = os.path.basename(mzml_file).split(".")[0]
    logger.info(f"[{file_idx}/{total_files}] Processing: {file_base} (Dataset: {dataset_id})")

    try:
        scan_list = load_and_preprocess_scans(mzml_file, max_peaks)
        if not scan_list:
            logger.warning(f"[{file_idx}/{total_files}] No scans found in {file_base}")
            return []

        ms2_data = load_ms2_data(csv_file)
        if ms2_data.empty:
            logger.warning(f"[{file_idx}/{total_files}] No MS2 data found in {file_base}")
            return []

        data_pairs = align_and_format_data(
            scan_list, ms2_data, feature_stats, file_base, dataset_id, mzml_file
        )
        mem_info = process.memory_info()
        logger.info(
            f"[{file_idx}/{total_files}] ✓ Generated {len(data_pairs)} pairs. Memory: {mem_info.rss / 1024 ** 3:.2f} GB"
        )

        del scan_list, ms2_data
        return data_pairs

    except Exception as e:
        logger.error(f"[{file_idx}/{total_files}] ✗ Error processing {file_base}: {e}")
        import traceback

        logger.error(traceback.format_exc())
        return []


def process_exceptional_dataset(
    pairs_for_dataset: List[Tuple[str, str, str]],
    feature_stats: Dict[int, Dict[str, float]],
    cap_value: int,
    max_peaks: int,
) -> Tuple[List[Dict[str, Any]], int, int]:
    """
    Process large datasets serially with MS1-grouped sampling.
    
    For datasets too large for standard file-level sampling, this loads all
    data first, then samples complete MS1 groups (preserving MS1-MS2 pairs)
    until reaching the cap.
    
    Returns:
        Tuple of (sampled_data_pairs, files_processed, ms2_count)
    """
    logger.warning(
        f"  Starting serial processing for {len(pairs_for_dataset)} files... This may take a while."
    )

    all_data_pairs = []
    total_files_processed = 0
    process = psutil.Process()

    for mzml_file, csv_file, dataset_id in pairs_for_dataset:
        file_base = os.path.basename(mzml_file).split(".")[0]
        logger.info(f"    Loading file: {file_base} (Dataset: {dataset_id})")

        scan_list = load_and_preprocess_scans(mzml_file, max_peaks)
        if not scan_list:
            logger.warning(f"    No scans in {file_base}, skipping.")
            continue

        ms2_data = load_ms2_data(csv_file)
        if ms2_data.empty:
            logger.warning(f"    No MS2 data in {file_base}, skipping.")
            continue

        data_pairs_from_file = align_and_format_data(
            scan_list, ms2_data, feature_stats, file_base, dataset_id, mzml_file
        )
        all_data_pairs.extend(data_pairs_from_file)
        total_files_processed += 1

        mem_info = process.memory_info()
        logger.info(
            f"    Loaded {len(data_pairs_from_file)} pairs. Total: {len(all_data_pairs)}. Mem: {mem_info.rss / 1024 ** 3:.2f} GB"
        )

    if not all_data_pairs:
        logger.warning("  No data pairs found in any file for this dataset.")
        return [], 0, 0

    logger.info(f"  Loaded {len(all_data_pairs)} total pairs. Grouping by MS1 scan number...")

    ms1_groups = {}
    for pair in all_data_pairs:
        ms1_key = (pair["source_file"], pair["ms1_scan_number"])
        if ms1_key not in ms1_groups:
            ms1_groups[ms1_key] = []
        ms1_groups[ms1_key].append(pair)

    logger.info(f"  Grouped into {len(ms1_groups)} unique MS1 scans. Shuffling...")

    ms1_keys = list(ms1_groups.keys())
    random.shuffle(ms1_keys)

    sampled_data_pairs = []
    current_ms2_count = 0

    for ms1_key in ms1_keys:
        child_ms2_pairs = ms1_groups[ms1_key]

        if current_ms2_count + len(child_ms2_pairs) <= cap_value:
            sampled_data_pairs.extend(child_ms2_pairs)
            current_ms2_count += len(child_ms2_pairs)
        elif current_ms2_count == 0:
            logger.warning(
                f"    First MS1 group (size {len(child_ms2_pairs)}) exceeds cap {cap_value}. Adding it anyway."
            )
            sampled_data_pairs.extend(child_ms2_pairs)
            current_ms2_count += len(child_ms2_pairs)
        elif current_ms2_count > 0:
            break

    logger.info(f"  Sampled {current_ms2_count} MS2 pairs from {total_files_processed} files.")

    return sampled_data_pairs, total_files_processed, current_ms2_count


def save_to_lance_batch(
    data_batch: List[Dict[str, Any]], dataset_path: str, mode: str = "overwrite"
):
    """Write a batch of processed data pairs to Lance dataset."""
    if not data_batch:
        return

    schema = pa.schema(
        [
            pa.field("ms1_scan_number", pa.int32()),
            pa.field("ms2_scan_number", pa.int32()),
            pa.field("mz_array", pa.list_(pa.float32())),
            pa.field("intensity_array", pa.list_(pa.float32())),
            pa.field("instrument_settings", pa.list_(pa.float32())),
            pa.field("precursor_mz", pa.float32()),
            pa.field("label", pa.float32()),
            pa.field("compound_name", pa.string()),
            pa.field("source_file", pa.string()),
            pa.field("dataset_id", pa.string()),
            pa.field("mzml_filepath", pa.string()),
        ]
    )

    pa_table = pa.Table.from_pylist(data_batch, schema=schema)

    if mode == "overwrite":
        lance.write_dataset(pa_table, dataset_path)
    elif mode == "append":
        lance.write_dataset(pa_table, dataset_path, mode="append")


def load_file_pairs(
    file_list_path: str, test_mode: bool = False, max_files: int = 3, exclude_blank: bool = False
) -> List[Tuple[str, str]]:
    """Load mzML/CSV file pairs from a text file (one pair per line, comma-separated)."""
    if not os.path.exists(file_list_path):
        logger.error(f"File list not found: {file_list_path}")
        return []

    with open(file_list_path, "r") as f:
        lines = f.readlines()

    file_pairs = []
    excluded_count = 0
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(",")
        if len(parts) >= 2:
            mzml_path = os.path.abspath(parts[0].strip())
            csv_path = os.path.abspath(parts[1].strip())

            if exclude_blank:
                mzml_filename = os.path.basename(mzml_path).lower()
                csv_filename = os.path.basename(csv_path).lower()

                if "blank" in mzml_filename or "blank" in csv_filename:
                    excluded_count += 1
                    logger.debug(
                        f"Excluding file pair with 'blank' in name: {os.path.basename(mzml_path)} / {os.path.basename(csv_path)}"
                    )
                    continue

            file_pairs.append((mzml_path, csv_path))

    if exclude_blank and excluded_count > 0:
        logger.info(f"Excluded {excluded_count} file pair(s) containing 'blank' in filename")

    if test_mode:
        file_pairs = file_pairs[:max_files]
        logger.info(f"⚠️  TEST MODE: Using only first {len(file_pairs)} file pairs")

    return file_pairs


def compute_global_stats(file_pairs: List[Tuple[str, str]]) -> Dict[int, Dict[str, float]]:
    """Compute global mean/std statistics across all CSV files for feature standardization."""
    logger.info(
        f"\n--- Step 1: Computing global feature statistics from {len(file_pairs)} files ---"
    )
    all_ms2_data = []
    for idx, (_, csv_file) in enumerate(file_pairs, 1):
        logger.debug(f"Loading stats from [{idx}/{len(file_pairs)}] {os.path.basename(csv_file)}")
        ms2_df = load_ms2_data(csv_file)
        if not ms2_df.empty:
            all_ms2_data.append(ms2_df)

    if not all_ms2_data:
        logger.error("No MS2 data found in any CSV file. Cannot compute stats.")
        return {}

    combined_ms2_df = pd.concat(all_ms2_data, ignore_index=True)
    stats = compute_stats(combined_ms2_df)
    del all_ms2_data, combined_ms2_df
    logger.info("Global feature statistics computed.")
    return stats


def process_file_list(
    file_pairs: List[Tuple[str, str]],
    global_feature_stats: Dict[int, Dict[str, float]],
    lance_uri: str,
    table_name: str,
    num_workers: int,
    max_peaks: int,
    batch_size: int = 10,
    cap_value: int = -1,
    reporting_df: pd.DataFrame = None,
    reporting_csv_path: str = None,
    exceptional_dataset_ids: List[str] = None,
):
    """
    Process file pairs in batches and write to Lance dataset.
    
    Groups files by dataset ID, applies per-dataset MS2 capping if specified,
    and processes files in parallel batches.
    """
    logger.info(f"\n{'=' * 80}")
    logger.info(f"Starting processing for: {table_name.upper()}")
    logger.info(f"{'=' * 80}")

    if exceptional_dataset_ids is None:
        exceptional_dataset_ids = []

    logger.info(f"Processing {len(file_pairs)} file pairs.")

    msv_regex = re.compile(r"(MSV\d{9})")
    datasets_map = {}
    for mzml_path, csv_path in file_pairs:
        match = msv_regex.search(mzml_path)
        dataset_id = match.group(1) if match else "unknown"
        if dataset_id not in datasets_map:
            datasets_map[dataset_id] = []
        datasets_map[dataset_id].append((mzml_path, csv_path, dataset_id))
    logger.info(f"Grouped {len(file_pairs)} files into {len(datasets_map)} datasets.")

    all_files_to_process = []
    dataset_path = os.path.join(lance_uri, table_name)
    first_write_done = False

    if cap_value == -1:
        logger.info("No cap applied (cap_value == -1). Processing all files.")
        for dataset_id, pairs_for_dataset in datasets_map.items():
            if dataset_id == "unknown" or (
                reporting_df is not None and dataset_id not in reporting_df.index
            ):
                if dataset_id != "unknown":
                    logger.warning(
                        f"Dataset {dataset_id} not in reporting CSV. Skipping report update."
                    )
            all_files_to_process.extend(pairs_for_dataset)
            if reporting_df is not None and dataset_id in reporting_df.index:
                reporting_df.loc[dataset_id, "number_of_mzmls_considered"] = len(pairs_for_dataset)
                reporting_df.loc[dataset_id, "number_of_MS2s_taken"] = reporting_df.loc[
                    dataset_id, "ms2"
                ]
    else:
        logger.info(f"\n--- Step 1: Applying sampling cap (cap = {cap_value}) ---")
        for dataset_id, pairs_for_dataset in datasets_map.items():
            if dataset_id == "unknown":
                logger.warning(
                    f"Skipping {len(pairs_for_dataset)} files with 'unknown' dataset ID."
                )
                continue
            if reporting_df is None or dataset_id not in reporting_df.index:
                logger.warning(f"Dataset {dataset_id} not in {reporting_csv_path}. Skipping.")
                continue

            if dataset_id in exceptional_dataset_ids:
                logger.warning(f"  Found exceptional dataset {dataset_id}. Processing serially...")
                sampled_pairs, files_processed, ms2s_taken = process_exceptional_dataset(
                    pairs_for_dataset, global_feature_stats, cap_value, max_peaks
                )
                logger.info(f"  {dataset_id}: Writing {len(sampled_pairs)} sampled pairs to Lance.")
                mode = "overwrite" if not first_write_done else "append"
                save_to_lance_batch(sampled_pairs, dataset_path, mode=mode)
                if sampled_pairs:
                    first_write_done = True
                reporting_df.loc[dataset_id, "number_of_mzmls_considered"] = files_processed
                reporting_df.loc[dataset_id, "number_of_MS2s_taken"] = ms2s_taken
                continue

            total_ms2_from_csv = reporting_df.loc[dataset_id, "ms2"]
            dataset_files_to_add = []
            dataset_ms2_count = 0

            if total_ms2_from_csv <= cap_value:
                logger.info(
                    f"  {dataset_id}: Has {total_ms2_from_csv} MS2 (<= cap). Taking all files."
                )
                actual_total_ms2 = 0
                for pair_with_id in pairs_for_dataset:
                    ms2_df = load_ms2_data(pair_with_id[1])
                    count = 0 if ms2_df.empty else len(ms2_df)
                    if count > 0:
                        dataset_files_to_add.append(pair_with_id)
                        actual_total_ms2 += count
                dataset_ms2_count = actual_total_ms2
            else:
                logger.info(
                    f"  {dataset_id}: Has {total_ms2_from_csv} MS2 (> cap). Sampling by file..."
                )
                ms2_counts_map = {}
                actual_total_ms2 = 0
                for pair_with_id in pairs_for_dataset:
                    csv_file = pair_with_id[1]
                    ms2_df = load_ms2_data(csv_file)
                    count = 0 if ms2_df.empty else len(ms2_df)
                    ms2_counts_map[pair_with_id] = count
                    actual_total_ms2 += count

                shuffled_pairs = random.sample(list(ms2_counts_map.keys()), len(ms2_counts_map))
                for pair_with_id in shuffled_pairs:
                    count = ms2_counts_map[pair_with_id]
                    if count == 0:
                        continue
                    if dataset_ms2_count + count <= cap_value:
                        dataset_ms2_count += count
                        dataset_files_to_add.append(pair_with_id)
                if dataset_ms2_count == 0 and actual_total_ms2 > 0:
                    for pair_with_id in shuffled_pairs:
                        if ms2_counts_map[pair_with_id] > 0:
                            logger.warning(
                                f"  {dataset_id}: First sampled file has {ms2_counts_map[pair_with_id]} scans. Taking it anyway."
                            )
                            dataset_ms2_count += ms2_counts_map[pair_with_id]
                            dataset_files_to_add.append(pair_with_id)
                            break
                logger.info(
                    f"  {dataset_id}: Sampled {dataset_ms2_count} MS2 from {len(dataset_files_to_add)} files."
                )

            reporting_df.loc[dataset_id, "number_of_mzmls_considered"] = len(dataset_files_to_add)
            reporting_df.loc[dataset_id, "number_of_MS2s_taken"] = dataset_ms2_count
            all_files_to_process.extend(dataset_files_to_add)

    logger.info(
        f"\n--- Step 2: Processing {len(all_files_to_process)} *normal* sampled files in batches of {batch_size} ---"
    )

    if not all_files_to_process and not first_write_done:
        logger.warning(
            f"No files or exceptional data selected for processing for {table_name}. Skipping."
        )
        return
    elif not all_files_to_process:
        logger.info("No *normal* files to process. Only exceptional data was written.")
        return

    for batch_idx in range(0, len(all_files_to_process), batch_size):
        batch_pairs = all_files_to_process[batch_idx : batch_idx + batch_size]
        logger.info(
            f"\n=== Processing batch {batch_idx // batch_size + 1} ({len(batch_pairs)} files) ==="
        )

        process_args = [
            (
                mzml_file,
                csv_file,
                dataset_id,
                global_feature_stats,
                idx,
                len(all_files_to_process),
                max_peaks,
            )
            for idx, (mzml_file, csv_file, dataset_id) in enumerate(batch_pairs, batch_idx + 1)
        ]

        if num_workers > 1:
            with Pool(processes=num_workers) as pool:
                results = pool.map(process_single_file, process_args)
        else:
            results = [process_single_file(args) for args in process_args]

        batch_data = []
        for result in results:
            batch_data.extend(result)

        mode = "append" if (batch_idx > 0 or first_write_done) else "overwrite"
        logger.info(f"Writing {len(batch_data)} records to Lance dataset (mode: {mode})")
        save_to_lance_batch(batch_data, dataset_path, mode)
        first_write_done = True
        del batch_data, results

    logger.info(f"\n✅ All batches processed for {table_name}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert mass spectrometry data (mzML + CSV) to Lance format",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--train_file_list", type=str, default="train_file_paths.txt")
    parser.add_argument("--val_file_list", type=str, default="validation_file_paths.txt")
    parser.add_argument("--lance_uri", type=str, default="./mass_spec_lance_store")
    parser.add_argument("--train_table", type=str, default="train_data")
    parser.add_argument("--val_table", type=str, default="validation_data")
    parser.add_argument("--workers", type=int, default=max(1, cpu_count() - 1))
    parser.add_argument("--max_peaks", type=int, default=400)
    parser.add_argument("--test_mode", action="store_true")
    parser.add_argument("--max_test_files", type=int, default=3)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_val", action="store_true")
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--cap_training_set", type=int, default=-1)
    parser.add_argument("--cap_val_set", type=int, default=-1)
    parser.add_argument("--training_set_csv", type=str, default="training_set.csv")
    parser.add_argument("--val_set_csv", type=str, default="val_set.csv")
    parser.add_argument("--exceptional_dataset_ids", type=str, nargs="*", default=[])
    parser.add_argument(
        "--add_filepath_column",
        action="store_true",
        help="(This is now default) Add mzml_filepath column.",
    )

    return parser.parse_args()


def load_reporting_df(csv_path: str) -> pd.DataFrame:
    """
    Load dataset reporting CSV with MS2 counts per dataset.
    
    Args:
        csv_path: Path to semicolon-delimited CSV with dataset_id and ms2 columns
        
    Returns:
        DataFrame indexed by dataset_id, or None if loading fails
    """
    if not os.path.exists(csv_path):
        logger.error(f"Reporting CSV not found: {csv_path}")
        return None
    try:
        df = pd.read_csv(csv_path, delimiter=";")

        if "ms2" in df.columns:
            df["ms2"] = pd.to_numeric(df["ms2"], errors="coerce").fillna(0).astype(int)

        df["number_of_mzmls_considered"] = (
            pd.to_numeric(df.get("number_of_mzmls_considered", 0), errors="coerce")
            .fillna(0)
            .astype(int)
        )
        df["number_of_MS2s_taken"] = (
            pd.to_numeric(df.get("number_of_MS2s_taken", 0), errors="coerce").fillna(0).astype(int)
        )

        df = df.set_index("dataset_id")
        return df
    except Exception as e:
        logger.error(f"Error loading reporting CSV {csv_path}: {e}")
        return None


if __name__ == "__main__":
    args = parse_args()
    start_time = datetime.now()
    logger.info(f"\n{'#' * 80}")
    logger.info(f"Mass Spectrometry Data → Lance Converter")
    logger.info(f"Started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"{'#' * 80}")
    logger.info(f"Configuration:")
    logger.info(f"  Lance URI: {args.lance_uri}")
    logger.info(f"  Workers: {args.workers}")
    logger.info(f"  Max peaks: {args.max_peaks}")
    logger.info(f"  Test mode: {args.test_mode}")
    if args.test_mode:
        logger.info(f"  Max test files: {args.max_test_files}")
    logger.info(f"  Train Cap: {args.cap_training_set if args.cap_training_set > 0 else 'None'}")
    logger.info(f"  Val Cap: {args.cap_val_set if args.cap_val_set > 0 else 'None'}")
    logger.info(f"  Train Report: {args.training_set_csv}")
    logger.info(f"  Val Report: {args.val_set_csv}")
    logger.info(f"  Exceptional Datasets: {args.exceptional_dataset_ids}")
    logger.info(f"{'#' * 80}")

    os.makedirs(args.lance_uri, exist_ok=True)

    train_report_df = None
    val_report_df = None
    if not args.skip_train:
        train_report_df = load_reporting_df(args.training_set_csv)
    if not args.skip_val:
        val_report_df = load_reporting_df(args.val_set_csv)

    training_stats = {}
    train_file_pairs = []
    val_file_pairs = []

    if not args.skip_train:
        if os.path.exists(args.train_file_list) and train_report_df is not None:
            train_file_pairs = load_file_pairs(
                args.train_file_list, args.test_mode, args.max_test_files, exclude_blank=True
            )
            if train_file_pairs:
                training_stats = compute_global_stats(train_file_pairs)
            else:
                logger.error("No training files found.")
                train_file_pairs = []
        else:
            if not os.path.exists(args.train_file_list):
                logger.warning(f"Training file list not found: {args.train_file_list}")
            if train_report_df is None:
                logger.warning(
                    f"Could not load or validate training report file: {args.training_set_csv}"
                )
            train_file_pairs = []

    if not args.skip_val:
        if os.path.exists(args.val_file_list) and val_report_df is not None:
            val_file_pairs = load_file_pairs(
                args.val_file_list, args.test_mode, args.max_test_files, exclude_blank=True
            )
        else:
            if not os.path.exists(args.val_file_list):
                logger.warning(f"Validation file list not found: {args.val_file_list}")
            if val_report_df is None:
                logger.warning(
                    f"Could not load or validate validation report file: {args.val_set_csv}"
                )

    if not training_stats and (not args.skip_train or not args.skip_val):
        logger.error("Could not compute training statistics. Aborting all processing.")
    else:
        if not args.skip_val and val_file_pairs:
            logger.info("\nProcessing validation set using TRAINING statistics...")
            process_file_list(
                file_pairs=val_file_pairs,
                global_feature_stats=training_stats,
                lance_uri=args.lance_uri,
                table_name=args.val_table,
                num_workers=args.workers,
                max_peaks=args.max_peaks,
                batch_size=args.batch_size,
                cap_value=args.cap_val_set,
                reporting_df=val_report_df,
                reporting_csv_path=args.val_set_csv,
                exceptional_dataset_ids=args.exceptional_dataset_ids,
            )
            val_report_df.reset_index().to_csv(args.val_set_csv, index=False, sep=";")
            logger.info(f"Updated validation report saved to {args.val_set_csv}")

        if not args.skip_train and train_file_pairs:
            process_file_list(
                file_pairs=train_file_pairs,
                global_feature_stats=training_stats,
                lance_uri=args.lance_uri,
                table_name=args.train_table,
                num_workers=args.workers,
                max_peaks=args.max_peaks,
                batch_size=args.batch_size,
                cap_value=args.cap_training_set,
                reporting_df=train_report_df,
                reporting_csv_path=args.training_set_csv,
                exceptional_dataset_ids=args.exceptional_dataset_ids,
            )
            train_report_df.reset_index().to_csv(args.training_set_csv, index=False, sep=";")
            logger.info(f"Updated training report saved to {args.training_set_csv}")

    end_time = datetime.now()
    duration = end_time - start_time
    logger.info(f"\n{'#' * 80}")
    logger.info(f"✨ All processing complete!")
    logger.info(f"Lance database location: {args.lance_uri}")
    logger.info(f"Total time: {duration}")

    for table_name in [args.train_table, args.val_table]:
        dataset_path = os.path.join(args.lance_uri, table_name)
        if os.path.exists(dataset_path):
            try:
                ds = lance.dataset(dataset_path)
                logger.info(f"{table_name}: {ds.count_rows():,} records")
            except Exception as e:
                logger.warning(f"Could not read final dataset {table_name}: {e}")
