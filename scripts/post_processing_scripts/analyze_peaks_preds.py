#!/usr/bin/env python3
"""
MS2 Spectral Quality Analysis

Analyzes MS2 spectra for quality indicators (few peaks, high charge state)
and generates probability distribution histograms for each quality metric.

Usage:
    python analyze_peaks_preds.py --input_csv predictions.csv --output_dir results/
"""

import os
import argparse
import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def analyze_scan_quality(df):
    """
    Extract quality metrics from MS2 spectra.
    
    Args:
        df: DataFrame with 'mzml_filepath' and 'scan_number' columns
        
    Returns:
        dict: Mapping of (mzml_filepath, scan_number) to quality flags
    """
    results = {}
    grouped = df.groupby('mzml_filepath')

    for mzml_file, group in tqdm(grouped, desc="Analyzing mzML files"):
        if not os.path.exists(mzml_file):
            logger.warning(f"File not found: {mzml_file}")
            continue

        scans_to_find = set(group['scan_number'].astype(int))

        try:
            for spectrum in read_one_spectrum(mzml_file):
                if not scans_to_find:
                    break

                if spectrum.get("_ms_level") == 2:
                    scan_num = spectrum.get("_scan_number")

                    if scan_num in scans_to_find:
                        peaks = spectrum.get("peaks", np.array([]))
                        charge = spectrum.get("charge")

                        few_peaks = len(peaks) < 3
                        high_charge = abs(charge) > 2

                        results[(mzml_file, scan_num)] = {
                            'few_peaks': few_peaks,
                            'high_charge': high_charge
                        }
                        scans_to_find.remove(scan_num)

        except Exception as e:
            logger.error(f"Error reading {mzml_file}: {e}")

    return results


def read_one_spectrum(file_input):
    """
    Generator yielding parsed spectra from an mzML file.
    
    Args:
        file_input: Path to mzML file
        
    Yields:
        dict: Spectrum data with ms_level, scan_number, peaks, charge, precursor_mz
    """
    import pyteomics.mzml
    with pyteomics.mzml.read(str(file_input)) as reader:
        for scan, spec in enumerate(reader):
            mz = spec['m/z array']
            intensity = spec['intensity array']
            peaks = np.asarray([mz, intensity], dtype=np.float32).T

            try:
                precursor_mz = spec['precursorList']['precursor'][0]['selectedIonList']['selectedIon'][0][
                    'selected ion m/z']
            except (KeyError, IndexError):
                precursor_mz = -1

            try:
                charge = spec['precursorList']['precursor'][0]['selectedIonList']['selectedIon'][0]['charge state']
            except (KeyError, IndexError):
                charge = 1

            yield {
                "_ms_level": spec.get("ms level", 0),
                "_scan_number": scan + 1,
                "peaks": peaks,
                "charge": charge,
                "precursor_mz": precursor_mz
            }


def plot_quality_histograms(df, output_dir: str, threshold: float = 0.767):
    """
    Generate probability distribution histograms for each quality metric.
    
    Args:
        df: DataFrame with 'probability' and quality flag columns
        output_dir: Directory to save output plots
        threshold: Decision threshold for visualization (default: 0.767)
    """
    plt.rcParams.update({"font.size": 11})
    os.makedirs(output_dir, exist_ok=True)

    plot_configs = [
        ('few_peaks', 'Distribution of Spectra with Few Fragment ions (<3)', '#d7191c', 'few_peaks_distribution.png'),
        ('high_charge', 'Distribution of Spectra with High Charge (>2)', '#2c7bb6', 'high_charge_distribution.png')
    ]

    bin_edges = np.linspace(0, 1, 51)

    for col_name, title, color, filename in plot_configs:
        fig, ax = plt.subplots(1, 1, figsize=(7, 4.5))

        subset = df[df[col_name] == True]['probability']

        ax.hist(subset, bins=bin_edges, color=color, alpha=0.7, edgecolor='white',
                label=f'Number of samples(n={len(subset)})', density=False)
        ax.axvline(threshold, color='black', linestyle='--', linewidth=2,
                   label=f'Threshold = {threshold}')

        ax.set_xlabel('Predicted Probability $P(y=1|x)$')
        ax.set_ylabel('Frequency')
        ax.set_title(title, fontweight='bold', pad=12)
        ax.set_xlim(0, 1)
        ax.margins(x=0)
        ax.legend(frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.18), ncol=2)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.tight_layout()
        output_path = os.path.join(output_dir, filename)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved: {output_path}")
        plt.close(fig)


def main():
    """Main entry point for MS2 quality analysis."""
    parser = argparse.ArgumentParser(description='Analyze MS2 scan quality metrics and generate distribution plots.')
    parser.add_argument('--input_csv', type=str, required=True, 
                        help='Path to CSV with predictions (requires: mzml_filepath, scan_number, label, probability)')
    parser.add_argument('--output_dir', type=str, default='results/quality_analysis',
                        help='Directory for output plots (default: results/quality_analysis)')
    parser.add_argument('--threshold', type=float, default=0.767,
                        help='Decision threshold for probability grouping (default: 0.767)')
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    df_neg = df[df['label'] == 0].copy()

    logger.info(f"Analyzing {len(df_neg)} unlabeled spectra...")
    quality_metrics = analyze_scan_quality(df_neg)

    df_neg['few_peaks'] = df_neg.apply(
        lambda x: quality_metrics.get((x['mzml_filepath'], int(x['scan_number'])), {}).get('few_peaks'), axis=1)
    df_neg['high_charge'] = df_neg.apply(
        lambda x: quality_metrics.get((x['mzml_filepath'], int(x['scan_number'])), {}).get('high_charge'), axis=1)

    df_neg = df_neg.dropna(subset=['few_peaks'])

    df_neg['Subgroup'] = np.where(df_neg['probability'] >= args.threshold, 'High Prob', 'Low Prob')
    summary = df_neg.groupby('Subgroup').agg({
        'few_peaks': ['sum', 'mean'],
        'high_charge': ['sum', 'mean']
    })
    
    summary_path = os.path.join(args.output_dir, "quality_summary_report.csv")
    os.makedirs(args.output_dir, exist_ok=True)
    summary.to_csv(summary_path)
    logger.info(f"Saved summary: {summary_path}")
    print(summary)

    plot_quality_histograms(df_neg, args.output_dir, args.threshold)


if __name__ == '__main__':
    main()