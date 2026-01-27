#!/usr/bin/env python3
"""
Spectral Interference Analysis 

Analyzes MS/MS spectra to quantify non-isotopic interference in the precursor
isolation window. Calculates the percentage of Total Ion Current (TIC) attributed
to peaks that are NOT part of the precursor's isotope envelope.


Usage:
    python percentage_tic_isotope.py --input_csv predictions.csv --delta 1.003355 --ppm 20
"""

import os
import argparse
import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from tqdm import tqdm
from scipy import stats

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def analyze_scan_quality(df, delta, ppm_tolerance):
    """
    Calculate interference TIC percentage for singly-charged precursors.
    
    For each MS/MS scan, computes the percentage of total ion current (TIC) 
    in the precursor isolation window (m/z > precursor - 5) that does NOT 
    belong to the precursor's isotope envelope.
    
    Args:
        df: DataFrame with columns 'mzml_filepath' and 'scan_number'
        delta: Mass difference between isotopes (typically 1.003355 Da for C13)
        ppm_tolerance: Mass tolerance in ppm for isotope peak matching
        
    Returns:
        dict: Mapping of (mzml_filepath, scan_number) to interference metrics
    """
    results = {}
    grouped = df.groupby('mzml_filepath')

    count_charge_1 = 0
    count_unknown = 0
    count_total_processed = 0

    for mzml_file, group in tqdm(grouped, desc="Analyzing charge 1 spectra"):
        if not os.path.exists(mzml_file):
            continue

        scans_to_find = set(group['scan_number'].astype(int))

        try:
            for spectrum in read_one_spectrum(mzml_file):
                if not scans_to_find:
                    break

                scan_num = spectrum.get("_scan_number")

                if scan_num in scans_to_find:
                    charge = spectrum.get("charge")

                    count_total_processed += 1
                    if charge == 0:
                        count_unknown += 1
                    elif abs(charge) == 1:
                        count_charge_1 += 1

                    # Only analyze singly-charged precursors (excludes unknown charge=0)
                    if abs(charge) == 1:
                        peaks = spectrum.get("peaks", np.array([]))
                        pmz = spectrum.get("precursor_mz")

                        tic_interference_pct = 0.0
                        if len(peaks) > 0 and pmz > 0:
                            total_intensity = np.sum(peaks[:, 1])

                            # Select peaks above precursor m/z - 5 Da
                            mask_window = (peaks[:, 0] > (pmz - 5))
                            window_peaks = peaks[mask_window]

                            if len(window_peaks) > 0:
                                is_interference = np.ones(len(window_peaks), dtype=bool)

                                # Exclude isotope peaks (M-5 to M+5) from interference
                                for i in range(-5, 6):
                                    iso_mz = pmz + (i * delta)
                                    da_tol = (iso_mz * ppm_tolerance) / 1e6

                                    in_iso_zone = (window_peaks[:, 0] > (iso_mz - da_tol)) & \
                                                  (window_peaks[:, 0] < (iso_mz + da_tol))
                                    is_interference &= ~in_iso_zone

                                interference_sum = np.sum(window_peaks[is_interference, 1])

                                if total_intensity > 0:
                                    tic_interference_pct = (interference_sum / total_intensity) * 100

                            results[(mzml_file, scan_num)] = {
                                'tic_above_pct': tic_interference_pct
                            }

                        scans_to_find.remove(scan_num)
        except Exception as e:
            logger.error(f"Error reading {mzml_file}: {e}")

    print("\n" + "=" * 40)
    print("CHARGE STATE SUMMARY")
    print(f"Total Scans Processed: {count_total_processed}")
    print(f"  - Charge 1 Scans:      {count_charge_1}")
    print(f"  - Unknown Charge:      {count_unknown}")
    print(f"  - Other Charges (>1):  {count_total_processed - count_charge_1 - count_unknown}")
    print("=" * 40 + "\n")

    return results


def read_one_spectrum(file_input):
    """
    Generator that yields parsed MS/MS spectra from an mzML file.
    
    Args:
        file_input: Path to mzML file
        
    Yields:
        dict: Spectrum data containing:
            - _ms_level: MS level (1, 2, etc.)
            - _scan_number: 1-based scan number
            - peaks: numpy array of shape (n, 2) with m/z and intensity
            - charge: Precursor charge state (0 if unknown)
            - precursor_mz: Precursor m/z (-1 if unavailable)
    """
    import pyteomics.mzml
    with pyteomics.mzml.read(str(file_input)) as reader:
        for scan, spec in enumerate(reader):
            mz = spec['m/z array']
            intensity = spec['intensity array']
            peaks = np.asarray([mz, intensity], dtype=np.float32).T

            try:
                precursor_info = spec['precursorList']['precursor'][0]['selectedIonList']['selectedIon'][0]
                pmz = precursor_info['selected ion m/z']
                charge = precursor_info.get('charge state', 0)
            except (KeyError, IndexError):
                pmz, charge = -1, 0

            if 'negative scan' in spec:
                charge = -abs(charge)

            yield {
                "_ms_level": spec.get("ms level", 0),
                "_scan_number": scan + 1,
                "peaks": peaks,
                "charge": charge,
                "precursor_mz": pmz
            }


def main():
    """
    Main entry point for spectral interference analysis.
    
    Reads prediction CSV, filters for unlabeled spectra (label=0), calculates
    interference metrics for charge-1 precursors, and generates correlation plots.
    """
    parser = argparse.ArgumentParser(
        description="Analyze spectral interference excluding isotope peaks"
    )
    parser.add_argument('--input_csv', type=str, required=True, 
                        help="Path to CSV with predictions (requires: mzml_filepath, scan_number, label, probability)")
    parser.add_argument('--delta', type=float, default=1.003355, 
                        help="Isotope mass difference in Da (default: 1.003355 for C13)")
    parser.add_argument('--ppm', type=float, default=20.0, 
                        help="PPM tolerance for isotope peak masking (default: 20)")
    args = parser.parse_args()

    output_dir = 'results/plots_14_01'
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(args.input_csv)
    df_neg = df[df['label'] == 0].copy()

    logger.info(f"Analyzing scans with delta={args.delta} Da and ppm={args.ppm}...")
    quality_results = analyze_scan_quality(df_neg, args.delta, args.ppm)

    df_neg['tic_above_pct'] = df_neg.apply(
        lambda x: quality_results.get((x['mzml_filepath'], int(x['scan_number'])), {}).get('tic_above_pct', np.nan),
        axis=1
    )

    df_c1 = df_neg.dropna(subset=['tic_above_pct']).copy()
    logger.info(f"Final charge-1 dataset size: {len(df_c1)} scans")

    # Correlation analysis
    if len(df_c1) > 1:
        pearson_r, pearson_p = stats.pearsonr(df_c1['probability'], df_c1['tic_above_pct'])
        spearman_rho, spearman_p = stats.spearmanr(df_c1['probability'], df_c1['tic_above_pct'])
        
        print("\n" + "=" * 40)
        print("CORRELATION ANALYSIS (Charge 1 Only)")
        print(f"Pearson r:   {pearson_r:.4f} (p = {pearson_p:.2e})")
        print(f"Spearman ρ:  {spearman_rho:.4f} (p = {spearman_p:.2e})")

        df_noisy = df_c1[df_c1['tic_above_pct'] > 5]
        if not df_noisy.empty:
            r_noisy, p_noisy = stats.pearsonr(df_noisy['probability'], df_noisy['tic_above_pct'])
            spearman_rho_noisy, spearman_p_noisy = stats.spearmanr(df_noisy['probability'], df_noisy['tic_above_pct'])
            print(f"\nFiltered (%TIC > 5%):")
            print(f"Pearson r:   {r_noisy:.4f} (p = {p_noisy:.2e})")
            print(f"Spearman ρ:  {spearman_rho_noisy:.4f} (p = {spearman_p_noisy:.2e})")
        print("=" * 40 + "\n")

    # Visualization
    plt.rcParams.update({"font.size": 11})

    # Plot 1: All charge-1 data
    fig, ax = plt.subplots(figsize=(7, 4.5))
    h = ax.hist2d(df_c1['probability'], df_c1['tic_above_pct'],
                  bins=[25, 25], cmap='viridis', cmin=1, norm=colors.LogNorm())

    cbar = plt.colorbar(h[3], ax=ax)
    cbar.set_label('Count (Log Scale)', labelpad=15)

    ax.axvline(x=0.767, color='black', linestyle='--', linewidth=2,
               label='Decision Threshold = 0.767')
    ax.set_title(f'Prediction Probability vs Spectral Interference (PPM={args.ppm})',
                 fontweight='bold', pad=12)
    ax.set_xlabel('Predicted Probability $P(y=1|x)$')
    ax.set_ylabel('% Interference TIC')
    ax.set_xlim(0, 1)
    ax.legend(frameon=True, loc='upper right', facecolor='white', edgecolor='black', framealpha=0.9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/tic_peaks_all_histogram.png", dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("Saved plot: tic_peaks_all_histogram.png")

    df_c1.to_csv("processed_quality_metrics_c1.csv", index=False)

    # Plot 2: Filtered data (TIC > 5%)
    df_filtered = df_c1[df_c1['tic_above_pct'] > 5]

    if not df_filtered.empty:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        h = ax.hist2d(df_filtered['probability'], df_filtered['tic_above_pct'],
                      bins=[25, 25], cmap='plasma', cmin=1, norm=colors.LogNorm())

        cbar = plt.colorbar(h[3], ax=ax)
        cbar.set_label('Count (Log Scale)', labelpad=15)

        ax.axvline(x=0.767, color='black', linestyle='--', linewidth=2,
                   label='Decision Threshold = 0.767')
        ax.set_title(f'Filtered (TIC > 5%): Probability vs Interference (PPM={args.ppm})',
                     fontweight='bold', pad=12)
        ax.set_xlabel('Predicted Probability $P(y=1|x)$')
        ax.set_ylabel('% Interference TIC')
        ax.set_xlim(0, 1)
        ax.legend(frameon=True, loc='upper right', facecolor='white', edgecolor='black', framealpha=0.9)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.tight_layout()
        plt.savefig(f"{output_dir}/tic_2d_hist_higher_than_5.png", dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("Saved plot: tic_2d_hist_higher_than_5.png")
    else:
        logger.warning("No scans found with TIC > 5%, skipping filtered plot")


if __name__ == '__main__':
    main()