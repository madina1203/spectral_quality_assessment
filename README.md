# Positive-Unlabeled Learning for Predicting Small Molecule MS2 Identifiability from MS1 Context and Acquisition Parameters

This repository contains the complete pipeline for training a model that predicts the identifiability of MS2 spectra based on provided MS1 spectra and instrument configurations used to generate consecutive MS2 scans.

The approach utilizes **Positive-Unlabeled (PU) Learning** to train models using only positive examples (library-matched spectra) and unlabeled data. It features a **Transformer-based architecture** (using the `depthcharge` library) to encode spectra and incorporates acquisition parameters as features. To handle large-scale spectral data efficiently, the project utilizes the **Lance** data format.

## Repository Structure

```
spectral_quality_assessment/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── environment.yml                    # Conda environment specification
│
├── src/                               # Core model implementations
│   └── transformers/
│       ├── model_bce_loss_one_hot.py              # BCE loss model (polarity-aware)
│       └── model_nn_pu_loss_detach_diff_polarity.py  # nnPU loss model
│
├── scripts/                           # Python scripts for each pipeline step
│   ├── data_download/
│   │   └── msv_download_datasets.py   # Download datasets from MassIVE using CSV metadata
│   │
│   ├── data_preprocessing/
│   │   ├── split_library.py           # Split GNPS library by polarity
│   │   ├── process_raw.py             # Convert .raw → .mzML, run ScanHeadsman
│   │   ├── library_matching_diff_polarity.py  # GNPS library matching
│   │   ├── data_processing_pipeline.py        # Complete data processing pipeline
│   │   └── create_lance_add_one_hot.py        # Create Lance dataset
│   │
│   ├── training/
│   │   ├── training_bce_loss_diff_polarity_one_hot.py  # Train BCE models
│   │   └── training_nn_pu_loss_detach_diff_polarity.py # Train nnPU model
│   │
│   └── inference/
│       ├── predict_lance_all.py                        # Run predictions
│       └── predict_lance_diff_polarity_one_hot.py      # Polarity-specific predictions
│
├── slurm_scripts/                     # Cluster job submission scripts
│   ├── data_download/
│   │   └── msv_download.sh
│   ├── data_preprocessing/
│   │   ├── run_process_raw.sh         # Convert raw files to mzML
│   │   ├── library_matching.sh        # Run library matching
│   │   ├── run_processing_pipeline.sh # Complete processing pipeline
│   │   └── run_build_lance.sh         # Build Lance datasets
│   ├── training/
│   │   ├── run_train_bce_loss_diff_polarity.sh
│   │   └── run_train_nnpu_loss.sh
│   └── inference/
│       ├── run_predict_lance.sh
│       └── run_predict_lance_val.sh
│
├── checkpoints/                       # Pre-trained model checkpoints (download from Zenodo)
│   └── README.md                      # Download instructions
│
├── tools/                             # External tools (download separately)
│   └── README.md                      # Installation guide for ThermoRawFileParser & ScanHeadsman
│
├── data/                              # Data and metadata
│   ├── README.md                      # Data directory documentation
│   ├── metadata/                      # Dataset metadata (in repo)
│   │   ├── train_datasets.csv
│   │   ├── val_datasets.csv
│   │   ├── test_1_metadata.csv
│   │   ├── test_2_metadata.csv
│   │   └── test_3_metadata.csv
│   ├── libraries/                     # GNPS libraries (download & split)
│   │   └── README.md                  # Download & split instructions
│   ├── file_paths/                    # [User-created] Lists of local file paths
│   │   ├── file_paths_train.txt
│   │   └── file_paths_val.txt
│   ├── lance_datasets/                # [External] Training & validation Lance data (download from Zenodo)
│   ├── lance_data_test_set_1/         # [External] Test Set 1 (download from Zenodo)
│   ├── lance_data_test_set_2/         # [External] Test Set 2 (download from Zenodo)
│   └── lance_data_test_set_3/         # [External] Test Set 3 (download from Zenodo)
│
└── docs/                              # Detailed documentation
    ├── DATA_PREPROCESSING.md          # Preprocessing pipeline
    ├── TRAINING.md                    # Model training guide
    └── INFERENCE.md                   # Running predictions
```

## Installation

### Prerequisites

- Python 3.11+
- CUDA 12.8+ (for GPU training)
- Conda 
- Access to a computing cluster (recommended for full pipeline)

### Environment Setup

```bash
# Clone the repository
git clone git@github.com:madina1203/spectral_quality_assessment.git
cd spectral_quality_assessment

# Create and activate conda environment
conda env create -f environment.yml
conda activate instrument_setting
```

**External Tools** :
If you intend to process **raw data** (convert `.raw` files to the Lance format used by the model), you must install the following tools in the `tools/` directory. If you only plan to use the pre-processed data from Zenodo, these are not required.

* **ThermoRawFileParser**: For converting `.raw` to `.mzML`.
* **ScanHeadsman**: For extracting MS1 spectra.

Please refer to [`tools/README.md`](tools/README.md) for installation instructions.
> **Note**: External tools are not required if using pre-processed data from Zenodo

## Data and Model Availability

All datasets and pre-trained models are hosted on Zenodo.

### 1. Pre-trained Models
Download the checkpoints to the `checkpoints/` directory.

* **nnPU Model (Recommended)**: The final model trained with non-negative PU loss.
* **BCE Models**: Polarity-specific models used for prior estimation.

[**Download Models (Zenodo Link)**](https://doi.org/10.5281/zenodo.18266932)

### 2. Datasets (Lance Format)
If you wish to reproduce the training or testing results without processing raw files, download the pre-processed Lance datasets.

* **Training/Validation Data**: `lance_data_train_validation.tar.gz`
* **Test Sets**: `lance_data_test_set_1.tar.gz`, `lance_data_test_set_2.tar.gz`, `lance_data_test_set_3.tar.gz`

[**Download Datasets (Zenodo Link)**](https://doi.org/10.5281/zenodo.18266932)

## Inference

For detailed inference instructions, see [`docs/INFERENCE.md`](docs/INFERENCE.md).

You can run the model on the provided test set or on your own custom data.

### Option A: Inference on Provided Test Sets
To evaluate the model on the provided Test Set 3 (Lance format downloaded from Zenodo: `lance_data_test_set_3`):

```bash
sbatch slurm_scripts/inference/run_predict_lance.sh
```

**Note**: Before running, edit `slurm_scripts/inference/run_predict_lance.sh` to configure:
- Paths to checkpoint and dataset
- Output directory
- Batch size and other parameters

### Option B: Inference on Custom Data
To run the model on your own data, you must first convert your `.raw` or `.mzML` files into the Lance format required by the model.

1. **Preprocess Data**: Follow the instructions in [`docs/DATA_PREPROCESSING.md`] to generate a Lance dataset from your files.
2. **Run Prediction**: Edit `slurm_scripts/inference/run_predict_lance.sh` to point to your custom dataset, then run:

   ```bash
   sbatch slurm_scripts/inference/run_predict_lance.sh
   ```

   Or run directly with Python:

   ```bash
   python scripts/inference/predict_lance_all.py \
       --checkpoint_path checkpoints/best_model_nnpu.ckpt \
       --lance_path path/to/your/custom_lance_dataset \
       --output_csv your_results.csv
   ```

**Output**: The script generates a CSV containing the `original_index`, `probability` (quality score), `mzml_filepath`, and `scan_number`.

## Training

For detailed training instructions, see [`docs/TRAINING.md`](docs/TRAINING.md).

Training involves a multi-stage pipeline designed for PU learning. You may train using the provided Zenodo datasets or your own preprocessed Lance datasets.

### Computational Environment

All models were trained and tested on a high-performance computing (HPC) cluster node equipped with:
- **CPUs**: Dual Intel Xeon Gold 5320 (2.20 GHz)
- **GPUs**: 4× NVIDIA A100 (80 GB)


Training was performed using distributed data parallelism across 2 GPUs with the following SLURM configuration:
```bash
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
```

### Training Pipeline Summary
1. **BCE Pre-training**: Train separate models for positive and negative polarities using Binary Cross-Entropy loss.
2. **Prior Estimation**: Use the best BCE models to estimate the class prior ($\pi$) on a held-out validation set (Test Set 1).
3. **nnPU Training**: Train the final model using the estimated priors.

### 1. BCE Pre-training (Polarity Specific)
Train separate models for positive (`--polarity 1`) and negative (`--polarity 0`) modes.

```bash
sbatch slurm_scripts/training/run_train_bce_loss_diff_polarity.sh
```

**Note**: Before running, edit `slurm_scripts/training/run_train_bce_loss_diff_polarity.sh` to configure:
- Polarity setting (`--polarity 0` for negative, `--polarity 1` for positive)
- Paths to Lance datasets

Use the trained BCE models to predict probabilities on Test Set 1, then calculate the average probability to estimate the priors.

```bash
# Predict on Test Set 1
python scripts/inference/predict_lance_diff_polarity_one_hot.py \
    --checkpoint_path logs/bce_pos/best_model.ckpt \
    --lance_path data/lance_data_test_set_1 \
    --output_csv predictions_prior_est.csv \
    --polarity 1
    
  ```

### 3. nnPU model Training
Train the final model using the priors estimates.

```bash
sbatch slurm_scripts/training/run_train_nnpu_loss.sh
```

**Note**: Before running, edit `slurm_scripts/training/run_train_nnpu_loss.sh` to configure:
- Prior estimates (`--prior_pos` and `--prior_neg`)
- Paths to Lance datasets
- Hyperparameters (learning rates, batch size, etc.)

### (Optional) Preparing Data from Scratch

For detailed data preprocessing instructions, see [`docs/DATA_PREPROCESSING.md`](docs/DATA_PREPROCESSING.md).

If you want to process raw data and create the Lance datasets yourself:

#### 1. Install External Tools

Install ThermoRawFileParser and ScanHeadsman (required for .raw file processing):

```bash
# See detailed installation instructions
cat tools/README.md

# Quick install (example for ThermoRawFileParser)
cd tools
wget https://github.com/compomics/ThermoRawFileParser/releases/download/v1.4.4/ThermoRawFileParser1.4.4.zip
unzip ThermoRawFileParser1.4.4.zip -d ThermoRawFileParser/
```

For detailed instructions, see [`tools/README.md`](tools/README.md)

#### 2. Download GNPS Spectral Libraries

For detailed instructions on downloading and processing GNPS spectral libraries, see [`data/libraries/README.md`](data/libraries/README.md).

#### 3. Run Data Processing Pipeline

See detailed instructions in `docs/DATA_PREPROCESSING.md` for:
- Converting raw files to mzML and mgf
- Running library matching
- Creating Lance datasets

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
