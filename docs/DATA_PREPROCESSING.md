# Data Preprocessing Pipeline

This document describes the complete data preprocessing pipeline from .raw MS files to Lance dataset format.

## Pipeline Overview

```
.raw files → .mzML, .mgf, .csv → Library Matching → Data Processing → Lance Dataset
```

**Quick Start** (complete workflow):

```bash
# Step 1: Convert .raw files and extract instrument settings
sbatch slurm_scripts/data_preprocessing/run_process_raw.sh

# Step 2: Run library matching to label spectra
sbatch slurm_scripts/data_preprocessing/library_matching.sh

# Step 3a: Process and prepare data for Lance creation
sbatch slurm_scripts/data_preprocessing/run_processing_pipeline.sh

# Step 3b: Create Lance datasets
sbatch slurm_scripts/data_preprocessing/run_build_lance.sh
```



## Prerequisites

### External Tools (Download Required)

- **ThermoRawFileParser** - Converts .raw files to .mzML and .mgf formats
- **ScanHeadsman** - Extracts instrument settings and scan metadata

See `tools/README.md` for download and installation instructions.



### Data Files

- **GNPS spectral library** - Download from GNPS2 (see `data/libraries/README.md`)
- **Raw .raw files** - Download from MassIVE (see `data/README.md`)

---

## Step 1: Convert .raw Files and Extract Settings

The `run_process_raw.sh` script automatically handles:
- ✅ .raw → .mzML conversion (using ThermoRawFileParser)
- ✅ .raw → .mgf conversion (using ThermoRawFileParser)
- ✅ .raw → .csv conversion (instrument settings extraction using ScanHeadsman)

```bash
# Process all .raw files in MSV folders
cd /path/to/your/working/directory
sbatch slurm_scripts/data_preprocessing/run_process_raw.sh
```


**Configuration** (edit the script if needed):
```bash
# Default input directory (where MSV folders are located)
MSV_PARENT_DEFAULT="/path/to/your/working/directory/data"

# Paths to external tools
SCANHEADSMAN="/path/to/tools/ScanHeadsman/ScanHeadsman.exe"
THERMORAWFILEPARSER="/path/to/tools/ThermoRawFileParser/ThermoRawFileParser.exe"
```

**Requirements**:
- ThermoRawFileParser and ScanHeadsman must be installed (see `tools/README.md`)
- Mono runtime (on Linux/macOS) for running .NET executables



**Output Files**:

For each `.raw` file (e.g., `sample.raw`), the script creates:
- `sample.mzML` - Spectral data in mzML format
- `sample.mgf` - Spectral data in MGF format (for library matching)
- `sample.csv` - Instrument settings and scan metadata

---

## Step 2: Library Matching for Data Labeling

The `library_matching.sh` script labels MS2 spectra by matching them against the GNPS spectral library using the matchms library.

```bash
sbatch slurm_scripts/data_preprocessing/library_matching.sh
```

**What this script does**:
- Matches MS2 spectra from .mgf files against GNPS spectral libraries (positive and negative modes)
- Uses cosine similarity to find the best match for each spectrum
- Processes spectra separately by polarity (positive/negative ionization mode)

**Matching Criteria**:
- **Cosine similarity > 0.7**
- **Minimum 6 matching peaks**
- **Precursor m/z tolerance**: 0.05 Da

**Requirements**:
- GNPS libraries must be downloaded and split (see `data/libraries/README.md`)
- `.mgf` files from Step 1 must be available

**Configuration** (edit the script if needed):
```bash
python scripts/data_preprocessing/library_matching_diff_polarity.py \
    --msv_folder /path/to/your/working/directory/data \
    --reference_mgf_positive data/libraries/spectral_db_positive.mgf \
    --reference_mgf_negative data/libraries/spectral_db_negative.mgf \
    --output_tsv results/spectral_matching_results.tsv \
    --num_cpus $SLURM_CPUS_PER_TASK
```

**Output**:

TSV file containing with library matching results. 

## Step 3: Process Data and Create Lance Dataset

After library matching, process all MSV folders and create the final Lance dataset.

### Step 3a: Process Files (Prepare for Lance Creation)

```bash
sbatch slurm_scripts/data_preprocessing/run_processing_pipeline.sh
```

This script processes all MSV folders, organizes the data structure, and labels samples based on GNPS library matching: samples with matches are assigned Label=1, while unmatched samples are assigned Label=0. Finally, it prepares the files for Lance dataset creation. 

### Step 3b: Create Lance Dataset for training the model

```bash
sbatch slurm_scripts/data_preprocessing/run_build_lance.sh
```

This combines mzML spectra, instrument settings, and labels into an efficient Lance format.



**Configuration** (edit `run_build_lance.sh` if needed):

```bash
python scripts/data_preprocessing/create_lance_add_one_hot.py \
    --train_file_list data/file_paths/file_paths_train.txt \
    --val_file_list data/file_paths/file_paths_val.txt \
    --lance_uri data/lance_data_train_validation \
    --train_table train_data \
    --val_table validation_data \
    --workers 16 \
    --cap_training_set 300000 \
    --cap_val_set 100000 \
    --training_set_csv data/metadata/train_datasets.csv \
    --val_set_csv data/metadata/val_datasets.csv
```

**Key Arguments:**

- `--train_file_list` / `--val_file_list`: Text files listing mzML files and their annotations (**User must create**)
- `--lance_uri`: Output directory for Lance dataset
- `--train_table` / `--val_table`: Table names within the Lance dataset
- `--workers`: Number of parallel workers
- `--cap_training_set` / `--cap_val_set`: Maximum spectra per split 

**Creating File Lists (Required Before Running)**:

The `--train_file_list` and `--val_file_list` text files must be created manually by the user. Each line should contain the path to an `.mzML` file and its corresponding `_annotated.csv` file, **comma-separated**, where `_annotated.csv` file is created by `data_processing_pipeline.py`.

**File Format**:
```
/path/to/MSV000012345/folder1/sample1.mzML,/path/to/MSV000012345/folder1/sample1_annotated.csv
/path/to/MSV000012345/folder2/sample2.mzML,/path/to/MSV000012345/folder2/sample2_annotated.csv
/path/to/MSV000067890/folder1/sample3.mzML,/path/to/MSV000067890/folder1/sample3_annotated.csv
```



**How to Create**:

1. **Identify Dataset IDs**: Check `data/metadata/train_datasets.csv` or `data/metadata/val_datasets.csv` to find which MassIVE dataset IDs (e.g., MSV000084346) belong to training or validation split.

2. **Locate Processed Files**: After running `run_processing_pipeline.sh`, your processed data will be in `data/new_data/MSV*/` directories. Each dataset will contain:
   - `.mzML` files (spectral data)
   - `_annotated.csv` files (labels library matching and instrument setting columns)
**Note**: Files in the `invalid/` folder should be ignored, as these `.mzML` files contain only MS1 data.
   - 
3. **Create File Lists**: For each dataset ID in your split, find all `.mzML` files and create a comma-separated pair with their corresponding `_annotated.csv` file.

4. **Save to File**: Save the list to `data/file_paths/file_paths_train.txt` (for training) or `data/file_paths/file_paths_val.txt` (for validation).



**Note**: Ensure you only include dataset IDs that belong to the correct split (training or validation) based on  metadata CSV files.


### Feature Encoding

**Numerical features** (standardized using training set statistics):
```python
normalized = (value - mean) / std
```

**Categorical features** (one-hot encoded):
- Polarity: [negative, positive]
- Activation Type: [CID, HCD, ETD, etc.]

Final `instrument_settings` vector concatenates:
```
[normalized_numerical_features (N), one_hot_polarity (2), one_hot_activation (K)]
```

### Data Sampling Strategy

**Training set** (42 datasets):
- Sample up to 300,000 MS2 scans per dataset
- Random sampling to balance dataset contributions

**Validation set** (11 datasets):
- Sample up to 100,000 MS2 scans per dataset

**Test sets** (7, 5, 7 datasets):
- Sample up to 100,000 MS2 scans per dataset

---

## Creating Lance Datasets for Test Sets

For creating test set Lance datasets, use the dedicated script `create_lance_test_one_hot.py`. This script computes feature statistics from training data but only creates Lance datasets for test sets.

**Key Features**:
- Computes normalization statistics from training files (required for consistent feature scaling)
- Creates Lance dataset only for test data (training Lance is not created)
- Extracts precursor m/z directly from mzML files
- Excludes blank files automatically

**Usage**:

```bash
python scripts/data_preprocessing/create_lance_test_one_hot.py \
    --train_file_list data/file_paths/file_paths_train.txt \
    --test_file_list data/file_paths/file_paths_test.txt \
    --lance_uri data/lance_data_test_set \
    --test_table test_data \
    --workers 16 \
    --cap_test_set 100000 \
    --test_set_csv data/metadata/test_datasets.csv \
    --skip_train
```

**Key Arguments**:

| Argument | Description |
|----------|-------------|
| `--train_file_list` | Path to training file list (used only for computing normalization statistics) |
| `--test_file_list` | Path to test file list (mzML,csv pairs) |
| `--lance_uri` | Output directory for Lance dataset |
| `--test_table` | Table name for test data (default: `test_data`) |
| `--cap_test_set` | Maximum MS2 scans per dataset (-1 for no cap) |
| `--test_set_csv` | CSV file with dataset metadata (must contain `dataset_id` and `ms2` columns, semicolon-delimited) |
| `--skip_train` | Skip training Lance creation (only compute stats from training files) |
| `--exceptional_dataset_ids` | List of dataset IDs requiring special serial processing (for very large datasets) |
| `--batch_size` | Number of files to process per batch (default: 10) |
| `--workers` | Number of parallel workers |
| `--max_peaks` | Maximum peaks per spectrum (default: 400) |

**Test Set CSV Format**:

The `--test_set_csv` file must be semicolon-delimited and contain at least:
- `dataset_id`: MassIVE dataset ID (e.g., MSV000012345)
- `ms2`: Total number of MS2 scans in the dataset

The script will add two columns to track sampling:
- `number_of_mzmls_considered`: Number of mzML files processed
- `number_of_MS2s_taken`: Number of MS2 scans included after sampling



---

## Creating Lance Datasets for Inference 

For creating inference or test Lance datasets without the need to download the Massive datasets from training split, use the dedicated script `create_lance_inference.py`. This script uses **precomputed training set statistics** from a JSON file to standardize features, ensuring consistency between our training and new inference data. The code randomly selects .csv files from each dataset to accumulate a total of 100,000 samples per dataset
**Key Features**:
- Uses precomputed training set statistics (mean/std) from JSON file
- Creates Lance dataset only for inference data
- Extracts precursor m/z directly from mzML files
- Excludes blank files automatically
- Applies same feature standardization as training data

**Usage**:

```bash
python scripts/data_preprocessing/create_lance_inference.py \
    --stats_file data/metadata/training_stats.json \
    --test_file_list data/file_paths/file_paths_inference.txt \
    --lance_uri data/lance_data_inference \
    --test_table test_data \
    --workers 16 \
    --cap_test_set 100000 \
    --test_set_csv data/metadata/inference_datasets.csv
```

**Key Arguments**:

| Argument | Description |
|----------|-------------|
| `--stats_file` | **Required.** Path to JSON file with training set statistics (mean/std for each feature) .Curent location: `data/metadata/training_stats.json` |
| `--test_file_list` | Path to inference file list (mzML,csv pairs) |
| `--lance_uri` | Output directory for Lance dataset |
| `--test_table` | Table name for inference data (default: `test_data`) |
| `--cap_test_set` | Maximum MS2 scans per dataset (-1 for no cap) |
| `--test_set_csv` | CSV file with dataset metadata (must contain `dataset_id` and `ms2` (indicates the total number of MS2 scans across all runs present in the dataset) columns, semicolon-delimited) |
| `--exceptional_dataset_ids` | List of dataset IDs requiring special serial processing (for very large datasets having many runs) |
| `--batch_size` | Number of files to process per batch (default: 10) |
| `--workers` | Number of parallel workers |
| `--max_peaks` | Maximum peaks per spectrum (default: 400) |

**Training Set Statistics JSON File**:

The `--stats_file` argument requires a JSON file containing mean and standard deviation for each numerical feature index. This file should be created from your training data.

**Recommended location**: `data/metadata/training_stats.json`

**JSON Format**:
```json
{
    "0": {"mean": 2.5, "std": 0.8},
    "1": {"mean": 3.2, "std": 1.1},
    "2": {"mean": 100.5, "std": 25.3},
    ...
}
```

Where:
- Keys are feature indices as strings ("0" through "13" for the 14 numerical features)
- Each value contains `"mean"` and `"std"` for that feature
- Features correspond to the columns returned by `get_instrument_settings_columns()`:
  0. MS2 Isolation Width
  1. Charge State
  2. Ion Injection Time (ms)
  3. Conversion Parameter C
  4. Energy1
  5. Orbitrap Resolution
  6. AGC Target
  7. HCD Energy(1)
  8. HCD Energy(2)
  9. HCD Energy(3)
  10. HCD Energy(4)
  11. HCD Energy(5)
  12. LM m/z-Correction (ppm)
  13. Micro Scan Count

**How to Generate Training Statistics JSON**:

You can generate this file from your training data using the `create_lance_add_one_hot.py` script, which computes statistics during training dataset creation. Alternatively, you can compute it manually from your training CSV files by calculating mean and std for each of the 14 numerical instrument setting columns.

**Using SLURM Script**:

For running on a SLURM cluster, use the provided script:

```bash
# Before running, edit the script to update:
# 1. Project directory path (cd /path/to/your/working/directory)
# 2. SLURM account name (--account=YOUR_ACCOUNT)
# 3. Email address (--mail-user=YOUR_EMAIL@example.com)
# 4. Adjust resource requirements if needed (cpus-per-task, mem, time)

sbatch slurm_scripts/data_preprocessing/run_build_lance_inference.sh
```

The script will:
- Load the conda environment
- Verify required packages are installed
- Run the inference Lance creation script
- Display dataset statistics upon completion

Make sure you have:
- Created the training statistics JSON file at `data/metadata/training_stats.json`
- Created the inference file list at `data/file_paths/file_paths_inference.txt`
- Created the inference metadata CSV at `data/metadata/inference_datasets.csv` (if using reporting)


