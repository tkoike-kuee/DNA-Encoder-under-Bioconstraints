# DNA Encoder under Bioconstraints

## Installation

### Clone the Repository
```bash
git clone https://github.com/tkoike-kuee/DNA-Encoder-under-Bioconstraints.git
cd DNA-Encoder-under-Bioconstraints
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

---

## Usage

### Data Preparation
[CIFAR-10 and CIFAR-100 datasets](https://www.cs.toronto.edu/~kriz/cifar.html)

### Training the Model
```bash
usage: train_encoder.py [-h] [--train_data TRAIN_DATA] [--test_data TEST_DATA] [-n NUM_CLASSES]
                        [--target_seqs TARGET_SEQS] [--query_seqs QUERY_SEQS]
                        [--encoder_path ENCODER_PATH] [--hp HP] [--epoch EPOCH] [--margin MARGIN]
                        [--hp_loss_flag] [--gc_loss_flag] [--encode-only]

options:
  -h, --help            show this help message and exit
  --train_data TRAIN_DATA
                        Path to the training data
  --test_data TEST_DATA
                        Path to the test data
  -n NUM_CLASSES, --num_classes NUM_CLASSES
                        Number of classes
  --target_seqs TARGET_SEQS
                        Output path for the target sequences
  --query_seqs QUERY_SEQS
                        Output path for the query sequences
  --encoder_path ENCODER_PATH
                        Path to the encoder
  --hp HP               Number of homopolymer
  --epoch EPOCH         Number of epochs
  --margin MARGIN       Margin
  --hp_loss_flag        Homopolymer loss flag
  --gc_loss_flag        GC loss flag
  --encode-only         Encode only

```
#### Training Options

| Option                        | Description                          |
|-------------------------------|--------------------------------------|
| `-h`, `--help`                | Show help message and exit           |
| `--train_data TRAIN_DATA`      | Path to the training data            |
| `--test_data TEST_DATA`        | Path to the test data                |
| `-n`, `--num_classes`          | Number of classes for image classification in the dataset |
| `--target_seqs TARGET_SEQS`    | Output path for the target sequences |
| `--query_seqs QUERY_SEQS`      | Output path for the query sequences  |
| `--encoder_path ENCODER_PATH`  | Path to save the encoder; if the file exists, training will resume from it |
| `--hp HP`                      | Maximum allowed homopolymer length in DNA sequences |
| `--epoch EPOCH`                | Number of epochs                     |
| `--margin MARGIN`              | Margin parameter for the triplet network loss function |
| `--hp_loss_flag`               | Homopolymer loss flag. If this option is set, **homopolymer loss will not be used**. |
| `--gc_loss_flag`               | GC loss flag. If this option is set, **GC loss will not be used**. |
| `--encode-only`                | If this flag is set, only encoding from features to DNA is performed without training |

### Simulation
Simulation is performed using [NUPACK](https://www.nupack.org/). By specifying `query_seqs` and `target_seqs`, the similarity between images is determined based on the hybridization yield.

### Running a Jupyter Notebook (Optional)
```bash
jupyter notebook results.ipynb
```

---

### References
- [Triplet Network-Based DNA Encoding for Enhanced Similarity Search](https://github.com/tkoike-kuee/dna-triplet-network/tree/master)