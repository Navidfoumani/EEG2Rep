## EEG2Rep: Enhancing Self-supervised EEG Representation Through Informative Masked Inputs
[![KDD 2024](https://img.shields.io/badge/KDD-2024-ff69b4.svg)](https://kdd2024.kdd.org/)
### ✨ **News:** This work has been accepted for publication in <span style="color:red;">**KDD24**</span>
### ✨ **Awards:** 🏆 1st Place in the [Audience Appreciation Awards](https://kdd2024.kdd.org/awards/)!

#### Authors: [Navid Mohammadi Foumani](https://scholar.google.com.au/citations?user=Ax62P1MAAAAJ&hl=en), 
[Geoffrey Mackellar](https://www.linkedin.com/in/geoffmackellar/?originalSubdomain=au), 
[Soheila Ghane](https://www.linkedin.com/in/soheila-ghane/?originalSubdomain=au), 
[Saad Irtza](),
[Nam Nguyen](),
[**Mahsa Salehi**](https://research.monash.edu/en/persons/mahsa-salehi)

This work follows from the project with [**Emotiv Research**](https://www.emotiv.com/neuroscience-research-education-solutions/), a bioinformatics research company based in Australia, and [**Emotiv**](https://www.emotiv.com/), 
a global technology company specializing in the development and manufacturing of wearable EEG products.

#### EEG2Rep Paper: [PDF](https://dl.acm.org/doi/pdf/10.1145/3637528.3671600)

This is a PyTorch implementation of **EEG2Rep: Enhancing Self-supervised EEG Representation Through Informative Masked Inputs**
<p align="center">
    <img src="Fig/EEG2Rep.png">
</p> 

## Datasets 

1. **Emotiv:**
   To download the Emotiv public datasets, please follow the link below to access the preprocessed datasets, which are split subject-wise into train and test sets. After downloading, copy the datasets to your Dataset directory.

   [Download Emotiv Public Datasets](https://drive.google.com/drive/folders/1KQyST6VJffWWD8r60AjscBy6MHLnT184?usp=sharing)

2. **Temple University Datasets:**
   Please use the following link to download and preprocess the TUEV and TUAB datasets.

   [Download Temple University Datasets](https://github.com/ycq091044/BIOT/tree/main/datasets)

## Setup

_Instructions refer to Unix-based systems (e.g. Linux, MacOS)._

This code has been tested with `Python 3.7` and `3.8`.

`pip install -r requirements.txt`

## Run

To see all command options with explanations, run: `python main.py --help`
In `main.py` you can select the datasets and modify the model parameters.
For example:

`self.parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')`

or you can set the parameters:

`python main.py --epochs 100 --data_dir Dataset/Crowdsource`

## Citation
If you find **EEG2Rep** useful for your research, please consider citing this paper using the following information:

````
```
@inproceedings{eeg2rep2024,
  title={Eeg2rep: enhancing self-supervised EEG representation through informative masked inputs},
  author={Mohammadi Foumani, Navid and Mackellar, Geoffrey and Ghane, Soheila and Irtza, Saad and Nguyen, Nam and Salehi, Mahsa},
  booktitle={Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
  pages={5544--5555},
  year={2024}
}

```
````
