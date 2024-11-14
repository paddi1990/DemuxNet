![Logo](images/logo.png)

## Machine learning augmented sample demultiplexing of pooled single-cell RNA-seq data
---


![GitHub release (latest SemVer)](https://img.shields.io/badge/Version-v1.1.0-yellowgreen) ![GitHub release (latest SemVer)](https://img.shields.io/badge/Language-python-yellowgreen)

DemuxNet is a computational tool for single-cell RNA sequencing (scRNA-seq) sample demultiplexing. It automates the process of assigning individual cells to their corresponding samples when multiple samples are pooled in a single sequencing run. DemuxNet predicts missing CMO (Cell Multiplexing Oligo) labels for barcodes with incomplete or missing information by leveraging machine learning algorithms. The tool works with sparse single-cell RNA expression matrices in RDS format and trains on known barcodes to predict sample identities for cells with missing or ambiguous CMO lables.
For more information, please refer to our manuscript: [******].

### Installation

#### Install from source code
```
git clone https://github.com/paddi1990/DemuxNet.git
cd DemuxNet
python setup.py install
```
#### Install from pip
```
pip install demuxnet
```

### Data preparation
DemuxNet takes a sparse single-cell expression matrix in RDS format as input. The sparse matrix can be prepared using the following pipeline:

```
************
************
************
************
************
```


### Usage

DemuxNet automatically detects the "CMO" keywords within barcode strings and uses them as training data to predict and fill in the missing CMO class for barcodes that do not contain the "CMO" information.

```
demuxnet -i gene_expressioin_matrix.rds -model DNN -feature 6000 -out prediction.csv
```

We also provide 


### Visualization
To do

### Contact
TandemMod is maintained by Hu lab.
If you use DemuxNet in your research, please cite *************************.