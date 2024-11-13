## Machine learning augmented sample demultiplexing of pooled single-cell RNA-seq data
---

![GitHub release (latest SemVer)](https://img.shields.io/badge/Version-v1.1.0-yellowgreen) ![GitHub release (latest SemVer)](https://img.shields.io/badge/Language-python-yellowgreen)

DemuxNet is a computational tool for single-cell RNA sequencing sample demultiplexing (for more info, please check our manuscript:******). DemuxNet is 

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
DemuxNet takes sparse single-cell expression matrix in `RDS` format as input. The sparse matrix can be prepared according to the following pipline:

```
************
************
************
************
************
```


### Usage
```
demuxnet -i gene_expressioin_matrix.rds -model DNN -out prediction.csv
```

### Contact
TandemMod is maintained by Hu lab.
If you use DemuxNet in your research, please cite *************************.