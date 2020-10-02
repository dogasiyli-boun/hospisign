# HospiSign - Semi-Supervised Discovery of Hand-Shapes in Turkish Sign Language
The code related to the paper "Semi-Supervised Discovery of Hand-Shapes in Turkish Sign Language"
The paper is in review stage at "Language Resources and Evaluation - Springer"

## Setup

Follow [this link](https://uoa-eresearch.github.io/eresearch-cookbook/recipe/2014/11/26/python-virtual-env/),
create a virtual environment 
and activate it
```bash
pip -h
pip install virtualenv
cd "the main folder you want to create the virtual environment in"
virtualenv hospisign
source hospisign/bin/activate
```

### Install dependencies

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install following pacakges.

```bash
pip3 install torch #installs future-0.18.2 numpy-1.19.2 torch-1.6.0
pip3 install sklearn #installs joblib-0.16.0 scikit-learn-0.23.2 scipy-1.5.2 sklearn-0.0 threadpoolctl-2.1.0
pip3 install pandas #installs pandas-1.1.2 python-dateutil-2.8.1 pytz-2020.1 six-1.15.0
pip3 install matplotlib #installs certifi-2020.6.20 cycler-0.10.0 kiwisolver-1.2.0 matplotlib-3.3.2 pillow-7.2.0 pyparsing-2.4.7
pip3 install wget #installs wget-3.2
pip3 install -U Cluster_Ensembles
pip3 install -U PyYAML #installs PyYAML-5.3.1
```
Cluster_Ensembles package needs 
```bash
sudo apt-get install metis
```
also in line 37 on Cluster_Ensembles.py change "import jaccard_similarity_score" -> "import jaccard_score as jaccard_similarity_score"

## Configuration File
Edit the "configs/conf.yaml" file accordingly before running any code
So that the folders will be correctly used by the code

## Dataset

The dataset details are at [Mendeley Data](https://data.mendeley.com/datasets/dymz94c393/draft?preview=1)
Files also uploaded to a website and data can be downloaded through the code below.
```python
import projRelatedHelperFuncs as prHF
import scriptFile as sf

sf.download_data_script()
prHF.combine_pca_hospisign_data(dataIdent=, pca_dim=256, nos=11, verbose=2)

```

## Running

### Clustering
```python
import projRelatedHelperFuncs as prHF

def cluster_hospisign(randomSeed,  #  some integer for reproducibility
                      dataset_ident_str,  #  'Dev' or 'Exp'
                      data_ident_vec=["hog", "sn", "sk", "hgsk", "hgsn", "snsk", "hgsnsk"],
                      clustCntVec=[128, 256],  #  96 for skeleton(sk), for others 256 with Dev, 1024 with Exp
                      verbose=0,
                      pca_dim=256)
```
This function runs clustering on :
- dataset_ident_str : Development or Expended Dataset
- data_ident_vec : Selected feature, raw or early fused
- clustCntVec : How many clusters to extract
- pca_dim : The dimension that will be used after getting the pca of the selected feature of dataset

After running this function with all the necessary parameters you wish then the last line will run
```python
import projRelatedHelperFuncs as prHF
prHF.traverse_base_results_folder_hospising()
```
which will list all the clustering results as a table

### Classification
```python
import scriptFile as sf

sf.mlp_hospisign(dropout_value=0.3, hidStateID=7, dataset_ident_str ="Dev", data_ident_vec=["hog", "sn", "sk", "hgsk", "hgsn", "snsk", "hgsnsk"])
sf.mlp_hospisign(dropout_value=0.3, hidStateID=0, dataset_ident_str ="Exp", data_ident_vec=["hog", "sn", "sk", "hgsk", "hgsn", "snsk", "hgsnsk"])

rs=0
dropout_value=0.3
hidStateID = 7
dataset_ident_str ="Dev"
data_va_te_str="te"
dv = [["hog", "sn", "sk", "hgsk", "hgsn", "snsk", "hgsnsk"], ["hog", "sn", "sk"],["hog", "sk"],["hog", "sk", "hgsk"],["hog", "sn"]]
for userTe in [2, 3, 4, 5, 6, 7]:
    for userVa in [2, 3, 4, 5, 6, 7]:
        if userTe==userVa:
            continue
        all_results = sf.append_to_all_results_dv_loop(userTe, userVa, dataset_ident_str, dv, data_va_te_str, dropout_value, rs, hidStateID)

hidStateID = 0
dataset_ident_str ="Exp"
dv = [["hog", "sn", "sk", "hgsk", "hgsn", "snsk", "hgsnsk"], ["hog", "sn", "sk"],["hog", "sk"],["hog", "sk", "hgsk"],["hog", "sn"]]
for userTe in [2, 3, 4, 5, 6, 7]:
    for userVa in [2, 3, 4, 5, 6, 7]:
        if userTe==userVa:
            continue
        all_results = sf.append_to_all_results_dv_loop(userTe, userVa, dataset_ident_str, dv, data_va_te_str, dropout_value, rs, hidStateID)

```
This function runs classification experiments as explained in the paper.


