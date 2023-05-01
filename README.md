# PhotonIDTraining
Example of photon ID MVA training by XGBoost under python3
## Set up
Suggest to use ```miniconda``` for python3 environment :

Step 1 : Download miniconda
```
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh # for Linux
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh # for Mac
```
Step 2 : Install miniconda
```
sh Miniconda3-latest-Linux-x86_64.sh # for Linux
sh Miniconda3-latest-MacOSX-arm64.sh # for Mac
```
Step 3 : Set ```conda``` environment (Actually do it whenever opening a new terminal)
```
source $HOME/miniconda3/etc/profile.d/conda.sh
```
Step 4 : Install all python modules
```
conda env create -f setup/environment.yml
```

Enter and exit miniconda

```
conda activate photon_id_xgboost # Enter conda
conda deactivate                 # Exit conda
```
