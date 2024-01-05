# DL23-Project

## Setup
### VC v14+ runtime
The package learn2learn (currently) requires the VCv 14+ runtime library to be installed on your PC. If you install Visual Studio with workload "Desktop development with C++", you will have all the necessary packages installed.

### Conda
```
conda create --name dl python=3.9
conda activate dl

pip install -r requirements.txt
```

## Reproduce the Results
### Adversarial Querying
```
python evaluate.py --name "AQ"
```