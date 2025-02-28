# Generalizing Constraint Models in Constraint Acquisition

This repository contains the code for the paper "Generalizing Constraint Models in Constraint Acquisition, AAAI 2025".

## Requirements

- Python 3.8+
- CPMpy 1.4.0
- scikit-learn 1.3.0
- pandas 2.0.0
- numpy 1.26.0

# Usage for experiments

You can run all the experiments presented in the paper using the shell script `run_experiments.sh`.

```bash
./run_experiments.sh
```

You can also run individual experiments by using the python script `main.py`.

```bash
python main.py [-h] [-b {sudoku,golomb,exam_timetabling,nurse_rostering}] [-ts TRAINING_SIZE] [-cp]
               [-c {random_forest,MLP,GaussianNB,CategoricalNB,SVM,DT,KNN,CN2,countcp}] [-n {fp,fn}] [-p {0,0.05,0.1,0.15,0.2}]
```

options:
  -h, --help            show this help message and exit

Experiment Configuration:
  -b {sudoku,golomb,exam_timetabling,nurse_rostering}, --benchmark {sudoku,golomb,exam_timetabling,nurse_rostering}
                        Benchmark to use
  -ts TRAINING_SIZE, --training-size TRAINING_SIZE
                        Number of training instances for reverse leave-p-out cross validation
  -cp, --custom-partitions
                        Use custom partitions

Classifier Configuration:
  -c {random_forest,MLP,GaussianNB,CategoricalNB,SVM,DT,KNN,CN2,countcp}, --classifier {random_forest,MLP,GaussianNB,CategoricalNB,SVM,DT,KNN,CN2,countcp} 
                        Machine learning classifier to use

Noise Configuration:
  -n {fp,fn}, --noise_mode {fp,fn}
                        Noise mode to use (fp: false positive, fn: false negative)
  -p {0,0.05,0.1,0.15,0.2}, --percentage {0,0.05,0.1,0.15,0.2}
                        Noise percentage to apply

Example:

```bash
python main.py --benchmark sudoku --classifier random_forest --training-size 1 --noise_mode fp --percentage 0.1
```

