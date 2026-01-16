# brainstorm

## issues
- unbalanced data

## ideas for solutions
- unbalanced data: best solution: undersampling -> use a threshold (doesn't work that good)
- class weighting: give undersampled classes a higher weight

## code structure

### 1. Get Dataset

### 2. Load Dataset

### 3. Prepare Dataset
- shuffling
- unbalanced data -> use a threshold of a maximum of 300 data samples (undersample)

### 4. classification
- model: LinearSVMClassifier (wrapper for LinearSVC)
- loss: L2
- hyperparameters:
```
    C = 0.008416315395715555,
    max_iter = 909,
    tol = 0.0007056401403169878,
    random_state = 42,
    verbose = 1,   
```
- extra functions like getter/setter 
### 5. results
- accuracy: 0.2884
  precision: 0.3106
  recall: 0.2884
  f1_score: 0.2948
- but ROC/AUROC curves look promising

### 6. Hyperparameter Search
- tried Bayes to find good parameters for our SVM