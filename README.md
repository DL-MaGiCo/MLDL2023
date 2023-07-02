# MLDL2023
Project for the class Machine Learning and Deep Learning - 2023 edition @PoliTo

This project focuses on the design choices of a DL model for the task of Visual Place Recognition (VPR) through image retrieval.

<b>First steps</b>
1. Download datasets by running the script *download_datasets.py*
2. Install the required packages by running
   
   > pip install -r requirements.txt

## Run an experiment
### Basic experiments
For the baseline of the project run
> python main.py --train_path /path/to/datasets/gsv_xs --val_path /path/to/datasets/tokyo_xs/test --test_path /path/to/datasets/tokyo_xs/test

### Architecture
You can choose the architecture using *exp_name* and run the experiment running
> python main.py --train_path /path/to/datasets/gsv_xs --val_path /path/to/datasets/tokyo_xs/test --test_path /path/to/datasets/tokyo_xs/test --exp_name NameOfArchitecture

### Loss function
You can choose the loss function using *loss* and run the experiment running
> python main.py --train_path /path/to/datasets/gsv_xs --val_path /path/to/datasets/tokyo_xs/test --test_path /path/to/datasets/tokyo_xs/test --loss LossFunction

### Optimizer and hyper-parameters
You can choose the optimizer using *optim*, the learning rate using *lr* and the weight decay using *wd*. You can run the experiment running
> python main.py --train_path /path/to/datasets/gsv_xs --val_path /path/to/datasets/tokyo_xs/test --test_path /path/to/datasets/tokyo_xs/test --optim NameOfOptimizer --lr lrValue --wd wdValue

### Number of epochs
You can choose the number of epochs using *max_epochs* and run the experiment running
> python main.py --train_path /path/to/datasets/gsv_xs --val_path /path/to/datasets/tokyo_xs/test --test_path /path/to/datasets/tokyo_xs/test --max_epochs NumberEpochs

### Learning rate and Weight Decay hyperparameters
You can choose the value of Learning Rate using *lr* and weight decay value using *wd* and run the experiment running
> python main.py --train_path /path/to/datasets/gsv_xs --val_path /path/to/datasets/tokyo_xs/test --test_path /path/to/datasets/tokyo_xs/test --lr lr --wd wd
