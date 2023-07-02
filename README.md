# MLDL2023
Project for the class Machine Learning and Deep Learning - 2023 edition @PoliTo

This project focuses on the design choices of a DL model for the task of Visual Place Recognition (VPR) through image retrieval.

<b>First steps</b>
1. Download datasets by running the script *download_datasets.py*
2. Install the required packages by running
   
   > pip install -r requirements.txt

## Run an experiment
### Basic experiments
For the baseline of the project run:
> python train.py --train_path /path/to/datasets/gsv_xs --val_path /path/to/datasets/tokyo_xs/test --test_path /path/to/datasets/tokyo_xs/test

### Architecture
You can choose the architecture using *--exp_name* among the values: 'default', 'gem', 'mix'.
> python train.py --train_path /path/to/datasets/gsv_xs --val_path /path/to/datasets/tokyo_xs/test --test_path /path/to/datasets/tokyo_xs/test --exp_name NameOfArchitecture

### Loss function
You can choose the loss function using *--loss* among the values: 'default', 'multisim', 'fast'.
> python train.py --train_path /path/to/datasets/gsv_xs --val_path /path/to/datasets/tokyo_xs/test --test_path /path/to/datasets/tokyo_xs/test --loss LossFunction

### Optimizer and hyper-parameters
You can choose the optimizer using *--optim* among the values 'sgs', 'adamw', the learning rate using *--lr* and the weight decay using *--wd*. 
> python train.py --train_path /path/to/datasets/gsv_xs --val_path /path/to/datasets/tokyo_xs/test --test_path /path/to/datasets/tokyo_xs/test --optim NameOfOptimizer --lr lrValue --wd wdValue

### Number of epochs
You can choose the number of epochs using *--max_epochs*.
> python train.py --train_path /path/to/datasets/gsv_xs --val_path /path/to/datasets/tokyo_xs/test --test_path /path/to/datasets/tokyo_xs/test --max_epochs NumberEpochs
