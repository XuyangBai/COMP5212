## Project Structure
- `main.py`:  parse the parameter and call the corresponding function
- `common.py`: define the constant we use in the project (*the variable RATIO is corresponding to p and the PRE_TRAINED is to decide whether we train the CNN from scratch or using pre-trained weights*)
- `ae.py`: implement the functions for train and test AutoEncoder including the hyperparameter searching.
- `cnn.py`: implement the functions for train and test CNN model(from scratch or using pretrained weights decided by the PRE_TRAINED variable defined in common.py)
- `AE/` : this dictionary  save the weight of AutoEncoder model with best hyperparameter. It is used in evaluate_ae function to reload the model and test the model's performance on test test.
    - `loss`: the figure recording the loss change with epoch num in the hyperparameter searching stage.
    - `best.png`: the loss curve during the training of model with best hyperparameters.
- `CNN/`: this dictionary save the weight of CNN model from scratch. Used in test_cnn function.
    - `loss`: the figure recording the loss change with epoch num in the hyperparameter searching stage.
    - `accuracy`: the figures recording the accuracy on validation set change with epoch num.
    - `best.png`: the loss curve and accuracy curve during the training of model with best hyperparameters.
- `CNN_pretrained`: the same to `CNN` dictionary but it used for CNN model wtih pretrained value. The AE model save the weights of three convolutional layers in this direction. And CNN model reload it and begin training from this checkpoint.

## How to run the scripts
```
python main.py --task [task name] --datapath [dataset path]
```
Parameter:
1. --task: name of task ('train_ae', 'evaluate_ae', 'train_cnn', 'test_cnn')
2. --dataset: the path to data direction 

### CNN model initializer

If you want to train the CNN model from scratch, please set the variable `PRE_TRAINED` to FALSE in `common.py`. The result will be saved in `CNN` dictionary

If you want to train the CNN model using pre-trained value, you have to train the AE model first to get the pretrained value. And then set `PRE_TRAINED` to True. And the result will be saved in 'CNN_pretrained' dictionary

### Training set size

By changing the value of `RATIO` defined in `common.py` you can explore the influence of training set size on model performance.