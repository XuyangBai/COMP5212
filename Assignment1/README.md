## Project Structure
- `main.py`:  parse the parameter and call the corresponding function
- `utility.py`: define read_data function
- `logistic_regresion.py`: realize the `LogisticRegression` class and `logistic_regression` function
- `svm.py`: realize the `svm_linear` and `svm_rbf` function
- `neural_network.py`: realize the `neural_network` function

## How to run the scripts

```
python main.py -m [model name] -d [dataset name]
```
Parameter:
1. -m --model: name of the model('logistic_regression', 'neural_network', 'svm_linear', 'svm_rbf')
2. -d --dataset: name of the dataset('mouse', 'pulsar_star', 'wine')

For svm_rbf and neural_network, the [dataset name] can be empty so that the visualization of cross validation on three 
different models is drawn on one single picture.

```
python main.py -m svm_rbf
python main.py -m neural_network
```
