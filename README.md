# Introduction to ML - Artificial Neural Networks Coursework

This is a general implementation of a neural network architecture for Python3, including the implementation of a neural network mini-library and training of a neural network for regression. The California House Prices Dataset and the Iris Dataset were used in this coursework. 

## Authors

Mingda Liu

Yizhou Wu

Jiaqi Zhao

Wenjia Wang

## Structure

> ```
> .
> ├── README.md
> ├── iris.dat
> ├── housing.csv
> ├── part1_nn_lib.py
> ├── part2_house_value_regression.py
> └── report.pdf
> ```

`part1_nn_lib.py ` contains a low-level implementation of a multi-layered neural network, including a basic implementation of the backpropagation algorithm. `part2_house_value_regression.py `contains the development and optimization of a neural network architecture to predict the price of houses in California using the California House Prices Dataset with PyTorch. `report.docx` contains the analysis of the results.

## Dependencies

`python3`

`numpy `

`PyTorch `

Last tested successfully using `Python 3.6`, `Numpy 1.19.2` and `PyTorch 1.10.0`.

## Implementation

1. Use command ***python3 part1_nn_lib.py*** in the terminal to run the script. It will output the evaluation results including Train loss, Validation loss and Validation accuracy for implementating the neural network mini-library on the iris dataset.
2. Use command ***python3 part2_house_value_regression.py*** in the terminal to run the script. It will output the architecture of the NN model and the best model with its loss results.
3. If you would like to try other dataset for `part1_nn_lib.py `, please modify the file path parameter of the read function "*example_main()*" , as shown below:

> ```
> dat = np.loadtxt("iris.dat")
> ```

5. If you would like to try other dataset for `part2_house_value_regression.py `, please modify the file path parameter of the read function "*example_main()*" , as shown below:

> ```
> data = pd.read_csv("housing.csv")
> ```

## Sample outputs

Test results on part1_nn_lib.py:

> ```
> Train loss =  0.057070835372143934
> Validation loss =  0.12058360180314974
> Validation accuracy: 0.9333333333333333
> ```

Test result on part2_house_value_regression.py:

> ```
> 63488.471678  with:   {'nb_epoch': 1500, 'learning_rate': 0.003, 'hidden_layer_3': 19, 'hidden_layer_2': 40, 'hidden_layer_1': 56, 'dropout': 0, 'batch_size': 1000}
> 63765.329736  with:   {'nb_epoch': 1000, 'learning_rate': 0.003, 'hidden_layer_3': 17, 'hidden_layer_2': 47, 'hidden_layer_1': 56, 'dropout': 0.5, 'batch_size': 1000}
> 63907.639297  with:   {'nb_epoch': 1500, 'learning_rate': 0.005, 'hidden_layer_3': 21, 'hidden_layer_2': 31, 'hidden_layer_1': 62, 'dropout': 0, 'batch_size': 2000}
> 63521.640616  with:   {'nb_epoch': 1500, 'learning_rate': 0.004, 'hidden_layer_3': 23, 'hidden_layer_2': 42, 'hidden_layer_1': 50, 'dropout': 0.5, 'batch_size': 2000}
> 63446.737448  with:   {'nb_epoch': 1500, 'learning_rate': 0.003, 'hidden_layer_3': 21, 'hidden_layer_2': 31, 'hidden_layer_1': 67, 'dropout': 0.5, 'batch_size': 1000}
> 62716.281993  with:   {'nb_epoch': 1500, 'learning_rate': 0.003, 'hidden_layer_3': 24, 'hidden_layer_2': 46, 'hidden_layer_1': 64, 'dropout': 0, 'batch_size': 1000}
> 64021.327159  with:   {'nb_epoch': 1000, 'learning_rate': 0.005, 'hidden_layer_3': 16, 'hidden_layer_2': 42, 'hidden_layer_1': 62, 'dropout': 0, 'batch_size': 2000}
> 64564.208870  with:   {'nb_epoch': 1000, 'learning_rate': 0.004, 'hidden_layer_3': 15, 'hidden_layer_2': 32, 'hidden_layer_1': 61, 'dropout': 0, 'batch_size': 1000}
> 63471.586759  with:   {'nb_epoch': 1000, 'learning_rate': 0.005, 'hidden_layer_3': 18, 'hidden_layer_2': 44, 'hidden_layer_1': 66, 'dropout': 0, 'batch_size': 1000}
> 63388.578844  with:   {'nb_epoch': 1000, 'learning_rate': 0.005, 'hidden_layer_3': 17, 'hidden_layer_2': 42, 'hidden_layer_1': 66, 'dropout': 0, 'batch_size': 1000}
> Best parameters set found: {'nb_epoch': 1500, 'learning_rate': 0.003, 'hidden_layer_3': 24, 'hidden_layer_2': 46, 'hidden_layer_1': 64, 'dropout': 0, 'batch_size': 1000}
> best model
> parameter:  1500 0.003 0 1000 24 46 64
> epoch: 100 loss: 46056.293
> epoch: 200 loss: 45583.426
> epoch: 300 loss: 44277.770
> epoch: 400 loss: 45151.773
> epoch: 500 loss: 47398.806
> epoch: 600 loss: 44758.760
> epoch: 700 loss: 44861.990
> epoch: 800 loss: 47648.720
> epoch: 900 loss: 45819.864
> epoch: 1000 loss: 46455.302
> epoch: 1100 loss: 46406.982
> epoch: 1200 loss: 48867.264
> epoch: 1300 loss: 47917.866
> epoch: 1400 loss: 44238.843
> epoch: 1500 loss: 42975.626
> 
> Saved model in part2_model.pickle
> 
> prediction
> [[184499.88]
>  [323543.4 ]
>  [255912.64]
>  ...
>  [130665.85]
>  [210473.89]
>  [ 76800.61]]
> 
> Regressor error: 52384.84788562433
> 
> ```

