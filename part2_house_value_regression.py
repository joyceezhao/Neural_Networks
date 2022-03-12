import sklearn
import torch
import pickle
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV
from torch import nn, optim, tensor
import math
from sklearn.model_selection import RandomizedSearchCV

device = torch.device("cpu")

# neural network architecture
class Net(nn.Module):
    def __init__(self, features, hidden_layer_1, hidden_layer_2, hidden_layer_3, dropout):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(features, hidden_layer_1)
        self.fc2 = nn.Linear(hidden_layer_1, hidden_layer_1)
        self.fc3 = nn.Linear(hidden_layer_1, hidden_layer_2)
        self.fc4 = nn.Linear(hidden_layer_2, hidden_layer_3)
        self.output_layer = nn.Linear(hidden_layer_3, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        pred = self.fc1(x)
        pred = nn.functional.relu(pred)
        pred = self.dropout(pred)

        pred = self.fc2(pred)
        pred = nn.functional.relu(pred)
        pred = self.dropout(pred)

        pred = self.fc3(pred)
        pred = nn.functional.relu(pred)
        pred = self.dropout(pred)

        pred = self.fc4(pred)
        pred = nn.functional.relu(pred)
        pred = self.dropout(pred)

        output = self.output_layer(pred)

        return output


class Regressor(sklearn.base.BaseEstimator, sklearn.base.RegressorMixin):

    def __init__(self, x, learning_rate=0.003, dropout=0, batch_size=1000, hidden_layer_1=64, hidden_layer_2=46,
                 hidden_layer_3=24, nb_epoch=1500):
        # You can add any input parameters you need
        # Remember to set them with a default value for LabTS tests
        """
        Initialise the model.

        Arguments:
            - x {pd.DataFrame} -- Raw input data of shape
                (batch_size, input_size), used to compute the size
                of the network.
            - network_parameters -- the network parameters
            - nb_epoch {int} -- number of epoch to train the network.

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        # Replace this code with your own
        # preprocessor_parameter {
        #     'text_label': "ocean_proximity",
        #     'x_attr_mean': None,
        #     'x_normalize': None,
        #     'y_normalize': None,
        #     'onehot': None
        # }
        self.preprocessor_parameter = None

        self.x = x
        self.train_x, _ = self._preprocessor(self.x, training=True)
        self.input_size = self.train_x.shape[1]
        self.output_size = 1

        self.learning_rate = learning_rate
        self.dropout = dropout
        self.batch_size = batch_size
        self.nb_epoch = nb_epoch
        self.hidden_layer_1 = hidden_layer_1
        self.hidden_layer_2 = hidden_layer_2
        self.hidden_layer_3 = hidden_layer_3
        self.net = Net(self.input_size, self.hidden_layer_1, self.hidden_layer_2, self.hidden_layer_3, self.dropout)

        return

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def _preprocessor(self, x, y=None, training=False):
        """
        Preprocess input of the network.

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw target array of shape (batch_size, 1).
            - training {boolean} -- Boolean indicating if we are training or
                testing the model.

        Returns:
            - {torch.tensor} or {numpy.ndarray} -- Preprocessed input array of
              size (batch_size, input_size).
            - {torch.tensor} or {numpy.ndarray} -- Preprocessed target array of
              size (batch_size, 1).

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        # Return preprocessed x and y, return None for y if it was None
        # avoiding modifying original data
        x = x.copy()
        if isinstance(y, pd.DataFrame):
            y = y.copy()

        # error control
        if self.preprocessor_parameter is None and training is False:
            raise Exception("No preprocessing parameters! Pre-precessed testing data before training data!")

        if training is True:
            self.preprocessor_parameter = {'text_label': 'ocean_proximity'}

            # set missing values to mean
            imp_x = SimpleImputer(strategy='mean')
            # only numerical attributes can calculate the median
            number_x = x.drop(self.preprocessor_parameter['text_label'], axis=1)
            imp_x.fit(number_x)
            processed_x = pd.DataFrame(imp_x.transform(number_x), columns=number_x.columns)
            # save the mean parameter for test data
            self.preprocessor_parameter['x_attr_mean'] = imp_x

            # normalize columns
            min_max_scaler = preprocessing.MinMaxScaler()
            processed_x = pd.DataFrame(min_max_scaler.fit_transform(processed_x), columns=processed_x.columns)
            self.preprocessor_parameter['x_normalize'] = min_max_scaler

            # deal with textual value, one hot encode
            text_x = x.loc[:, [self.preprocessor_parameter['text_label']]]
            lb = preprocessing.LabelBinarizer()
            lb.fit(text_x)
            text_x_onehot = pd.DataFrame(lb.transform(text_x, ), columns=lb.classes_)
            # save one hot coding
            self.preprocessor_parameter['onehot'] = lb

            # concatenate
            processed_x.reset_index(drop=True, inplace=True)
            processed_x = processed_x.join(text_x_onehot)
            # for i in range(0, text_x_onehot.shape[1]):
            #     processed_x[self.preprocessor_parameter['text_label'] + '_' + str(i)] = text_x_onehot[i]

        else:
            # set missing values to mean
            # only numerical attributes can calculate the median
            number_x = x.drop(self.preprocessor_parameter['text_label'], axis=1)
            processed_x = pd.DataFrame(self.preprocessor_parameter['x_attr_mean'].transform(number_x),
                                       columns=number_x.columns)

            # normalize columns
            processed_x = pd.DataFrame(self.preprocessor_parameter['x_normalize'].transform(processed_x),
                                       columns=processed_x.columns)

            # deal with textual values
            text_x = x.loc[:, [self.preprocessor_parameter['text_label']]]
            text_x_onehot = pd.DataFrame(self.preprocessor_parameter['onehot'].transform(text_x),
                                         columns=self.preprocessor_parameter['onehot'].classes_)

            # concatenate
            processed_x.reset_index(drop=True, inplace=True)
            processed_x = processed_x.join(text_x_onehot)

        processed_x = torch.tensor(processed_x.to_numpy(), dtype=torch.float32)

        if isinstance(y, pd.DataFrame):
            processed_y = torch.tensor(y.to_numpy(), dtype=torch.float32)
        else:
            processed_y = None

        return processed_x, processed_y

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def fit(self, x, y):
        """
        Regressor training function

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw output array of shape (batch_size, 1).

        Returns:
            self {Regressor} -- Trained model.

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        print("parameter: ", self.nb_epoch, self.learning_rate, self.dropout, self.batch_size,
              self.hidden_layer_3, self.hidden_layer_2, self.hidden_layer_1)
        # training (tensor)
        x_train, y_train = self._preprocessor(x, y=y, training=True)
        self.net.train()

        optimizer = optim.Adam(self.net.parameters(), self.learning_rate)
        loss_function = nn.MSELoss(reduction='mean')

        for i in range(self.nb_epoch):
            running_loss = 0.0
            x_train, y_train = sklearn.utils.shuffle(x_train, y_train)
            iterations = len(x_train) // self.batch_size
            for j in range(iterations):
                # zero the parameter gradients
                optimizer.zero_grad()

                x_batch = x_train[j * self.batch_size:, :]
                y_batch = y_train[j * self.batch_size:, :]

                # if j + 1 == iterations:
                # else:
                #     x_batch = x_train[j * self.batch_size:(j + 1) * self.batch_size, :]
                #     y_batch = y_train[j * self.batch_size:(j + 1) * self.batch_size, :]

                outputs = self.net(x_batch)
                loss = loss_function(outputs, y_batch)
                loss.backward()
                optimizer.step()
                running_loss = loss.item()
                # print statistics
            if i % 100 == 99:
                print('epoch: %d loss: %.3f' %
                      (i + 1, math.sqrt(running_loss)))

        return self

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def predict(self, x):
        """
        Ouput the value corresponding to an input x.

        Arguments:
            x {pd.DataFrame} -- Raw input array of shape
                (batch_size, input_size).

        Returns:
            {np.darray} -- Predicted value for the given input (batch_size, 1).

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        # test data
        x_train, _ = self._preprocessor(x, training=False)
        # evaluation mode
        self.net.eval()
        with torch.no_grad():
            outputs = self.net(x_train)
            outputs = outputs.detach().numpy()

        return outputs

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def score(self, x, y):
        """
        Function to evaluate the model accuracy on a validation dataset.

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw ouput array of shape (batch_size, 1).

        Returns:
            {float} -- Quantification of the efficiency of the model.

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        x_test, y_test = self._preprocessor(x, y=y, training=False)  # Do not forget

        self.net.eval()
        with torch.no_grad():
            # outputs = self.predict(x)
            outputs = self.net(x_test)
            loss_function = nn.MSELoss(reduction='mean')
            loss = loss_function(outputs, y_test).detach().numpy()
            # RMSE
            loss_value = np.sqrt(loss.item())

        # RMSE
        # loss = math.sqrt(mean_squared_error(Y, otuputs))
        return loss_value

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################


def save_regressor(trained_model):
    """
    Utility function to save the trained regressor model in part2_model.pickle.
    """
    # If you alter this, make sure it works in tandem with load_regressor
    with open('part2_model.pickle', 'wb') as target:
        pickle.dump(trained_model, target)
    print("\nSaved model in part2_model.pickle\n")


def load_regressor():
    """
    Utility function to load the trained regressor model in part2_model.pickle.
    """
    # If you alter this, make sure it works in tandem with save_regressor
    with open('part2_model.pickle', 'rb') as target:
        trained_model = pickle.load(target)
    print("\nLoaded model in part2_model.pickle\n")
    return trained_model


def RegressorHyperParameterSearch(x_train, y_train):
    # Ensure to add whatever inputs you deem necessary to this function
    """
    Performs a hyper-parameter for fine-tuning the regressor implemented
    in the Regressor class.

        Arguments:
            - x_train {pd.DataFrame} -- Raw input array of shape
                (batch_size, input_size).
            - y_train {pd.DataFrame} -- Raw ouput array of shape (batch_size, 1).

        Returns:
            - best_parameter {dict} -- searched best parameter combination of the model.
            - best_model {Regressor} -- best model.

    """

    #######################################################################
    #                       ** START OF YOUR CODE **
    #######################################################################

    param_dist = {
        'learning_rate': [0.005, 0.0025, 0.001],
        'dropout': [0, 0.5],
        'nb_epoch': [1000, 1500],
        'batch_size': [100, 200],
        'hidden_layer_1': np.arange(50, 70),
        'hidden_layer_2': np.arange(30, 50),
        'hidden_layer_3': np.arange(15, 25)
    }

    rand_search = RandomizedSearchCV(Regressor(x_train), param_distributions=param_dist, cv=3, scoring='neg_mean_squared_error')
    rand_search.fit(x_train, y_train)

    means = rand_search.cv_results_['mean_test_score']
    params = rand_search.cv_results_['params']
    for mean, param in zip(means, params):
        print("%f  with:   %r" % (np.sqrt(-mean), param))

    best_parameter = rand_search.best_params_
    best_model = rand_search.best_estimator_

    print("Best parameters set found:", best_parameter)

    return best_parameter, best_model

    #######################################################################
    #                       ** END OF YOUR CODE **
    #######################################################################


def example_main():
    output_label = "median_house_value"

    # Use pandas to read CSV data as it contains various object types
    # Feel free to use another CSV reader tool
    # But remember that LabTS tests take Pandas Dataframe as inputs
    data = pd.read_csv("housing.csv")

    # Spliting input and output
    x_train = data.loc[:, data.columns != output_label]
    y_train = data.loc[:, [output_label]]

    # spliting into train and test data
    train_features, test_features, train_labels, test_labels = train_test_split(x_train, y_train, test_size=0.2)

    # search for best network patameters
    best_parameter, best_model = RegressorHyperParameterSearch(train_features, train_labels)

    # Training
    # This example trains on the whole available dataset.
    # You probably want to separate some held-out data
    # to make sure the model isn't overfitting
    # regressor = Regressor(train_features)

    print("best model")
    regressor = best_model
    # run one more time
    regressor.fit(train_features, train_labels)
    save_regressor(regressor)

    # predict
    print("prediction")
    print(regressor.predict(train_features))

    # Error
    train_error = regressor.score(train_features, train_labels)
    print("\nRegressor error on training data: {}\n".format(train_error))
    test_error = regressor.score(test_features, test_labels)
    print("\nRegressor error on testing data: {}\n".format(test_error))


if __name__ == "__main__":
    example_main()
