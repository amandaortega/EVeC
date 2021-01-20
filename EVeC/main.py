import matplotlib
matplotlib.use('Agg')
import csv
from EVeC import EVeC
from math import sqrt
import matplotlib.pyplot as plt
import mlflow
import numpy as np
from numpy import loadtxt
import os
from sklearn.metrics import mean_squared_error
import time
from tqdm import tqdm
import warnings

# databases IDs
EMOTIONS = 1

TRAINING = 0
TEST = 1

def read_csv(file):
    try:
        return loadtxt(file)
    except Exception:
        return loadtxt(file, delimiter=',')

def plot_graph(y, y_label, x_label, file_name, y_aux=None, legend=None, legend_aux=None):
    plt.plot(y)
    plt.ylabel(y_label)
    plt.xlabel(x_label)    

    if y_aux is not None:    
        plt.plot(y_aux)
        plt.legend([legend, legend_aux])
    else:
        plt.annotate(str(round(y[-1, 0], 3)), xy=(y.shape[0], y[-1, 0]), ha='center')
    
    plt.savefig(file_name)
    plt.close()

def read_parameters():
    try:
        mode = int(input('Run the 1- training or 2- test (default) dataset?\n')) - 1
    except ValueError:
        mode = TEST  

    try:
        dataset = int(input('Enter the dataset to be tested:\n1- Emotions (default)\n'))
    except ValueError:
        dataset = EMOTIONS

    dim = -1
    N_default = 4

    if dataset == EMOTIONS:
        input_path_default = '../data/emotions/'
        experiment_name = 'Emotions'

        dim = 72
        sigma_default = 0.5
        delta_default = 50
        N_default = 5

    input_path = input('Enter the dataset path (default = ' + input_path_default + '): ')
    if input_path == '':
        input_path = input_path_default

    experiment_name_complement = input('Add a complement for the experiment name (default = None): ')
    if experiment_name_complement != '':
        experiment_name = experiment_name + " - " + experiment_name_complement    

    sigma = list(map(float, input('Enter the sigma (default value = ' + str(sigma_default) + '): ').split()))
    if len(sigma) == 0:
        sigma = [sigma_default]

    delta = list(map(int, input('Enter the delta (default value = ' + str(delta_default) + '): ').split()))
    if len(delta) == 0:
        delta = [delta_default]

    N = list(map(int, input('Enter the size of the window (default value = ' + str(N_default) + '): ').split()))
    if len(N) == 0:
        N = [N_default]

    register_experiment = input('Register the experiment? (default value = true): ')

    if register_experiment in ['No', 'no', 'false', 'False']:
        register_experiment = False
    else:
        register_experiment = True
    
    if dim == 2 and register_experiment:
        plot_frequency = list(map(int, input('Enter the frequency or the intervals you want to generate the plots (default = -1 in case of no plots): ').split()))
        if len(plot_frequency) == 0:
            plot_frequency = -1
        elif len(plot_frequency) != 1:
            plot = list()

            for i in range(0, len(plot_frequency), 2):
                plot = plot + list(range(plot_frequency[i], plot_frequency[i + 1]))
            
            plot_frequency = plot
    else:
        plot_frequency = -1
    
    return [dataset, mode, input_path, experiment_name, dim, sigma, delta, N, register_experiment, plot_frequency]

def run(dataset, mode, input_path, experiment_name, dim, sigma, delta, N, register_experiment, plot_frequency):
    mlflow.set_experiment(experiment_name)

    print("EVeC - " + experiment_name + ": sigma = " + str(sigma) + ", delta = " + str(delta) + ", N = " + str(N))

    X_train = read_csv(input_path + 'X_train.csv')

    if mode == TRAINING:
        X = X_train
        y = read_csv(input_path + 'Y_train.csv').astype(int)
    else:
        X = read_csv(input_path + 'X_test.csv')
        y = read_csv(input_path + 'Y_test.csv').astype(int)

    X_min = np.tile(np.min(X_train, axis=0), (X.shape[0], 1))
    X_max = np.tile(np.max(X_train, axis=0), (X.shape[0], 1))

    X = (X - X_min) / (X_max - X_min)

    dim = X.shape[1]

    if dim != 2:
        plot_frequency = -1

    if register_experiment:
        mlflow.start_run()    

        if plot_frequency == -1:
            mlflow.set_tag("plots", 'no')
        else:
            mlflow.set_tag("plots", 'yes')

        if mode == TRAINING:
            mlflow.set_tag("mode", "training")
        else:
            mlflow.set_tag("mode", "test")

        artifact_uri = mlflow.get_artifact_uri()
        # removing the 'file://'
        artifact_uri = artifact_uri[7:] + '/'            

        mlflow.log_param("sigma", sigma)
        mlflow.log_param("delta", delta)
        mlflow.log_param("N", N)

    model = EVeC(y.shape[1], sigma, delta, N)

    predictions = np.zeros((y.shape[0], y.shape[1]))
    number_of_rules = np.zeros((y.shape[0], 1))
    time_ = np.zeros((y.shape[0], 1))

    for i in tqdm(range(y.shape[0])):
        start = time.time()
        predictions[i, :] = model.predict(X[i, :].reshape(1, -1))
        model.train(X[i, :].reshape(1, -1), y[i].reshape(-1), np.array([[i]]))

        end = time.time()
        time_[i, 0] = end - start

        # Saving statistics for the step i
        number_of_rules[i, 0] = model.c

        if plot_frequency != -1: 
            if len(plot_frequency) == 1:
                if (i % plot_frequency[0]) == 0:
                    model.plot(artifact_uri + str(i) + '_input.png', artifact_uri + str(i) + '_output.png', i)
            elif i in plot_frequency:
                model.plot(artifact_uri + str(i) + '_input.png', artifact_uri + str(i) + '_output.png', i)

    if register_experiment:
        np.savetxt(artifact_uri + 'predictions.csv', predictions)
        np.savetxt(artifact_uri + 'rules.csv', number_of_rules)
        np.savetxt(artifact_uri + 'time.csv', time_)

        plot_graph(number_of_rules, 'Number of rules', 'Step', artifact_uri + 'rules.png')

        mlflow.log_metric('Mean_rules', np.mean(number_of_rules))
        mlflow.log_metric('Last_No_rule', number_of_rules[-1, 0])

        mlflow.end_run()        

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    abspath = os.path.abspath(__file__)
    os.chdir(os.path.dirname(abspath)) 

    [dataset, mode, input_path, experiment_name, dim, sigmas, deltas, Ns, register_experiment, plot_frequency] = read_parameters()

    for sigma in sigmas:
        for delta in deltas:
            for N in Ns:
                run(dataset, mode, input_path, experiment_name, dim, sigma, delta, N, register_experiment, plot_frequency)