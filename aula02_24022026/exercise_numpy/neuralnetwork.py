#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from anyio import Path
import numpy as np
from pathlib import Path
from layers import DenseLayer


class NeuralNetwork:
 
    def __init__(self):
        # attributes
        self.layers = []

    def add(self, layer, biases = None, weights = None):
        if self.layers:
            layer.set_input_shape(input_shape=self.layers[-1].output_shape())
        if biases is not None: layer.set_biases(biases)
        if weights is not None: layer.set_weigths(weights)
        self.layers.append(layer)
        return self

    def forward_propagation(self, X, training):
        output = X
        for layer in self.layers:
            output = layer.forward_propagation(output, training)
        return output

    def predict(self, dataset):
        return self.forward_propagation(dataset.X, training=False)

    def score(self, dataset, predictions):
        if self.metric is not None:
            return self.metric(dataset.y, predictions)
        else:
            raise ValueError("No metric specified for the neural network.")


if __name__ == '__main__':
    from activation import SigmoidActivation
    from data import read_csv

    # training data
    # dataset = read_csv('xnor.data', sep=',', features=False, label=True)
    data_path = Path(__file__).with_name('xnor.data')
    dataset = read_csv(data_path, sep=',', features=False, label=True)

    net = NeuralNetwork()
    n_features = dataset.X.shape[1]

    # Hidden layer (2 neurónios)
    w1 = np.array([[20, -20], [20, -20]])
    b1 = np.array([[-30, 10]])
    net.add(DenseLayer(2, (n_features,)), biases=b1, weights=w1)
    net.add(SigmoidActivation())

    # Output layer (1 neurónio)
    w2 = np.array([[20], [-20]])
    b2 = np.array([[-10]])
    net.add(DenseLayer(1), biases=b2, weights=w2)
    net.add(SigmoidActivation())
    
    print("Predictions for the training dataset:")
    pred = net.predict(dataset)
    print(pred)
    print((pred >= 0.5).astype(int))

    
    

