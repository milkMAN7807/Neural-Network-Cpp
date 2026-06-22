#include <iostream>
#include <cstdlib>

#include <random>
#include "Network.h"
#include "CustomFunctions.h"

using namespace std;

// Layer
NeuralNet::Layer::Layer(int size, int batchSize, function<double(double)> activationFunc, bool isInput, Layer* prevLayer)
{
    this->size = size;
    this->activationFunc = activationFunc;
    this->isInput = isInput;

    if (isInput) return;

    // Random Distribution for weights
    random_device random;
    mt19937 gen(random());
    uniform_real_distribution<double> distrib(-invSqrt(prevLayer->size), invSqrt(prevLayer->size));

    // Resize weight matrix according to in/out
    weights.resize(size);
    for (int i = 0; i < size; i++) weights[i].resize(prevLayer->size);

    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < prevLayer->size; j++)
        {
            weights[i][j] = distrib(gen);
        }
    }

    // Scale up bias matrix
    biases.resize(size);
    for (int i = 0; i < size; i++) biases[i].resize(batchSize);  
}

// Z = sigmoid(WX + B)
matrix NeuralNet::Layer::getLayerOutput(bool withGrad)
{   
    if (isInput) return MatrixLib::apply(input, activationFunc);

    matrix multResultant = MatrixLib::mult(weights, input);
    matrix addResultant = MatrixLib::add(multResultant, biases); // Z

    if (withGrad) linearOutput = addResultant;

    return MatrixLib::apply(addResultant, activationFunc);
}

//Network
NeuralNet::Network::Network(int batchSize)
{
    this->batchSize = batchSize;
}

NeuralNet::Network::~Network()
{
    for (Layer* layer : layers) delete layer;
}

matrix NeuralNet::Network::forward(matrix m, bool withGrad)
{    
    layers[0]->input = m;

    for (int i = 1; i < layers.size(); i++)
    {
        Layer* layer = layers[i];
        layer->input = layers[i - 1]->getLayerOutput(withGrad);
    }

   return layers[layers.size() - 1]->getLayerOutput(withGrad);
}