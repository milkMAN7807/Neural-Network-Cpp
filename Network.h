#pragma once

#include <vector>
#include <functional>
#include "Matrix.h"

using namespace std;

class Layer
{
    //private:
    public:
        // Column = neuron num, row = weight num 
        matrix weights, biases, input;
        function<double(double)> activationFunc;
        matrix linearOutput, activationOutput;
        matrix weightGrads, biasGrads;

        matrix getLayerOutput(bool withGrad);

    //public:
        int size;
        bool isInput;

        Layer(int size, int batchSize, function<double(double)> activationFunc, bool isInput, Layer* prevLayer);
};

class Network
{
    public:
        int batchSize;    
        vector<Layer*> layers;    

        Network(int batchSize);
        ~Network();

        matrix forward(matrix m, bool withGrad);
};