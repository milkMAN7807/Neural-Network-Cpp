#pragma once
#include <vector>
#include <functional>
#include "Matrix.h"

using namespace std;

namespace NeuralNet
{
    class Layer
    {
        public:
            int size;
            matrix weights, biases, input;
            function<double(double)> activationFunc;
            matrix linearOutput;
            matrix weightGrads, biasGrads;

            Layer(int size, int batchSize, function<double(double)> activationFunc, bool isInput, Layer* prevLayer);
            matrix getLayerOutput(bool withGrad);

        private:
            bool isInput;
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
}