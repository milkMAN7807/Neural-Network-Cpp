#include <iostream>
#include <vector>

#include "CustomFunctions.h"
#include "Network.h"
#include "Matrix.h"
#include "Optimizer.h"

using namespace NeuralNet;

int main()
{
    // Create Model
    Network* digitClassificationModel = new Network(3);
    int mBatchSize = digitClassificationModel->batchSize;
    vector<Layer*>& mLayers = digitClassificationModel->layers;

    mLayers.push_back(new Layer(784, mBatchSize, identity, true, nullptr));
    mLayers.push_back(new Layer(30, mBatchSize, relu, false, mLayers[0]));
    mLayers.push_back(new Layer(30, mBatchSize, relu, false, mLayers[1]));
    mLayers.push_back(new Layer(30, mBatchSize, relu, false, mLayers[2]));
    mLayers.push_back(new Layer(10, mBatchSize, sigmoid, false, mLayers[3]));

    


    return 0;
}