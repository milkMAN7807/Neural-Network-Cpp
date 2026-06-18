#pragma once
#include "Network.h"
#include "Matrix.h"

namespace Optimizer
{
    matrix applyLoss(matrix input, matrix groundTruth, function<double(double, double)> lossFunc);
    void backward(Network* network, matrix modelOutput, matrix groundTruth, function<double(double, double)> lossDerivative);
    void step(Network* Network, double learnRate);
}