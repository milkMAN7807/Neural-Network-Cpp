#pragma once
#include "Network.h"
#include "Matrix.h"

namespace Optimizer
{
    matrix applyLoss(const matrix& input, const matrix& groundTruth, function<double(double, double)> lossFunc);
    void backward(Network* network, const matrix& modelOutput, const matrix& groundTruth, function<double(double, double)> lossDerivative);
    void step(Network* Network, double learnRate);
}