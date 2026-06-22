#include "Optimizer.h"
#include "Matrix.h"
#include "CustomFunctions.h"

matrix Optimizer::applyLoss(const matrix& input, const matrix& groundTruth, function<double(double, double)> lossFunc)
{
    matrix r(input.size(), row(input[0].size()));

    for (int i = 0; i < input.size(); i++)
    {
        for (int j = 0; j < input[0].size(); j++)
        {
            r[i][j] = lossFunc(input[i][j], groundTruth[i][j]);
        }
    }

    return r;
}

// Backpropogation
void Optimizer::backward(Network* network, const matrix& modelOutput, const matrix& groundTruth, function<double(double, double)> lossFunc)
{
    double(*lossFuncPtr)(double, double) = *lossFunc.target<double(*)(double, double)>(); // This is awful omg, hope I dont change the type of my funcs!!!!
    function<double(double, double)> lossDerivative = lossDerivativeMap[lossFuncPtr];
    
    matrix cur = modelOutput;
    cur = applyLoss(cur, groundTruth, lossDerivative); // Applies loss derivative, now have dL/da where a is final activation layer 

    int depth = network->layers.size();
    for (int i = depth - 1; i > 0; i--)
    {
        Layer* layer = network->layers[i];

        function<double(double)>& activationFunc = layer->activationFunc;
        double(*activationFuncPtr)(double) = *activationFunc.target<double(*)(double)>();
        function<double(double)> activationDerivative = derivativeMap[activationFuncPtr];

        matrix update = MatrixLib::apply(layer->linearOutput, activationDerivative);
        cur = MatrixLib::multElement(cur, update); // da/dz change of activation with respect to linear output applied to gradient vector

        // Grab this layers gradients
        matrix inputT = MatrixLib::transpose(layer->input);
        matrix weightGrads = MatrixLib::mult(cur, inputT);
        weightGrads = MatrixLib::multScalar(1.0 / (double)network->batchSize, weightGrads);

        matrix biasGrads = cur; // dz/db is 1 so its whatever the current gradients are

        layer->weightGrads = weightGrads;
        layer->biasGrads = biasGrads;

        // Continue backprop "flow"
        matrix weightsT = MatrixLib::transpose(layer->weights);
        cur = MatrixLib::mult(weightsT, cur); // dz/dx change of linear output with respect to last input 
    }
}

void Optimizer::step(Network* network, double learnRate)
{
    int depth = network->layers.size();

    for (int l = depth - 1; l > 0; l--) // Layer
    {
        Layer* layer = network->layers[l];
        
        matrix& weights = layer->weights;
        matrix& biases = layer->biases;
        matrix& weightGrads = layer->weightGrads;
        matrix& biasGrads = layer->biasGrads;

        // Update weights
        for (int i = 0; i < weights.size(); i++)
        {
            for (int j = 0; j < weights[0].size(); j++)
            {
                double newW = weights[i][j] - (learnRate * weightGrads[i][j]);
                weights[i][j] = newW;
            }
        }

        // Update biases
        for (int i = 0; i < biases.size(); i++)
        {
            double trueGrad = 0.0;
            for (int j = 0; j < biases[0].size(); j++) trueGrad += biasGrads[i][j];
            trueGrad /= network->batchSize; // Normalize

            for (int j = 0; j < biases[0].size(); j++)
            {
                double newB = biases[i][j] - (learnRate * trueGrad);
                biases[i][j] = newB;
            }
        }
    }
}