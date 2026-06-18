#pragma once
#include <cmath>
#include <unordered_map>
#include <functional>

using namespace std;

inline double invSqrt(int n)
{
    return 1.0 / sqrt(n);
}

// Activation functions
inline double sigmoid(double x)
{
    return 1 / (1 + exp(-x));
}

inline double relu(double x)
{
    return x > 0 ? x : 0;
}

inline double identity(double x)
{
    return x;
}

// Loss
inline double mseLoss(double x, double groundTruth)
{
    return (x - groundTruth) * (x - groundTruth);
}

// Derivative funcs
inline double sigmoidDerivative(double x)
{
    return exp(-x) / pow((1 + exp(-x)), 2);
}

inline double reluDerivative(double x)
{
    return x > 0 ? 1 : 0;
}

inline double oneFunc(double x)
{
    return 1.0;
}

inline double mseLossDerivative(double x, double groundTruth)
{
    return 2 * (x - groundTruth);
}

inline unordered_map<double(*)(double), double(*)(double)> derivativeMap = 
{ 
    { sigmoid, sigmoidDerivative }, 
    { relu, reluDerivative },
    { identity, oneFunc }
};

inline unordered_map<double(*)(double, double), double(*)(double, double)> lossDerivativeMap = 
{ 
    { mseLoss, mseLossDerivative } 
};