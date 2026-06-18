#include <iostream>

#include "CustomFunctions.h"
#include "Network.h"
#include "Matrix.h"
#include "Optimizer.h"

int main()
{
    Network* myModel = new Network(1);

    myModel->layers.push_back(new Layer(5, 1, identity, true, nullptr));
    myModel->layers.push_back(new Layer(15, 1, relu, false, myModel->layers[0]));
    myModel->layers.push_back(new Layer(15, 1, relu, false, myModel->layers[1]));
    myModel->layers.push_back(new Layer(3, 1, sigmoid, false, myModel->layers[2]));

    matrix in1 = matrixlib::transpose({ { 1, 2, 3, 4, 5 } });

    matrix truth = matrixlib::transpose({ { 0.2, 0.4, 0.4 } });

    for (int i = 0; i < 1000; i++)
    {
        matrix out1 = myModel->forward(in1, true);
        Optimizer::backward(myModel, out1, truth, mseLoss);
        Optimizer::step(myModel, 0.1);
    }
    
    matrix final = myModel->forward(in1, false);

    matrixlib::debug::show(Optimizer::applyLoss(final, truth, mseLoss));


    return 0;
}