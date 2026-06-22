#include <iostream>
#include <vector>

#include "mnist/mnist_reader.hpp"

#include "CustomFunctions.h"
#include "Network.h"
#include "Matrix.h"
#include "Optimizer.h"

#define BATCH_SIZE 3
#define EPOCHS 10
#define EXPECTED_OUTPUTS 10
#define LEARN_RATE 0.01

using namespace std;
using namespace NeuralNet;

int main()
{
    // Create Model
    Network* imgClassificationModel = new Network(BATCH_SIZE);

    int mBatchSize = imgClassificationModel->batchSize;
    vector<Layer*>& mLayers = imgClassificationModel->layers;

/*
    mLayers.push_back(new Layer(5, mBatchSize, identity, true, nullptr));
    mLayers.push_back(new Layer(15, mBatchSize, relu, false, mLayers[0]));
    mLayers.push_back(new Layer(15, mBatchSize, relu, false, mLayers[1]));
    mLayers.push_back(new Layer(3, mBatchSize, sigmoid, false, mLayers[2]));

    matrix testInput = MatrixLib::transpose(
        {
            { 1, 2, 3, 4, 5 },
            { 5, 4, 3, 2, 1 },
            { 3, 1, 5, 4, 2 }
        }
    );
    matrix tru = MatrixLib::transpose(
    {
        { 0.2, 0.5, 0.2 },
        { 0.5, 0.2, 0.2 },
        { 0.1, 0.1, 0.8 }
    }
    );

    matrix mOut = imgClassificationModel->forward(testInput, false);
    MatrixLib::Debug::show(Optimizer::applyLoss(mOut, tru, mseLoss));


    for (int i = 0; i < 10000; i++)
    {
        matrix mOutG = imgClassificationModel->forward(testInput, true);
        Optimizer::backward(imgClassificationModel, mOutG, tru, mseLoss);
        Optimizer::step(imgClassificationModel, 0.1);
    }

    mOut = imgClassificationModel->forward(testInput, false);
    MatrixLib::Debug::show(Optimizer::applyLoss(mOut, tru, mseLoss));
*/



    mLayers.push_back(new Layer(784, mBatchSize, identity, true, nullptr));
    mLayers.push_back(new Layer(30, mBatchSize, relu, false, mLayers[0]));
    mLayers.push_back(new Layer(30, mBatchSize, relu, false, mLayers[1]));
    mLayers.push_back(new Layer(30, mBatchSize, relu, false, mLayers[2]));
    mLayers.push_back(new Layer(10, mBatchSize, sigmoid, false, mLayers[3]));

    // Load data
    auto dataset = mnist::read_dataset<vector, vector, uint8_t, uint8_t>("data");

    // vector<uint8_t>
    auto& trainImages = dataset.training_images;
    auto& testImages = dataset.test_images;
    auto& trainLabels = dataset.training_labels;
    auto& testLabels = dataset.test_labels;

    int TD_SIZE_TEMP = trainImages.size() - (trainImages.size() % BATCH_SIZE); //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    // Train
    for (int curEpoch = 1; curEpoch <= EPOCHS; curEpoch++)
    {
        // Batch
        for (int batchIdx = 0; batchIdx < TD_SIZE_TEMP; batchIdx += BATCH_SIZE) // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        {
            // Create input matrix
            matrix modelIn(mBatchSize, row(trainImages[batchIdx].size()));
            matrix truth(mBatchSize, row(EXPECTED_OUTPUTS));

            int bNum = 0;
            for (int i = batchIdx; i < batchIdx + BATCH_SIZE; i++) // Convert vector<uint8_t> to double
            {
                vector<uint8_t>& img = trainImages[i];
                modelIn[i - batchIdx] = vector<double>(img.begin(), img.end());
                for (double& v : modelIn[i - batchIdx]) v /= 255.0;

                uint8_t idx = trainLabels[i];
                truth[i - batchIdx][(int)idx] = 1.0;

                cout << "TRAIN ON INPUT " << i << " EPOCH " << curEpoch << '\n';
            }

            modelIn = MatrixLib::transpose(modelIn);
            truth = MatrixLib::transpose(truth);

            matrix modelOut = imgClassificationModel->forward(modelIn, true);
            Optimizer::backward(imgClassificationModel, modelOut, truth, mseLoss);
            Optimizer::step(imgClassificationModel, LEARN_RATE);
        }
    }

    int numCorrect = 0;

    // Eval
    for (int batchIdx = 0; batchIdx < testImages.size() - (testImages.size() % BATCH_SIZE); batchIdx += BATCH_SIZE)// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    {
        matrix modelIn(mBatchSize, row(testImages[batchIdx].size()));
        matrix truth(mBatchSize, row(EXPECTED_OUTPUTS));

        for (int i = batchIdx; i < batchIdx + BATCH_SIZE; i++) // Convert vector<uint8_t> to double
        {
            vector<uint8_t>& img = trainImages[i];
            modelIn[i - batchIdx] = vector<double>(img.begin(), img.end());
            for (double& v : modelIn[i - batchIdx]) v /= 255.0;

            uint8_t idx = trainLabels[i];
            truth[i - batchIdx][(int)idx] = 1.0;
            cout << "TEST ON INPUT " << i << '\n';
        }

        modelIn = MatrixLib::transpose(modelIn);
        truth = MatrixLib::transpose(truth);
        matrix modelOut = imgClassificationModel->forward(modelIn, false);

        // EDIT LATER:
        modelOut = MatrixLib::transpose(modelOut);
        truth = MatrixLib::transpose(truth);

        for (int i = 0; i < truth.size(); i++)
        {
            int predIdx = 0, trueIdx = 0;
            double maxVal = modelOut[i][0];

            for (int j = 1; j < modelOut[i].size(); j++)
            {
                if (modelOut[i][j] > maxVal)
                {
                    maxVal = modelOut[i][j];
                    predIdx = j;
                }
            }

            for (int j = 0; j < truth[i].size(); j++)
            {
                if (round(truth[i][j]) == 1) { trueIdx = j; break; }
            }

            if (predIdx == trueIdx) numCorrect++;
        }
    }

    std::cout << "Model Accuracy: ";
    std::cout << (double)numCorrect / (double)testImages.size() << '\n';

    std::cout << "success";
    return 0;
}