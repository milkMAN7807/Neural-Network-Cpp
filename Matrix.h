#pragma once
#include <vector>

using namespace std;

typedef vector<vector<double>> matrix;
typedef vector<double> row;
 
namespace matrixlib
{
    matrix add(matrix a, matrix b);
    matrix mult(matrix a, matrix b);
    void multScalar(double s, matrix& m);
    matrix multElement(matrix a, matrix b);
    matrix transpose(matrix m);
    
    // Compiler throwing BS if this is in the .cpp file, this is the easy fix, oh well
    template <typename T>
    matrix apply(matrix in, T func)
    {
        matrix r(in.size(), row(in[0].size()));
    
        for (int i = 0; i < in.size(); i++)
        {
            for (int j = 0; j < in[0].size(); j++)
            {
                r[i][j] = func(in[i][j]);
            }
        }

        return r;
    }

    namespace debug
    {
        void showDim(matrix m);
        void show(matrix m);
    }
}