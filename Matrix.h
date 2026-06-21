#pragma once
#include <vector>
#include <functional>

using namespace std;

typedef vector<vector<double>> matrix;
typedef vector<double> row;
 
namespace MatrixLib
{
    matrix add(const matrix& a, const matrix& b);
    matrix mult(const matrix& a, const matrix& b);
    matrix multScalar(double s, const matrix& m);
    matrix multElement(const matrix& a, const matrix& b);
    matrix transpose(const matrix& m);
    matrix apply(const matrix& in, function<double(double)> func);

    namespace Debug
    {
        void showDim(const matrix& m);
        void show(const matrix& m);
    }
}