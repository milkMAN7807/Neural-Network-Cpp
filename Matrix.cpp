#include <cstdlib>
#include <iostream>
#include "Matrix.h"

matrix matrixlib::add(matrix a, matrix b)
{
    if (a.size() != b.size() || a[0].size() != b[0].size())
    {
        std::cout << "MATRIX ADDITION SIZE MISMATCH";
        std::exit(EXIT_FAILURE);
    }
    
    matrix r(a.size(), row(a[0].size()));

    for (int i = 0 ; i < a.size(); i++)
    {
        for (int j = 0; j < a[0].size(); j++)
        {
            r[i][j] = a[i][j] + b[i][j];
        }
    }

    return r;
}

matrix matrixlib::mult(matrix a, matrix b)
{
    if (a[0].size() != b.size())
    {
        std::cout << "MATRIX MULTIPLICATON MISMATCH";
        std::exit(EXIT_FAILURE);
    }

    matrix r(a.size(), row(b[0].size()));

    // Yeah im not using no O(n^2.8) algorithm lmao
    for (int bCol = 0; bCol < b[0].size(); bCol++)
    {
        for (int aRow = 0; aRow < a.size(); aRow++)
        {
            double sum = 0.0;
            for (int i = 0; i < a[0].size(); i++) sum += a[aRow][i] * b[i][bCol]; // Iter therough the common values

            r[aRow][bCol] = sum;
        }
    }

    return r;
}

void matrixlib::multScalar(double s, matrix& m)
{
    for (int i = 0; i < m.size(); i++)
    {
        for (int j = 0; j < m[0].size(); j++)
        {
            m[i][j] *= s;
        }
    }
}

matrix matrixlib::multElement(matrix a, matrix b)
{
    if (a.size() != b.size() || a[0].size() != b[0].size())
    {
        std::cout << "MATRIX MULTPLICATION_EW SIZE MISMATCH";
        std::exit(EXIT_FAILURE);
    }

    matrix r(a.size(), row(a[0].size()));

    for (int i = 0; i < a.size(); i++)
    {
        for (int j = 0; j < a[0].size(); j++)
        {
            r[i][j] = a[i][j] * b[i][j];
        }
    }

    return r;
}

matrix matrixlib::transpose(matrix m)
{
    matrix r(m[0].size(), row(m.size()));

    for (int i = 0; i < m.size(); i++)
    {
        for (int j = 0; j < m[0].size(); j++)
        {
            r[j][i] = m[i][j];
        }
    }

    return r;
}

// matrixlib::apply definition in header file

void matrixlib::debug::showDim(matrix m)
{
    std::cout << "num rows: ";
    std::cout << m.size();
    std::cout << " , num cols: ";
    std::cout << (m.size() > 0 ? m[0].size() : 0);
    std::cout << '\n';
}

void matrixlib::debug::show(matrix m)
{
    for (int i = 0; i < m.size(); i++)
    {
        std::cout << "[ ";
        
        for (int j = 0; j < m[0].size(); j++)
        {
            std::cout << m[i][j];
            std::cout << ", ";
        }

        std::cout << "]\n";
    }

    std::cout << '\n';
}