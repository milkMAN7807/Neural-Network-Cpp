#include <cstdlib>
#include <iostream>
#include "Matrix.h"

matrix MatrixLib::add(const matrix& a, const matrix& b)
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

matrix MatrixLib::mult(const matrix& a, const matrix& b)
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

matrix MatrixLib::multScalar(double s, const matrix& m)
{
    matrix r(m.size(), row(m[0].size()));
    
    for (int i = 0; i < m.size(); i++)
    {
        for (int j = 0; j < m[0].size(); j++)
        {
            r[i][j] = m[i][j] * s;
        }
    }

    return r;
}

matrix MatrixLib::multElement(const matrix& a, const matrix& b)
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

matrix MatrixLib::transpose(const matrix& m)
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

matrix MatrixLib::apply(const matrix& in, function<double(double)> func)
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

void MatrixLib::Debug::showDim(const matrix& m)
{
    std::cout << "num rows: ";
    std::cout << m.size();
    std::cout << " , num cols: ";
    std::cout << (m.size() > 0 ? m[0].size() : 0);
    std::cout << '\n';
}

void MatrixLib::Debug::show(const matrix& m)
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