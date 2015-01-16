#ifndef _BINOMIALSOLVER_HPP
#define _BINOMIALSOLVER_HPP

#include "Solver.hpp"
#include "../OptionTypes/AmericanOption.hpp"
#include "../OptionTypes/EuroOption.hpp"
#include "../OptionTypes/VanillaOption.hpp"


#include <iostream>
#include <math.h>
#include <algorithm>

template<class OptionType> class BinomialSolver : private Solver<OptionType>
{
    private:
	int N;
	double **assetValues;
	double **optionValues;

    public:
	void operator()(OptionType *option);
	BinomialSolver(int _N);
	~BinomialSolver();
};

template<class OptionType> BinomialSolver<OptionType>::BinomialSolver(int _N)
{
    N = _N;

    assetValues = new double*[N+1];
    optionValues = new double*[N+1];

    assetValues[0] = new double[1];
    optionValues[0] = new double[1];
    for(int i = 1; i < N+1; i++)
    {
	assetValues[i] = new double[i+1];
	optionValues[i] = new double[i+1];
    }
};

template<class OptionType> BinomialSolver<OptionType>::~BinomialSolver()
{
    for(int i = 0; i < N+1; i++)
    {
	delete assetValues[i];
	delete optionValues[i];
    }
    delete[] assetValues;
    delete[] optionValues;
};

#endif
