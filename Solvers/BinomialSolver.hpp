#ifndef _BINOMIALSOLVER_HPP
#define _BINOMIALSOLVER_HPP

#include "Solver.hpp"
#include "../OptionTypes/AmericanOption.hpp"
#include "../OptionTypes/EuroOption.hpp"
#include "../OptionTypes/VanillaOption.hpp"


#include <algorithm>
#include <assert.h>
#include <iostream>
#include <math.h>

template<class OptionType> class BinomialSolver : private Solver<OptionType>
{
    private:
	int N;
	double **assetValues;
	double **optionValues;

    public:
	void operator()(OptionType *option);
	void init();
	BinomialSolver(int _N);
	~BinomialSolver();
};

template<class OptionType> BinomialSolver<OptionType>::BinomialSolver(int _N)
{
    N = _N;
    assetValues = NULL;
    optionValues = NULL;
};

template<class OptionType> void BinomialSolver<OptionType>::init()
{
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
	if(assetValues[i])
	    delete assetValues[i];
	if(optionValues[i])
	    delete optionValues[i];
    }

    if(assetValues)
	delete[] assetValues;
    if(optionValues)
	delete[] optionValues;
};

#endif
