/*
The MIT License (MIT)

Copyright (c) 2014 Mathias Helsen

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include <iostream>
#include <fstream>
#include <string>
#include <cstdlib>
#include <vector>
#include <string>

#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>

/*
#include "Generic_PDF.hpp"
#include "NIT_PDF.hpp"
#include "TimeSeries.hpp"
#include "European.hpp"
#include "American.hpp"
#include "BlackScholes.hpp"
#include "Binomial.hpp"
#include "FiniteDiff.hpp"
*/

#include "Solvers/BinomialSolver.hpp"
#include "Solvers/FiniteDiffSolver.hpp"
#include "Solvers/BlackScholesSolver.hpp"
#include "Solvers/MCMCSolver.hpp"
#include "OptionTypes/VanillaOption.hpp"
#include "Gaussian_PDF.hpp"

using namespace std;

void readFile(ifstream *file, vector<double> *data);

int main(int argc, char **argv)
{
    double r = 3.0/3.6e4;
    double T = 250.0;
    double S0 = 20.0;
    double K = 100.0;
    double sigma = 0.02;

    double price, delta, gamma, theta;


    VanillaOption trialOption(S0, K, sigma, r, T, true, true);
    BinomialSolver solver(1000);
    FiniteDiffSolver diffSolve(1000, 1000);
    BlackScholesSolver *bsSolve = new BlackScholesSolver;

    Gaussian_PDF mcmcPDF(r, sigma/sqrt(1.0));
    MCMCSolver mcmcSolver(&mcmcPDF, 50, 4000, (int)T);
    while(S0 < 120.0)
    {
	trialOption.setUnderlying(S0); 

	solver(&trialOption);
	trialOption.getPrice(&price);
	trialOption.getGreeks(&delta, &gamma, &theta);
	std::cout << S0 << "\t" << price << "\t"; 


	diffSolve(&trialOption);
	trialOption.getPrice(&price);
	trialOption.getGreeks(&delta, &gamma, &theta);
	std::cout << price << "\t";


	(*bsSolve)(&trialOption);
	trialOption.getPrice(&price);
	trialOption.getGreeks(&delta, &gamma, &theta);
	std::cout << price << "\t";


	mcmcSolver(&trialOption);
	trialOption.getPrice(&price);
	trialOption.getGreeks(&delta, &gamma, &theta);
	std::cout << price << "\t" << mcmcSolver.errCalculation() << std::endl;


	S0 += 1.0;
    }

    return 0;
}

void readFile(ifstream *file, vector<double> *data)
{
    if(file->is_open())
    {
	string line;
	while( getline( *file, line ) )
	{
	    data->push_back( atof( line.c_str() ) );
	}
    }
}
