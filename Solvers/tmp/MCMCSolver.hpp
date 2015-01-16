#ifndef _MCMCSOLVER_HPP
#define _MCMCSOLVER_HPP

#include "../OptionTypes/VanillaOption.hpp"
#include "../Generic_PDF.hpp"
#include "Solver.hpp"

#include "../linalg/ap.h"
#include "../linalg/interpolation.h"

#include <boost/random.hpp>
#include <boost/random/mersenne_twister.hpp>

#include <gsl/gsl_histogram.h>

#include <vector>

using namespace std;

class MCMCSolver : public Solver<VanillaOption>
{
    private:
	Generic_PDF *p;
	gsl_histogram *priceHistogram;
	int Nbins;
	int Nseries;
	int Nsteps;
	double dt;
	double **assetValues;
	double *payoffs;
	void performCalculation(VanillaOption *option);
    public:
	MCMCSolver(Generic_PDF *_p, int _Nbins, int _NSeries, int _Nsteps);
	~MCMCSolver();
	void operator()(VanillaOption *option);
	double errCalculation();
};

#endif
