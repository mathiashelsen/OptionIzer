#ifndef _MCMCSOLVER_HPP
#define _MCMCSOLVER_HPP

#include "../OptionTypes/VanillaOption.hpp"
#include "Solver.hpp"
#include "../Generic_PDF.hpp"

#include "../linalg/ap.h"
#include "../linalg/interpolation.h"

#include <gsl/gsl_histogram.h>
#include <vector>

#include <boost/random.hpp>
#include <boost/random/mersenne_twister.hpp>

using namespace std;

class MCMCSolver : public Solver<VanillaOption>
{
    private:
	Generic_PDF *p;
	gsl_histogram *priceHistogram;
	int Nbins;
	int Nseries;
	int Nsteps;
	double **assetValues;
	double *payoffs;
    public:
	MCMCSolver(Generic_PDF *_p, int _Nbins, int _NSeries, int _Nsteps);
	~MCMCSolver();
	void operator()(VanillaOption *option);
};

#endif
