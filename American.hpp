#ifndef _AMERICAN_HPP
#define _AMERICAN_HPP

#include <assert.h>

#include "Option.hpp"
#include "ap.h"
#include "interpolation.h"

#include <gsl/gsl_sf.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>

static double WeighedLaguerre(double x, int n);

class AmericanOption : public Option
{
    protected:
	double strike; // The strike price of the underlying
	void LSE_estimate(vector<double> *x, vector<double> *y, vector<double> *ybar);
    public:
	AmericanOption(double rate, double underlying, double strike);
	virtual void evaluate();
};

#endif
