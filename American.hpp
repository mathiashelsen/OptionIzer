#ifndef _AMERICAN_HPP
#define _AMERICAN_HPP

#include <assert.h>

#include "Option.hpp"
#include "linalg/ap.h"
#include "linalg/interpolation.h"

#include <gsl/gsl_sf.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>

class AmericanOption : public Option
{
    protected:
	double strike; // The strike price of the underlying
	void LSE_estimate(vector<double> *x, vector<double> *y, vector<double> *ybar);
    public:
	AmericanOption(double _rate, double _underlying, double _strike, int bins);
	~AmericanOption();
	virtual void evaluate();
};

#endif
