#ifndef _GAUSSIANPDF_HPP
#define _GAUSSIANPDF_HPP

#include "Generic_PDF.hpp"

#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>

class Gaussian_PDF : public Generic_PDF
{
    private:
	double mu;
	double sigma;
	gsl_rng *r;
	const gsl_rng_type * T;
    public:
	double drawRandom();
	Gaussian_PDF(double _mu, double _sigma);
	~Gaussian_PDF();

};

#endif
