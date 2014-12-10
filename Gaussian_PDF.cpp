#include "Gaussian_PDF.hpp"

Gaussian_PDF::Gaussian_PDF(double _mu, double _sigma)
{
    mu = _mu;
    sigma = _sigma;

    gsl_rng_env_setup();
    T = gsl_rng_default;
    r = gsl_rng_alloc (T);
};

Gaussian_PDF::~Gaussian_PDF()
{
    gsl_rng_free(r);
};

double Gaussian_PDF::drawRandom()
{
    return (gsl_ran_gaussian(r, sigma) + mu);
};
