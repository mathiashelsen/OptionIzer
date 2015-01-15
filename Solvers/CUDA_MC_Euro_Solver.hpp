#ifndef _CUDA_MC_EURO_SOLVER
#define _CUDA_MC_EURO_SOLVER

#include "../OptionTypes/VanillaOption.hpp"

#include <assert.h>
#include <iostream>
#include <stdio.h>

#include <cuda.h>
#include <curand.h>

class CUDA_MC_Euro_Solver : public Solver<VanillaOption>
{
private:
    int Nseries;
    int Nsteps;

    float *returns, *assets, *payoffs;
    curandGenerator_t gen;
public:
    CUDA_MC_Euro_Solver(int _NSeries, int _NStep);
    ~CUDA_MC_Euro_Solver();
    void operator()(VanillaOption *option);
    
};

#endif
