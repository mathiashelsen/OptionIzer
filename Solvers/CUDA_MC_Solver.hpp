#ifndef _CUDA_MC_SOLVER_HPP
#define _CUDA_MC_SOLVER_HPP

#include "../OptionTypes/VanillaOption.hpp"

#include <assert.h>
#include <iostream>
#include <stdio.h>

#include <cuda.h>
#include <curand.h>

template<class OptionType> class CUDA_MC_Solver : public Solver<OptionType>
{
public:
    virtual ~CUDA_MC_Solver(){ return; }; 
    virtual void operator()(OptionType *option);
};

template<> class CUDA_MC_Solver<VanillaOption> : public Solver<VanillaOption>
{
private:
    int Nseries, Nsteps;
    float *returns, *assets, *payoffs;
    curandGenerator_t gen;
public:
    CUDA_MC_Solver(int _NSeries, int _NStep);
    ~CUDA_MC_Solver();
    void operator()(VanillaOption *option);
};

#endif
