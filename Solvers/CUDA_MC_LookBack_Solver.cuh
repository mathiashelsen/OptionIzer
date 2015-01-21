#ifndef _CUDA_MC_LOOKBACK_SOLVER_CUH
#define _CUDA_MC_LOOKBACK_SOLVER_CUH

#include "CUDA_MC_Solver.hpp"
#
enum LookBackType
{
    LookBackTypeMin, LookBackTypeMax
};


template<LookBackType type> __global__ void LookBackKernel(float *_x, float *_assets, float *_payoffs,
    float r,
    float S0,
    float sigma,
    float T,
    float call,
    int Nseries,
    int Nsteps)
{

}

#endif
