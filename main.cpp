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

#include "Generic_PDF.hpp"
#include "NIT_PDF.hpp"
#include "TimeSeries.hpp"
#include "European.hpp"
#include "American.hpp"
#include "BlackScholes.hpp"

using namespace std;

void readFile(ifstream *file, vector<double> *data);

int main(int argc, char **argv)
{
    ifstream inputFile( argv[1] );
    vector<double> values;

    readFile( &inputFile, &values );

    double r = 3.0;
    NIT_PDF *newPDF = new NIT_PDF(200);
    newPDF->generatePDF(&values);
    newPDF->setDrift(exp(r/3.6e4) - 1.0);
    double sigma = newPDF->getStandardDev();

    double i = 0.1;
    double S0 = 100.0;
    double T = 65.0;
    BlackScholes bs(S0, 100.0, sigma, r/3.6e4, 65.0);
    while( T > 0.0 )
    {
	i = 0.1;
	while( i <= 2.0 )
	{
	bs.setS0(S0*i);	
	double a = 0.0, b = 0.0;

	std::cout << T << "\t" << S0*i << "\t";
	bs.calcDelta(&a, &b);
	std::cout << a << "\t";
	bs.calcGamma(&a, &b);
	std::cout << a << "\t";
	bs.calcVega(&a, &b);
	std::cout << a << "\t";
	bs.calcTheta(&a, &b);
	std::cout << a << "\n";

	i += 0.005;
	}
	std::cout << "\n";
	T -= 1.0;
	bs.setT(T);
    }

    delete newPDF;
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
