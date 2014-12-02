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
#include "Binomial.hpp"

using namespace std;

void readFile(ifstream *file, vector<double> *data);

int main(int argc, char **argv)
{
    ifstream inputFile( argv[1] );
    vector<double> values;

    readFile( &inputFile, &values );

    double r = 3.0;
    double T = 250.0;
    double S0 = 100.0;
    NIT_PDF *newPDF = new NIT_PDF(200);
    newPDF->generatePDF(&values);
    newPDF->setDrift(exp(r/3.6e4) - 1.0);
    TimeSeries series(250, 1000, S0, (Generic_PDF *)newPDF);
    double sigma = newPDF->getStandardDev();
    /*
    for(int i = 10; i < 1000; i+=20)
    {
	double price = 0.0;
	trial.setN(i);
	trial.calcPrice(&price);
	std::cout << i << "\t" << price << std::endl;
    }
    */
    double i = 0.1;
    BlackScholes bs(S0, 100.0, sigma, r/3.6e4, T);
    Binomial trial(S0, 100.0, sigma, r/3.6e4, T, 1000);
    AmericanOption mc(r/3.6e4, S0, 100.0, 100);
    mc.setWalk(&series);
    mc.evaluate();
    NIT_PDF *putDist = mc.getPutPriceDist();


    double a,b;
    bs.calcPrice(&a, &b);
    std::cout << b << "\t";

    trial.evaluate();
    trial.calcPrice(&a);
    std::cout << a << "\t";

    std::cout << putDist->getAverage() << std::endl;

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
