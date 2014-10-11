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

#include "MyPDF.hpp"
#include "TimeSeries.hpp"
#include "European.hpp"
#include "American.hpp"

using namespace std;

void readFile(ifstream *file, vector<double> *data);

int main(int argc, char **argv)
{
    ifstream inputFile( argv[1] );
    vector<double> values;

    readFile( &inputFile, &values );

    MyPDF *newPDF = new MyPDF(200, &values, true);
    TimeSeries *newSeries = new TimeSeries(65, 10000, newPDF);
    EuropeanOption *option = new EuropeanOption( 0.05, 191.28, 190.0, 100);
    AmericanOption *american = new AmericanOption( 0.05, 191.28, 190.0, 100);
    option->setWalk(newSeries);
    option->evaluate();
    american->setWalk(newSeries);
    american->evaluate();

    //cout << "Call avg: " << callPDF->getAverage() << " +/- " << callPDF->getStandardDev() << "\n";
    cout << "European put  avg: " << option->getPutPriceDist()->getAverage() << "\n";
    cout << "American put  avg: " << american->getPutPriceDist()->getAverage() << "\n";


    delete option;
    delete newSeries;
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
