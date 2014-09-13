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
#include "Vanilla.hpp"

using namespace std;

void readFile(ifstream *file, vector<double> *data);

int main(int argc, char **argv)
{
    ifstream inputFile( argv[1] );
    vector<double> values;

    readFile( &inputFile, &values );

    MyPDF *newPDF = new MyPDF(50, &values);
    TimeSeries *newSeries = new TimeSeries(90, 2000, newPDF);
    VanillaOption *option = new VanillaOption( 0.05, 191.28, 170.0 );
    MyPDF *callPDF = new MyPDF(100);
    MyPDF *putPDF = new MyPDF(100);
    option->getValueDistribution(newSeries, callPDF, putPDF);

    cout << "Call avg: " << callPDF->getAverage() << "\n";
    cout << "Put avg: " << putPDF->getAverage() << "\n";

    for(int i = 0 ; i < 1000; i++ )
    {
	double x = 0.1 * (double)i; // Price between 0 and 30$
//	cout << x << "\t" << callPDF->getPDFValue(x) << "\t" << putPDF->getPDFValue(x) << "\n";
    }

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
