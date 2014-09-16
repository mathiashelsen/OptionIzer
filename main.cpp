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
    option->setWalk(newSeries);
    option->evaluate();

    //cout << "Call avg: " << callPDF->getAverage() << " +/- " << callPDF->getStandardDev() << "\n";
    //cout << "Put  avg: " << putPDF->getAverage() << " +/- " << putPDF->getStandardDev() << "\n";

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
