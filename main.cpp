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

using namespace std;

void readFile(ifstream *file, vector<double> *data);

int main(int argc, char **argv)
{
    ifstream inputFile( argv[1] );
    vector<double> values;

    readFile( &inputFile, &values );

    MyPDF *newPDF = new MyPDF(50, &values);
    TimeSeries *newSeries = new TimeSeries(90, 2000, newPDF);

    cout << newSeries->series[0][0] << ", " << newSeries->series[0][1] << ", " << newSeries->series[1][0] << "\n";

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
