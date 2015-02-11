#include <iostream>
#include <vector>
#include <cstdlib>
#include <cassert>
#include <cmath>
#include <ctime>
#include <random>

#include "neuron.hpp"
#include "layer.hpp"
#include "multilayer_perceptron.hpp"
#include "training_data.hpp"

using namespace std;

int epoch;
int eta;
int alpha;
string csv_file;
string output_file;

void syntax()
{
  cout << "trainin_data [csv_file] [output_file] [eta] [alpha]" << endl;
}

int main(int argc, char **argv)
{
  if(argc != 5) {
    syntax();
    exit(-1);
  }

  csv_file = argv[1];
  output_file = argv[2];
  eta = stod(argv[3]);
  alpha = stod(argv[4]);

  TrainingData *td = new TrainingData(argv[1]);

  return 0;
}
