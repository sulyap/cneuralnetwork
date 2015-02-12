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
double eta;
double alpha;
string csv_file;
string output_file;

void syntax()
{
  cout << "train_data [csv_file] [output_file] [eta] [alpha] [epoch]" << endl;
}

int main(int argc, char **argv)
{
  if(argc != 6) {
    syntax();
    exit(-1);
  }

  csv_file = argv[1];
  output_file = argv[2];
  eta = stod(argv[3]);
  cout << "ETA: " << eta << endl;
  alpha = stod(argv[4]);
  cout << "ALPHA: " << alpha << endl;
  epoch = atoi(argv[5]);
  cout << "EPOCH: " << epoch << endl;

  vector<int> topology;
  topology.push_back(24);
  topology.push_back(25);
  topology.push_back(1);

  TrainingData td(argv[1]);

  MultilayerPerceptron *mp = new MultilayerPerceptron(topology, eta, alpha);
  mp->train(td, epoch);

  return 0;
}
