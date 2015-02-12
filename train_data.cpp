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
int num_layers;
double eta;
double alpha;
string csv_file;
string output_file;

void syntax()
{
  cout << "train_data [csv_file] [output_file] [eta] [alpha] [epoch] [num_layers]" << endl;
}

int main(int argc, char **argv)
{
  if(argc != 7) {
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
  num_layers = atoi(argv[6]);
  cout << "NUM LAYERS: " << num_layers << endl;

  // Build neuron count for each layer
  vector<int> topology;
  for(int i = 0; i < num_layers; i++) {
    int neuron_count;
    if(i == 0) {
      cout << "[Layer " << i + 1 << " Input] Neuron count: ";
    } else if(i == num_layers - 1) {
      cout << "[Layer " << i + 1 << " Output] Neuron count: ";
    } else {
      cout << "[Layer " << i + 1 << " Hidden] Neuron count: ";
    }
    cin >> neuron_count;
    topology.push_back(neuron_count);
  }

  TrainingData td(argv[1]);

  MultilayerPerceptron *mp = new MultilayerPerceptron(topology, eta, alpha);
  mp->train(td, epoch);

  return 0;
}
