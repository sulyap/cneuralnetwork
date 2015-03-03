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
string validation_file;
vector<vector<double> > validation_data;
vector<double> labels;

void syntax()
{
  cout << "train_data [csv_file] [output_file] [eta] [alpha] [epoch] [num_layers] [validation_file]" << endl;
}

int main(int argc, char **argv)
{
  if(argc != 8) {
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

  // Validation
  ifstream file(argv[7]);
  string line;

  vector<vector<double> > buffData;
  while(getline(file, line)) {
    vector<double> dataLine;

    std::istringstream iss(line);
    std::string result;
    while(std::getline(iss, result, ',')) {
      dataLine.push_back(stod(result));
    }

    buffData.push_back(dataLine);
  }

  for(int i = 0; i < buffData.size(); i++) {
    vector<double> training_data_vector;
    for(int j = 0; j < buffData.at(i).size() - 1; j++) {
      training_data_vector.push_back(buffData.at(i).at(j));
    }
    
    validation_data.push_back(training_data_vector);
    labels.push_back(buffData.at(i).at(buffData.at(i).size() - 1));
  }

  int positiveHits = 0;
  int negativeHits = 0;

  for(int i = 0; i < validation_data.size(); i++) {
    double result = mp->predict(validation_data.at(i));
    double expected = labels.at(i);
    cout << "Result: " << result << " Expected: " << expected << endl;
    if(result > 0.5 and expected == 1) {
      positiveHits++;
    }

    if(result <= 0.5 and expected == 0) {
      negativeHits++;
    }
  }

  int totalHits = positiveHits + negativeHits;
  double accuracy = (double)totalHits / (double)validation_data.size();
  
  cout << "Final accuracy: " << accuracy * 100 << "%" << endl;

  mp->save(output_file);

  return 0;
}
