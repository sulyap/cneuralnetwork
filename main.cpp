#include <iostream>
#include <vector>
#include <cstdlib>
#include <cassert>
#include <cmath>
#include <ctime>
#include <random>

using namespace std;

#include "neuron.hpp"
#include "layer.hpp"
#include "multilayer_perceptron.hpp"

using namespace std;

int main(int argc, char **argv)
{
  vector<int> topology;
  topology.push_back(2);
  topology.push_back(6);
  topology.push_back(4);
  topology.push_back(4);
  topology.push_back(6);
  topology.push_back(4);
  topology.push_back(4);
  topology.push_back(1);
  MultilayerPerceptron network(topology);

  int epoch = 1000;
  for(int i = 0; i < epoch; i++) {
    vector<vector<double> > trainingData;

    vector<double> inputVals;
    vector<double> targetVals;

    inputVals.push_back(0);
    inputVals.push_back(0);
    targetVals.push_back(0);

    network.feedForward(inputVals);
    network.backPropagation(targetVals);

    vector<double> resultVals = network.getResults();
    for(int i = 0; i < resultVals.size(); i++) {
      cout << "INPUT: 0 0" << endl;
      cout << "OUTPUT[" << i << "]: " << resultVals.at(i) << endl;
    }
    resultVals.clear();

    inputVals.clear();
    targetVals.clear();

    inputVals.push_back(0);
    inputVals.push_back(1);
    targetVals.push_back(1);

    network.feedForward(inputVals);
    network.backPropagation(targetVals);

    resultVals = network.getResults();
    for(int i = 0; i < resultVals.size(); i++) {
      cout << "INPUT: 0 1" << endl;
      cout << "OUTPUT[" << i << "]: " << resultVals.at(i) << endl;
    }
    resultVals.clear();

    inputVals.clear();
    targetVals.clear();

    inputVals.push_back(1);
    inputVals.push_back(0);
    targetVals.push_back(1);

    network.feedForward(inputVals);
    network.backPropagation(targetVals);

    resultVals = network.getResults();
    for(int i = 0; i < resultVals.size(); i++) {
      cout << "INPUT: 1 0" << endl;
      cout << "OUTPUT[" << i << "]: " << resultVals.at(i) << endl;
    }
    resultVals.clear();

    inputVals.clear();
    targetVals.clear();

    inputVals.push_back(1);
    inputVals.push_back(1);
    targetVals.push_back(0);

    network.feedForward(inputVals);
    network.backPropagation(targetVals);

    resultVals = network.getResults();
    for(int i = 0; i < resultVals.size(); i++) {
      cout << "INPUT: 1 1" << endl;
      cout << "OUTPUT[" << i << "]: " << resultVals.at(i) << endl;
    }
    resultVals.clear();
  }



  return 0;
}
