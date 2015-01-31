#ifndef MULTILAYER_PERCEPTRON_HPP_
#define MULTILAYER_PERCEPTRON_HPP_

#include <iostream>
#include <vector>
#include <cstdlib>
#include <cassert>
#include <cmath>

using namespace std;

#include "layer.hpp"

class MultilayerPerceptron
{
public:
  // constructor
  MultilayerPerceptron(vector<int> topology);

  // Left to right transfer of information
  void feedForward(vector<double> inputVals);

  // Learning method
  void backPropagation(vector<double> targetVals);

  vector<double> getResults();

  // setters
  void setTopology(vector<int> topology) { this->topology = topology; }
  void setLayers(vector<Layer> layers) { this->layers = layers; }

  // getters
  vector<int> getTopology() { return topology; }
  vector<Layer> getLayers() { return layers; }

  // utility methods
  void printOutputLayer();

  double overallNetError;
  double recentAverageError;
  double recentAverageSmoothingFactor;

private:
  vector<int> topology;       // topology of this network
  vector<Layer> layers;       // collection of layers
};

#endif
