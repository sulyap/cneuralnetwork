#ifndef LAYER_HPP_
#define LAYER_HPP_

#include <iostream>
#include <vector>
#include <cstdlib>
#include <cassert>
#include <cmath>
#include <ctime>
#include <random>

using namespace std;

#include "neuron.hpp"

class Layer
{
public:
  // add neuron
  void addNeuron(Neuron n) { neurons.push_back(n); }

  // Getters
  vector<Neuron> getNeurons() { return neurons; }

  // Setters
  void setNeurons(vector<Neuron> neurons) { this->neurons = neurons; }

  // Constructor
  Layer();

  vector<Neuron> neurons;

  /* UTILITY METHODS */
  // As previous layer, return the expected inputs heading towards a current neuron
  vector<double> getInputs();

  // As previous layer, return the expected weights heading towards a current neuron
  vector<double> getWeights(int index);

private:
  vector<int> topology;       // topology of this network
};

#endif
