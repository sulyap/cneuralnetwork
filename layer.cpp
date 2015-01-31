#include "layer.hpp"

Layer::Layer()
{
}

// Utility method to get the input vector of this layer
vector<double> Layer::getInputs()
{
  vector<double> vals;

  for(int i = 0; i < neurons.size(); i++) {
    vals.push_back(neurons.at(i).getOutputValue());
  }

  return vals;
}

// Utility method to get the weights headed for a neuron given by index from this layer
vector<double> Layer::getWeights(int index)
{
  vector<double> vals;

  for(int i = 0; i < neurons.size(); i++) {
    vals.push_back(neurons.at(i).getOutputWeights().at(index).value);
  }

  return vals;
}
