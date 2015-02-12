#ifndef NEURON_HPP_
#define NEURON_HPP_

#include <iostream>
#include <vector>
#include <cstdlib>
#include <cassert>
#include <cmath>
#include <ctime>
#include <random>

using namespace std;

#include "weight.hpp"

class Neuron
{
public:
  int getId() { return id; }

  // Getters
  double getOutputValue() { return outputValue; }
  bool isBias() { return flagBias; }
  bool isInput() { return flagInput; }
  bool isHidden() { return flagHidden; }
  vector<Weight> getOutputWeights() { return outputWeights; }

  // Setters
  void setOutputValue(double val) { this->outputValue = val; }
  void setFlagBias(bool val) { this->flagBias = val; }
  void setFlagInput(bool val) { this->flagInput = val; }
  void setFlagHidden(bool val) { this->flagHidden = val; }

  // Calculate output gradients
  void calculateOutputGradients(double targetVal);

  // Calculate hidden gradients
  //void calculateHiddenGradients(Layer &nextLayer);
  void calculateHiddenGradients(vector<Neuron> nextLayerNeurons);

  // TODO: Document this
  void updateInputWeights(vector<Neuron> &previousLayerNeurons);

  // Summation of all the derivatives  of weights
  double sumDOW(vector<Neuron> nextLayerNeurons);

  // Feed Forward operation
  void feedForward(vector<double> inputs, vector<double> weights);

  // Transfer Function
  static double transferFunction(double x);

  // TODO: Why do we need the derivative?
  static double transferFunctionDerivative(double x);

  // Constructor that accepts flags to determine Neuron type
  // Bias - Extra with constant 1 as its value
  // Input - Accepts only one value
  // Hidden - Accepts multiple inputs
  // eta - magnifying factor
  // alpha - momentum
  Neuron(bool flagBias, bool flagInput, bool flagHidden, int numOutputs, int id, double eta, double alpha);

  double outputValue;           // the output value of this neuron
  bool flagBias;                // flag for bias neuron
  bool flagInput;               // flag for input neuron
  bool flagHidden;              // flag for hidden neuron
  vector<Weight> outputWeights; // output weights

  static double randomWeight(); // random weight generator

  double gradient;

  double eta;  // [0,1]
  double alpha;
private:
  int id;                       // the id of this neuron
};

#endif
