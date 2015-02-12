#include "neuron.hpp"

// Constructor
Neuron::Neuron(bool flagBias, bool flagInput, bool flagHidden, int numOutputs, int id, double eta, double alpha)
{
  setFlagBias(flagBias);
  setFlagInput(flagInput);
  setFlagHidden(flagHidden);

  for(int i = 0; i < numOutputs; i++) {
    this->outputWeights.push_back(Weight(Neuron::randomWeight(), 0));
  }

  this->id = id;

  // instantiate outputValue to 0
  if(flagBias) {
    setOutputValue(1);
  } else {
    setOutputValue(0);
  }

  this->eta = eta;
  this->alpha = alpha;
}

// Random weight generator
double Neuron::randomWeight()
{
  double min = -1.0;
  double max = 1.0;

  static bool first = true;
  if(first)
  {
    srand(time(NULL));
    first = false;
  }

  double f = (double)rand() / RAND_MAX;

  return min + f * (max - min);
}

// Transfer Function
// tanh - output range [-1, 1]
// NOTE: Make sure to scale values that are acceptible by tanh
double Neuron::transferFunction(double x)
{
  //return tanh(x);
  return 1 / (1 + exp(-x));
}

// Drivative of transfer function
double Neuron::transferFunctionDerivative(double x)
{
  //return 1.0 - x * x;
  return x * (1 - x);
}

// Feed forward operation for a neuron
void Neuron::feedForward(vector<double> inputs, vector<double> weights)
{
  // inputs.size == outputWeights.size
  assert(inputs.size() == weights.size());

  double summ = 0;
  for(int i = 0; i < inputs.size(); i++) {
    summ += inputs.at(i) * weights.at(i);
  }

  this->outputValue = Neuron::transferFunction(summ);
}

// Calculate output gradients
void Neuron::calculateOutputGradients(double targetVal)
{
  double delta = targetVal - outputValue;
  gradient = delta * Neuron::transferFunctionDerivative(outputValue);
}

void Neuron::calculateHiddenGradients(vector<Neuron> nextLayerNeurons)
{
  double dow = sumDOW(nextLayerNeurons);
  gradient = dow * Neuron::transferFunctionDerivative(outputValue);
}

double Neuron::sumDOW(vector<Neuron> nextLayerNeurons)
{
  double sum = 0.0;

  for(int n = 0; n < nextLayerNeurons.size() - 1; n++) {
    sum += outputWeights[n].value * nextLayerNeurons[n].gradient;
  }

  return sum;
}

void Neuron::updateInputWeights(vector<Neuron> &previousLayerNeurons) {
  // Weights are updated in neurons in the preceeding layer including bias
  for(int i = 0; i < previousLayerNeurons.size() - 1; i++) {
    Neuron &n = previousLayerNeurons[i];
    double oldDeltaWeight = n.outputWeights[id].deltaWeight;
    double newDeltaWeight = 
            // Individual input, magnified by gradient and train rate
            eta
            * n.getOutputValue()
            * gradient
            // Momentum
            + alpha
            * oldDeltaWeight;

    n.outputWeights[id].deltaWeight = newDeltaWeight;
    n.outputWeights[id].value += newDeltaWeight;
  }
}
