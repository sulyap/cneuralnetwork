#include "multilayer_perceptron.hpp"

/*
  CONSTRUCTOR
*/
MultilayerPerceptron::MultilayerPerceptron(vector<int> topology)
{
  setTopology(topology);

  int num_layers = topology.size();


  for(int i = 0; i < num_layers; i++) {
    this->layers.push_back(Layer()); 
    int num_outputs = i == topology.size() - 1 ? 0 : topology.at(i + 1);

    // inner loop
    // plus bias for <=
    for(int neuron_num = 0; neuron_num <= topology.at(i); neuron_num++) {
      // Check if bias or input or hidden
      bool isBias = false;
      bool isInput = false;
      bool isHidden = false;

      if(i == 0) {
        isInput = true;
      } 
      
      if(neuron_num == topology.at(i)) {
        isBias = true;
      }

      if(i != (topology.size() - 1) && i != 0) {
        isHidden = true;
      }

      this->layers.back().addNeuron(Neuron(isBias, isInput, isHidden, num_outputs, neuron_num));
    }
  }
}

/*
  FEED FORWARD
*/
void MultilayerPerceptron::feedForward(vector<double> inputVals)
{
  /*
   * Assertion to check for consistency in data input
   * - size of neurons at input layer - 1 assuming bias neuron is present
   */
  assert(inputVals.size() == (this->layers.at(0).getNeurons().size() - 1));

  // Latch inputs to neurons in input layer
  for(int i = 0; i < inputVals.size(); i++) {
    this->layers[0].neurons[i].outputValue = inputVals.at(i);
  }
 
  // bias neuron value
  this->layers[0].neurons[inputVals.size()].outputValue = 1;

  // Feed Forward for succeeding layers
  for(int i = 1; i < this->layers.size(); i++) {
    Layer previousLayer = this->layers.at(i - 1);

    // minus the bias
    for(int j = 0; j < this->layers.at(i).getNeurons().size() - 1; j++) {

      // Build the vector of inputs from previous layer outputs
      vector<double> inputs = previousLayer.getInputs();
      vector<double> weights = previousLayer.getWeights(j);

      // Update the outputValue of this neuron j
      this->layers[i].neurons[j].feedForward(inputs, weights);
    }
  }
}

/*
  PRINT OUTPUT LAYER
  Utility method to print out the result/s of the output layer
 */
void MultilayerPerceptron::printOutputLayer()
{
  for(int i = 0; i < this->topology.back(); i++) {
    Neuron outputNeuron = this->layers.back().getNeurons().at(i);
    cout << "OUTPUT NEURON " << i << ": " << outputNeuron.getOutputValue() << endl;
  }
}

/*
  BACK PROPAGATION
*/
void MultilayerPerceptron::backPropagation(vector<double> targetVals)
{
  // Calculate overall net error (root mean square of output neuron errors)
  // Each neuron has an error term
  Layer &outputLayer = layers.back();
  overallNetError = 0.0;
  for(int n = 0; n < outputLayer.getNeurons().size() - 1; n++) {
    double delta = targetVals[n] - outputLayer.getNeurons().at(n).getOutputValue();
    overallNetError += delta * delta;
  }

  // Average error squared
  overallNetError /= outputLayer.getNeurons().size() - 1;

  // Root mean square
  overallNetError = sqrt(overallNetError);

  // Recent average measurement
  recentAverageError = 
    (recentAverageError * recentAverageSmoothingFactor + overallNetError)
    / (recentAverageSmoothingFactor + 1.0);

  // Calculate output layer gradients
  for(int n = 0; n < outputLayer.getNeurons().size() - 1; n++) {
    outputLayer.neurons[n].calculateOutputGradients(targetVals[n]);
  }

  // Calculate gradients on hidden layer
  for(int layer_index = layers.size() - 2; layer_index > 0; layer_index--) {
    Layer &hiddenLayer = layers[layer_index];
    Layer &nextLayer = layers[layer_index + 1];

    for(int n = 0; n < hiddenLayer.getNeurons().size(); n++) {
      hiddenLayer.neurons[n].calculateHiddenGradients(nextLayer.neurons);
    }
  }

  // For all layers from output to first hidden layer, update connection weights
  // Update connection weights
  for(int layer_index = layers.size() - 1; layer_index > 0; layer_index--) {
    Layer &layer = layers[layer_index];
    Layer &previousLayer = layers[layer_index - 1];
    vector<Neuron> &previousLayerNeurons = previousLayer.neurons;

    // minus bias
    for(int n = 0; n < layer.neurons.size() - 1; n++) {
      layer.neurons[n].updateInputWeights(previousLayerNeurons);
    }
  }
}

/*
  GET RESULTS
*/
vector<double> MultilayerPerceptron::getResults()
{
  vector<double> resultVals;

  for(int i = 0; i < layers.back().neurons.size() - 1; i++) {
    resultVals.push_back(layers.back().neurons[i].getOutputValue());
  }

  return resultVals;
}
