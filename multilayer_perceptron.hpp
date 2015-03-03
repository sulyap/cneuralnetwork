#ifndef MULTILAYER_PERCEPTRON_HPP_
#define MULTILAYER_PERCEPTRON_HPP_

#include <iostream>
#include <vector>
#include <cstdlib>
#include <cassert>
#include <cmath>

using namespace std;

#include "layer.hpp"
#include "training_data.hpp"

class MultilayerPerceptron
{
public:
  // constructor
  MultilayerPerceptron(vector<int> topology, double eta, double alpha);

  // Left to right transfer of information
  void feedForward(vector<double> inputVals);

  // Learning method
  void backPropagation(vector<double> targetVals);

  vector<double> getResults();

  // setters
  void setTopology(vector<int> topology) { this->topology = topology; }
  void setLayers(vector<Layer> layers) { this->layers = layers; }
  void setTrainingData(TrainingData trainingData) { this->trainingData = trainingData; }

  // getters
  vector<int> getTopology() { return topology; }
  vector<Layer> getLayers() { return layers; }
  TrainingData getTrainingData() { return trainingData; }

  // utility methods
  void printOutputLayer();

  double overallNetError;
  double recentAverageError;
  double recentAverageSmoothingFactor;

  // method for training. requires an instance of TrainingData to be passed
  void train(TrainingData trainingData, int epoch);

  // save to configuration to file
  /* Format:
  minVal=min
  maxVal=max
  topology={n1,weight1,0},{n2,weight,0}|{n1,weight1,0},{n2,weight2,0}
  */
  void save(string filename);

  double predict(vector<double> inputVals);

private:
  vector<int> topology;       // topology of this network
  vector<Layer> layers;       // collection of layers
  TrainingData trainingData;  // training data object
  int epoch;                  // epoch value for training
  double minVal;              // min val for normalization
  double maxVal;              // max val for normalization
};

#endif
