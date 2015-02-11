#ifndef TRAINING_DATA_HPP_
#define TRAINING_DATA_HPP_

#include <iostream>
#include <vector>
#include <cstdlib>
#include <cassert>
#include <cmath>
#include <ctime>
#include <random>
#include <fstream>
#include <sstream>

using namespace std;

class TrainingData
{
public:
  vector<vector<double> > trainingData;
  vector<double> labels;
  int dimensionCount;

  TrainingData();
  TrainingData(const char *filename);
};

#endif
