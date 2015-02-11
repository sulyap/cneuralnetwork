#include "training_data.hpp"

// Default constructor
TrainingData::TrainingData() : trainingData(0), labels(0), dimensionCount(0)
{
}

TrainingData::TrainingData(const char *filename)
{
  ifstream file(filename);
  string line;

  vector<vector<double> > buffData;
  while(getline(file, line)) {
    //cout << line << endl;
    vector<double> dataLine;

    std::istringstream iss(line);
    std::string result;
    while(std::getline(iss, result, ',')) {
      dataLine.push_back(stod(result));
    }

    buffData.push_back(dataLine);
  }

  // load training data
  for(int i = 0; i < buffData.size(); i++) {
    vector<double> training_data_vector;
    for(int j = 0; j < buffData.at(i).size() - 1; j++) {
      training_data_vector.push_back(buffData.at(i).at(j));
    }
    
    trainingData.push_back(training_data_vector);
    labels.push_back(buffData.at(i).at(buffData.at(i).size() - 1));
  }
}
