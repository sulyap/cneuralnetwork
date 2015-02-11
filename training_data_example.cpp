#include <iostream>
#include <vector>
#include <cstdlib>
#include <cassert>
#include <cmath>
#include <ctime>
#include <random>

using namespace std;

#include "training_data.hpp"

using namespace std;

void syntax()
{
  cout << "training_data [csv_file]" << endl;
}

int main(int argc, char **argv)
{
  if(argc != 2) {
    syntax();
    exit(-1);
  }

  TrainingData *td = new TrainingData(argv[1]);

  return 0;
}
