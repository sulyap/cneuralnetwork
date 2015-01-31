#ifndef WEIGHT_HPP_
#define WEIGHT_HPP_

#include <iostream>
#include <vector>
#include <cstdlib>
#include <cassert>
#include <cmath>
#include <ctime>
#include <random>

using namespace std;

class Weight
{
public:
  double value;
  double deltaWeight;
  Weight(double value, double deltaWeight);
};

#endif
