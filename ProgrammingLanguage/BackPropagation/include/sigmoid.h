#ifndef SIGMOID_H
#define SIGMOID_H

#include <math.h>

static inline double sigmoid(double x) { return 1.0 / (1.0 + exp(-x)); }
static inline double sigmoid_derivative(double x) { return sigmoid(x) * (1.0 - sigmoid(x)); }

#endif