#ifndef TRAINING_H
#define TRAINING_H

#include "config.h"

typedef struct {
    double** w_input_to_hidden;
    double** w_hidden_to_output;
} Model;

Model*  training(double target_precision_ratio);
void free_model(Model* model);

#endif
