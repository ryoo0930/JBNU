#include <stdio.h>
#include <stdlib.h>

#include "test.h"
#include "sigmoid.h"

void test(Model* model) {
    double* input_layer   = (double*)calloc(INPUT_SIZE  + 1, sizeof(double));
    double* hidden_layer  = (double*)calloc(HIDDEN_SIZE + 1, sizeof(double));
    double* output_layer  = (double*)calloc(OUTPUT_SIZE + 1, sizeof(double));
    input_layer[0]  = 1.0;
    hidden_layer[0] = 1.0;

    FILE* fp = fopen("data/test.txt", "r");
    if (fp == NULL) { printf("test.txt 파일이 존재하지 않습니다.\n"); return; }

    int total = 0, correct = 0;
    int target_number;

    while (fscanf(fp, "%d", &target_number) != EOF) {
        for (int i = 1; i <= INPUT_SIZE; i++)
            fscanf(fp, "%lf", &input_layer[i]);

        /* FORWARD */
        for (int j = 1; j <= HIDDEN_SIZE; j++) {
            double sum = model->w_input_to_hidden[0][j];
            for (int i = 1; i <= INPUT_SIZE; i++)
                sum += input_layer[i] * model->w_input_to_hidden[i][j];
            hidden_layer[j] = sigmoid(sum);
        }

        for (int k = 1; k <= OUTPUT_SIZE; k++) {
            double sum = model->w_hidden_to_output[0][k];
            for (int j = 1; j <= HIDDEN_SIZE; j++)
                sum += hidden_layer[j] * model->w_hidden_to_output[j][k];
            output_layer[k] = sigmoid(sum);
        }

        /* 예측 */
        int predicted = 0;
        double max_val = output_layer[1];
        for (int k = 2; k <= OUTPUT_SIZE; k++) {
            if (output_layer[k] > max_val) { max_val = output_layer[k]; predicted = k - 1; }
        }

        if (predicted == target_number) correct++;
        total++;
    }

    fclose(fp);

    printf("Test Precision: %.4f (%d / %d)\n", (double)correct / total, correct, total);

    free(input_layer);
    free(hidden_layer);
    free(output_layer);
}
