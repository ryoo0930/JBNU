#include <stdio.h>
#include <stdlib.h>

#include "training.h"
#include "sigmoid.h"

static int** make_target_vector();
static double*** load_samples(FILE* fp, int counts[OUTPUT_SIZE]);

void free_model(Model* model);

Model* training(double target_precision_ratio) {
    /* 작업에 필요한 배열들 생성 */
    double* input_layer   = (double*)calloc(INPUT_SIZE  + 1, sizeof(double));
    double* hidden_layer  = (double*)calloc(HIDDEN_SIZE + 1, sizeof(double));
    double* output_layer  = (double*)calloc(OUTPUT_SIZE + 1, sizeof(double));
    int**   target_vector = make_target_vector();
    input_layer[0]  = 1.0;
    hidden_layer[0] = 1.0;

    double** w_input_to_hidden = (double**)malloc(sizeof(double*) * (INPUT_SIZE + 1));
    for (int i = 0; i <= INPUT_SIZE; i++) {
        w_input_to_hidden[i] = (double*)malloc(sizeof(double) * (HIDDEN_SIZE + 1));
        for (int j = 0; j <= HIDDEN_SIZE; j++) {
            w_input_to_hidden[i][j] = ((double)rand() / RAND_MAX) * 0.2 - 0.1;
        }
    }

    double** w_hidden_to_output = (double**)malloc(sizeof(double*) * (HIDDEN_SIZE + 1));
    for (int j = 0; j <= HIDDEN_SIZE; j++) {
        w_hidden_to_output[j] = (double*)malloc(sizeof(double) * (OUTPUT_SIZE + 1));
        for (int k = 0; k <= OUTPUT_SIZE; k++) {
            w_hidden_to_output[j][k] = ((double)rand() / RAND_MAX) * 0.2 - 0.1;
        }
    }

    FILE* fp = fopen("data/train.txt", "r");
    if (fp == NULL) { printf("train.txt 파일이 존재하지 않습니다.\n"); return NULL; }

    int counts[OUTPUT_SIZE];
    double*** samples = load_samples(fp, counts);
    fclose(fp);

    int max_count = 0;
    for (int d = 0; d < OUTPUT_SIZE; d++) {
        if (counts[d] > max_count) max_count = counts[d];
    }

    double current_precision_ratio = 0.0;
    while (current_precision_ratio < target_precision_ratio) {
        int total = 0, correct = 0;

        for (int round = 0; round < max_count; round++) {
            for (int d = 0; d < OUTPUT_SIZE; d++) {
                if (round >= counts[d]) continue;

                /* FORWARD */
                for (int i = 1; i <= INPUT_SIZE; i++) {
                    input_layer[i] = samples[d][round][i - 1];
                }
                
                for (int j = 1; j <= HIDDEN_SIZE; j++) {
                    double sum = w_input_to_hidden[0][j];
                    for (int i = 1; i <= INPUT_SIZE; i++)
                        sum += input_layer[i] * w_input_to_hidden[i][j];
                    hidden_layer[j] = sigmoid(sum);
                }

                for (int k = 1; k <= OUTPUT_SIZE; k++) {
                    double sum = w_hidden_to_output[0][k];
                    for (int j = 1; j <= HIDDEN_SIZE; j++)
                        sum += hidden_layer[j] * w_hidden_to_output[j][k];
                    output_layer[k] = sigmoid(sum);
                }

                /* BACKWARD */
                double* delta_output = (double*)calloc(OUTPUT_SIZE + 1, sizeof(double));
                double* delta_hidden = (double*)calloc(HIDDEN_SIZE + 1, sizeof(double));

                double** dw = (double**)calloc(HIDDEN_SIZE + 1, sizeof(double*));
                for (int j = 0; j <= HIDDEN_SIZE; j++) {
                    dw[j] = (double*)calloc(OUTPUT_SIZE + 1, sizeof(double));
                }

                double** dv = (double**)calloc(INPUT_SIZE + 1, sizeof(double*));
                for (int i = 0; i <= INPUT_SIZE; i++) {
                    dv[i] = (double*)calloc(HIDDEN_SIZE + 1, sizeof(double));
                }

                for (int k = 1; k <= OUTPUT_SIZE; k++) {
                    double y = output_layer[k];
                    double t = (double)target_vector[d][k - 1];
                    delta_output[k] = (t - y) * y * (1.0 - y);
                    for (int j = 0; j <= HIDDEN_SIZE; j++)
                        dw[j][k] = ALPHA * delta_output[k] * hidden_layer[j];
                }

                for (int j = 1; j <= HIDDEN_SIZE; j++) {
                    double delta_in = 0.0;
                    for (int k = 1; k <= OUTPUT_SIZE; k++) {
                        delta_in += delta_output[k] * w_hidden_to_output[j][k];
                    }
                    double z = hidden_layer[j];
                    delta_hidden[j] = delta_in * z * (1.0 - z);
                    for (int i = 0; i <= INPUT_SIZE; i++) {
                        dv[i][j] = ALPHA * delta_hidden[j] * input_layer[i];                
                    }
                }

                for (int k = 1; k <= OUTPUT_SIZE; k++) {
                    for (int j = 0; j <= HIDDEN_SIZE; j++) {
                        w_hidden_to_output[j][k] += dw[j][k];
                    }
                }
                for (int j = 1; j <= HIDDEN_SIZE; j++) {
                    for (int i = 0; i <= INPUT_SIZE; i++) {
                        w_input_to_hidden[i][j] += dv[i][j];
                    }
                }

                /* 메모리 할당 해제 */
                for (int j = 0; j <= HIDDEN_SIZE; j++) free(dw[j]);
                free(dw);
                for (int i = 0; i <= INPUT_SIZE; i++) free(dv[i]);
                free(dv);
                free(delta_output);
                free(delta_hidden);

                /* 정확도 집계 */
                int predicted = 0;
                double max_val = output_layer[1];
                for (int k = 2; k <= OUTPUT_SIZE; k++) {
                    if (output_layer[k] > max_val) { max_val = output_layer[k]; predicted = k - 1; }
                }
                if (predicted == d) correct++;
                total++;
            }
        }

        current_precision_ratio = (total > 0) ? (double)correct / total : 0.0;
        printf("Precision: %.4f\n", current_precision_ratio);
    }

    /* 메모리 할당 해제 */
    for (int d = 0; d < OUTPUT_SIZE; d++) {
        for (int idx = 0; idx < counts[d]; idx++)
            free(samples[d][idx]);
        free(samples[d]);
    }
    free(samples);

    for (int i = 0; i < OUTPUT_SIZE; i++) free(target_vector[i]);
    free(target_vector);
    free(input_layer);
    free(hidden_layer);
    free(output_layer);

    /* 학습 정보 넘겨주기 */
    Model* model = (Model*)malloc(sizeof(Model));
    model->w_input_to_hidden  = w_input_to_hidden;
    model->w_hidden_to_output = w_hidden_to_output;
    return model;
}

static double*** load_samples(FILE* fp, int counts[OUTPUT_SIZE]) {
    for (int d = 0; d < OUTPUT_SIZE; d++) counts[d] = 0;

    int label;
    double dummy;
    /* 샘플 갯수 파악 */
    while (fscanf(fp, "%d", &label) != EOF) {
        counts[label]++;
        for (int i = 0; i < INPUT_SIZE; i++)
            fscanf(fp, "%lf", &dummy);
    }

    /* samples[OUTPUT_SIZE][SAMPLE_SIZE][INPUT_SIZE] */
    double*** samples = (double***)malloc(sizeof(double**) * OUTPUT_SIZE);
    for (int d = 0; d < OUTPUT_SIZE; d++) {
        samples[d] = (double**)malloc(sizeof(double*) * counts[d]);
        for (int idx = 0; idx < counts[d]; idx++)
            samples[d][idx] = (double*)malloc(sizeof(double) * INPUT_SIZE);
    }

    /* 실제 데이터 넣기 */
    rewind(fp);
    int cursor[OUTPUT_SIZE] = {0};
    while (fscanf(fp, "%d", &label) != EOF) {
        int idx = cursor[label]++;
        for (int i = 0; i < INPUT_SIZE; i++)
            fscanf(fp, "%lf", &samples[label][idx][i]);
    }

    return samples;
}

static int** make_target_vector() {
    int** tv = (int**)malloc(sizeof(int*) * OUTPUT_SIZE);
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        tv[i] = (int*)calloc(OUTPUT_SIZE, sizeof(int));
        tv[i][i] = 1;
    }
    return tv;
}

void free_model(Model* model) {
    for (int i = 0; i <= INPUT_SIZE;  i++) free(model->w_input_to_hidden[i]);
    free(model->w_input_to_hidden);
    for (int j = 0; j <= HIDDEN_SIZE; j++) free(model->w_hidden_to_output[j]);
    free(model->w_hidden_to_output);
    free(model);
}