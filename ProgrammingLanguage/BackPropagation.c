#include <stdio.h>
#include "SIGMOID.h"

#define INPUT_SIZE 100
#define OUTPUT_SIZE 10
#define HIDDEN_SIZE (OUTPUT_SIZE * 2)

void make_target_vector(int** target_vector);
void training(double target_precision_ratio);

int main() {
    double target_precison_ratio = 0;
    scanf("%lf", &target_precison_ratio);

    training(target_precison_ratio);
    

    
}

void make_target_vector(int** target_vector) {
    target_vector = (int**)malloc(sizeof(int*) * OUTPUT_SIZE);
    for(int i = 0; i < 10; i++) {
        target_vector[i] = (int*)calloc(OUTPUT_SIZE, sizeof(int));
        target_vector[i][i] = 1;
    }
}

void training(double target_precision_ratio) {
    double* input_layer = (double*)calloc(INPUT_SIZE + 1, sizeof(double));
    double* hidden_layer = (double*)calloc(HIDDEN_SIZE + 1, sizeof(double)); 
    double* output_layer = (double*)calloc(OUTPUT_SIZE + 1, sizeof(double));
    int** target_vector; make_target_vector(target_vector);
    input_layer[0] = 1.0; // Bias
    hidden_layer[0] = 1.0; // Bias

    double** w_input_to_hidden = (double**)malloc(sizeof(double*) * HIDDEN_SIZE + 1);
    for(int i = 0; i <= HIDDEN_SIZE; i++) { w_input_to_hidden[i] = (double*)malloc(sizeof(double) * INPUT_SIZE + 1); }
    double** w_hidden_to_output = (double**)malloc(sizeof(double*) * OUTPUT_SIZE + 1);
    for(int i = 0; i <= OUTPUT_SIZE; i++) { w_hidden_to_output[i] = (double*)malloc(sizeof(double) * HIDDEN_SIZE + 1); }

    int target_number = 0;
    double current_precision_ratio = 0;
    
    FILE* fp;
    fp = fopen("test.txt", "r");
    if(fp == NULL) { printf("test.txt 파일이 존재하지 않습니다."); return; }

    while(current_precision_ratio >= target_precision_ratio) {
        rewind(fp);

        while(fscanf(fp, "%d", &target_number) != EOF) {
            /* FORWARD */
            for(int i = 1; i <= INPUT_SIZE; i++) { fscanf(fp, "%lf", &input_layer[i]); }

            for(int j = 1; j <= HIDDEN_SIZE; j++) {
                double sum = w_input_to_hidden[0][j];
                for(int i = 1; i <= INPUT_SIZE; i++) { sum += input_layer[i] * w_input_to_hidden[i][j]; }
                hidden_layer[j] = sigmoid(sum);
            }

            for(int k = 1; k <= OUTPUT_SIZE; k++) {
                double sum = w_hidden_to_output[0][k];
                for(int j = 1; j <= HIDDEN_SIZE; j++) { sum += hidden_layer[j] * w_hidden_to_output[j][k]; }
                output_layer[k] = sigmoid(sum);
            }

            /* BACKWARD */
            // PPT에서는 Delta값 계산하고 마지막에 업데이트하는데 Delta값 계산하고 바로 update해도 될 듯? 
            for(int i = 1; i <= OUTPUT_SIZE; i++) {
            }




        }
    }
    





    


    fclose(fp);
}
