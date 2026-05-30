#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "config.h"
#include "training.h"
#include "test.h"



int main() {
    srand(time(NULL));

    printf("BackPropagation Network\n");
    Model* model = NULL;
    
    while(1) {
        printf("1 : Training\n");
        printf("2 : Testing\n");
        printf("3 : Quit\n");

        int menu = 0;
        printf("Enter Your Choice (1-3) >> ");
        scanf("%d", &menu);
        switch(menu){
            case 1 :
                double target_precision_ratio = 0;
                printf("Attainment Target >> ");
                scanf("%lf", &target_precision_ratio); target_precision_ratio /= 100.0;

                model = training(target_precision_ratio);
                break;
            case 2 :
                if(model == NULL) { printf("There are no trained models.\n"); break; }
                test(model);
                break;
            case 3 :
                if(model != NULL) free_model(model);
                return 0;
        }
    }
    
    return 0;
}
