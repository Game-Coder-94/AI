// defineDataset.h
#ifndef DEFINEDATASET_H
#define DEFINEDATASET_H 

#define NUM_SAMPLES 4
#define INPUT_SIZE 2
#define OUTPUT_SIZE 1

// XOR operation dataset

double inputs[NUM_SAMPLES][INPUT_SIZE] = {
    {0.0, 0.0},
    {0.0, 1.0},
    {1.0, 0.0},
    {1.0, 1.0}
};

double targets[NUM_SAMPLES][OUTPUT_SIZE] = {
    {0.0},
    {1.0},
    {1.0},
    {0.0}
};

#endif // defineDataset.h