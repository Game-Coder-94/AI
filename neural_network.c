#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "layers.h"

#define NUM_SAMPLES 4
#define INPUT_SIZE 2
#define OUTPUT_SIZE 1

// Random weight initilizer
double randWeight(){
    return ((double) rand() / RAND_MAX) * 2 - 1;    // Returns value between -1 to 1
}

// Function to initilize Hidden layer
void initializeLayer(Layer *layer, int numInputs, int numNeurons){
    layer -> numInputs = numInputs;
    layer -> numNeurons = numNeurons;
    layer -> weights = malloc(numInputs * numNeurons * sizeof(double));
    layer -> biases = malloc(numNeurons * sizeof(double));
    layer -> outputs = malloc(numNeurons * sizeof(double));
    layer -> deltas = malloc(numNeurons * sizeof(double));
    
    // Add Randomize weight
    for (int i = 0; i < numInputs * numNeurons; i++){
        layer -> weights[i] = randWeight();
    }

    // Add randomize bias
    for (int i = 0; i < numNeurons; i++){
        layer -> biases[i] = randWeight();
    }

    // Initilize deltas with zero
    for (int i = 0; i < numNeurons; i++){
        layer -> deltas[i] = 0;
    }
}

void initilizeNetwork(NeuralNetwork *nn, int *layerSizes, int numLayers){
    nn -> numLayers = numLayers;
    nn -> layers = malloc(numLayers * sizeof(Layer));

    // Didn't understand what this loop is doing
    for (int i = 0; i < numLayers; i++){
        int numInputs = (i == 0) ? layerSizes[i] : layerSizes[i - 1];
        initializeLayer(&nn -> layers[i], numInputs, layerSizes[i]);
    }
}

/*
    <----------------------------------------->
                Forward Propagation
    <----------------------------------------->
*/

// Activation function
double sigmoid(double x){
    return 1.0 / (1.0 + exp(-x));
}

void forwardLayer(Layer *layer, double *inputs){
    for (int i = 0; i < layer -> numNeurons; i++){
        // Adding bias to sum (raw output)
        double sum = layer -> biases[i];
        for (int j = 0; j < layer -> numInputs; j++){
            sum += inputs[j] * layer -> weights[j + i * layer -> numInputs];    // Didn't understand how weight is selected
        }
        layer -> outputs[i] = sigmoid(sum); //Apply activation function to normalize in range(-1, 1)
    }
}

void forwardNetwork(NeuralNetwork *nn, double *inputs){
    double *currentInputs = inputs;
    for (int i = 0; i < nn -> numLayers; i++){
        forwardLayer(&nn -> layers[i], currentInputs);
        currentInputs = nn -> layers[i].outputs;
    }
}

// Load dataset function
void loadDataset(const char *filename, double inputs[][INPUT_SIZE], double targets[][OUTPUT_SIZE]){
    FILE *fp = fopen(filename, "r");
    if(fp == NULL){
        perror("Dataset can't be opened!");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < NUM_SAMPLES; i++){
        for (int j = 0; j < INPUT_SIZE; j++){
            fscanf(fp, "%lf", &inputs[i][j]);
        }
        for (int k = 0; k < OUTPUT_SIZE; k++){
            fscanf(fp, "%lf", &targets[i][k]);
        }
    }
    fclose(fp);
    // Add this function in main function
}

// Compute Error
double computeError(double *predicted, double *actual, int size){
    double error = 0.0;
    for (int i = 0; i < size; i++){
        double diff = predicted[i] - actual[i];
        error += diff * diff;   // Squared error
    }
    return error / size;    // Mean of squared error
}

/*
        <-------------------------------------->

                    Back Propagation

        <-------------------------------------->
*/

// Calculate Gradient for Output nodes
void computeOutputLayerGradients(Layer *outputLayer, double *actual){
    for (int i = 0; i < outputLayer -> numNeurons; i++){
        double error = outputLayer -> outputs[i] - actual[i];
        double derivative = outputLayer -> outputs[i] * (1 - outputLayer -> outputs[i]);
        outputLayer -> deltas[i] = error * derivative;
    }
}

// Calculate Gradient for hidden layers
void computeHiddenLayerGradients(Layer *currentLayer, Layer *nextLayer){
    for (int i = 0; i < currentLayer -> numNeurons; i++){
        double sum = 0.0;
        for (int j = 0; j < nextLayer -> numNeurons; j++){
            sum += nextLayer -> weights[i + j * currentLayer -> numNeurons] * nextLayer -> deltas[j];
        }
        double derivate = currentLayer -> outputs[i] * (1 - currentLayer -> outputs[i]);
        currentLayer -> deltas[i] = sum * derivate;
    }
}

// Main Function


int main(){
    srand(time(NULL));  // Set random seed generator at NULL

    // Define network structure (input = 2, hidden neurons = 2, output = 1)
    int layersizes[] = {2, 2, 1};
    NeuralNetwork nn;
    initilizeNetwork(&nn, layersizes, 3);

    // Input (Format)
    double inputs[] = {0.0, 1.0};

    // Passing inputs
    forwardNetwork(&nn, inputs);

    // Output for result
    printf("Output: %f\n", *nn.layers[nn.numLayers - 1].outputs);

    // Free memory
    for (int i = 0; i < nn.numLayers; i++){
        free(nn.layers[i].weights);
        free(nn.layers[i].biases);
        free(nn.layers[i].outputs);
    }
    free(nn.layers);

    return 0;
}