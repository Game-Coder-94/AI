#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "layers.h"
#include "defineDataset.h"

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

void storeWeightsAndBiases(const char *filename, NeuralNetwork *nn){
    FILE *fp = fopen(filename, "w");
    if(fp == NULL){
        perror("File can't able to open");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < nn -> numLayers; i++){
        Layer *layer = &nn -> layers[i];

        // Store weights
        for (int j = 0; j < layer -> numNeurons; j++){
            for (int k = 0; k < layer -> numInputs; k++){
                fprintf(fp, "%lf  ", layer -> weights[k + j * layer -> numInputs]);
            }
            fprintf(fp, "\n");
        }

        fprintf(fp, "\n");

        // Store biases
        for (int j = 0; j < layer -> numNeurons; j++){
            fprintf(fp, "%lf\n", layer -> biases[j]);
        }
        
        fprintf(fp, "\n\n");
    }
    fclose(fp);
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

// Udate weights and biases
void updateWeightsAndBiases(Layer *layer, double *inputs, double learningRate){
    for (int i = 0; i < layer -> numNeurons; i++){
        // Update Biases
        layer -> biases[i] -= learningRate * layer -> deltas[i];

        // Update weights
        for (int j = 0; j < layer -> numInputs; j++){
            // Function understood
            layer -> weights[j + i * layer -> numNeurons] -= learningRate * layer -> deltas[i] * inputs[j]; // But this statement didn't understand
        }
    }
}

// Train Function
void train(NeuralNetwork *nn, double inputs[NUM_SAMPLES][INPUT_SIZE], double targets[NUM_SAMPLES][OUTPUT_SIZE], int numSamples, int numEpochs, double learningRate){
    for (int epoch = 0; epoch < numEpochs; epoch++){
        double totalError = 0.0;

        for (int sample = 0; sample < numSamples; sample++){
            // Forward Propagation (Calculation of final output)
            forwardNetwork(nn, inputs[sample]);

            // Calculate Error (Cost)
            Layer *outputLayer = &nn -> layers[nn -> numLayers - 1];
            totalError += computeError(outputLayer -> outputs, targets[sample], outputLayer -> numNeurons);

            // Backpropation
            computeOutputLayerGradients(outputLayer, targets[sample]);
            for (int i = nn -> numLayers - 2; i >= 0; i--){
                computeHiddenLayerGradients(&nn -> layers[i], &nn -> layers[i + 1]);
            }

            // Update Weights and Biases
            double *currentInputs = inputs[sample];
            for (int i = 0; i < nn -> numLayers; i++){
                updateWeightsAndBiases(&nn -> layers[i], currentInputs, learningRate);
                currentInputs = nn -> layers[i].outputs; 
            }
        }

        // Print error for epoch
        printf("Epoch = %d, Error = %f\n", epoch, totalError / numSamples);
    }
}





// Main Function


int main(){
    srand(time(NULL));  // Set random seed generator at NULL

    // Define network structure (input = 2, hidden neurons = 2, output = 1)
    int layersizes[] = {2, 2, 1};
    NeuralNetwork nn;
    initilizeNetwork(&nn, layersizes, 3);

    /*
    // Input (Format)
    double inputs[] = {0.0, 1.0};

    // Passing inputs
    forwardNetwork(&nn, inputs);

    // Output for result
    printf("Output: %f\n", *nn.layers[nn.numLayers - 1].outputs);
    */
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

    // Train a network
    train(&nn, inputs, targets, 4, 10000, 0.5);

    // Test the network
    for (int i = 0; i < 4; i++){
        forwardNetwork(&nn, inputs[i]);
        Layer *outputLayer = &nn.layers[nn.numLayers - 1];
        printf("Input => %0.1f, %0.1f Output => %0.5f\n", inputs[i][0], inputs[i][1], outputLayer -> outputs[0]);
    }

    // Store Weight and Biases
    storeWeightsAndBiases("weightsandbiases.txt", &nn);
    

    // Free memory
    for (int i = 0; i < nn.numLayers; i++){
        free(nn.layers[i].weights);
        free(nn.layers[i].biases);
        free(nn.layers[i].outputs);
    }
    free(nn.layers);

    return 0;
}