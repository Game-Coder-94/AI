// layers.h
#ifndef LAYERS_H
#define LAYERS_H

typedef struct{
   int numInputs, numNeurons;
   double *weights;
   double *biases;
   double *outputs;
} Layer;

typedef struct{
    int numLayers;
    Layer *layers;
} NeuralNetwork;

#endif // LAYERS.H