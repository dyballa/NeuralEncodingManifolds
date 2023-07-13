# NeuralEncodingManifolds
Here you will find the code for the analysis of neural populations in response to an ensemble of stimuli. For more information, please refer to the paper:

> Dyballa, L. et al. (2023), "Population encoding of stimulus features along the visual hierarchy", Preprint available on bioRxiv: https://www.biorxiv.org/content/10.1101/2023.06.27.545450v2

## Contents

Code files are organized into folders:

[`creating-the-tensor`](/creating-the-tensor) -  Creating the tensor of response maps from spike files (includes kernel smoothing, displaying response maps for multiple stimuli, building the tensor).

[`permuted-decomposition`](/permuted-decomposition) - Running the permuted factorization (MATLAB files), selection of the number of tensor components to use, and plotting resulting factors.

[`encoding-manifold`](/encoding-manifold) - Inferring the encoding manifold (building the neural factor matrix, handling of non-significant responses, inferring the data graph and underlying manifold, dimensionality).

[`CNNs`](/CNNs) - Convolutional neural networks (activity across layer, stimulus classification, and sampling procedure).


## Installation

Simply copy the functions used througout the examples in each folder. You will still need to install the following dependencies to run all the necessary examples:
- Python >= 3.9
- NumPy >= 1.21.2
- SciPy >= 1.7.3
- scikit-learn >= 1.1.2
- Matplotlib >= 3.4.2
- tensor_toolbox >= v3.1 (MATLAB, for permuted decomposition)
- IAN >= 1.0.2 (https://github.com/dyballa/IAN)
- tensorflow >= 2.10 (for running the CNN example)


## Documentation

You will find detailed usage examples in the python notebooks present in each folder. Feel free to contact me by email if you have any questions.