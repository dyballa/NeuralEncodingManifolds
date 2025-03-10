# NeuralEncodingManifolds
Here you will find code and data for the analysis of neural populations in response to an ensemble of stimuli. For more information, please refer to the papers:

> Dyballa, L., Field, G. D., Stryker, M. P., & Zucker, S. W. (2024). Functional organization and natural scene responses across mouse visual cortical areas revealed with encoding manifolds. bioRxiv, 2024-10. https://doi.org/10.1101/2024.10.24.620089

__Note:__ Code for the preprint above (using data from the Allen Institute) is currently being added to the folder ``allen-data-analysis` in this repository. We expect to be done very soon.

> Dyballa, L., Rudzite, A. M., Hoseini, M. S., Thapa, M., Stryker, M. P., Field, G. D., & Zucker, S. W. (2024), "Population encoding of stimulus features along the visual hierarchy", _Proceedings of the National Academy of Sciences_, 121(4), e2317773121. https://doi.org/10.1073/pnas.2317773121

Additional code for analyzing spike waveforms and CSD can be found [`here`](https://github.com/Mahmood-Hoseini/NeuralEncodingManifolds).



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
- scikit-learn >= 1.0.2
- Matplotlib >= 3.4.2
- tensor_toolbox >= v3.1 (MATLAB, for permuted decomposition)
- IAN >= 1.1.2 (https://github.com/dyballa/IAN)
- tensorflow >= 2.10 (for running the CNN example)


## Documentation

You will find detailed usage examples in the python notebooks present in each folder. Feel free to contact me by email if you have any questions.
