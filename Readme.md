# QPSK Signal Classification using Variational Quantum Classifier (VQC)

This project, created by Jesús Vilela Jato, is dedicated to simulating a Quadrature Phase Shift Keying (QPSK) signal with added noise, and classifying the signal-to-noise ratio (SNR) of each packet into two classes based on a certain threshold. A Variational Quantum Classifier (VQC) is trained on this data, and its performance is evaluated on a separate test set.

## Prerequisites

The project is implemented in Python and requires the following libraries:

- Qiskit
- Numpy
- Scikit-learn

## Installation

You can install the required packages using pip:

```bash
pip install qiskit numpy scikit-learn
```
## Running the Code

To run the code, simply navigate to the directory containing the script and run:

```bash
python main.py
```
## Project Structure

The project consists of the following steps:

1. **Simulate a QPSK signal and add noise**: Generates a QPSK signal with noise.
2. **Calculate SNR and define labels**: Calculates the SNR for each packet and labels it based on the SNR threshold.
3. **Prepare data for VQC**: The labels are converted to strings, and a feature map, a variational form, and an optimizer are created.
4. **Train the VQC**: The VQC is trained on the training set and the parameters of the variational form are adjusted to minimize the loss function.
5. **Test the VQC**: The VQC is tested on the test set, and the accuracy of the model is calculated.

## License

Copyright (c) 2023 Jesús Vilela Jato. All Rights Reserved.

Distributed under the MIT license. See `LICENSE` for more information.
