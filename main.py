'''
In this script, we simulate a QPSK signal, add noise, and then classify the signal-to-noise ratio (SNR) of each packet into two classes based on a threshold. We then train a Variational Quantum Classifier (VQC) on this data, and test it on a separate test set to calculate the model's accuracy.

Here's a brief overview of the key steps in the script:

1. Simulate a QPSK signal and add noise: This step generates a Quadrature Phase Shift Keying (QPSK) signal with noise. In real-world scenarios, this could represent a signal transmission over a noisy channel, such as in wireless communication.

2. Calculate SNR and define labels: The script then calculates the Signal-to-Noise Ratio (SNR) for each packet of symbols and labels each packet based on whether the SNR is above or below a certain threshold.

3. Prepare data for VQC: This involves converting the labels to strings (as required by the VQC), creating a feature map and a variational form (which are the two main components of the VQC), and creating an optimizer. The optimizer is used in the training process to adjust the parameters of the variational form to minimize the loss function.

4. Train the VQC: The training data is split into a training set and a testing set, and the VQC is then trained on the training set. The fit method is used to train the VQC, which adjusts the parameters of the variational form to minimize the loss function.

5. Test the VQC: After training, the VQC is tested on the test set. The predict method is used to classify the test data, and the accuracy of the model is then calculated.

In the problem at hand, classifying the SNR of QPSK signals, quantum computing can potentially excel.
This is because quantum systems naturally exhibit interference and superposition, key principles that underpin the behavior of QPSK signals.

'''
# (c) Jesús Vilela Jato, all rights reserved.

import numpy as np
from qiskit import Aer
from qiskit.utils import QuantumInstance
from qiskit.algorithms.optimizers import COBYLA
from qiskit.circuit.library import RealAmplitudes
from qiskit_machine_learning.algorithms.classifiers import VQC
from qiskit.circuit.library import PauliFeatureMap
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score
from qiskit.algorithms.optimizers import SPSA

# Define the coherent QPSK FEC 1Tb baud subaquatic cable system parameters
num_symbols = 10000
symbol_rate = 1e9
baud_rate = 2 * symbol_rate
symbols_per_packet = 10  # Define the number of symbols per packet

# Define the QPSK signal
signal_I = np.random.choice([-1, 1], num_symbols)
signal_Q = np.random.choice([-1, 1], num_symbols)

# Simulate noise and calculate SNR
noise_I = np.random.normal(scale=np.sqrt(signal_I.var()/10**(10/10)), size=num_symbols)
noise_Q = np.random.normal(scale=1.1*np.sqrt(signal_Q.var()/10**(10/10)), size=num_symbols)

# Convert to binary sequence
binary_sequence = np.array([(i > 0) + 2 * (q > 0) for i, q in zip(signal_I, signal_Q)])

# Split into packets
packets_I = np.array([signal_I[i:i+symbols_per_packet] for i in range(0, len(signal_I), symbols_per_packet)])
packets_Q = np.array([signal_Q[i:i+symbols_per_packet] for i in range(0, len(signal_Q), symbols_per_packet)])
noise_packets_I = np.array([noise_I[i:i+symbols_per_packet] for i in range(0, len(noise_I), symbols_per_packet)])
noise_packets_Q = np.array([noise_Q[i:i+symbols_per_packet] for i in range(0, len(noise_Q), symbols_per_packet)])

# Calculate SNR for each packet
snr = 10 * np.log10((packets_I.var(axis=1) + packets_Q.var(axis=1)) / (noise_packets_I.var(axis=1) + noise_packets_Q.var(axis=1)))

# Now, you can set the snr_threshold to the median of snr
snr_threshold = np.median(snr)  # dB

# Prepare labels
labels = snr >= snr_threshold

# Convert labels to integers
int_labels = np.array([int(i) for i in labels])

# Prepare labels and convert them into strings (required by VQC)
str_labels = np.array([str(i) for i in int_labels])

# Convert to binary sequence
binary_sequence = np.array([(i > 0) + 2 * (q > 0) for i, q in zip(signal_I, signal_Q)])

# Split into packets
packets = np.array([binary_sequence[i:i+symbols_per_packet] for i in range(0, len(binary_sequence), symbols_per_packet)])

backend = Aer.get_backend('statevector_simulator')
print(backend.configuration())

feature_dim = 4
num_qubits = 4  # Set the number of qubits

feature_map = PauliFeatureMap(feature_dimension=num_qubits, reps=4, paulis=['X', 'Y', 'Z'])
var_form = RealAmplitudes(num_qubits, entanglement='linear', reps=4)

optimizer = COBYLA(maxiter=2000)

# Prepare data for VQC
train_data = np.array([np.histogram(packet, bins=4, range=(0, 4))[0]/symbols_per_packet for packet in packets])
training_input = {str(i): train_data[int_labels == i] for i in np.unique(int_labels)}

print(np.unique(str_labels))
print(train_data.shape)
print(int_labels.shape)
print(np.unique(int_labels, return_counts=True))

for i in np.unique(str_labels):  # Here is the correction: use str_labels instead of labels
    print(f"Shape of class {i} data:", training_input[i].shape)

print("Feature map circuit:")
print(feature_map.draw())
print("Variational form circuit:")
print(var_form.draw())
print(f"Number of features in data: {train_data.shape[1]}")
print(f"Number of qubits in feature map: {feature_map.num_qubits}")
print(f"Number of qubits in ansatz: {var_form.num_qubits}")

# split data into training and testing sets
train_data, test_data, train_labels, test_labels = train_test_split(train_data, str_labels, test_size=0.2, random_state=42)

# prepare training and test datasets in the required format
training_input = {str(i): train_data[train_labels == str(i)] for i in np.unique(train_labels)}
test_input = {str(i): test_data[test_labels == str(i)] for i in np.unique(test_labels)}

# Convert training_input dictionary to X_train and y_train
X_train = np.concatenate([training_input[i] for i in training_input])
y_train = np.concatenate([np.full(shape=(len(training_input[i]),), fill_value=i) for i in training_input])

# Convert test_input dictionary to X_test and y_test
X_test = np.concatenate([test_input[i] for i in test_input])
y_test = np.concatenate([np.full(shape=(len(test_input[i]),), fill_value=i) for i in test_input])

# VQC training
vqc = VQC(num_qubits, feature_map, var_form, 'cross_entropy', optimizer)
vqc.fit(X_train, y_train)

# model testing
predicted_labels = vqc.predict(X_test)

# calculate accuracy
accuracy = np.sum(predicted_labels == y_test) / len(y_test)
print('Accuracy:', accuracy)

# Predicted labels
predicted_labels = vqc.predict(X_test)

# Confusion Matrix
print('Confusion Matrix:')
print(confusion_matrix(y_test, predicted_labels))

# Precision
precision = precision_score(y_test, predicted_labels, pos_label='1')
print('Precision: ', precision)

# Recall
recall = recall_score(y_test, predicted_labels, pos_label='1')
print('Recall: ', recall)

# F1 Score
f1 = f1_score(y_test, predicted_labels, pos_label='1')
print('F1 Score: ', f1)

# Analyze the incorrect predictions made by the VQC model
predicted_labels = vqc.predict(X_test)
incorrect_predictions = X_test[predicted_labels != y_test]
print("Incorrect predictions:")
print(incorrect_predictions.shape)


# Initialize a variable to store the parameters
params = None

while incorrect_predictions.shape[0] > 0:
    # If we have parameters from the previous iteration, use them as the initial point
    if params is not None:
        optimizer.initial_point = params

    # Train the VQC model
    vqc.fit(X_train, y_train)

    # Store the trained parameters for the next iteration
    params = vqc.weights

    # Add regularization to the optimizer
    regularized_optimizer = SPSA(maxiter=2000, last_avg=5,perturbation=0.1,
                                    regularization=True, learning_rate=0.01)
    # Retrain the VQC model with the regularized optimizer
    vqc = VQC(num_qubits, feature_map, var_form, optimizer=regularized_optimizer)
    vqc.fit(X_train, y_train)
    # model testing
    predicted_labels = vqc.predict(X_test)

    # calculate accuracy
    accuracy = np.sum(predicted_labels == y_test) / len(y_test)
    print('Accuracy:', accuracy)

    # Predicted labels
    predicted_labels = vqc.predict(X_test)

    # Confusion Matrix
    print('Confusion Matrix:')
    print(confusion_matrix(y_test, predicted_labels))

    # Precision
    precision = precision_score(y_test, predicted_labels, pos_label='1')
    print('Precision: ', precision)

    # Recall
    recall = recall_score(y_test, predicted_labels, pos_label='1')
    print('Recall: ', recall)

    # F1 Score
    f1 = f1_score(y_test, predicted_labels, pos_label='1')
    print('F1 Score: ', f1)

    # Analyze the incorrect predictions made by the VQC model
    predicted_labels = vqc.predict(X_test)
    incorrect_predictions = X_test[predicted_labels != y_test]
    print("Incorrect predictions:")
    print(incorrect_predictions.shape)
