Protein Sequence Regression Model
This project implements a deep learning model for predicting target values (tm) from protein sequences using the ESM (Evolutionary Scale Modeling) pre-trained model and a Multi-Layer Perceptron (MLP).

Table of Contents：
1、Project Overview
2、File Structure
3、Installation
4、Usage
5、Model Architecture
6、Dataset
7、Training
8、Results
9、License

Project Overview
The goal of this project is to predict a target value (tm) from protein sequences using a deep learning approach. The model leverages the ESM pre-trained model to extract features from protein sequences and uses an MLP for regression.

File Structure
The project consists of the following files:

train.py: The main script for training the model.

model.py: Defines the model architecture using ESM and MLP.

dataset.py: Handles data loading and preprocessing.

mlp.py: Implements the Multi-Layer Perceptron (MLP) used in the model.

loss_his.csv: Contains the training and testing loss history.

README.md: This file, providing an overview of the project.

Installation
To run this project, you need to install the required dependencies. Follow these steps:


cd your-repo-name
Set up a Python environment (optional but recommended):

python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
Install the required packages:

pip install torch pandas numpy
pip install esm  # Install the ESM library


Usage
Training the Model
To train the model, run the following command:

python train.py
This script will:

Load the dataset from human_cell.csv.

Train the model for 10 epochs.

Save the training and testing loss history to loss_his.csv.

Dataset
The dataset should be in a CSV file (human_cell.csv) with the following columns:

sequence: The protein sequence.

target: The target value (tm) to predict.

set: Indicates whether the sample is for train or test.

validation: Indicates whether the sample is for validation.

Model Architecture
The model consists of two main components:

ESM Model: A pre-trained ESM model (esm2_t6_8M_UR50D) is used to extract features from protein sequences.

MLP: A Multi-Layer Perceptron is used to predict the target value (tm) from the ESM features.

Training Process
The model is trained using the Adam optimizer with a learning rate of 0.001.

The loss function is Mean Squared Error (MSE).

Training and testing losses are logged and saved in loss_his.csv.

Results
The training and testing losses are saved in loss_his.csv. Here is an example of the loss history:

Epoch	Training Loss	Testing Loss
0	143.73016	31.597523
1	24.81714	24.452393
2	22.905607	21.789206
License
This project is licensed under the MIT License. See the LICENSE file for details.

Acknowledgments
The ESM model is provided by Meta AI's ESM repository.

This project uses PyTorch for deep learning.
