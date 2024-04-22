# Comparative Analysis of Deepfake Detection Models

In this project we aim to conduct a comprehensive comparative analysis of deepfake video detection models, leveraging the open-source Deepstar toolkit. This toolkit provides training data and two example neural networks for deepfake detection. Our goal is to train the two networks and compare their effectiveness, and then use our insights to design our own model and compare it to the original two. We will speculate about the impact a given model’s structure has on the given model’s performance. Our aim is to improve our model to be as accurate as possible, and ideally at least as accurate as the two examples.

## Installation

This project relies on several Python packages to run. These can be found in the requirements.txt file.

```bash
# To install all the required dependencies, run this command in the project directory
pip install -r requirements. txt
```

## Usage

To train the models and see the results first install the dependencies in the requirements file.

1. Run the train_deepstar_models.ipynb Jupyter Notebook file to train the base Mesonet and Mouthnet models. 
    - 2 models will be trained for each - One for regular frame generator, One for sample frame generator
2. Evaluate the trained models in evaluate_deepstar_models.ipynb file
3. Run the train_temporal_models.ipynb file in order to train the RNN model
    - Can modify which base model is used by referencing the specific model in custom_RNN_model.py file
4. Evaluate the RNN model in evaluate_temporal_models.ipynb file
5. Run the train