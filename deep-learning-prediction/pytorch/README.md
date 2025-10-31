### Overview

This folder contains a deep learning implementation using PyTorch and brainwave data to classify drone commands and a trained model (as noted below). 
- Jupyter notebook: Includes data preparation steps to process the brainwave data, electroencephalography (EEG) brain signals, use of common python libraries to create train/test datasets for model training, and the use of PyTorch to train a Convolutional neural network (CNN) model. 
- FlexibleCNNClassifier.pth: This is the trained model, which was created using the Jupyter notebook and 100 Epochs. 

### Key Features

- Data preparation steps can dynamically identify brain command (.txt) files & load them into a dataframe. 
- Data preprocessing steps take raw data and transforms it into Tensors for model training
- PyTorch Dataset and Data Loader utilities were used to prep train/test splits to access & pass samples in batches, reshuffle at every epoch to reduce model overfitting.
- PyTorch framework was used to create and train the CNN model.
- Trained model was evaluated using sklearn's accuracy_score, classification_report, and confusion_matrix functions (matplotlib and seaborn were used for visualizations as well).

### Usage

- The Jupyter notebook can be run on an operating system that has the python libraries included installed. There are only a couple directory references that would need to be updated prior to starting as well, one to set location of raw brain signal data files, and another to set location of where to save the trained model file.
- Note the the raw brain signal files or directories/subdirectories must have at least one reference to the corresponding drone command for the data to be processed correctly. The following would need to be included: backward, forward, landing, left, right, or takeoff. If additional commands are ever recorded they can be added.

### Author

Original Author: Giovanni Antunez

### Demo 

- Coming soon

### References

References:
<br>
- Scikit-learn: https://scikit-learn.org/stable/
<br>
- Pandas: https://pandas.pydata.org/
<br>
- Numpy: https://numpy.org
<br>
- PyTorch: https://pytorch.org
<br>
- Zero to mastery learn pytorch for deep learning. Zero to Mastery Learn PyTorch for Deep Learning. (n.d.). https://www.learnpytorch.io/ 
  

