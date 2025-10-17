### Overview

This folder contains a random forest (rf) model implementation using PyTorch, common python libraries, and brainwave data to classify drone commands and a trained model (as noted below). While PyTorch was used to create and train the rf model it is more a custom model than one that relies soley on PyTorch given PyTorch is intended for deep learning models, so it doesn't have features "out of the box" to use to create models like decision trees or random forest. 
- Jupyter notebook: Includes the use of PyTorch to create a custom random forest model. 
- custom_rf.pt.gz: Contains the trained model, which presented with an accuracy of 83% (on training data) and a test accuracy of 49% (on unseen data) so there's plenty of room for improvement, however adding this into the Avatar project's repo to show where it stands. This will need to be revisited at a later date. 

### Key Features

- Data preparation steps can dynamically identify brain command (.txt) files & load them into a dataframe. 
- Data preprocessing steps take raw data and transforms it into PyTorch tensors for model training
- Various PyTorch operations are included in the model such as randint and randperm to randomly subsample indices for bootstrap/bagging and shuffle feature indices subsets.
- Applied the use of vectorization to reduce model training time. Initially when using python loops training was taking days, now it's down to hours though more can be done. 

### Usage

- The Jupyter notebook can be run on an operating system that has the python libraries included installed. There are only a few directory references that would need to be updated prior to starting. One reference to set the location of raw brain signal data files, and another to set location of where to save the trained model file.
- The the raw brain signal files or directories/subdirectories must have at least one reference to the corresponding drone command for the data to be processed correctly. The following would need to be included: backward, forward, landing, left, right, or takeoff. If additional commands are ever recorded they can be added.
- In order to load and use the model the RandomForest class must be imported or defined using existing code. 
- Recommend using this code with cpu given some issues may be observed, specially when using mps (e.g. errors due to torch.mode, memory crashes).


### Author

Original Author: Giovanni Antunez
<br>
<br>
 
 ### References

- Pandas: https://pandas.pydata.org/
- Numpy: https://numpy.org
- PyTorch: https://pytorch.org
- Scikit-learn: https://scikit-learn.org/stable/
- Scikit-learn RandomForestClassifier: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
- python docs: https://docs.python.org
- Mendoza, A. (2020, September 23). Ensemble Models From Scratch With PyTorch. ConsciousML. https://www.axelmendoza.com/posts/ensemble-models-from-scratch-pytorch/ 
- Yadav, A. (2024, November 6). How can I use KNN and Random Forest models in PyTorch? Medium. https://medium.com/we-talk-data/how-can-i-use-knn-and-random-forest-models-in-pytorch-6083f5ef370a
- Breiman, L. (2001). Random forests. Machine Learning, 45(1), 5–32. https://doi.org/10.1023/A:1010933404324
- Breiman, L., & Cutler, A. (2004). Random forests – classification description. University of California, Berkeley. Retrieved from https://www.stat.berkeley.edu/~breiman/RandomForests/cc_home.htm
- Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., & Liu, T.-Y. (2017). LightGBM: A highly efficient gradient boosting decision tree. Advances in Neural Information Processing Systems, 30. https://proceedings.neurips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boosting-decision-tree.pdf
- Zhang, H., Si, S., & Hsieh, C.-J. (2017). GPU acceleration for large-scale tree boosting. arXiv preprint arXiv:1706.08359. https://arxiv.org/abs/1706.08359
- Genuer, R., & Poggi, J.-M. (2016). Arbres CART et Forêts aléatoires: Importance et sélection de variables. arXiv preprint arXiv:1610.08203. https://arxiv.org/abs/1610.08203
- Ehrlinger, J., Ishwaran, H., & Gerds, T. A. (2015). ggRandomForests: Visually exploring a random forest for regression. arXiv preprint arXiv:1501.07196. https://arxiv.org/abs/1501.07196
- Google Developers. (n.d.). Random forests. In Machine Learning Crash Course. Retrieved October 14, 2025, from https://developers.google.com/machine-learning/decision-forests/random-forests
- IBM. (n.d.). What is random forest? IBM Think Blog. Retrieved October 14, 2025, from https://www.ibm.com/think/topics/random-forest
- Stack Overflow. (2018, September 26). What is the meaning of bins of LightGBM? Stack Exchange Inc. Retrieved from https://stackoverflow.com/questions/52529021/what-is-the-meaning-of-bins-of-lightgbm
- Data Science Made Simple. (2019, May 5). Understanding LightGBM. Medium. https://medium.com/data-science/understanding-the-lightgbm-772ca08aabfa
- GeeksforGeeks. (2020, March 12). LightGBM histogram-based learning. https://www.geeksforgeeks.org/machine-learning/lightgbm-histogram-based-learning