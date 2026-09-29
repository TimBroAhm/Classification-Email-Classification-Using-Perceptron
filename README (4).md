# Email Classification Using Perceptron

A machine learning project that classifies emails using the Perceptron algorithm, a linear binary classifier that learns a weight vector by iteratively correcting misclassified samples. The full workflow is implemented in a Jupyter notebook.

## Repository Structure

```
Classification-Email-Classification-Using-Perceptron/
├── Email_Classification_Using_Perceptron.ipynb   # Data preparation, training, evaluation
├── index.html                                     # Project web page
└── README.md                                      # Project documentation
```

## How the Perceptron Works

For an input feature vector `x`, weights `w`, and bias `b`, the model predicts:

```
ŷ = sign(w · x + b)
```

During training, each misclassified sample updates the weights:

```
w ← w + η · (y − ŷ) · x
b ← b + η · (y − ŷ)
```

where `η` is the learning rate. Training repeats over the dataset until the error stops improving or a maximum number of epochs is reached.

## Workflow

1. **Load the email dataset**
2. **Preprocess the text**: clean and normalize email content
3. **Extract features**: convert text into numerical vectors
4. **Split the data** into training and test sets
5. **Train the Perceptron** classifier
6. **Evaluate** with accuracy, precision, recall, F1-score, and a confusion matrix

## Requirements

- Python 3.9+
- Jupyter Notebook / JupyterLab, or Google Colab
- `numpy`, `pandas`, `matplotlib`, `scikit-learn`

```bash
pip install numpy pandas matplotlib scikit-learn jupyter
```

## Getting Started

**Run locally**

```bash
git clone https://github.com/TimBroAhm/Classification-Email-Classification-Using-Perceptron.git
cd Classification-Email-Classification-Using-Perceptron
jupyter notebook Email_Classification_Using_Perceptron.ipynb
```

**Run in Google Colab**

1. Open [colab.research.google.com](https://colab.research.google.com).
2. Choose **File → Open notebook → GitHub** and paste the repository URL.
3. Select `Email_Classification_Using_Perceptron.ipynb` and run all cells (**Runtime → Run all**).

## Author

**TimBro**
GitHub: [@TimBroAhm](https://github.com/TimBroAhm)
