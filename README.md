# ft_linear_regression_42

A small learning project that implements linear regression from scratch to predict car prices using gradient descent. This repository is designed for people who are taking their first steps in machine learning and want to understand how linear regression works by implementing the algorithm themselves (no high-level ML libraries required).

## Contents

- `data.csv` - the dataset used for training and testing (CSV). Expected to contain at least two numeric columns: one feature and one target (price). Adjust as needed for multiple features.
- `README.md` - this file.

## Project goal

Build and train a linear regression model (using gradient descent) to predict the price of a car from one or more features. The main learning goals are:

- Understand and implement the hypothesis function for linear regression.
- Implement mean squared error (MSE) as the loss function.
- Implement gradient descent to minimize MSE and learn model parameters (weights and bias).
- Learn how to normalize features, choose learning rates, and diagnose convergence issues.

## Expected dataset format

The repository includes `data.csv`. A minimal, expected format is a header row followed by numeric values, for example:

feature,price
1200,15000
1500,18000
...

Where `feature` is a numeric explanatory variable (e.g., engine size, mileage, year difference) and `price` is the target variable to predict. If you have multiple features, use multiple columns (e.g., `mileage,year,engine_size,price`). When training with multiple features, make sure the training script handles multiple columns.

If your file uses a different delimiter or has non-numeric columns, clean or convert it before training.

## How to run (suggested, minimal Python workflow)

This project intentionally keeps dependencies minimal. You can use plain Python (3.8+) and common packages such as `numpy` and `pandas` for data loading and numeric ops.

1. Create a virtual environment (recommended):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install numpy pandas matplotlib
```

2. Add or create a training script (suggested name: `train.py`) that:

- Reads `data.csv` with `pandas`.
- Splits features and target (X and y).
- Normalizes features (mean/std or min/max).
- Initializes parameters (weights and bias).
- Runs gradient descent for a specified number of iterations and learning rate.
- Saves or prints final parameters and training loss history.

3. Example minimal training loop (conceptual):

```python
# Pseudocode - implement in train.py
X, y = load_csv('data.csv')
X = normalize(X)
theta = zeros(n_features)
b = 0
for epoch in range(epochs):
	y_pred = X.dot(theta) + b
	error = y_pred - y
	loss = (error**2).mean()
	# gradients
	grad_theta = (2/len(y)) * X.T.dot(error)
	grad_b = (2/len(y)) * error.sum()
	# update
	theta -= lr * grad_theta
	b -= lr * grad_b

	if epoch % 100 == 0:
		print(epoch, loss)
```

4. Plot the loss history to verify convergence using `matplotlib` (optional).

## Implementation tips and common pitfalls

- Feature scaling: Gradient descent converges faster if features are scaled. Use standardization (subtract mean, divide by std) or min-max scaling.
- Learning rate: If loss diverges or becomes NaN, reduce the learning rate. If training is very slow, increase it a little.
- Initialization: Initialize weights to zeros or small random values. For linear regression zeros are fine.
- Overfitting: With few features and a simple model, overfitting is less of a concern. Still consider splitting data into train/test to evaluate generalization.
- Multiple features: Remember to add a column for the bias term (or keep a separate bias variable) when vectorizing updates.
- Numerical stability: Use float64 if possible to reduce numerical issues.

## Evaluation

- Use mean squared error (MSE) or root mean squared error (RMSE) to evaluate model quality on a held-out test set.
- Optionally compute R^2 score for a measure of explained variance.

## Example commands to try

After implementing `train.py` you can run:

```bash
source .venv/bin/activate
python train.py --data data.csv --lr 0.01 --epochs 2000
```

Add CLI flags for dataset path, learning rate, epochs, and output file to make experiments easy.

## What you can add next (suggested improvements)

- Implement normal equation solution for linear regression and compare to gradient descent.
- Add L2 regularization (Ridge) to reduce overfitting.
- Add support for polynomial features to capture non-linear relationships.
- Write unit tests for gradient and loss computations.
- Add a small Jupyter notebook that walks through the dataset, training, and plots predictions vs ground truth.

## Assumptions

- The assignment expects a single-file project and learning-focused implementation. I assumed Python is the intended language since it's common for ML introductions. If you prefer another language, adapt the pseudocode accordingly.

## License

This repository is for educational purposes. Add an appropriate license if you plan to publish the code.

---

If you'd like, I can also scaffold a minimal `train.py` and a small notebook that demonstrates training and plotting — tell me whether you prefer a full implementation or a guided template.
