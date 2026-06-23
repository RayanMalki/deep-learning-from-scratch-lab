## Noisy linear Regression Result

For the noisy dataset, the target was generated from:

y=3x + 5 + noisy

The model learned approximately:

w = 2.84%
b = 5.96%

The learned parameters are close to the original hidden relationship, but not exact because the data contains random noise.

Unlike the clean dataset, the loss does not reach zero. This is expected because no straight line can perfectly fit all noisy points. Instead, gradient descent finds the line that minimizes the average squared error.

This experiment shows that real machine learning models usually optimize for the best approximation, not a perfect answer.
