# Stage 00 Notes

## Why the model improves

The model predicts `y_pred = X @ w + b` and measures error with mean squared error:

```text
loss = mean((y_pred - y)^2)
```

Each step computes the gradient of the loss with respect to `w` and `b` — the direction in which the loss increases fastest — and then takes a small step in the *opposite* direction:

```text
w = w - learning_rate * dw
b = b - learning_rate * db
```

Because the gradient points uphill, subtracting it moves the parameters downhill on the loss surface. Repeating this many times slides `w` and `b` toward the values that minimize the average squared error, which is why the loss curve falls steadily over the epochs.

## Clean linear regression result

The clean dataset was generated from `y = 2x` with no noise.

The model learned approximately:

```text
w = 1.995
b = 0.017
```

Predictions line up almost exactly with the targets, and the unseen point `x = 6` is predicted as `11.99` (true value `12`). Since the data is perfectly linear, the loss can be driven very close to zero — by epoch 900 it has dropped from ~30 to ~0.0001.

## Noisy linear regression result

The noisy dataset was generated from:

```text
y = 3x + 5 + noise
```

The model learned approximately:

```text
w = 2.84
b = 5.96
```

The learned parameters are close to the hidden relationship (`w = 3`, `b = 5`) but not exact, because the data contains random noise (and the exact values shift between runs since the noise isn't seeded).

Unlike the clean dataset, the loss does **not** reach zero. This is expected: no straight line can pass through every noisy point. Instead, gradient descent finds the line that minimizes the average squared error.

This is the key takeaway — real machine learning models usually optimize for the best approximation, not a perfect answer.

## Things to remember

- `loss` decreasing is the signal that training is working.
- A learning rate that's too large makes the loss diverge instead of converge.
- Matrix form (`X @ w`) scales to many features without changing the logic — the same loop works whether `w` is a scalar or a vector.
