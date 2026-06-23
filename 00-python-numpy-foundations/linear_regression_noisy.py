import numpy as np
import matplotlib.pyplot as plt


def main():
    X = np.linspace(0, 10, 100)
    noise = np.random.normal(0, 2, size=100)
    y = 3 * X + 5 + noise

    w = 0.0
    b = 0.0

    learning_rate = 0.01
    epochs = 1000

    losses = []

    for epoch in range(epochs):
        # Forwared Pass
        y_pred = w * X + b

        # Loss
        loss = np.mean((y_pred - y) ** 2)
        losses.append(loss)

        # Gradients
        dw = np.mean(2 * (y_pred - y) * X)
        db = np.mean(2 * (y_pred - y))

        # Gradient descent update

        w -= learning_rate * dw
        b -= learning_rate * db

        if epoch % 100 == 0:
            print(f"epoch={epoch}, loss = {loss:.6f}, w={w:.4f}, b ={b:.4f}")

    print("\nFinal parameters:")
    print(f"w = {w:.4f}")
    print(f"b = {b:.4f}")

    print("\nPredictions:")
    final_predictions = w * X + b
    for xi, yi, pred in zip(X, y, final_predictions):
        print(f"x={xi:.1f}, true={yi:.1f}, predicted={pred:.2f}")

    # Predict unseen value
    x_new = 6
    y_new = w * x_new + b
    print(f"\nPrediction for x={x_new}: {y_new:.2f}")

    # Plot loss
    plt.figure()
    plt.scatter(X, y, label="Noisy data")
    plt.plot(X, final_predictions, label="Learned line")
    plt.xlabel("X")
    plt.ylabel("y")
    plt.title("Linear Regression on Noisy Data")
    plt.legend()
    plt.savefig("00-python-numpy-foundations/results/noisy_regression_line.png")
    plt.show()


if __name__ == "__main__":
    main()
