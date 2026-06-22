# Deep Learning From Scratch Lab

A long-term AI/ML learning project focused on understanding the math and implementation behind deep learning.

This is not an “AI app” project. The goal is not to wrap an API or build a web frontend. The goal is to learn how neural networks actually work by implementing the core ideas step by step:

1. Autograd engine from scratch
2. Neural networks in NumPy
3. CNN / computer vision models in PyTorch
4. Tiny GPT-style Transformer from scratch

The project is designed to force deep understanding of:

- Python
- NumPy
- PyTorch
- Backpropagation
- Gradient descent
- Matrix calculus
- Neural network training loops
- Computer vision
- Transformers
- Experiment tracking
- Debugging ML models

---

# Project Motivation

I started becoming less interested in web development and more interested in the math behind AI and machine learning. This repo is my structured path to build deep learning knowledge from the ground up.

Instead of jumping directly into high-level libraries, this project starts from the basics:

- How gradients are computed
- How backpropagation works
- How neural networks learn
- How CNNs extract visual features
- How Transformers model sequences
- How to debug and evaluate models properly

This project is inspired by learning paths and ideas from people like:

- Andrej Karpathy — neural networks from scratch, micrograd, nanoGPT
- Andrew Ng — foundational ML/deep learning concepts and practical model development
- Yoshua Bengio — representation learning and deep learning theory

---

# Expected Timeline

This project is expected to take around **12 to 16 weeks**.

Estimated workload:

| Hours per week | Expected duration |
|---:|---:|
| 5 hours/week | 4–5 months |
| 10 hours/week | 3–4 months |
| 15 hours/week | 2–3 months |
| 20+ hours/week | 6–8 weeks, intense |

The goal is not to rush. The goal is to understand.

---

# Repo Structure

```txt
deep-learning-from-scratch-lab/
│
├── 00-python-numpy-foundations/
│   ├── README.md
│   ├── notes.md
│   ├── numpy_basics.ipynb
│   ├── linear_regression_numpy.py
│   └── results/
│
├── 01-autograd-engine/
│   ├── README.md
│   ├── math_notes.md
│   ├── engine.py
│   ├── nn.py
│   ├── train_mlp.py
│   ├── tests/
│   └── results/
│
├── 02-numpy-neural-network/
│   ├── README.md
│   ├── math_notes.md
│   ├── layers.py
│   ├── losses.py
│   ├── optimizers.py
│   ├── train_mnist.py
│   ├── tests/
│   └── results/
│
├── 03-cnn-vision/
│   ├── README.md
│   ├── math_notes.md
│   ├── models.py
│   ├── train.py
│   ├── evaluate.py
│   ├── visualize.py
│   └── results/
│
├── 04-tiny-transformer/
│   ├── README.md
│   ├── math_notes.md
│   ├── tokenizer.py
│   ├── model.py
│   ├── train.py
│   ├── generate.py
│   ├── tests/
│   └── results/
│
├── experiments/
│   ├── experiment_log.md
│   └── results_summary.md
│
├── notes/
│   ├── backpropagation.md
│   ├── optimization.md
│   ├── cnns.md
│   ├── transformers.md
│   └── debugging_ml.md
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

# Main Learning Goals

By the end of this project, I should be able to explain and implement:

- What a neural network is
- What a tensor is
- What a computation graph is
- How gradients are computed
- How backpropagation works
- Why gradient descent works
- How to train an MLP
- How softmax and cross-entropy work
- Why CNNs work well for images
- What convolution does
- How self-attention works
- How a Transformer block works
- How a GPT-style model generates text
- How to debug exploding/vanishing gradients
- How to compare models with experiments

---

# Weekly Roadmap

## Week 1 — Python and NumPy Foundations

### Goal

Become comfortable enough with Python and NumPy to start implementing ML code without constantly fighting the language.

### Learn

Python:

- Variables
- Functions
- Classes
- Lists
- Dictionaries
- Tuples
- Loops
- Type hints
- Basic file organization
- Importing modules

NumPy:

- Arrays
- Shapes
- Matrix multiplication
- Broadcasting
- Indexing
- Reshaping
- Axis operations
- Random initialization
- Mean, sum, max, argmax
- Dot product

Math review:

- Vectors
- Matrices
- Matrix multiplication
- Derivatives
- Partial derivatives
- Chain rule
- Mean squared error
- Gradient descent intuition

### Implement

In `00-python-numpy-foundations/`:

1. A NumPy basics notebook
2. Linear regression from scratch
3. Gradient descent for linear regression
4. Plot loss over time
5. Compare manual predictions before and after training

### Suggested files

```txt
00-python-numpy-foundations/
├── numpy_basics.ipynb
├── linear_regression_numpy.py
├── notes.md
└── results/
```

### How to implement

Start with simple data:

```txt
x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]
```

Train a model:

```txt
y_pred = w * x + b
```

Use mean squared error:

```txt
loss = mean((y_pred - y)^2)
```

Update:

```txt
w = w - learning_rate * dw
b = b - learning_rate * db
```

### Good enough criteria

This week is good enough if:

- I can write basic Python without copying everything
- I can explain what a NumPy shape is
- I can multiply matrices using NumPy
- I can train a linear regression model from scratch
- My loss decreases over time
- I can plot the loss curve
- I understand why the model improves

### Common mistakes

- Confusing Python lists with NumPy arrays
- Not understanding shape errors
- Forgetting that matrix multiplication is not the same as element-wise multiplication
- Using too large a learning rate
- Not plotting the loss

---

# Stage 1 — Autograd Engine

## Week 2 — Scalar Autograd Engine

### Goal

Build a small automatic differentiation engine that can compute gradients automatically.

This is inspired by micrograd.

### Learn

Core concepts:

- Computational graph
- Forward pass
- Backward pass
- Chain rule
- Local derivative
- Global derivative
- Topological sort
- Backpropagation

Python concepts:

- Classes
- Operator overloading
- `__add__`
- `__mul__`
- `__pow__`
- `__repr__`
- Sets
- Recursion

### Implement

In `01-autograd-engine/engine.py`, implement a `Value` class.

The `Value` class should support:

- Storing data
- Storing gradient
- Addition
- Multiplication
- Power
- Negation
- Subtraction
- Division
- ReLU
- Tanh
- Backward pass

Example usage:

```python
a = Value(2.0)
b = Value(-3.0)
c = a * b
d = c + Value(10.0)
d.backward()

print(a.grad)
print(b.grad)
```

### Required operations

Implement:

```python
__add__
__mul__
__pow__
__neg__
__sub__
__truediv__
relu()
tanh()
backward()
```

### Good enough criteria

This week is good enough if:

- I can create a computation graph
- I can call `.backward()`
- Gradients are calculated correctly for simple expressions
- I can manually verify gradients for small examples
- I can explain what each node stores
- I can explain why topological order is needed

### Tests to write

In `tests/test_engine.py`:

- Test addition gradient
- Test multiplication gradient
- Test power gradient
- Test ReLU gradient
- Test tanh gradient
- Compare simple gradients to manual calculations

Example:

```python
def test_multiplication_gradient():
    a = Value(3.0)
    b = Value(4.0)
    c = a * b
    c.backward()

    assert a.grad == 4.0
    assert b.grad == 3.0
```

### Common mistakes

- Forgetting to accumulate gradients
- Overwriting gradients instead of adding them
- Calling backward in the wrong order
- Forgetting to initialize final node gradient to 1
- Not understanding why `dL/dL = 1`

---

## Week 3 — MLP Using Autograd Engine

### Goal

Use the autograd engine to build and train a tiny neural network.

### Learn

- Neuron
- Layer
- MLP
- Parameters
- Loss function
- Gradient descent
- Activation functions
- Training loop

### Implement

In `01-autograd-engine/nn.py`:

Implement:

- `Neuron`
- `Layer`
- `MLP`

Example:

```python
model = MLP(nin=2, nouts=[4, 4, 1])
```

Each neuron should:

- Have weights
- Have bias
- Apply weighted sum
- Apply activation

### Train on toy data

Example dataset:

```python
xs = [
    [2.0, 3.0],
    [1.0, -1.0],
    [-2.0, 1.0],
    [0.5, 1.0],
]

ys = [1.0, -1.0, -1.0, 1.0]
```

Training loop:

1. Forward pass
2. Compute loss
3. Zero gradients
4. Backward pass
5. Update parameters

### Good enough criteria

This stage is good enough if:

- The MLP loss decreases
- The model can fit a tiny toy dataset
- I can explain each step of the training loop
- I can explain what weights and biases are
- I can explain why gradients must be reset
- I can compare my code conceptually with PyTorch

### Stretch goals

- Add graph visualization
- Add support for different activations
- Add mean squared error loss
- Add binary classification example
- Compare gradients against PyTorch

### Common mistakes

- Forgetting to call `zero_grad()`
- Updating parameters before backward
- Using a learning rate that is too high
- Not understanding why the model overfits the tiny dataset
- Not separating model code from training code

---

# Stage 2 — NumPy Neural Network

## Week 4 — Vectorized Neural Network Basics

### Goal

Move from scalar autograd to matrix-based neural networks using NumPy.

Instead of computing one scalar at a time, implement neural networks using arrays and matrix operations.

### Learn

- Matrix-based forward propagation
- Batch dimension
- Vectorized operations
- Dense layers
- Matrix calculus basics
- Shape debugging
- Forward pass vs backward pass
- Activation functions

### Important shape conventions

Use this convention:

```txt
X shape:      (batch_size, input_dim)
W shape:      (input_dim, output_dim)
b shape:      (1, output_dim)
Z = X @ W + b
Z shape:      (batch_size, output_dim)
```

### Implement

In `02-numpy-neural-network/layers.py`:

Implement:

- Dense layer
- ReLU activation
- Sigmoid activation
- Tanh activation

Each layer should have:

```python
forward(x)
backward(dout)
```

Example:

```python
class Dense:
    def __init__(self, input_dim, output_dim):
        self.W = ...
        self.b = ...

    def forward(self, x):
        self.x = x
        return x @ self.W + self.b

    def backward(self, dout):
        self.dW = self.x.T @ dout
        self.db = np.sum(dout, axis=0, keepdims=True)
        dx = dout @ self.W.T
        return dx
```

### Good enough criteria

This week is good enough if:

- I understand batch size
- I understand why `X @ W + b` works
- I can debug shape errors
- I can implement forward and backward for a Dense layer
- I can pass gradients backward through multiple layers
- I can explain the dimensions of every tensor

### Tests to write

- Dense forward output shape
- Dense backward gradient shape
- ReLU forward
- ReLU backward
- Numerical gradient check for Dense layer

### Common mistakes

- Mixing up `(input_dim, output_dim)` and `(output_dim, input_dim)`
- Forgetting `keepdims=True`
- Not averaging gradients over batch size when needed
- Reusing cached values incorrectly
- Debugging by guessing instead of printing shapes

---

## Week 5 — Losses, Optimizers, and Training Loop

### Goal

Implement a full MLP training pipeline in NumPy.

### Learn

Loss functions:

- Mean squared error
- Softmax
- Cross-entropy
- Softmax + cross-entropy derivative

Optimizers:

- SGD
- SGD with momentum
- Adam

Training concepts:

- Epochs
- Mini-batches
- Learning rate
- Overfitting
- Train/validation split
- Accuracy

### Implement

Files:

```txt
02-numpy-neural-network/
├── losses.py
├── optimizers.py
├── model.py
├── train_mnist.py
```

Implement:

- `softmax`
- `cross_entropy`
- `softmax_cross_entropy_backward`
- `SGD`
- `Momentum`
- `Adam`
- `Sequential` model class
- Mini-batch training loop
- Accuracy function

### Dataset

Use one of:

- MNIST
- Fashion-MNIST
- sklearn digits dataset

Start with sklearn digits if MNIST setup becomes annoying.

### Good enough criteria

This week is good enough if:

- The model trains on a real dataset
- Loss decreases
- Accuracy improves
- I can explain softmax
- I can explain cross-entropy
- I can explain why mini-batches are used
- I can save loss/accuracy curves

Minimum target:

```txt
Dataset: sklearn digits
Target accuracy: 85%+
```

Better target:

```txt
Dataset: MNIST
Target accuracy: 90%+
```

Strong target:

```txt
Dataset: MNIST
Target accuracy: 95%+
```

### Common mistakes

- Applying softmax twice
- Taking `log(0)` without numerical stability
- Forgetting to shuffle data
- Not normalizing pixel values
- Using too large a learning rate
- Wrong labels format
- Confusing one-hot labels and class indices

---

## Week 6 — Regularization and Gradient Checking

### Goal

Make the NumPy neural network more robust and verify that the math is correct.

### Learn

- Train/validation/test split
- Overfitting
- L2 regularization
- Dropout
- Weight initialization
- Xavier initialization
- He initialization
- Gradient checking
- Numerical gradients

### Implement

Add:

- L2 regularization
- Dropout layer
- Xavier initialization
- He initialization
- Gradient checking utility

Gradient checking idea:

```txt
numerical_gradient = (loss(w + epsilon) - loss(w - epsilon)) / (2 * epsilon)
```

Compare it with backprop gradient.

### Good enough criteria

This stage is good enough if:

- I can detect whether gradients are wrong
- I can explain why gradient checking is useful
- I can show that dropout changes training behavior
- I can show that L2 regularization reduces overfitting
- I can explain why initialization matters
- My README includes experiment results

### Experiment examples

| Experiment | Accuracy | Notes |
|---|---:|---|
| Basic MLP | 91% | No regularization |
| MLP + L2 | 92% | Less overfitting |
| MLP + Dropout | 91% | More stable validation |
| MLP + He init | 94% | Faster convergence |

### Common mistakes

- Doing dropout during evaluation
- Not scaling dropout outputs
- Applying regularization to biases
- Using gradient checking with too large a network
- Forgetting that numerical gradient checking is slow

---

# Stage 3 — CNN / Vision Model

## Week 7 — PyTorch Foundations and CNN Basics

### Goal

Learn PyTorch and train a basic CNN for image classification.

### Learn

PyTorch:

- Tensors
- `torch.Tensor`
- `nn.Module`
- `forward`
- `torch.optim`
- `DataLoader`
- `Dataset`
- `.train()`
- `.eval()`
- `torch.no_grad()`
- GPU usage with CUDA or MPS if available

CNN concepts:

- Convolution
- Kernels/filters
- Channels
- Feature maps
- Padding
- Stride
- Pooling
- Flattening
- Fully connected layer

### Implement

In `03-cnn-vision/`:

- CNN model in PyTorch
- Training loop
- Evaluation loop
- Accuracy calculation
- Loss curve plot
- Confusion matrix
- Save model checkpoint

Start with:

- MNIST or Fashion-MNIST

Then try:

- CIFAR-10

### Basic model

```txt
Conv2d → ReLU → MaxPool
Conv2d → ReLU → MaxPool
Flatten
Linear → ReLU
Linear → Output
```

### Good enough criteria

This week is good enough if:

- I can train a CNN in PyTorch
- I can explain what convolution does
- I can explain channels and filters
- I can explain why CNNs work better than MLPs for images
- I can compare CNN accuracy against my NumPy MLP
- I save plots and results

Minimum target:

```txt
Fashion-MNIST CNN accuracy: 85%+
```

Good target:

```txt
Fashion-MNIST CNN accuracy: 90%+
```

Strong target:

```txt
CIFAR-10 CNN accuracy: 65%+
```

### Common mistakes

- Forgetting `.train()` and `.eval()`
- Forgetting `torch.no_grad()` during evaluation
- Wrong input shape
- Not normalizing images
- Flattening to the wrong size
- Accidentally training on test data

---

## Week 8 — CNN Experiments and Visualizations

### Goal

Go beyond “I trained a CNN” by analyzing what it learns.

### Learn

- Data augmentation
- Overfitting
- Batch normalization
- Dropout in CNNs
- Learning rate schedules
- Feature map visualization
- Misclassified examples
- Confusion matrix

### Implement

Add experiments:

1. CNN without augmentation
2. CNN with augmentation
3. CNN with dropout
4. CNN with batch normalization
5. CNN with different learning rates

Add visualizations:

- Training loss curve
- Validation accuracy curve
- Confusion matrix
- Misclassified images
- First-layer filters
- Feature maps from convolution layers

### Good enough criteria

This stage is good enough if:

- I can explain why some images are misclassified
- I can show at least 3 experiments
- I can explain overfitting from loss curves
- I can explain the effect of augmentation
- I can show visualizations in the README

### Strong README section

Include:

```txt
What worked:
- Data augmentation improved validation accuracy.
- Batch normalization made training more stable.
- Dropout reduced overfitting.

What failed:
- Too large a learning rate caused unstable loss.
- A deeper model overfit quickly without augmentation.
```

### Common mistakes

- Adding experiments without recording results
- Only showing the best run
- Not explaining failed runs
- Not saving hyperparameters
- Thinking accuracy is the only useful metric

---

# Stage 4 — Tiny Transformer

## Week 9 — Language Modeling Foundations

### Goal

Understand and prepare the data pipeline for a small language model.

### Learn

- Language modeling
- Tokenization
- Character-level tokenization
- Vocabulary
- Encoding and decoding text
- Context length
- Sequence batches
- Next-token prediction
- Logits
- Cross-entropy for language modeling

### Implement

In `04-tiny-transformer/tokenizer.py`:

- Character tokenizer
- `encode(text)`
- `decode(tokens)`
- Vocabulary creation
- Train/validation split
- Batch generation

Dataset options:

- Tiny Shakespeare
- Small French text dataset
- Public domain books
- Small code dataset

Start with character-level tokenization because it is easier.

### Batch format

For language modeling:

```txt
Input:  [t0, t1, t2, t3]
Target: [t1, t2, t3, t4]
```

Example:

```txt
Text: "hello"
Input:  "hell"
Target: "ello"
```

### Good enough criteria

This week is good enough if:

- I can encode text into integers
- I can decode integers back into text
- I can create input/target batches
- I understand next-token prediction
- I understand context length
- I can explain logits

### Common mistakes

- Confusing input tokens and target tokens
- Not splitting train/validation data
- Using a dataset that is too large too early
- Starting with BPE tokenization before understanding character tokenization
- Not checking decoded samples

---

## Week 10 — Bigram Model and Training Loop

### Goal

Build the simplest possible language model before implementing a Transformer.

This makes the Transformer easier to debug later.

### Learn

- Embedding table
- Bigram language model
- Cross-entropy loss
- Sampling
- Temperature
- Text generation

### Implement

In `04-tiny-transformer/model.py`:

Start with a Bigram model:

```txt
token_id → embedding table → logits for next token
```

Implement:

- Forward pass
- Loss calculation
- Training loop
- Text generation
- Sampling from probabilities

### Good enough criteria

This week is good enough if:

- The bigram model trains
- Loss decreases
- Generated text is bad but not random
- I can explain why bigram models are limited
- I have a baseline before building the Transformer

### Common mistakes

- Expecting good text from a bigram model
- Sampling incorrectly
- Forgetting to use validation loss
- Not setting model to eval mode during generation
- Not understanding output shape

Expected output quality:

```txt
Bad, but somewhat character-distribution-aware.
```

That is fine.

---

## Week 11 — Self-Attention and Transformer Block

### Goal

Implement the core of the Transformer: causal self-attention.

### Learn

- Embeddings
- Positional embeddings
- Query, Key, Value
- Attention scores
- Scaled dot-product attention
- Causal mask
- Multi-head attention
- Residual connections
- Layer normalization
- Feedforward networks

### Implement

Core components:

- `Head`
- `MultiHeadAttention`
- `FeedForward`
- `Block`
- `LayerNorm`
- `TinyGPT`

Architecture:

```txt
Token embedding
+ Positional embedding
→ Transformer block
→ Transformer block
→ LayerNorm
→ Linear vocab projection
```

### Self-attention formula

```txt
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V
```

### Causal masking

The model should not see future tokens.

For a sequence:

```txt
[t0, t1, t2, t3]
```

Token `t1` can only attend to:

```txt
[t0, t1]
```

It cannot attend to:

```txt
[t2, t3]
```

### Good enough criteria

This week is good enough if:

- The model runs without shape errors
- Training loss decreases
- Validation loss is tracked
- Generated text becomes better than bigram output
- I can explain Q, K, and V in simple words
- I can explain why the causal mask is needed
- I can explain every tensor shape in the model

### Common mistakes

- Wrong attention mask shape
- Forgetting to scale by `sqrt(d_k)`
- Applying softmax on the wrong dimension
- Mixing batch, time, and channel dimensions
- Forgetting residual connections
- Forgetting positional embeddings
- Not understanding why the model sees only the past

---

## Week 12 — TinyGPT Training and Generation

### Goal

Train a small GPT-style model and make the results presentable.

### Learn

- Hyperparameters
- Context length
- Batch size
- Number of layers
- Number of heads
- Embedding dimension
- Dropout
- Sampling temperature
- Top-k sampling
- Model checkpointing
- Train/validation loss

### Implement

Add:

- Training script
- Generation script
- Save/load checkpoints
- Config file
- Experiment logging
- Temperature sampling
- Optional top-k sampling

Example config:

```python
batch_size = 64
block_size = 128
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
n_embd = 128
n_head = 4
n_layer = 4
dropout = 0.2
```

### Good enough criteria

This stage is good enough if:

- The model trains end-to-end
- Validation loss improves
- It generates text with some structure
- I can save and load checkpoints
- I can compare it to the bigram model
- I can explain why increasing model size helps or hurts
- I can explain the role of context length

### Strong criteria

This stage is strong if:

- I run at least 3 experiments
- I compare different context lengths
- I compare different model sizes
- I compare different temperatures
- I include generated samples
- I explain failure cases honestly

Example experiment table:

| Experiment | Layers | Heads | Embedding | Context | Val Loss | Notes |
|---|---:|---:|---:|---:|---:|---|
| Bigram baseline | 0 | 0 | 0 | 1 | X.XX | Weak baseline |
| TinyGPT small | 2 | 2 | 64 | 64 | X.XX | Learns simple patterns |
| TinyGPT medium | 4 | 4 | 128 | 128 | X.XX | Better structure |
| TinyGPT dropout | 4 | 4 | 128 | 128 | X.XX | Less overfitting |

### Common mistakes

- Training too large a model too early
- Not using validation loss
- Only judging by generated text
- Using too high a learning rate
- Not saving checkpoints
- Not comparing to a baseline
- Thinking the model failed because the generated text is not perfect

---

# Week 13 — Cleanup, Documentation, and Resume Polish

### Goal

Turn the project from “code that works” into a strong portfolio project.

### Learn

- How to write technical documentation
- How to present ML experiments
- How to explain failures
- How to organize a GitHub repo
- How to write a strong resume bullet

### Implement

Add:

- Main README
- Stage-specific READMEs
- Math notes
- Diagrams
- Loss curves
- Accuracy curves
- Experiment tables
- Setup instructions
- How to run each stage
- Results summary
- Future improvements

### Main README should include

1. Project overview
2. Motivation
3. What I built
4. What I learned
5. Repo structure
6. Installation
7. Results
8. Experiments
9. Math notes
10. Future work

### Good enough criteria

This project is good enough for GitHub if:

- Someone can clone the repo and run at least one script
- Each stage has a README
- Results are shown with plots/tables
- I explain what worked and what failed
- I explain the math in my own words
- The repo does not look like copied tutorial code
- The commit history shows progression
- The code is organized and readable

### Resume bullet draft

```txt
Deep Learning From Scratch Lab — Built a four-stage deep learning project implementing an autograd engine, NumPy-based neural networks, CNN image classifiers, and a GPT-style Transformer from scratch. Implemented backpropagation, gradient descent, softmax cross-entropy, self-attention, and training loops; validated gradients against PyTorch and tracked experiments with loss/accuracy curves.
```

Shorter version:

```txt
Built a deep learning lab from scratch, implementing autograd, NumPy neural networks, CNNs, and a GPT-style Transformer with custom training loops, gradient checks, and experiment tracking.
```

---

# Final Project Success Criteria

The full project is successful if I can confidently answer these questions:

## Autograd

- What is a computation graph?
- What does `.backward()` do?
- Why do we need topological sorting?
- What does the gradient of a variable mean?
- Why do we initialize the final node’s gradient to 1?

## Neural Networks

- What are weights and biases?
- What does an activation function do?
- Why do we need nonlinearities?
- What is a loss function?
- Why does gradient descent work?
- What is backpropagation?

## NumPy MLP

- What is the shape of each matrix?
- What is a batch?
- How does softmax work?
- How does cross-entropy work?
- Why do we normalize inputs?
- How do we check if gradients are correct?

## CNNs

- What is a convolution?
- What is a filter/kernel?
- What is a feature map?
- Why are CNNs good for images?
- What does pooling do?
- How do CNNs differ from MLPs?

## Transformers

- What is tokenization?
- What is an embedding?
- What is self-attention?
- What are Q, K, and V?
- Why do we need positional embeddings?
- Why do we need a causal mask?
- What is a Transformer block?
- How does text generation work?

## ML Engineering

- How do I structure a training loop?
- How do I evaluate a model?
- How do I detect overfitting?
- How do I compare experiments?
- How do I save and load models?
- How do I debug a model that is not learning?

---

# Definition of “Good Enough”

This project does not need to be perfect.

It is good enough when:

- Every stage runs
- Every stage has documentation
- Every stage has at least one result
- I understand the code I wrote
- I can explain the math behind the code
- I have plots or tables proving that training worked
- I can discuss bugs and failures
- I can compare my implementation with PyTorch
- I can show this repo to someone technical without being embarrassed

The project is not good enough if:

- I only copied code without understanding it
- There are no explanations
- There are no results
- There are no experiments
- The README is empty
- The model only works by accident
- I cannot explain tensor shapes
- I cannot explain the training loop
- I cannot explain why loss decreases

---

# Suggested Resources

## Python and NumPy

- Python official tutorial
- NumPy documentation
- “Python Data Science Handbook” sections on NumPy

## Deep Learning Foundations

- Andrew Ng Machine Learning Specialization
- Andrew Ng Deep Learning Specialization
- 3Blue1Brown Neural Networks series

## Autograd and Backpropagation

- Andrej Karpathy — Micrograd
- Andrej Karpathy — Neural Networks: Zero to Hero
- CS231n backpropagation notes

## CNNs

- Stanford CS231n
- PyTorch tutorials
- Papers/articles explaining LeNet, AlexNet, ResNet

## Transformers

- Andrej Karpathy — nanoGPT
- “Attention Is All You Need”
- The Illustrated Transformer
- PyTorch Transformer examples

---

# Development Rules

## Rule 1 — Do not skip the math

For every important concept, write a small explanation in `math_notes.md`.

Examples:

```txt
Why does ReLU need a backward function?
What is the derivative of tanh?
Why does softmax need numerical stability?
Why do we divide attention scores by sqrt(d_k)?
```

## Rule 2 — Always track experiments

Every experiment should include:

- Date
- Model
- Dataset
- Hyperparameters
- Train loss
- Validation loss
- Accuracy if applicable
- Notes
- What changed from previous run

Example:

```txt
## Experiment 003

Date: YYYY-MM-DD
Stage: CNN
Dataset: Fashion-MNIST
Model: 2-layer CNN
Learning rate: 0.001
Batch size: 64
Epochs: 10

Result:
Train accuracy: 94%
Validation accuracy: 90%

Notes:
Validation accuracy improved after adding batch normalization.
```

## Rule 3 — Compare against a baseline

Never just say “my model works.”

Compare against:

- Random guessing
- Linear model
- MLP
- Bigram model
- PyTorch equivalent
- Smaller model

## Rule 4 — Plot results

Every major stage should have at least one plot:

- Loss curve
- Accuracy curve
- Confusion matrix
- Generated text samples
- Feature map visualization
- Gradient comparison

## Rule 5 — Write explanations in my own words

The README should prove understanding.

Bad explanation:

```txt
Backpropagation computes gradients.
```

Better explanation:

```txt
Backpropagation applies the chain rule from the loss backward through the computation graph. Each node receives the gradient of the loss with respect to its output, then uses its local derivative to pass gradients to its inputs.
```

## Rule 6 — Keep the first version simple

Do not start with the hardest version.

Correct order:

```txt
Linear regression → tiny autograd → tiny MLP → NumPy MLP → CNN → Bigram → TinyGPT
```

Wrong order:

```txt
Start with huge Transformer and debug everything at once.
```

---

# Final Deliverables

By the end, the repo should contain:

## Code

- Autograd engine
- MLP using autograd
- NumPy neural network library
- CNN training code
- Tiny Transformer
- Training loops
- Evaluation scripts
- Tests

## Documentation

- Main README
- Stage READMEs
- Math notes
- Experiment logs
- Results summary

## Results

- Loss curves
- Accuracy curves
- Gradient checks
- CNN results
- Transformer generated samples
- Experiment tables

## Resume Material

- Resume bullet
- Project summary
- Technical explanation
- Possible demo script

---

# Possible Final README Summary

This project is a four-stage deep learning lab designed to build neural network understanding from the ground up. I implemented an automatic differentiation engine, a NumPy-based neural network library, CNN image classifiers in PyTorch, and a GPT-style Transformer trained from scratch.

The project focuses on the mathematical and engineering foundations of deep learning: computation graphs, backpropagation, gradient descent, vectorized neural networks, convolution, self-attention, and sequence modeling. Each stage includes implementation notes, math explanations, tests, and experiments.

The goal was not to create another AI-powered app, but to understand how deep learning systems work internally.

---

# Future Extensions

After finishing the main project, possible extensions include:

## Extension 1 — Model Compression

Add pruning, quantization, and knowledge distillation.

Possible goals:

- Compress CNN
- Compress TinyGPT
- Compare size vs accuracy
- Measure inference time
- Try 8-bit and 4-bit quantization

## Extension 2 — Vision Transformer

Implement a small Vision Transformer and compare it against CNNs.

Possible goals:

- Patch embeddings
- Transformer encoder
- CIFAR-10 classification
- Attention visualization
- CNN vs ViT comparison

## Extension 3 — Representation Learning Lab

Implement:

- Autoencoder
- Denoising autoencoder
- Variational autoencoder
- Contrastive learning
- Latent space visualization

## Extension 4 — Distributed Training Mini-Lab

Implement:

- Gradient accumulation
- Mixed precision
- Data parallel training
- Checkpointing
- Multi-GPU if available

## Extension 5 — Inference Optimization

Implement:

- Batching
- KV cache for Transformer generation
- TorchScript or ONNX export
- Latency benchmarking
- CPU vs GPU inference comparison

---

# Personal Rule

The goal is not to finish fast.

The goal is to become the kind of person who can look at a neural network and understand what is happening mathematically and computationally.

Every time I get stuck, I should ask:

```txt
Is this a Python problem?
Is this a shape problem?
Is this a math problem?
Is this an optimization problem?
Is this a data problem?
```

That debugging mindset is part of the project.
