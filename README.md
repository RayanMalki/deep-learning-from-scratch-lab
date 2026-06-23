# Deep Learning From Scratch Lab

Building the core machinery of modern deep learning from first principles — autograd, neural networks, CNNs, and a GPT-style Transformer — with as few high-level libraries as possible.

This is not an AI-powered app. There is no API wrapper and no web frontend. The point is to understand _how_ neural networks actually work by implementing the ideas behind them, one layer of abstraction at a time.

## Why I'm building this

I wanted to stop treating deep learning as a black box. It's easy to call `model.fit()` and watch a number go down; it's much harder to explain what every tensor, gradient, and update is actually doing. This project is my way of earning that understanding — by writing the backpropagation, the attention, and the training loops myself instead of importing them.

The guiding rule: I shouldn't move on from a stage until I can explain the math and the code behind it in my own words.

## Inspiration

The structure and philosophy of this project draw heavily on people who taught these ideas by building them from scratch:

- **Andrej Karpathy** — micrograd and nanoGPT, and the "build it yourself to understand it" approach
- **Andrew Ng** — foundational ML and deep learning intuition
- **Yoshua Bengio** — representation learning and deep learning theory

## What I'm building

The project moves deliberately from a single scalar gradient up to a working language model. Each stage builds on the last, and each one comes with its own notes, math explanations, and experiments measured against simple baselines.

### Stage 00 — Foundations

**What I'm coding:** Linear regression trained with gradient descent in pure NumPy — predictions, a mean squared error loss, manual gradients, and a plotted loss curve.

**What I'm learning:** How a model actually "learns" — how gradients drive parameter updates, why the loss goes down, and how to think in vectors and array shapes instead of loops.

**Inspiration:** Andrew Ng's machine learning courses, which frame everything around the loss function and gradient descent as the engine underneath it all.

### Stage 01 — Autograd Engine

**What I'm coding:** A small automatic differentiation engine built around a `Value` class that records operations into a computation graph, plus a tiny MLP trained on top of it with a real training loop.

**What I'm learning:** What `.backward()` is really doing — the chain rule applied node by node, why gradients must accumulate, and why a topological order is needed to propagate them correctly.

**Inspiration:** Andrej Karpathy's micrograd and his _Neural Networks: Zero to Hero_ series.

### Stage 02 — NumPy Neural Network

**What I'm coding:** A vectorized neural network library — dense layers with forward/backward passes, ReLU and other activations, softmax cross-entropy, optimizers (SGD, momentum, Adam), and gradient checking — trained on a real dataset like MNIST.

**What I'm learning:** How to move from scalar autograd to batched matrix math, how softmax and cross-entropy fit together, and how to verify my gradients are correct instead of just hoping they are.

**Inspiration:** Stanford's CS231n notes on backpropagation and neural networks.

### Stage 03 — CNN Vision

**What I'm coding:** Convolutional image classifiers in PyTorch, with training and evaluation loops, confusion matrices, and visualizations of filters and feature maps. Then a set of experiments — augmentation, dropout, batch norm, learning-rate changes — with honest write-ups of what worked and what didn't.

**What I'm learning:** What convolution actually computes, why CNNs beat MLPs on images, and how to analyze a model rather than just report its best accuracy.

**Inspiration:** CS231n and the classic vision architectures (LeNet, AlexNet, ResNet).

### Stage 04 — Tiny Transformer

**What I'm coding:** A GPT-style language model from scratch — a character tokenizer, a bigram baseline, then self-attention, multi-head attention, causal masking, and a full Transformer block, with training, sampling, and text generation.

**What I'm learning:** How self-attention works, what Q/K/V mean, why positional embeddings and causal masks are necessary, and how a model generates text one token at a time.

**Inspiration:** Andrej Karpathy's nanoGPT and the _Attention Is All You Need_ paper.

## Tech

Python · NumPy · PyTorch · Matplotlib

## Repository structure

```txt
deep-learning-from-scratch-lab/
├── 00-python-numpy-foundations/   # Linear regression, gradient descent
├── 01-autograd-engine/            # Scalar autograd + MLP
├── 02-numpy-neural-network/       # Vectorized NN library
├── 03-cnn-vision/                 # CNNs in PyTorch
├── 04-tiny-transformer/           # GPT-style Transformer
├── experiments/                   # Experiment logs and results
└── notes/                         # Math and concept notes
```

## Getting started

```bash
git clone <repo-url>
cd deep-learning-from-scratch-lab
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run the first stage:

```bash
python 00-python-numpy-foundations/linear_regression_numpy.py
```

## Status

Work in progress, built stage by stage. Each stage is treated as done only when it runs end to end, has documented results, and the math behind it is something I can explain — not just code that happens to work.
