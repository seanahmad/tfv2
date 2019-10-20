# Sections 1, 2, and 3 - TensorFlow 2.0 essentials: What's new (Python)

## Section 1: Installation and Migration

This section provides an overview of the key differences between TensorFlow v1.0 and v2.0 (hereafter referrred to as TF 1.0 and TF 2.0).

Specifically, you will learn:

- How to convert TF 1.0 code to the TF 2.0 standard
- Main differences between TF 1.0 and TF 2.0 syntax
- What is eager exectuion and how to implement it

## Section 2: Model Compilation and Execution

This section introduces the power of AutoGraph.

AutoGraph allows for conversion of standard Python code to TensorFlow graph code - which significantly enhances performance when running complex models.

The advantages and disadvantages of AutoGraph relative to eager execution will also be discussed.

## Section 3: Applied Machine Learning Methods with TensorFlow

With *tf.keras* now the default API for building machine learning models, this section explores how to generate a machine learning model with this API in order to solve a classification problem using machine learning.

Specifically, a neural network is used to predict hotel cancellations based on various features for each customer, e.g. lead time, country of origin, deposit type, etc.

You will see how to:

- Implement feature selection
- The cornerstone principles behind running neural networks
- Differences between executing such models across TF 1.0 and TF 2.0
- How to validate and test model predictions

This example builds upon a previous study by original authors Antonio, Almeida, and Nunes (2018).

Sources:

- https://www.sciencedirect.com/science/article/pii/S2352340918315191?via%3Dihub
- https://www.researchgate.net/publication/329286343_Hotel_booking_demand_datasets