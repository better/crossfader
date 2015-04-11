What's this?
------------

An experimental model for robust dimensionality reduction of arbitrary data sets.
It lets you explore the full joint probability distribution as well as the marginal distributions with any parameter fixed.
It's also extremely fast at computing these probability distributions.
It assumes nothing about the distribution of the features, which can have any units and have any scale.

What can it be used for?
------------------------

* Predicting missing values
* Visualizing probability distributions
* Finding outliers
* Modelling uncertainties

Demo Time
---------

`Here's a demo <https://rawgit.com/erikbern/random-forests/master/demo.html>` trained on a bunch of different data sets.

How does it work?
-----------------

It builds an `autoencoder <http://en.wikipedia.org/wiki/Autoencoder>` that learns to reconstruct missing data.

To be able to work with any distributions, it reduces all inputs to a series of binary values.
Every feature is encoded as a binary feature vector by constructing splits from the empirical distribution of the training data.
The predicted probabilities for each split then gives the `CDF <http://en.wikipedia.org/wiki/Cumulative_distribution_function>` directly.

The autoencoder has a series of hidden bottleneck layers (typically 2-5 layers with 20-100 units).
One way to think of it is that the autoencoder finds a low-dimensional manifold in the high-dimensional space.
This manifold can be highly nonlinear due to the nonlinearities in the autoencoder.
The autoencoder then essentially learns a projection from the high dimensional space onto the manifold and another projection back to the original space.

