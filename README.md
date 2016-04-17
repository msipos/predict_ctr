## Predict Click-Through-Rate from ad impression data

In this project, I try to predict clicks based on impression data.

The click data is very "sparse" in the sense that there is roughly 1 click in every 1000 impressions.

Considering the size of the dataset, I decided that I would try to use incremental (online) learning.  First step was to look at the features in the dataset.  All features that were too "diverse", such as `IP`, or `timestamp`, were dropped.  After dropping those and using "1 of n" encoding for all the categorical features, I ended up with a list of 124 features.

Also considering I wanted to use `scikit-learn`, the idea to use online learning meant that the number of algorithms available was relatively small.  It was limited to Naive Bayes and mini-batch stochastic gradient descent.  I unfortunately couldn't get SGD to work very well with log-loss (i.e. logistic regression).  I think I may not understand how to use SGD to learn in an online fashion, but I ran out of time to investigate this in detail.

To compare the algorithms, I trained on first 2 million data points and tested on the last 800000. The result ROC curves seem to indicate that Gaussian and Bernoulli Naive Bayes algorithms seem to perform well.

I then trained the final predictor on all 2.8 million data points and used it to predict probabilities for `test.csv`.

Unfortunately, I ran out of time to do anything more (I have by this point spent 6 or so hours on the project).  If I had more time I would have considered:

* Getting the SGD algorithm to work
* Using feature selection to cut down on features and size of the dataset
* Investigate using offline learning algorithms, and see how well they perform compare to NB.
* Perform a grid search to optimize algorithm parameters and AUROC.
