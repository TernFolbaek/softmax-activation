# Softmax

## Define softmax function
We begin by definiting the soft maz function:
The softmax function can be represented as:

$$
\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_{j=1}^{n} e^{z_j}}
$$

Where:
- $z_i$ is the input element.
- $e^{z_i}$ represents the exponential of the input.
- $\sum_{j=1}^{n} e^{z_j}$ is the sum of all exponentials of the elements in the input vector $z$.

## Plot the values
We create a plot function where we plot the output of values which are inserted into our softmax function

## Create Blobs
We will be creating blobs for our data samples, each blob will be centered around their own specfic coordinates

## Define Unpreffered Model
We will define our model which will be of  3 layers, the first layer containing 25 neurons, then 15, then 4, where the activation function
for the last layer will be softmax.

## Define Preffered Model
This preffered model will be using the linear activation function for the last layer, this is to allow our compile method to use (from_logits=true). Using logits (logits=True) allows for better numerical stability when passing the output directly to a loss function like tf.nn.softmax_cross_entropy_with_logits, avoiding potential issues with very small values. We use linear activation (no activation) for the last layer because it outputs raw scores, which the loss function can then transform into probabilities internally, ensuring both precision and efficiency in training.

## Conclusion

You can see the difference between the outputs from the preffered and the unpreffered models, where the outputs of the preffered model give a more distinct answer.