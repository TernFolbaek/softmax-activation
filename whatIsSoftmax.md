## Softmax

#### Define softmax function
We begin by definiting the soft maz function:
The softmax function can be represented as:

$$
\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_{j=1}^{n} e^{z_j}}
$$

Where:

- \( z_i \) is the input element.
- \( e^{z_i} \) represents the exponential of the input.
- $\sum_{j=1}^{n} e^{z_j}$ is the sum of all exponentials of the elements in the input vector \( z \).
