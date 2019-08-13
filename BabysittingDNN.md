# Babysitting Process of Training Neural Networks

## Possible Pre-processing you must be aware of.

- Data Normalization.
- Whitening (projection in the first eigenvectors of Singular Value Decomposition U,S,V = <X.T,X> -> <X,U>).
- Reduce Noise <X,U> -> <X,U[:,:first_eigenvecs]>.
- Batch Normalization.
  - The core observation is that this is possible because normalization is a simple differentiable operation. In the implementation, applying this technique usually amounts to insert the BatchNorm layer immediately after layers
- Initialization and Activation Function Choosing
  - Relevant for weights and gradients not dying out (see below)

### Gradient Checking

- Important since will choosen initialization and activation function can lead to zero gradient.
  - E.G: Sigmoid basis has zero gradient directions, ReLU can die in negative part and is not simetrical along 0.
  - LeakyReLU does not die out. 
  - Some others activations has nice properties for gradients but are computational expensive.
 
### Regularization 

- Prevents overfitting, increases generalization capacities.
  - L2 regularization: enforce linear decay on weights  W += - lambda*W.
  - L1 regularization: do worst them L2 but is better for feature selection since it give sparse vectors (easy separable).
  - Dropout: cut of connections during the training with probability p.
  
### Batch Normalization

- BatchNorm layer immediately after fully connected layers or convolutional layers, and before non-linearities.
  - Smooth gradients
  - Helps to use bigger learning rates.
  
## Babysitting

- I recommend you to do the babysitting process following the [TensorBoard Interface](https://github.com/tensorflow/tensorboard/blob/master/README.md) for TensorFlow. You can find a [CheatSheet here](https://github.com/Uiuran/Terminal-Dev-Utils/edit/master/tensorflow.md)

### Sanity Checks
  
- Make sure you can overfit small portions of data.
  - Use gradative weight of Regularization parameters to look for the regularization behavior.
    
### Loss-Function 

- NaNs usually means high learning-rate. Not going down too low learning-rate.

### Hyper-Parametrization
