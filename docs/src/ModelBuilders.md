# Building a model 

SciMLFisheries has one primary function for building a fisheries assessment model, `SurplusProduction.` This function only requires a data set to build a model but offers several keyword arguments to allow users to modify the model structure, incorperate prior information, and optimize model performance. The function of these arguments depends on the model type and is discussed in detail below.


```@docs
SciMLFisheries.SurplusProduction(data;kwargs ...)
```


## Production models
The core of the surplus production model is a function that estimates the growth rate of the population in each time step ``r_t``. SciMLFisheries had six choices in production models that fall into one of three primary categories: time delay embedding models, recurrent neural networks, and feed-forward neural networks.  


### Time Delay Embedding
Time delay embedding models estimate the growth rate at time ``t``  time step as a function of the current biomass ``B_t`` and prior observations of the biomass and fishing mortality up to time ``t-\tau`` where ``tau`` is the "embedding" dimension, a use specified parameter that determines how much of the population’s history is included in the model.


The simplest model uses a neural network parameterized with weight ``w`` and biases ``b`` to approximate a function that maps from the prior observations to the population growth rate.


```math
r_t = NN(B_t,\{B_{t-i}\}_{i=1}^{\tau}, \{H_{t-i}\}_{i=1}^{\tau};w,b).
```


We can build a delay embedding model using the SurplusProduction function by specifying "DelayEmbedding" as the     `production_model` argument. The Delay embedding model had five hyperparameters that can be tuned to optimize the model's performance. In this example, all five parameters are listed in the NamedTuple `produciton_hyper_parameters` and set to their default values. The embedding dimension ``\tau`` is set equal to `lags` and must take an integer value. The next argument, `hidden,` influences how complex (degree of nonlinearity) the production function can be by controlling the number of hidden units in the neural network. The argument `seed` initializes the random number generator used to sample the initial neural network parameters, and the parameters `extrap_value` and `extrap_length` determine the behavior of the model predictions when extrapolating outside of the observed data set. The predicted growth rate will revert to `extrap_value` in the limit as the abundance of the stock becomes small, and it will revert to `-1*extra_value` in the limit as the stock becomes large. `extrap_length` determines how quickly the model reverts to this default behavior outside the range of historical observations. When forecasting, the production model is determined by the fitted neural network ``F(B,...)`` and the extrapolation parameters

```math
r_t =   \left\{
\begin{array}{ll}
      F(B,...) & min(B_t) < B < max(B_t) \\
      \omega(B)F(B,...) + (1-\omega(B)) \rho & B < min(B_t) \\
      \omega(B)F(B,...) - (1-\omega(B)) \rho & B > max(B_t) \\
\end{array} 
\right.  \\
\omega(B) = e^{-\frac{1}{2}\left( \frac{min(min(B_t)-B),B - max(B_t)}{extrap\_length} \right)^2}
```

where ``min(B_t)`` is the smallest observation in the data set and ``max(B_t)`` is the largest. We can also specify the functional form and weight for regularizing the neural network parameters when building the models, using the `regualrizaiton_type` and `regularization_weight` arguments. When using the default L2 regularization, the sum of squares penalizes the neural network weights, and L1 uses the sum of absolute values times the `regularization_weight.`

```julia
using UniversalDiffEq
model = SurplusProduction(data,
                        production_model = "DelayEmbedding", # options Int
                        produciton_hyper_parameters = (lags=5,hidden=10,seed=123,extrap_value=0.0,extrap_length=0.5),
                        regularizaiton_type = "L2", # options ["L2", "L1"]
                        regularizaiton_weight = 10^-4 # options Real {x | x >= 0}
                        )
```

Overfitting and variable selection are common issues for delay embedding models. We may wish to account for many possible time lags and allow for nonlinear functional forms in the model, but these model features also add many degrees of freedom that allow the model to identify patterns in the training data that do not generalize. This limitation can be addressed by controlling the complexity of the neural network in the standard delay embedding model through the choice of regularization parameters, the number of lags included in the model, and the number of hidden units in the neural network. However, we also developed alternative model structures that take additional approaches to solving the overfitting problem. The approach adds a dropout layer to the neural network. Drop-out layers randomly switch some of the neural network weights to zero during the training process; the model is prevented from learning overly complex functions that depend on the interaction between network parameters, reducing overfitting. To build a model that includes a drop-out layer, choose `"DelayEmbeddingDropOut"` as the model production model. The dropout model has all the same hyperparameters as the standard delay embedding model, plus an additional parameter `drop_prob` that defines the probability network weights are set to zero during the training process. The following cell builds a model with a drop-out layer setting all hyperparameters to their default values, `drop_prob` is included as a keyword argument to show how it can be modified.

```julia
using UniversalDiffEq
model = SurplusProduction(data,
                        production_model = "DelayEmbeddingDropOut", # options Int
                        produciton_hyper_parameters = (drop_prob=0.1))
```

Regularization and dropout can control the degree of nonlinearity incorporated in the model, but they do not directly control the number of inputs. To handle this challenge, we developed a regularization method that attempts to detect the relevant model inputs automatically and sets the effect of all other inputs to zero. This is done by multipying the neural network inputs ``X_t = {B_t, B_{t-1}, ..., F_{t-1}, F_{t-2}, ...}`` by a vector ``I_r``, before the are fed into the neural network. This produces a model of the form


```math
   r_t = NN(I_r \circ X_t; w, b)
```
On its own, this model structure does not reduce overfitting, but it allows us to directly regularize the network inputs. Specifically, we use L1 regularization on the input vector ``I_r`` and L2 regularization on the neural network weights. L1 regularization will force parameters to equal zero if they are not contributing adequately to the model performance; this allows the model to detect and remove irrelevant variables automatically. Applying L2 regularization to the network weights controls the degree of non-linearity of the fitted model. This regularization scheme can independently address the two primary degrees of freedom that lead to overfitting, variable selection, and nonlinearity. To construct a model of the form, use `"DelayEmbeddingARD"` as the process model. All keyword arguments are the same as the standard Delay embedding model, except it is possible to specify differnt weights for the L1 and L2 regularization steps by passing a named tuple to the `regularizaiton_weight` argument. If only a single value is supplied it will be applied to both L1 and L2 staged


```julia
using UniversalDiffEq
model = SurplusProduction(data,
                        production_model = "DelayEmbeddingDropOut", # options Int
                        regularizaiton_weight = (L1 = 10^-3.5, L2 = 10^-3.5))
```


### Recurrent nerual networks


### Feed Forward Networks


## Observation models 

### Discrete approximation 

### Linear approximation 


## Uncertianty quantification 

### Process errors

### Observation errors

### Priors  