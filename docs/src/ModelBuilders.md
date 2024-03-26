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
   r_t = \[   \left\{
\begin{array}{ll}
     \omega(B)F(B,...) + (1-\omega(B)) extrap\_value & B < min(B_t) \\
     F(B,...) &  min(B_t) \leq B_t \leq max(B_t) \\
     \omega(B)F(B,...) - (1-\omega(B)) extrap\_value & B >  max(B_t) 
\end{array}
\right. \] \\
w(B) = e^{-\frac{1}{2}\left( \frac{min(min(B_t)-B),B - max(B_t))^2}{extrap\_length} \right)^2}
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


### Recurrent nerual networks



### Feed Forward Networks


## Observation models 

### Discrete approximation 

### Linear approximation 


## Uncertianty quantification 

### Process errors

### Observation errors

### Priors  