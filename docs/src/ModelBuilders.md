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


We can build a delay embedding model using the SurplusProduction function by specifying "DelayEmbedding" as the     `production_model` argument. The Delay embedding model had six hyperparameters that can be tuned to optimize the model's performance. In this example, all six parameters are listed in the NamedTuple `produciton_hyper_parameters` and set to their default values. The embedding dimension ``\tau`` is determined by the argument `lags` and must take an integer value. The next argument, `hidden,` influences how complex (degree of nonlinearity) the production function can be by controlling the number of hidden units in the neural network. The argument `seed` initializes the random number generator used to sample the initial neural network parameters, and the parameters `extrap_value` and `extrap_length` determine the behavior of the model predictions when extrapolating outside of the observed data set. The predicted growth rate will revert to `extrap_value` in the limit as the abundance of the stock becomes small, and it will revert to `-1*extra_value` in the limit as the stock becomes large. `extrap_length` determines how quickly the model reverts to this default behavior outside the range of historical observations. When forecasting, the production model is determined by the fitted neural network ``F(B,...)`` and the extrapolation parameters

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

where ``min(B_t)`` is the smallest observation in the data set and ``max(B_t)`` is the largest. The weight given to regularizing the neural network is determined by `regularizaiton_weight`. The functional form of regularization is set by a seperate key word argument `regularization_type`. Useres can choose between L1 (`"L1"`) and L2 regularization (`"L2"`). When using the default L2 regularization, the sum of squares penalizes the neural network weights, and L1 uses the sum of absolute values times the `regularization_weight.`

```julia
using UniversalDiffEq
model = SurplusProduction(data,
            production_model = "DelayEmbedding", # options Int
            regularizaiton_type = "L2", # options ["L2", "L1"]
            produciton_parameters = (lags=5,hidden=10,seed=123,extrap_value=0.0,extrap_length=0.5,regularizaiton_weight=10^-4)
            )
```

Overfitting and variable selection are common issues for delay embedding models. We may wish to account for many possible time lags and allow for nonlinear functional forms in the model, but these model features also add many degrees of freedom that allow the model to identify patterns in the training data that do not generalize. This limitation can be addressed by controlling the complexity of the neural network in the standard delay embedding model through the choice of regularization parameters, the number of lags included in the model, and the number of hidden units in the neural network. However, we also developed alternative model structures that take additional approaches to solving the overfitting problem. The approach adds a dropout layer to the neural network. Drop-out layers randomly switch some of the neural network weights to zero during the training process; the model is prevented from learning overly complex functions that depend on the interaction between network parameters, reducing overfitting. To build a model that includes a drop-out layer, choose `"DelayEmbeddingDropOut"` as the model production model. The dropout model has all the same hyperparameters as the standard delay embedding model, plus an additional parameter `drop_prob` that defines the probability network weights are set to zero during the training process. The following cell builds a model with a drop-out layer setting all hyperparameters to their default values, `drop_prob` is included as a keyword argument to show how it can be modified.

```julia
using UniversalDiffEq
model = SurplusProduction(data,
                        production_model = "DelayEmbeddingDropOut", # options Int
                        produciton_parameters = (drop_prob=0.1))
```

Regularization and dropout can control the degree of nonlinearity incorporated in the model, but they do not directly control the number of inputs. To handle this challenge, we developed a regularization method that attempts to detect the relevant model inputs automatically and sets the effect of all other inputs to zero. This is done by multipying the neural network inputs ``X_t = {B_t, B_{t-1}, ..., F_{t-1}, F_{t-2}, ...}`` by a vector ``I_r``, before the are fed into the neural network. This produces a model of the form

```math
   r_t = NN(I_r \circ X_t; w, b)
```
On its own, this model structure does not reduce overfitting, but it allows us to directly regularize the network inputs. Specifically, we use L1 regularization on the input vector ``I_r`` and L2 regularization on the neural network weights. L1 regularization will force parameters to equal zero if they are not contributing adequately to the model performance; this allows the model to detect and remove irrelevant variables automatically. Applying L2 regularization to the network weights controls the degree of non-linearity of the fitted model. This regularization scheme can independently address the two primary degrees of freedom that lead to overfitting, variable selection, and nonlinearity. To construct a model of the form, use `"DelayEmbeddingARD"` as the process model. All keyword arguments are the same as the standard Delay embedding model.

```julia
using UniversalDiffEq
model = SurplusProduction(data,
                        production_model = "DelayEmbeddingARD")
```

### Recurrent nerual networks

Another way to include prior observations in the model predictions is with recurrent neural networks (RNN). RNNs predict the value of a set of hidden variables ``h_t`` each time step based on the value of the hidden variables in the prior time step ``h_{t-1}`` and the biomass and fishing mortality. The value of the hidden states is combined with biomass to predict each time step. The model learns a function ``f_h(B_t,F_t,h_t)`` to update the hidden states along with a function ``f_r(B_t,h_t)`` to make prediction ``r_t`` using the hidden states The full model can be written as a system of two equations, one for the hidden states and one for the predictions
```math
h_t = F_h(B_t,F_t,h_t) \\
r_t = F_r(B_t,h_t).
```
In the training process, the model can learn what information it needs to use from prior observations to make accurate predictions and store that information in the hidden states.

There are many possible architectures for recurrent neural networks, but we have chosen to focus on long short-term memory networks (LSTMs) since they are known to perform well on a variety of tasks. SciMLFisheries has two LSTM architectures, a standard LSTM model `"LSTM"` and an LSTM with dropout `"LSTMDropOut".` For LSTM models, the number of hidden variables is chosen by adding `cell_dim` to the production_hyper_parameters argument. All other key work arguments have their usual function except for `hidden`, which does not affect the model.
```julia
using UniversalDiffEq
model = SurplusProduction(data,
                       production_model = "LSTM", # options Int
                       produciton_parameters = (cell_dim=10))
```
When a dropout layer is included, the `drop_prob` parameter determines the probability a weight is set to zero in the training process.
```julia
using UniversalDiffEq
model = SurplusProduction(data,
                       production_model = "LSTMDropOut", # options Int
                       produciton_parameters = (cell_dim = 10, drop_prob = 0.1))
```

### Feed-Forward Networks

the feed-forward class of models uses a feed-forward neural network to represent a nonlinear relationship between the current stock and the growth rate without including any prior observations. These are effectively non-parametric versions of standard surplus production models. Feed-forward networks take all of the same keyword arguments as the standard delay embedding model, except for `lags`, which does not apply.
```julia
using UniversalDiffEq
model = SurplusProduction(data,
                       production_model = "FeedForward")
```


## Observation models


The observation model describes two data sources, harvest ``H`` and an abundance index ``y``. There are two options for each of these data sources for a total of four possible models. We assume the true harvest process can be described in continuous time by integrating the product of fishing mortality and biomass over time between observations. The two harvest models provide alternative approximations for this integral. The index models represent the relationship between the population biomass and the abundnace index. The default model `"Linear"` assumes a proportional relationship, and the alternative `"HyperStability"` allows for a small amount of nonlinearity.

The two observations models `"DiscreteAprox"` and `"LinearAprox"` approximate the integral by assuming fishing mortality is constant over the time interval; `"DiscreteAprox"` assumes biomass is constant as well while `"LinearAprox"` assumes the stock grows or declines exponentially. 

### Harvest: discrete approximation

The discrete approximation to the harvest model is used by default. It approximates the fishing mortality and biomass as constants across each time period. The harvest model has one hyper parameter `theta`, which it determines the proportion of catch reported in the harvest statistics; `theta < 1.0` implies that some fish that are caught by the fishery are not included in the landings statistics. This may be useful for fisheries where the harvest is calculated by sampling a subset of anglers, which is common in recreational fisheries or in fisheries where a portion of the catch is discarded. The value of `theta` is set using the `harvest_parameters` keyword argument.


```julia
using UniversalDiffEq
model = SurplusProduction(data,
                       harvest_model = "DiscreteAprox",
                       harvest_parameters = (theta = 1.0,) # the fraction of the catch that is reported. options: Real
                  )
```

### Harvest: linear approximation

The linear approximation assumes that fishing mortality is constant but assumes the abundance of the stock to grow or decay exponentially over the time step. This approximation is useful for stocks with high growth rates when the length between observations is long. The stock’s exponential growth rate is determined by the growth rate estimate `r_t` and fishing mortality `F_t`
```math
H_t \approx \int_{t}^{t+\Delta t}\theta F_t B(t)e^{(r-F/\theta)u} du = \theta F_t B_t \frac{e^{(r_t-F_t)\Deltat} - 1}{r_t-F_t}.
```
A model based on the linear approximation to the harvest model is built by supplying `"LinearAprox"` to the harvest model argument.
```julia
using UniversalDiffEq
model = SurplusProduction(data,harvest_model = "LinearAprox")
```

### Index: linear

The default model for the relative abundance index assumes the index ``y_t`` proportional to the abundance of the stock. The model-building function automatically log transforms the abundance index so the proportional relationship becomes additive.
```math
y_t = log(B_t) + q.
```
The choice in the index model is determined by the value given to the `index_model` keyword argument. Two additional parameters specified using the `index_priors` key word determine the behavior of the index model, `q` and `sigma_q.` These two parameters specify a prior distribution for the scaling parameter ``q.`` 
```julia
using UniversalDiffEq
model = SurplusProduction(data,
                             index_model="Linear",
                             index_priors = (q = 0.0, sigma_q = 10.0)
                             )
```

### Index: Hyperstabillity 

The abundance index may be more sensitive to changes in the stock biomass when the stock is scarce or when the stock is abundant. We can account for this source of non-linearity by adding an exponent ``b`` to the index model.  
```math
y_t = b log(B_t) + q.
```
When `b` is less than one, the index is more sensitive to changes in abundnace when the stock is rare, and when it is greater than one, it is more sensitive to changes in abundance when the stock is large. Passing `"HyperStability"` to the `index_model` argument will build a model using this index model. When using the Hyperstability model there are two additional values that can be passed to the `index_priors` key word: `b` and `sigma_b` that specify the mean and variance of a normal prior distribution. 

```julia
using UniversalDiffEq
model = SurplusProduction(data,
                             index_model="HyperStability",
                             index_priors = (q = 0.0, sigma_q = 10.0, b = 1.0, sigma_b = 10.0))
```

## Uncertianty quantification 

The state space model structure used by SciMLFisheries models allows the models to quantify too account for two types of uncertianty, observational errors and process errors. Observaitonal errors decribe an imperfect relatioship between the value of th state variables, biomass ``B`` and fishing mortaltiy ``F`` and the observaitons, Harvest ``H`` and relative abundnace ``y``. Processes errors describe imperfect predictions of the change in state variables over time. Four variacne parameters quantify the effect of these two forms of unceritnaty ``\sigma_H`` and ``\sigma_y`` describe the observational errors associated with harvest and the abundance index and ``\sigma_B`` and ``sigma_F`` describe the process errors assocaited with biomass and fishing mortality. 

There are two options for handeling uncertianty, fixed variances or estimated variances. The model type is specified by the key word argument `likelihood`, the string `"FixedVariance"` will initialize a model with fixed variances and `"EstimateVariance"` will initialize a model that estimates the variance. The value of the varinace parameters are set using the `variance_priors` key word argument which is a named tuple. For the fixe variance case the arguments are `sigma_H` for the harvest observaiton error,   `sigma_y` for the index observaiton error, `sigma_B` for the biomass process errors and `sigma_F` for the fishing mortaltiy process errors. 

```julia
using UniversalDiffEq
model = SurplusProduction(data,
                         likelihood="FixedVariance",
                         variance_priors = (sigma_H=0.1, sigma_y = 0.1, sigma_B = 0.05, sigma_F = 0.2))
```

For models with estimated variance terms we allow gamma prior distributions to be set on the index observaiton errors ``\sigma_y`` with mean ``\hat{\sigma}_y`` and standard deviation ``sd(\sigma_y)``. Priors for the remainin varance terms are set as a ratio ``r_{i:y}`` of the index observation errors. These priors also take the form of gamma distributions with means ``\hat{r_{i:y}}`` and standard deviation ``sd(r_{i:y})``.

```math
\sigma_y \sim gamma(\hat{\sigma}_y, sd(\sigma_y)) \\
\frac{\sigma_H}{\sigma_y} \sim gamma(\hat{r_{H:y}}, sd(r_{H:y})) \\
\frac{\sigma_B}{\sigma_y} \sim gamma(\hat{r_{B:y}}, sd(r_{B:y})) \\
\frac{\sigma_F}{\sigma_y} \sim gamma(\hat{r_{F:y}}, sd(r_{F:y})) 
```

The parameters for the prior distributions are set using the `variance_priors` argument as illustrated below. 

```julia
using UniversalDiffEq
model = SurplusProduction(data,
                         likelihood="EstimateVariance",
                         variance_priors = (sigma_y = 0.1, sd_sigma_y=0.05,rH=0.25,sd_rH=0.025,rB=1.0,sd_rB=0.1,rF=5.0,sd_rF=0.2))
```
# Model fitting

Currently SciMLFisheries has two algorithms to estimate the model paramters, `gradient_decent!` and `BFGS!`. the gradient decent function using the ADAM algorithm to fid the maximimum likeihood esitmate of the model parameters, while BFGS acciomplishes the same task with a quasi-Newton algorithm. In general the best results are achieved by first running `gradient_decent!` and then `BFGS!`. You can track the progress of both algorithms by setting the key word `verbose = true`, the number of iteratins and step size for gradent decent are adjusted with the key words `maxiter` and `step_size`.

```julia
gradient_decent!(UDE; step_size = 0.05, maxiter = 500, verbose = false)
```

The BFGS algorithm has a single tuning parameter `initial_step_norm` with a default value of 0.01.
 
```julia
BFGS!(UDE; verbose = true, initial_step_norm = 0.01)
```