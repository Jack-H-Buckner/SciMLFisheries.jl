# SciMlFisheries

Many interacting factors, including harvest, physical climate and oceanographic conditions, and species interactions, can influence the abundance of fish populations. Two important goals in fisheries science are understanding how these complex interactions influence fish population dynamics and designing management strategies that account for them. We aim to address this challenge by combining standard fisheries models with novel machine-learning techniques.


Machine learning methods have been very successful at reconstructing the dynamics of complex systems, including popautlion dynamics. Machine learning techniques have been so effective because they are flexible and, therefore, can represent complex nonlinear interactions when they are present. Machine learning techniques can also incorporate information from past observations to inform predictions, which is very helpful when that data set does not include all relevant state variables. This may often be the case in fisheries science because fish populations are embedded in complex ecosystems and interact with species that may or may not be observed directly.


Here, we aim to develop new models that combine standard fisheries modeling techniques with machine learning tools to leverage the potential advantages of these methods while accounting for the multiple sources of uncertainty common in fisheries data. The models are built using a surplus production or "Biomass dynamics” framework, where machine learning is used to estimate the production function.  These machine learning-based production models can incorporate prior observations that implicitly account for species interactions and other ecosystem feedbacks that operate on extended time scales. Models like these that combine machine learning with parametric models are often called Scientific Machine Learning, which lends its name to our package.


## Model structure 

SciMLFisheries uses a state space modeling framework. State space models are a class of time series models that describe a sequence of observations by combining two models: 1) the process model that describes how the state of a system changes over time, and 2) the data model that describes how the state of a system determines the observations. Combining the two models allows the state space model to account for noisy, imperfectly measured data and random variability in the system state.

The surplus production models describe two data sources, harvest ``H_t`` and a relative abundance index ``y_t``. Using these two data sources, the models estimate two state variables, the population biomass ``B_t`` and fishing mortality rate ``F_t``. There are two built-in observation models for the abundance index. The first assumes a proportional relationship between biomass and the abundance index with a scaling factor ``q`` and with normally distributed observaiton errors with variance ``\sigma^2``
```math
log(B_t) = t_t - q + \epsilon_{y,t}\\
\epsilon_{y,t} \sim N(0,\sigma_y).
```
The second model allows for some nonlinearity in the relationship by adding a third parameter ``b``
```math
log(B_t) = b y_t - q + \epsilon_{y,t}\\
\epsilon_{y,t} \sim N(0,\sigma_y).
```
When ``b`` is less than one, the index is more sensitive to changes in biomass when the stock has a low abundance, and when ``b`` is less than one, the index is more sensitive to changes when the stock is large.

We also include two models for harvest. In general, harvest can be modeled in continuous time as a function of biomass and the fishing mortality rate
```math
H_t = \int_{t}{t+\Delta t} \theta B(u)F(u)du,
```
where ``\theta`` is a conversion factor that accounts for the portion of fish killed by the fishery that is not landed and counted in harvest statistics. Our modeling framework uses a discrete-time formulation, so biomass and fishing mortality are only estimated at a single point during each period, and we must approximate the integral in the harvest equations. The simplest approximation is the product of the fishing mortality, biomass, scaling parameter, and a long normal error term with variance ``\sigma_H``
```math
log(H_t) = log(B_t) + log(F_t) + log(\theta) + \epsilon_{H_t} \\
    \epsilon_{H,t} \sim N(0,\sigma_H).
```
We also provide an approximation that assumes fishing mortality is constant over the interval and that the population dynamics can be approximated over the interval with exponential growth (or decay). This results in a more complicated expression that includes the per capita growth rate of the population ``r_t`` and adds the additional assumption the abundance index ``i_t`` is measured at the beginning of the period
```math
H_t = log(\theta) + log(F_t) +  log(B_t) + log(e^{(r_t-F_t)* \Delta t} - 1 ) - log(r_t-F_t) + \epsilon_{H_t} \\
    \epsilon_{H,t} \sim N(0,\sigma_H).
```

The change in biomass between periods is determined by the per capita population growth rate ``r_t``, the fishing mortality rate ``F_t``, process errors ``\nu_t``, and the length of time between observations ``\Delta t``
```math
log(B_{t+1}) = log(B_t) + \Delta t \times (r_t - F_t) + \nu_{B,t} \\
    \nu_{B,t} \sim N(0,\sigma_B)
```
The growth rates ``r_t`` are modeld as a function of the current biomass ``log(B_t)`` and the biomass and fishing mortality in earlier time periods ``{ log(B_{t-1},B_{t-1},...,\B_{t-\tau}, F_{t-1}, F_{t-2},...,F_{t-\tau}) }``. This function is estimated using neural networks. SciMLFishieres provides several neural network architectures, which are discussed in the following section.


The fishing mortality rates ``F_t`` are given a random walk prior with variance parameter ``\sigma_F`` that controls how rapidly the estimated fishing mortality rates change over time. 
```math
log(F_{t+1}) = log(F_{t-1}) + \nu_{F,t} \\
    \nu_{F,t} \sim N(0,\sigma_F).
```

## Produciton models


## Package Contents
```@contents
Pages = ["index.md","ModelBuilders.md","Modeltesting.md","ModelEvaluation.md"]
```