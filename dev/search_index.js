var documenterSearchIndex = {"docs":
[{"location":"ModelBuilders/#Building-a-model","page":"Building a model","title":"Building a model","text":"","category":"section"},{"location":"ModelBuilders/","page":"Building a model","title":"Building a model","text":"SciMLFisheries has one primary function for building a fisheries assessment model, SurplusProduction. This function only requires a data set to build a model but offers several keyword arguments to allow users to modify the model structure, incorperate prior information, and optimize model performance. The function of these arguments depends on the model type and is discussed in detail below.","category":"page"},{"location":"ModelBuilders/","page":"Building a model","title":"Building a model","text":"SciMLFisheries.SurplusProduction(data;kwargs ...)","category":"page"},{"location":"ModelBuilders/#SciMLFisheries.SurplusProduction-Tuple{Any}","page":"Building a model","title":"SciMLFisheries.SurplusProduction","text":"SurplusProduction(data;kwargs ...)\n\nInitailizes a surplus production model to fit to the data set with a colum for time, harvest and the abundnace index. \n\ntable 1: example data set \n\nt y H\n0 1.0 0.1\n1 0.95 0.15\n2 0.925 0.125\n... ... ...\n\nA number of key work arguments are used to modify the models behavior. Each of the key words specifies a specific model sturcture or model behavior, see the section on model types for details. \n\nSurplusProduction(data;\n        production_model = \"DelayEmbedding\", # options = [\"FeedForward\",\"LSTM\",\"DelayEmbeddingARD\",\"DelayEmbeddingDropOut\",\"LSTMDropOut\"]\n        harvest_model = \"DiscreteAprox\", # options = [\"FeedForward\"]\n        index_model = \"Linear\", # index_model = [\"Nonlinear\"]\n        regularizaiton_type = \"L2\", # options = [\"L1\"]\n        regularizaiton_weight = 10^-4, # options Real\n        loss = \"FixedVariance\", # options = [\"EstimateVariance\"]\n        process_weights = [0.5,1.0], # options:  Vector{Real}\n        observation_weights = [0.25,0.1], # options: Vector{Real}\n        produciton_hyper_parameters = NamedTuple(), # options: Naned tuple with keys (lags=Int,hidden=Int,cell_dim=Int,seed=Int,drop_prob=Real in [0,1],extrap_value=Real,extrap_length=Real)\n        prior_q = 0.0, # options: Real\n        prior_b = 0.0 # options: Real\n        prior_weight = 0.0 # options = Real\n        variance_priors = NamedTuple() # named tuple with keys (var_y=Real,sigma_y=Real,rH=Real,sigma_rH=Real,rB=Real,sigma_rB=Real,rF=Real,sigma_rF=Real)\n    )\n\n\n\n\n\n","category":"method"},{"location":"ModelBuilders/#Production-models","page":"Building a model","title":"Production models","text":"","category":"section"},{"location":"ModelBuilders/","page":"Building a model","title":"Building a model","text":"The core of the surplus production model is a function that estimates the growth rate of the population in each time step r_t. SciMLFisheries had six choices in production models that fall into one of three primary categories: time delay embedding models, recurrent neural networks, and feed-forward neural networks.  ","category":"page"},{"location":"ModelBuilders/#Time-Delay-Embedding","page":"Building a model","title":"Time Delay Embedding","text":"","category":"section"},{"location":"ModelBuilders/","page":"Building a model","title":"Building a model","text":"Time delay embedding models estimate the growth rate at time t  time step as a function of the current biomass B_t and prior observations of the biomass and fishing mortality up to time t-tau where tau is the \"embedding\" dimension, a use specified parameter that determines how much of the population’s history is included in the model.","category":"page"},{"location":"ModelBuilders/","page":"Building a model","title":"Building a model","text":"The simplest model uses a neural network parameterized with weight w and biases b to approximate a function that maps from the prior observations to the population growth rate.","category":"page"},{"location":"ModelBuilders/","page":"Building a model","title":"Building a model","text":"r_t = NN(B_tB_t-i_i=1^tau H_t-i_i=1^tauwb)","category":"page"},{"location":"ModelBuilders/","page":"Building a model","title":"Building a model","text":"We can build a delay embedding model using the SurplusProduction function by specifying \"DelayEmbedding\" as the     production_model argument. The Delay embedding model had five hyperparameters that can be tuned to optimize the model's performance. In this example, all five parameters are listed in the NamedTuple produciton_hyper_parameters and set to their default values. The embedding dimension tau is set equal to lags and must take an integer value. The next argument, hidden, influences how complex (degree of nonlinearity) the production function can be by controlling the number of hidden units in the neural network. The argument seed initializes the random number generator used to sample the initial neural network parameters, and the parameters extrap_value and extrap_length determine the behavior of the model predictions when extrapolating outside of the observed data set. The predicted growth rate will revert to extrap_value in the limit as the abundance of the stock becomes small, and it will revert to -1*extra_value in the limit as the stock becomes large. extrap_length determines how quickly the model reverts to this default behavior outside the range of historical observations. When forecasting, the production model is determined by the fitted neural network F(B) and the extrapolation parameters","category":"page"},{"location":"ModelBuilders/","page":"Building a model","title":"Building a model","text":"r_t =   left\nbeginarrayll\n      F(B)  min(B_t)  B  max(B_t) \n      omega(B)F(B) + (1-omega(B)) rho  B  min(B_t) \n      omega(B)F(B) - (1-omega(B)) rho  B  max(B_t) \nendarray \nright  \nomega(B) = e^-frac12left( fracmin(min(B_t)-B)B - max(B_t)extrap_length right)^2","category":"page"},{"location":"ModelBuilders/","page":"Building a model","title":"Building a model","text":"where min(B_t) is the smallest observation in the data set and max(B_t) is the largest. We can also specify the functional form and weight for regularizing the neural network parameters when building the models, using the regualrizaiton_type and regularization_weight arguments. When using the default L2 regularization, the sum of squares penalizes the neural network weights, and L1 uses the sum of absolute values times the regularization_weight.","category":"page"},{"location":"ModelBuilders/","page":"Building a model","title":"Building a model","text":"using UniversalDiffEq\nmodel = SurplusProduction(data,\n                        production_model = \"DelayEmbedding\", # options Int\n                        produciton_hyper_parameters = (lags=5,hidden=10,seed=123,extrap_value=0.0,extrap_length=0.5),\n                        regularizaiton_type = \"L2\", # options [\"L2\", \"L1\"]\n                        regularizaiton_weight = 10^-4 # options Real {x | x >= 0}\n                        )","category":"page"},{"location":"ModelBuilders/","page":"Building a model","title":"Building a model","text":"Overfitting and variable selection are common issues for delay embedding models. We may wish to account for many possible time lags and allow for nonlinear functional forms in the model, but these model features also add many degrees of freedom that allow the model to identify patterns in the training data that do not generalize. This limitation can be addressed by controlling the complexity of the neural network in the standard delay embedding model through the choice of regularization parameters, the number of lags included in the model, and the number of hidden units in the neural network. However, we also developed alternative model structures that take additional approaches to solving the overfitting problem. The approach adds a dropout layer to the neural network. Drop-out layers randomly switch some of the neural network weights to zero during the training process; the model is prevented from learning overly complex functions that depend on the interaction between network parameters, reducing overfitting. To build a model that includes a drop-out layer, choose \"DelayEmbeddingDropOut\" as the model production model. The dropout model has all the same hyperparameters as the standard delay embedding model, plus an additional parameter drop_prob that defines the probability network weights are set to zero during the training process. The following cell builds a model with a drop-out layer setting all hyperparameters to their default values, drop_prob is included as a keyword argument to show how it can be modified.","category":"page"},{"location":"ModelBuilders/","page":"Building a model","title":"Building a model","text":"using UniversalDiffEq\nmodel = SurplusProduction(data,\n                        production_model = \"DelayEmbeddingDropOut\", # options Int\n                        produciton_hyper_parameters = (drop_prob=0.1))","category":"page"},{"location":"ModelBuilders/","page":"Building a model","title":"Building a model","text":"Regularization and dropout can control the degree of nonlinearity incorporated in the model, but they do not directly control the number of inputs. To handle this challenge, we developed a regularization method that attempts to detect the relevant model inputs automatically and sets the effect of all other inputs to zero. This is done by multipying the neural network inputs X_t = B_t B_t-1  F_t-1 F_t-2  by a vector I_r, before the are fed into the neural network. This produces a model of the form","category":"page"},{"location":"ModelBuilders/","page":"Building a model","title":"Building a model","text":"   r_t = NN(I_r circ X_t w b)","category":"page"},{"location":"ModelBuilders/","page":"Building a model","title":"Building a model","text":"On its own, this model structure does not reduce overfitting, but it allows us to directly regularize the network inputs. Specifically, we use L1 regularization on the input vector I_r and L2 regularization on the neural network weights. L1 regularization will force parameters to equal zero if they are not contributing adequately to the model performance; this allows the model to detect and remove irrelevant variables automatically. Applying L2 regularization to the network weights controls the degree of non-linearity of the fitted model. This regularization scheme can independently address the two primary degrees of freedom that lead to overfitting, variable selection, and nonlinearity. To construct a model of the form, use \"DelayEmbeddingARD\" as the process model. All keyword arguments are the same as the standard Delay embedding model, except it is possible to specify differnt weights for the L1 and L2 regularization steps by passing a named tuple to the regularizaiton_weight argument. If only a single value is supplied it will be applied to both L1 and L2 staged","category":"page"},{"location":"ModelBuilders/","page":"Building a model","title":"Building a model","text":"using UniversalDiffEq\nmodel = SurplusProduction(data,\n                        production_model = \"DelayEmbeddingARD\", # options Int\n                        regularizaiton_weight = (L1 = 10^-3.5, L2 = 10^-3.5))","category":"page"},{"location":"ModelBuilders/#Recurrent-nerual-networks","page":"Building a model","title":"Recurrent nerual networks","text":"","category":"section"},{"location":"ModelBuilders/","page":"Building a model","title":"Building a model","text":"Another way to include prior observations in the model predictions is with recurrent neural networks (RNN). RNNs predict the value of a set of hidden variables h_t each time step based on the value of the hidden variables in the prior time step h_t-1 and the biomass and fishing mortality. The value of the hidden states is combined with biomass to predict each time step. The model learns a function f_h(B_tF_th_t) to update the hidden states along with a function f_r(B_th_t) to make prediction r_t using the hidden states The full model can be written as a system of two equations, one for the hidden states and one for the predictions","category":"page"},{"location":"ModelBuilders/","page":"Building a model","title":"Building a model","text":"h_t = F_h(B_tF_th_t) \nr_t = F_r(B_th_t)","category":"page"},{"location":"ModelBuilders/","page":"Building a model","title":"Building a model","text":"In the training process, the model can learn what information it needs to use from prior observations to make accurate predictions and store that information in the hidden states.","category":"page"},{"location":"ModelBuilders/","page":"Building a model","title":"Building a model","text":"There are many possible architectures for recurrent neural networks, but we have chosen to focus on long short-term memory networks (LSTMs) since they are known to perform well on a variety of tasks. SciMLFisheries has two LSTM architectures, a standard LSTM model \"LSTM\" and an LSTM with dropout \"LSTMDropOut\". For LSTM models, the number of hidden variables is chosen by adding cell_dim to the productionhyperparameters argument. All other key work arguments have their usual function except for hidden, which does not affect the model.","category":"page"},{"location":"ModelBuilders/","page":"Building a model","title":"Building a model","text":"using UniversalDiffEq\nmodel = SurplusProduction(data,\n                       production_model = \"LSTM\", # options Int\n                       produciton_hyper_parameters = (cell_dim=10))","category":"page"},{"location":"ModelBuilders/","page":"Building a model","title":"Building a model","text":"When a dropout layer is included, the drop_prob parameter determines the probability a weight is set to zero in the training process.","category":"page"},{"location":"ModelBuilders/","page":"Building a model","title":"Building a model","text":"using UniversalDiffEq\nmodel = SurplusProduction(data,\n                       production_model = \"LSTMDropOut\", # options Int\n                       produciton_hyper_parameters = (cell_dim = 10, drop_prob = 0.1))","category":"page"},{"location":"ModelBuilders/#Feed-Forward-Networks","page":"Building a model","title":"Feed-Forward Networks","text":"","category":"section"},{"location":"ModelBuilders/","page":"Building a model","title":"Building a model","text":"the feed-forward class of models uses a feed-forward neural network to represent a nonlinear relationship between the current stock and the growth rate without including any prior observations. These are effectively non-parametric versions of standard surplus production models. Feed-forward networks take all of the same keyword arguments as the standard delay embedding model, except for lags, which does not apply.","category":"page"},{"location":"ModelBuilders/","page":"Building a model","title":"Building a model","text":"using UniversalDiffEq\nmodel = SurplusProduction(data,\n                       production_model = \"FeedForward\",\n                       produciton_hyper_parameters = (hidden=10,seed=123,extrap_value=0.0,extrap_length=0.5),\n                       regularizaiton_type = \"L2\", # options [\"L2\", \"L1\"]\n                       regularizaiton_weight = 10^-4 # options Real {x | x >= 0}\n                       )","category":"page"},{"location":"ModelBuilders/#Observation-models","page":"Building a model","title":"Observation models","text":"","category":"section"},{"location":"ModelBuilders/","page":"Building a model","title":"Building a model","text":"The observation model describes two data sources, harvest H and an abundance index y. There are two options for each of these data sources for a total of four possible models. We assume the true harvest process can be described in continuous time by integrating the product of fishing mortality and biomass over time between observations. The two harvest models provide alternative approximations for this integral. The index models represent the relationship between the population biomass and the abundnace index. The default model \"Linear\" assumes a proportional relationship, and the alternative \"HyperStability\" allows for a small amount of nonlinearity.","category":"page"},{"location":"ModelBuilders/","page":"Building a model","title":"Building a model","text":"The two observations models \"DiscreteAprox\" and \"LinearAprox\" approximate the integral by assuming fishing mortality is constant over the time interval; \"DiscreteAprox\" assumes biomass is constant as well while \"LinearAprox\" assumes the stock grows or declines exponentially. ","category":"page"},{"location":"ModelBuilders/#Harvest:-discrete-approximation","page":"Building a model","title":"Harvest: discrete approximation","text":"","category":"section"},{"location":"ModelBuilders/","page":"Building a model","title":"Building a model","text":"The discrete approximation to the harvest model is used by default. It approximates the fishing mortality and biomass as constants across each time period. The keyword argument theta is associated with both harvest models. It determines the proportion of catch reported in the harvest statistics; theta < 1.0 implies that some fish that are caught by the fishery are not included in the landings statistics. This may be useful for fisheries where the harvest is calculated by sampling a subset of anglers, which is common in recreational fisheries or in fisheries where a portion of the catch is discarded.","category":"page"},{"location":"ModelBuilders/","page":"Building a model","title":"Building a model","text":"using UniversalDiffEq\nmodel = SurplusProduction(data,\n                       harvest_model = \"DiscreteAprox\",\n                       theta = 1.0 # the fraction of the catch that is reported. options: Real\n                        )","category":"page"},{"location":"ModelBuilders/#Harvest:-linear-approximation","page":"Building a model","title":"Harvest: linear approximation","text":"","category":"section"},{"location":"ModelBuilders/","page":"Building a model","title":"Building a model","text":"The linear approximation assumes that fishing mortality is constant but assumes the abundance of the stock to grow or decay exponentially over the time step. This approximation is useful for stocks with high growth rates when the length between observations is long. The stock’s exponential growth rate is determined by the growth rate estimate r_t and fishing mortality F_t","category":"page"},{"location":"ModelBuilders/","page":"Building a model","title":"Building a model","text":"H_t approx int_t^t+Delta ttheta F_t B(t)e^(r-Ftheta)u du = theta F_t B_t frace^(r_t-F_t)Deltat - 1r_t-F_t","category":"page"},{"location":"ModelBuilders/","page":"Building a model","title":"Building a model","text":"A model based on the linear approximation to the harvest model is built by supplying \"LinearAprox\" to the harvest model argument.","category":"page"},{"location":"ModelBuilders/","page":"Building a model","title":"Building a model","text":"using UniversalDiffEq\nmodel = SurplusProduction(data,harvest_model = \"LinearAprox\")","category":"page"},{"location":"ModelBuilders/#Index:-linear","page":"Building a model","title":"Index: linear","text":"","category":"section"},{"location":"ModelBuilders/","page":"Building a model","title":"Building a model","text":"The default model for the relative abundance index assumes the index y_t proportional to the abundance of the stock. The model-building function automatically log transforms the abundance index so the proportional relationship becomes additive.","category":"page"},{"location":"ModelBuilders/","page":"Building a model","title":"Building a model","text":"y_t = log(B_t) + q","category":"page"},{"location":"ModelBuilders/","page":"Building a model","title":"Building a model","text":"The choice in the index model is determined by the value given to the index_model keyword argument. Two additional parameters determine the behavior of the index model, prior_q and prior_weight. These two parameters specify a prior distribution for the scaling parameter q. prior_q is the a priori expected value, and prior_sigma is the standard deviation of a normal distribution. The default sets prior_q = 0, which implies that the abundance index is equal to abundance, and prior_sigma = Inf, which implies that the priors do not affect the parameter estimates.   ","category":"page"},{"location":"ModelBuilders/","page":"Building a model","title":"Building a model","text":"using UniversalDiffEq\nmodel = SurplusProduction(data,\n                             index_model=\"Linear\",\n                             prior_q = 0.0,\n                             prior_sigma = 0.1)","category":"page"},{"location":"ModelBuilders/#Index:-Hyperstabillity","page":"Building a model","title":"Index: Hyperstabillity","text":"","category":"section"},{"location":"ModelBuilders/","page":"Building a model","title":"Building a model","text":"The abundance index may be more sensitive to changes in the stock biomass when the stock is scarce or when the stock is abundant. We can account for this source of non-linearity by adding an exponent b to the index model.  ","category":"page"},{"location":"ModelBuilders/","page":"Building a model","title":"Building a model","text":"y_t = b log(B_t) + q","category":"page"},{"location":"ModelBuilders/","page":"Building a model","title":"Building a model","text":"When b is less than one, the index is more sensitive to changes in abundnace when the stock is rare, and when it is greater than one, it is more sensitive to changes in abundance when the stock is large. Passing \"HyperStability\" to the index_model argument will build a model using this index model. When using the Hyperstability model, additional keyword arguments are available to set a prior distribution over the exponent b. The prior mean for b is given by prior_b, and the prior weights for both q and b are set by passing a NameTuple to the prior_sigma argument with keys q and b specifying the priors for the two parameters. Otherwise the model defualts to uniformative priors. ","category":"page"},{"location":"ModelBuilders/","page":"Building a model","title":"Building a model","text":"using UniversalDiffEq\nmodel = SurplusProduction(data,\n                             index_model=\"HyperStability\",\n                             prior_q = 0.0,\n                             prior_b = 1.0,\n                             prior_sigma = (q = 0.1, b = 0.05))","category":"page"},{"location":"ModelBuilders/#Uncertianty-quantification","page":"Building a model","title":"Uncertianty quantification","text":"","category":"section"},{"location":"ModelBuilders/#Process-errors","page":"Building a model","title":"Process errors","text":"","category":"section"},{"location":"ModelBuilders/#Observation-errors","page":"Building a model","title":"Observation errors","text":"","category":"section"},{"location":"ModelBuilders/#Priors","page":"Building a model","title":"Priors","text":"","category":"section"},{"location":"Modeltesting/#Testing-model-performance","page":"Testing model performance","title":"Testing model performance","text":"","category":"section"},{"location":"ModelEvaluation/#Using-the-fitted-model","page":"Using the fitted model","title":"Using the fitted model","text":"","category":"section"},{"location":"#SciMlFisheries","page":"SciMlFisheries","title":"SciMlFisheries","text":"","category":"section"},{"location":"","page":"SciMlFisheries","title":"SciMlFisheries","text":"Many interacting factors, including harvest, physical climate and oceanographic conditions, and species interactions, can influence the abundance of fish populations. Two important goals in fisheries science are understanding how these complex interactions influence fish population dynamics and designing management strategies that account for them. We aim to address this challenge by combining standard fisheries models with novel machine-learning techniques.","category":"page"},{"location":"","page":"SciMlFisheries","title":"SciMlFisheries","text":"Machine learning methods have been very successful at reconstructing the dynamics of complex systems, including popautlion dynamics. Machine learning techniques have been so effective because they are flexible and, therefore, can represent complex nonlinear interactions when they are present. Machine learning techniques can also incorporate information from past observations to inform predictions, which is very helpful when that data set does not include all relevant state variables. This may often be the case in fisheries science because fish populations are embedded in complex ecosystems and interact with species that may or may not be observed directly.","category":"page"},{"location":"","page":"SciMlFisheries","title":"SciMlFisheries","text":"Here, we aim to develop new models that combine standard fisheries modeling techniques with machine learning tools to leverage the potential advantages of these methods while accounting for the multiple sources of uncertainty common in fisheries data. The models are built using a surplus production or \"Biomass dynamics” framework, where machine learning is used to estimate the production function.  These machine learning-based production models can incorporate prior observations that implicitly account for species interactions and other ecosystem feedbacks that operate on extended time scales. Models like these that combine machine learning with parametric models are often called Scientific Machine Learning, which lends its name to our package.","category":"page"},{"location":"#Model-structure","page":"SciMlFisheries","title":"Model structure","text":"","category":"section"},{"location":"","page":"SciMlFisheries","title":"SciMlFisheries","text":"SciMLFisheries uses a state space modeling framework. State space models are a class of time series models that describe a sequence of observations by combining two models: 1) the process model that describes how the state of a system changes over time, and 2) the data model that describes how the state of a system determines the observations. Combining the two models allows the state space model to account for noisy, imperfectly measured data and random variability in the system state.","category":"page"},{"location":"","page":"SciMlFisheries","title":"SciMlFisheries","text":"The surplus production models describe two data sources, harvest H_t and a relative abundance index y_t. Using these two data sources, the models estimate two state variables, the population biomass B_t and fishing mortality rate F_t. There are two built-in observation models for the abundance index. The first assumes a proportional relationship between biomass and the abundance index with a scaling factor q and with normally distributed observaiton errors with variance sigma^2","category":"page"},{"location":"","page":"SciMlFisheries","title":"SciMlFisheries","text":"log(B_t) = log(y_t) - q + epsilon_yt\nepsilon_yt sim N(0sigma_y)","category":"page"},{"location":"","page":"SciMlFisheries","title":"SciMlFisheries","text":"The second model allows for some nonlinearity in the relationship by adding a third parameter b","category":"page"},{"location":"","page":"SciMlFisheries","title":"SciMlFisheries","text":"log(B_t) = blog(y_t) - q + epsilon_yt\nepsilon_yt sim N(0sigma_y)","category":"page"},{"location":"","page":"SciMlFisheries","title":"SciMlFisheries","text":"When b is less than one, the index is more sensitive to changes in biomass when the stock has a low abundance, and when b is greater than one, the index is more sensitive to changes when the stock is large.","category":"page"},{"location":"","page":"SciMlFisheries","title":"SciMlFisheries","text":"We also include two models for harvest. In general, harvest can be modeled in continuous time as a function of biomass and the fishing mortality rate","category":"page"},{"location":"","page":"SciMlFisheries","title":"SciMlFisheries","text":"H_t = int_tt+Delta t theta B(u)F(u)du","category":"page"},{"location":"","page":"SciMlFisheries","title":"SciMlFisheries","text":"where theta is a conversion factor that accounts for the portion of fish killed by the fishery that is not landed and counted in harvest statistics. Our modeling framework uses a discrete-time formulation, so biomass and fishing mortality are only estimated at a single point during each period, and we must approximate the integral in the harvest equations. The simplest approximation is the product of the fishing mortality, biomass, scaling parameter, and a normally distributed error term with variance sigma_H","category":"page"},{"location":"","page":"SciMlFisheries","title":"SciMlFisheries","text":"H_t = theta B_t F_t+ + epsilon_H_t \n    epsilon_Ht sim N(0sigma_H)","category":"page"},{"location":"","page":"SciMlFisheries","title":"SciMlFisheries","text":"We also provide an approximation that assumes fishing mortality is constant over the interval and that the population dynamics can be approximated over the interval with exponential growth (or decay). This results in a more complicated expression that includes the per capita growth rate of the population r_t and adds the additional assumption the abundance index i_t is measured at the beginning of the period","category":"page"},{"location":"","page":"SciMlFisheries","title":"SciMlFisheries","text":"H_t = theta F_t B_t (e^(r_t-F_t)* Delta t - 1 )(r_t-F_t) + epsilon_H_t \n    epsilon_Ht sim N(0sigma_H)","category":"page"},{"location":"","page":"SciMlFisheries","title":"SciMlFisheries","text":"The change in biomass between periods is determined by the per capita population growth rate r_t, the fishing mortality rate F_t, process errors nu_t, and the length of time between observations Delta t","category":"page"},{"location":"","page":"SciMlFisheries","title":"SciMlFisheries","text":"log(B_t+1) = log(B_t) + Delta t times (r_t - F_t) + nu_Bt \n    nu_Bt sim N(0sigma_B)","category":"page"},{"location":"","page":"SciMlFisheries","title":"SciMlFisheries","text":"The growth rates r_t are modeld as a function of the current biomass log(B_t) and the biomass and fishing mortality in earlier time periods  log(B_t-1B_t-1B_t-tau F_t-1 F_t-2F_t-tau) . This function is estimated using neural networks. SciMLFishieres provides several neural network architectures, which are discussed in the following section.","category":"page"},{"location":"","page":"SciMlFisheries","title":"SciMlFisheries","text":"The fishing mortality rates F_t are given a random walk prior with variance parameter sigma_F that controls how rapidly the estimated fishing mortality rates change over time. ","category":"page"},{"location":"","page":"SciMlFisheries","title":"SciMlFisheries","text":"log(F_t+1) = log(F_t-1) + nu_Ft \n    nu_Ft sim N(0sigma_F)","category":"page"},{"location":"","page":"SciMlFisheries","title":"SciMlFisheries","text":"The model building seciton that follows includes a detailed description of the options avaible for the production and observation models along with the code needed to implement them. ","category":"page"},{"location":"#Package-Contents","page":"SciMlFisheries","title":"Package Contents","text":"","category":"section"},{"location":"","page":"SciMlFisheries","title":"SciMlFisheries","text":"Pages = [\"index.md\",\"ModelBuilders.md\",\"Modeltesting.md\",\"ModelEvaluation.md\"]","category":"page"}]
}
