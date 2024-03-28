
include("ProductionModels.jl")
include("DataModels.jl")
include("Regularization.jl")
include("Priors.jl")
include("Likelihoods.jl")

function init_loss(times,dt_final,data,predict,process_loss,link,observation_loss,process_regularization,observation_regularization,variance_prior)
    
    function loss_function(parameters)
        
        # initialize process model 
        ut = parameters.uhat[:,1]
        dt = times[2] - times[1]
        uhat, r, aux = predict(ut,dt,parameters.predict) 
        
        # calcualte loss for first observation 
        yhat = link(parameters.uhat[:,1],r,dt,parameters.link)
        L_obs = observation_loss(data[:,1], yhat,parameters.observation_loss)
        
        # initalize process loss accumulator 
        L_proc = 0 
        for t in 2:(size(data)[2])
            # calcualte forecasting error 
            ut = parameters.uhat[:,t]
            L_proc += process_loss(ut,uhat,parameters.process_loss)
            if t < size(data)[2] 
                # calcualte forecast and obervational loss using time between observations
                dt = times[t] - times[t-1]
                uhat, r, aux = predict(ut,aux,dt,parameters.predict) 
                yhat = link(ut,r,dt,parameters.link)
                L_obs += observation_loss(data[:,t], yhat,parameters.observation_loss)
            else
                # calcualte forecast and obervational loss using final value of delta t
                uhat, r, aux = predict(ut,aux,dt_final,parameters.predict) 
                yhat = link(ut,r,dt_final,parameters.link)
                L_obs += observation_loss(data[:,t], yhat,parameters.observation_loss) 
            end
        end
        
        # regularization
        L_reg = process_regularization(parameters.predict)
        L_reg += observation_regularization(parameters.link)
        L_reg += variance_prior(parameters.observation_loss, parameters.process_loss)

        return L_obs + L_proc + L_reg
    end
    
end 

mutable struct SurplusProduction
    times
    dt_final
    data
    dataframe
    X
    parameters
    predict
    forecast_F
    forecast_H
    link
    observation_loss
    process_loss
    loss_function
    constructor
end

"""
    SurplusProduction(data;kwargs ...)

Initailizes a surplus production model to fit to the data set with a colum for time, harvest and the abundnace index. 
    
table 1: example data set 

|t  | y  |H   |
|---|----|----|
|0  | 1.0|0.1 |
|1  |0.95|0.15|
|2  |0.925|0.125|
|...|...|...|    

A number of key work arguments are used to modify the models behavior. Each of the key words specifies a specific model sturcture or model behavior, see the section on model types for details. 

```julia
SurplusProduction(data;
        production_model = "DelayEmbedding", # options = ["FeedForward","LSTM","DelayEmbeddingARD","DelayEmbeddingDropOut","LSTMDropOut"]
        harvest_model = "DiscreteAprox", # options = ["FeedForward"]
        index_model = "Linear", # index_model = ["Nonlinear"]
        regularizaiton_type = "L2", # options = ["L1"]
        regularizaiton_weight = 10^-4, # options Real
        loss = "FixedVariance", # options = ["EstimateVariance"]
        process_weights = [0.5,1.0], # options:  Vector{Real}
        observation_weights = [0.25,0.1], # options: Vector{Real}
        produciton_hyper_parameters = NamedTuple(), # options: Naned tuple with keys (lags=Int,hidden=Int,cell_dim=Int,seed=Int,drop_prob=Real in [0,1],extrap_value=Real,extrap_length=Real)
        prior_q = 0.0, # options: Real
        prior_b = 0.0 # options: Real
        prior_weight = 0.0 # options = Real
        variance_priors = NamedTuple() # named tuple with keys (var_y=Real,sigma_y=Real,rH=Real,sigma_rH=Real,rB=Real,sigma_rB=Real,rF=Real,sigma_rF=Real)
    )
```

"""
function SurplusProduction(data;
        production_model = "DelayEmbedding",
        harvest_model = "DiscreteAprox",
        regularizaiton_type = "L1",
        regularizaiton_weight = 10.0^-4,
        index_model="Linear",
        likelihood="FixedVariance",
        process_weights = [0.5,1.0],
        observation_weights = [0.25,0.1],
        produciton_hyper_parameters = NamedTuple(),
        theta = 1.0,
        prior_q = 0.0,
        prior_b = 1.0,
        prior_weight = 0.0,
        variance_priors = NamedTuple()
    )
    # process data
    times,data,dataframe,T = process_surplus_production_data(data)

    # update default hyper-paramters with user inputs 
    new_produciton_hyper_parameters = ComponentArray(produciton_hyper_parameters)
    produciton_hyper_parameters = ComponentArray((lags=5,hidden=10,cell_dim=10,seed=1,drop_prob=0.1,extrap_value=0.1,extrap_length=0.25))
    produciton_hyper_parameters[keys(new_produciton_hyper_parameters)] .= new_produciton_hyper_parameters
    
    # production model 
    predict,parameters,forecast_F,forecast_H,process_loss,loss_params = ProductionModel(production_model,likelihood,process_weights[1],process_weights[2],produciton_hyper_parameters)
    
    # observaiton model
    link,observation_loss,loss_params_obs,link_params=DataModel(harvest_model,index_model,likelihood,observation_weights[1],observation_weights[2],theta)

    # production regularization
    if (production_model == "DelayEmbeddingARD") && (typeof(regularizaiton_weight) == Float64)
        regularizaiton_weight = (L1 = regularizaiton_weight, L2 = regularizaiton_weight)
    end 
    process_regularization = Regularization(regularizaiton_type,production_model,regularizaiton_weight)

    # observation model priors
    observation_regularization = q_prior(prior_q,prior_weight)
    if (index_model == "HyperStability") && (typeof(prior_weight) == Float64)
        prior_weight = (q = prior_weight, b = prior_weight)
    end 

    if index_model == "HyperStability"
        observation_regularization = q_and_b_prior(prior_q,prior_b,prior_weight)
    end 
    
    # variance prior
    variance_prior = (observation,process) -> 0.0
    if likelihood=="EstimateVariance"
        new_variance_priors = ComponentArray(variance_priors)
        variance_priors = ComponentArray((var_y=0.05,sigma_y=0.05,rH=0.25,sigma_rH=0.025,rB=1.0,sigma_rB=0.1,rF=5.0,sigma_rF=0.25))
        variance_priors[keys(new_variance_priors)] .= new_variance_priors
        variance_prior = init_variance_prior(variance_priors.var_y, variance_priors.sigma_y, variance_priors.rH, variance_priors.sigma_rH, variance_priors.rB,variance_priors.sigma_rB, variance_priors.rF, variance_priors.sigma_rF)
    end 

    # loss function 
    dt_final = times[end-1]-times[end]
    loss_function = init_loss(times,dt_final,data,predict,process_loss,link,observation_loss,process_regularization,observation_regularization,variance_prior)

    # parameters
    parameters = ComponentArray((uhat = zeros(size(data)),predict = parameters, process_loss = loss_params, link = link_params, observation_loss = loss_params_obs))

    constructor = data -> SurplusProduction(data;production_model=production_model,harvest_model=harvest_model,regularizaiton_type=regularizaiton_type,regularizaiton_weight=regularizaiton_weight,index_model=index_model,likelihood=likelihood,process_weights=process_weights,observation_weights=observation_weights,produciton_hyper_parameters=produciton_hyper_parameters,prior_q=prior_q,prior_weight=prior_weight,variance_priors=variance_priors)
   
    return SurplusProduction(times,dt_final,data,dataframe,[],parameters,predict,forecast_F,forecast_H,link,observation_loss, process_loss,loss_function,constructor)

end 