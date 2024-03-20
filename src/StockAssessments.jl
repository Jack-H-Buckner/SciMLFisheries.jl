
include("ProductionModels.jl")
include("DataModels.jl")
include("Regularization.jl")
include("Priors.jl")
include("Likelihoods.jl")

function init_loss(times,dt_final,data,predict,process_loss,link,observation_loss,process_regularization,observation_regularization)
    
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
    loss_function
    constructor
end

"""
    SurplusProduction(data;kwargs ...)

Initailizes a surplus production model to fit to the data set.

...
# Arguments
 - data, a DataFrames.jl data frame with columns for time `t`, harvest `H`, and a relative abundance index `y`.

# Key words
- production_model, a string specifying the produciton model. Default = "DelayEmbedding", 
    ## options:
    - "DelayEmbedding" 
    - "FeedForward"
    -  "LSTM"
    - "DelayEmbeddingARD"
    - "DelayEmbeddingDropOut"
    - "LSTMDropOut
- harvest_model, a string specifying the harvest model. Default = "DiscreteAprox"
    ## options 
        - "DiscreteAprox"
        - "LinearAprox"
- index_model
    ## options 
    - "Linear"
    - "Nonlinear
- regularizaiton_type
    ## options
    - "L1"
    - "L2"
- regularizaiton_weight = 10.0^-4,
- loss="MSE",
- process_weights = [2.0,0.1],
- observation_weights = [0.5,0.5],
- produciton_hyper_parameters = NamedTuple(),
- prior_q = 0.0,
- prior_b = 1.0,
- prior_weight = 0.0
...
"""
function SurplusProduction(data;
        production_model = "DelayEmbedding",
        harvest_model = "DiscreteAprox",
        regularizaiton_type = "L1",
        regularizaiton_weight = 10.0^-4,
        index_model="Linear",
        loss="FixedVariance",
        process_weights = [0.5,1.0],
        observation_weights = [0.25,0.1],
        produciton_hyper_parameters = NamedTuple(),
        prior_q = 0.0,
        prior_b = 1.0,
        prior_weight = 0.0
    )
    # process data
    times,data,dataframe,T = process_surplus_production_data(data)

    # update default hyper-paramters with user inputs 
    new_produciton_hyper_parameters = ComponentArray(produciton_hyper_parameters)
    produciton_hyper_parameters = ComponentArray((lags=5,hidden=10,cell_dim=10,seed=1,drop_prob=0.1,extrap_value=0.1,extrap_length=0.25))
    produciton_hyper_parameters[keys(new_produciton_hyper_parameters)] .= new_produciton_hyper_parameters
    
    # production model 
    predict,parameters,forecast_F,forecast_H,process_loss,loss_params = ProductionModel(production_model,loss,process_weights[1],process_weights[2],T,produciton_hyper_parameters)
    
    # observaiton model 
    link,observation_loss,loss_params_obs,link_params=DataModel(harvest_model,index_model,loss,observation_weights[1],observation_weights[2],T)

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
    
    # loss function 
    dt_final = times[end-1]-times[end]
    loss_function = init_loss(times,dt_final,data,predict,process_loss,link,observation_loss,process_regularization,observation_regularization)

    # parameters
    parameters = ComponentArray((uhat = zeros(size(data)),predict = parameters, process_loss = loss_params, link = link_params, observation_loss = loss_params_obs))

    constructor = data -> SurplusProduction(data;production_model=production_model,harvest_model=harvest_model,regularizaiton_type=regularizaiton_type,regularizaiton_weight=regularizaiton_weight,index_model=index_model,loss=loss,process_weights=process_weights,observation_weights=observation_weights,produciton_hyper_parameters=produciton_hyper_parameters,prior_q=prior_q,prior_weight=prior_weight)
   
    return SurplusProduction(times,dt_final,data,dataframe,[],parameters,predict,forecast_F,forecast_H,link,loss_function,constructor)

end 