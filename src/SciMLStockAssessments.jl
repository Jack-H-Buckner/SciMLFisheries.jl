"""
"""
module SciMLStockAssessments


using Optimization, OptimizationOptimisers, OptimizationOptimJL, ComponentArrays, Zygote, Plots, LaTeXStrings, DataFrames, Lux, Random, Statistics, Distributions

include("Regularization.jl")


"""
"""
function harvest(B,r,F,dt)
    if (r-F) == 0 
        return dt*F*B
    end
    return dt*F*B #*(exp((r-exp(F))*dt)-1)/(r-exp(F)) #exp(F)*B #
end 


mutable struct FeedForwardGrowth
    NN # lux neural network object 
    parameters #::ComponentArray # nerual network paramters
    predict::Function # neural network 
    forecast
end 


function FeedForwardGrowth(;hidden = 10, seed = 1, dyn_reg = 0.01)
    
    # initial neurla Network
    NN = Lux.Chain(Lux.Dense(1,hidden,tanh), Lux.Dense(hidden,1))
    
    # parameters 
    Random.seed!(seed)  # set seed for reproducibility 
    rng = Random.default_rng() 
    parameters, states = Lux.setup(rng,NN) 
    
    parameters = (NN = parameters, rho = 0.0)
    
    
    function forecast(u,aux,dt,parameters,umax,umin)
        # eval network
        r = NN([u[1]],parameters.NN,states)[1][1]
        
        if u[1] > umax
            w = exp(-0.5*((u[1]-umax)/(umax-umin))^2)
            r = w*r -dyn_reg*(1-w)
        elseif u[1] < umin
            w = exp(-0.5*((u[1]-umin)/(umax-umin))^2)
            r = w*r + dyn_reg*(1-w)
        end 
        
        # update states
        x = u[1] .+ dt*r .- dt*u[2]
        f = u[2] 
        return [x,f], r,  0
    end
    
    function predict(u,dt,parameters) 
        # eval network
        r = NN([u[1]],parameters.NN,states)[1][1]
        # state transiitons
        x = u[1] .+ dt*r .- dt*u[2]
        f = u[2] 
        return [x,f], r,  0
    end 

    function predict(u,aux,dt,parameters) 
        # eval network
        r = NN([u[1]],parameters.NN,states)[1][1]
        # state transiitons
        x = u[1] .+ dt*r .- dt*u[2]
        f = u[2] 
        return [x,f], r,  0
    end 
    
    return FeedForwardGrowth(NN,parameters,predict,forecast)
    
end 


mutable struct LogisticGrowth
    parameters #::ComponentArray # nerual network paramters
    predict::Function # neural network 
    forecast
    forecast_H
end 


function LogisticGrowth(;K0 = 10.0, r0 = 0.5, dyn_reg = 0.01)
    
    parameters = (r = r0, K = K0)
    
    function forecast(u,aux,dt,parameters,umax,umin)
        # eval network
        r = parameters.r; K = parameters.K
        r = r*(1-exp(u[1])/K)
        
        if u[1] > umax
            w = exp(-0.5*((u[1]-umax)/(umax-umin))^2)
            r = w*r -dyn_reg*(1-w)
        elseif u[1] < umin
            w = exp(-0.5*((u[1]-umin)/(umax-umin))^2)
            r = w*r + dyn_reg*(1-w)
        end 
        
        # update states
        x = u[1] .+ dt*r- dt*u[2]
        f = u[2] #+ dt*parameters.rho*u[2]
        return [x,f],r,0
    end
    
    function forecast_H(u,aux,H,dt,parameters,umax,umin)
        # eval network
        
        r = parameters.r; K = parameters.K
        r = r*(1-exp(u[1])/K)
        
        if u[1] > umax
            w = exp(-0.5*((u[1]-umax)/(umax-umin))^2)
            r = w*r -dyn_reg*(1-w)
        elseif u[1] < umin
            w = exp(-0.5*((u[1]-umin)/(umax-umin))^2)
            r = w*r + dyn_reg*(1-w)
        end 
        
        x = u[1] .+ dt*r - H/exp(u[1])
        
        return [x],r,0
    end
    
    function predict(u,dt,parameters) 
        r = parameters.r; K = parameters.K
        r = r*(1-exp(u[1])/K)
        # state transiitons
        x = u[1] .+ dt*r- dt*u[2]
        f = u[2] #+ dt*parameters.rho*u[2]
        return [x,f],r,0
    end 

    function predict(u,aux,dt,parameters) 
        # eval network
        r = parameters.r; K = parameters.K
        r = r*(1-exp(u[1])/K)
        # state transiitons
        x = u[1] .+ dt*r .- dt*u[2]
        f = u[2] 
        return [x,f], r, 0
    end 
    
    return LogisticGrowth(parameters,predict,forecast,forecast_H)
    
end 


mutable struct LSTM
    net
    parameters #::ComponentArrasy # nerual network paramters
    predict::Function # neural network 
    forecast
end 

function LSTM(;cell_dim = 10, seed = 1,dyn_reg=0.01,l=0.5)
    
    # initial neurla Network
    LSTM_ = Lux.LSTMCell(1=>cell_dim)
    DenseLayer = Lux.Dense((cell_dim+1)=>1,tanh)
    
    # parameters 
    Random.seed!(seed)  # set seed for reproducibility 
    rng = Random.default_rng() 
    LSTM_parameters, LSTM_states = Lux.setup(rng,LSTM_) 
    dense_parameters, dense_states = Lux.setup(rng, DenseLayer)
    rng = Random.default_rng()  
    parameters = (Dense = dense_parameters, LSTM = LSTM_parameters, x0 = [0.0])
    
    function forecast(u,aux,dt,parameters,umax,umin)
        # eval network
        x = reshape(u[1:1],1,1); c, st_lstm, ut1, ft1 = aux
        rt1 = u[1] - ut1 + ft1
        (y, c), st_lstm = LSTM_((reshape([rt1],1,1),c),parameters.LSTM, st_lstm)
        r, states = DenseLayer(vcat(x,y),parameters.Dense,dense_states);
        
        if u[1] > umax
            w = exp(-0.5/l^2 * ((u[1]-umax)/(umax-umin))^2)
            r = [w*r[1] -dyn_reg*(1-w)]
        elseif u[1] < umin
            w = exp(-0.5/l^2 * ((u[1]-umin)/(umax-umin))^2)
            r = [w*r[1] + dyn_reg*(1-w)]
        end 
        
        x = u[1] + dt*r[1] - dt*u[2]
        f = u[2] 
        return [x,f],r[1],(c, st_lstm, u[1], u[2])
    end
    
    function predict(u,dt,parameters)
        # eval neural net
        x = reshape(u[1:1],1,1)
        (y, c), st_lstm = LSTM_(reshape(parameters.x0,1,1),parameters.LSTM, LSTM_states)
        r, states = DenseLayer(vcat(x,y),parameters.Dense,dense_states)
        # state transitions
        x = u[1] + dt*r[1] - dt*u[2]
        f = u[2] 
        return [x,f],r[1],(c, st_lstm, u[1], u[2])
    end 
    
    function predict(u,aux,dt,parameters)
        # evaluate neural network 
        x = reshape(u[1:1],1,1); c, st_lstm, ut1, ft1 = aux
        rt1 = u[1] - ut1 + ft1
        (y, c), st_lstm = LSTM_((reshape([rt1],1,1),c),parameters.LSTM, st_lstm)
        r, states = DenseLayer(vcat(x,y),parameters.Dense,dense_states);
        # calcualte state transitions
        x = u[1] + dt*r[1] - dt*u[2]
        f = u[2] 
        return [x,f], r[1], (c, st_lstm, u[1], u[2])
    end 
    
    return LSTM((LSTM_,DenseLayer),parameters,predict,forecast)
    
end 


function LSTMDropOut(;cell_dim = 10, seed = 1, drop_prob = 0.1)
    
    # initial neurla Network
    LSTM_ = Lux.LSTMCell(2=>cell_dim)
    DenseLayer = Lux.Chain( Lux.Dense( cell_dim +1 => cell_dim), 
                            Lux.Dropout(drop_prob), 
                            Lux.Dense(cell_dim=>1,tanh)
                            )
    
    # parameters 
    Random.seed!(seed)  # set seed for reproducibility 
    rng = Random.default_rng() 
    LSTM_parameters, LSTM_states = Lux.setup(rng,LSTM_) 
    dense_parameters, dense_states = Lux.setup(rng, DenseLayer)
    rng = Random.default_rng()  
    parameters = (Dense = dense_parameters,LSTM = LSTM_parameters, x0 = [4.0,0])
    
    function predict(u,dt,parameters)
        # eval neural net
        x = reshape(u[1:1],1,1)
        (y, c), st_lstm = LSTM_(reshape(parameters.x0,2,1),parameters.LSTM, LSTM_states)
        r, states = DenseLayer(vcat(x,y),parameters.Dense,dense_states)
        # state transitions
        x = u[1] + dt*r[1] - dt*u[2]
        f = u[2] 
        return [x,f],r[1],(c, st_lstm, u[1], u[2])
    end 
    
    function predict(u,aux,dt,parameters)
        # evaluate neural network 
        x = reshape(u[1:1],1,1); c, st_lstm, ut1, ft1 = aux
        (y, c), st_lstm = LSTM_((reshape([ut1,ft1],2,1),c),parameters.LSTM, st_lstm)
        r, states = DenseLayer(vcat(x,y),parameters.Dense,dense_states);
        # calcualte state transitions
        x = u[1] + dt*r[1] - dt*u[2]
        f = u[2] 
        return [x,f], r[1], (c, st_lstm, u[1], u[2])
    end 
    
    
    return LSTM((LSTM_,DenseLayer),parameters,predict)
    
end 







function K_prior(parameters,Khat)
    K = parameters.process_model.K - parameters.observation_model.q
    return (K-Khat)^2
end 

function r_prior(parameters,rhat)
    return (parameters.process_model.r-rhat)^2
end 

mutable struct NeuralNetTimeDelays
    lags::Int
    NN
    parameters #::ComponentArrasy # nerual network paramters
    predict::Function # neural network 
    forecast
    forecast_H
end 



function DelayEmbedding(;lags=5,hidden = 10, seed = 1, dyn_reg=0.01, l = 0.25)
    
    # initial neurla Network
    NN = Lux.Chain(Lux.Dense(1+2*lags,hidden,tanh), Lux.Dense(hidden,1))
    
    # parameters 
    Random.seed!(seed)  # set seed for reproducibility 
    rng = Random.default_rng() 
    NN_parameters, NN_states = Lux.setup(rng,NN) 
    parameters = (NN = NN_parameters, aux0 = zeros(2*lags),rho = 0.0)
    
    
    function forecast(u,aux,dt,parameters,umax,umin)
        # eval network
        lags,ut1,ft1=aux
        #rt1 = u[1] - ut1 + ft1
        lags = vcat([ut1,ft1],lags[1:(end-2)])
        r = NN(vcat(u[1:1],lags),parameters.NN,NN_states)[1][1]
        
        if u[1] > umax
            w = exp(-0.5/l^2*((u[1]-umax)/(umax-umin))^2)
            r = w*r -dyn_reg*(1-w)
        elseif u[1] < umin
            w = exp(-0.5/l^2*((u[1]-umin)/(umax-umin))^2)
            r = w*r + dyn_reg*(1-w)
        end 
        
        # update states
        x = u[1] .+ dt*r - dt*u[2]
        f = u[2] .+ parameters.rho*u[2]
        return [x,f],r, (lags,u[1],u[2])
    end
    
    
    function forecast_H(u,aux,H,dt,parameters,umax,umin)
        # eval network
        lags,ut1,ft1=aux
        
        lags = vcat([ut1,ft1],lags[1:(end-2)])
        r = NN(vcat(u[1:1],lags),parameters.NN,NN_states)[1][1]
        
        if u[1] > umax
            w = exp(-0.5/l^2*((u[1]-umax)/(umax-umin))^2)
            r = w*r -dyn_reg*(1-w)
        elseif u[1] < umin
            w = exp(-0.5/l^2*((u[1]-umin)/(umax-umin))^2)
            r = w*r + dyn_reg*(1-w)
        end 
        
        x = u[1] .+ dt*r - H/exp(u[1])
        
        return [x],r, (lags,u[1],H/exp(u[1]))
    end
    
    
    function predict(u,dt,parameters) 
        # eval network 
        r = NN(vcat(u[1:1],parameters.aux0),parameters.NN,NN_states)[1][1]
        # update states
        x = u[1] + dt*r - dt*u[2]
        f = u[2] + parameters.rho*u[2]
        return [x,f], r, (parameters.aux0,u[1],u[2])
    end 
    
    function predict(u,aux,dt,parameters) 
        # eval network
        lags,ut1,ft1=aux
        #rt1 = u[1] - ut1 + ft1
        lags = vcat([ut1,ft1],lags[1:(end-2)])
        r = NN(vcat(u[1:1],lags),parameters.NN,NN_states)[1][1]
        # update states
        x = u[1] .+ dt*r - dt*u[2]
        f = u[2] .+ parameters.rho*u[2]
        return [x,f],r, (lags,u[1],u[2])
    end 
    
    return NeuralNetTimeDelays(lags,NN,parameters,predict,forecast,forecast_H)
    
end 


mutable struct LinkFunction
    parameters
    link
end

function BiomassAndFishing()
        
    # parameters
    parameters = (q = 0.0, )
    vals = NamedTuple()
    
    # link 
    function link(u,r,dt,p)
        yt = u[1] .- p.q
        H = harvest(exp(u[1]),r,u[2],dt)
        return [yt,H]
    end
    
    return LinkFunction(parameters,link)
end 

mutable struct WeightedMSE
    parameters
    loss
end


function WeightedMSE(weights;N = 1, )
    parameters = NamedTuple()
    loss = (u,uhat,parameters) -> sum(weights .* (u .- uhat).^2)/N
    return WeightedMSE(parameters,loss)
end


function Normal(sigma_0;N = 1, )
    parameters = (sigma = sigma_0)
    
    function loss(u,uhat,parameters) 
        nugget = 10^-6.0
        Z = 1 ./ (sqrt(2*3.14159)*parameters.sigma .+ nugget)
        ll = -0.5 * (u .- uhat).^2 ./ (parameters.sigma .^2 .+ nugget)
        return sum(ll .+ log.(Z))  
    end
        
    return WeightedMSE(parameters,loss)
end

mutable struct FixedObsProcRatio
    process_loss
    observation_loss
    parameters
    proc_BH_ratios
    obs_BH_ratio
    PO_ratio
end 

function FixedObsProcRatio(weight_0, proc_BH_ratios, obs_BH_ratios, PO_ratio ;)
    parameters = (weight = weight_0,)
    
    function proc_loss(u,uhat,parameters) 
        sigmas = exp(-1*parameters.weight)
        sigmas = abs.(sigmas.*proc_BH_ratios)
        Z = 1 ./ (sqrt(2*3.14159)*sigmas)
        ll = -0.5 * (u .- uhat).^2 ./ sigmas .^2
        return -1*sum(ll .+ log.(Z))  
    end
    
    function obs_loss(u,uhat,parameters) 
        sigmas = exp(-1*parameters.weight)
        sigmas =  abs.(sigmas.*obs_BH_ratios*PO_ratio)
        Z = 1 ./ (sqrt(2*3.14159).*sigmas)
        ll = -0.5 * (u .- uhat).^2 ./ sigmas .^2
        return -1*sum(ll .+ log.(Z))  
    end
        
    return FixedObsProcRatio(proc_loss,obs_loss,parameters,proc_BH_ratios, obs_BH_ratios, PO_ratio)
end



function FixedObsProcRatio(weight_0, proc_BH_ratios, obs_BH_ratios;)
    parameters = (weight = weight_0,PO_ratio = 1.0,)
    
    function proc_loss(u,uhat,parameters) 
        sigmas = exp(-1*parameters.weight)
        sigmas = abs.(sigmas.*proc_BH_ratios)
        Z = 1 ./ (sqrt(2*3.14159)*sigmas)
        ll = -0.5 * (u .- uhat).^2 ./ sigmas .^2 .+ log(Z)
        return -1*sum(ll ) 
    end
    
    function obs_loss(u,uhat,parameters) 
        sigmas = exp(-1*parameters.weight)
        sigmas =  abs.(sigmas.*obs_BH_ratios*parameters.PO_ratio)
        Z = 1 ./ (sqrt(2*3.14159).*sigmas)
        ll = -0.5 * (u .- uhat).^2 ./ sigmas .^2 .+ log.(Z)
        return -1*sum(ll)  
    end
        
    return FixedObsProcRatio(proc_loss,obs_loss,parameters,proc_BH_ratios, obs_BH_ratios, "estimated")
end


function init_loss(times,dt_final,data,process_model,process_loss,observation_model,observation_loss,
                        process_regularization,observation_regularization)
    
    function loss_function(parameters)
        
        # initialize process model 
        ut = parameters.uhat[:,1]
        dt = times[2] - times[1]
        uhat, r, aux = process_model.predict(ut,dt,parameters.process_model) 
        
        # calcualte loss for first observation 
        yhat = observation_model.link(parameters.uhat[:,1],r,dt,parameters.observation_model)
        L_obs = observation_loss.loss(data[:,1], yhat,parameters.observation_model)
        
        # initalize process loss accumulator 
        L_proc = 0 
        for t in 2:(size(data)[2])
            # calcualte forecasting error 
            ut = parameters.uhat[:,t]
            L_proc += process_loss.loss(ut,uhat,parameters.process_loss)
            if t < size(data)[2] 
                # calcualte forecast and obervational loss using time between observations
                dt = times[t] - times[t-1]
                uhat, r, aux = process_model.predict(ut,aux,dt,parameters.process_model) 
                yhat = observation_model.link(ut,r,dt,parameters.observation_model)
                L_obs += observation_loss.loss(data[:,t], yhat,parameters.observation_model)
            else
                # calcualte forecast and obervational loss using final value of delta t
                uhat, r, aux = process_model.predict(ut,aux,dt_final,parameters.process_model) 
                yhat = observation_model.link(ut,r,dt_final,parameters.observation_model)
                L_obs += observation_loss.loss(data[:,t], yhat,parameters.observation_model) 
            end
        end
        
        # regularization
        L_reg = process_regularization.loss(parameters.process_model,parameters.process_regularization)
        L_reg += observation_regularization.loss(parameters.observation_model,parameters.observation_regularization)
        
        return L_obs + L_proc + L_reg
    end
    
end 

function init_obs_and_reg(data,proc_weights,obs_weights, reg_weight, prior_weight, q_prior)
    data = transpose(Matrix(data[:,2:size(data)[2]]))
    process_loss = WeightedMSE(proc_weights,N = size(data)[1]-1)
    observation_model = BiomassAndFishing()
    observation_loss = WeightedMSE(obs_weights,N = size(data)[1]-1)
    process_regularization = Regularization.L2(weight=reg_weight)
    observation_regularization = Regularization.q_prior(q_prior,prior_weight)
    return process_loss, observation_model,observation_loss,process_regularization,observation_regularization
end 


function init_normal_obs_and_reg(data,proc_sigma,obs_sigma,reg_weight, prior_weight, q_prior)
    data = transpose(Matrix(data[:,2:size(data)[2]]))
    process_loss = Noraml(proc_sigma,N = size(data)[1]-1)
    observation_model = BiomassAndFishing()
    observation_loss = Noraml(obs_sigma,N = size(data)[1]-1)
    process_regularization = Regularization.L2(weight=reg_weight)
    observation_regularization = Regularization.q_prior(q_prior,prior_weight)
    return process_loss, observation_model,observation_loss,process_regularization,observation_regularization
end 


function init_normal_fixed_ratio(data,proc_BH_ratios, obs_BH_ratios, PO_ratio,reg_weight, prior_weight, q_prior)
    data = transpose(Matrix(data[:,2:size(data)[2]]))
    observation_model = BiomassAndFishing()
    loss = FixedObsProcRatio(0.1, proc_BH_ratios, obs_BH_ratios, PO_ratio ;)
    process_regularization = Regularization.L2(weight=reg_weight)
    observation_regularization = Regularization.q_prior(q_prior,prior_weight)
    return loss, observation_model,process_regularization,observation_regularization
end 


function init_normal_fixed_ratio(data,proc_BH_ratios, obs_BH_ratios, reg_weight, prior_weight, q_prior)
    data = transpose(Matrix(data[:,2:size(data)[2]]))
    observation_model = BiomassAndFishing()
    loss = FixedObsProcRatio(0.1, proc_BH_ratios, obs_BH_ratios;)
    process_regularization = Regularization.L2(weight=reg_weight)
    observation_regularization = Regularization.q_prior(q_prior,prior_weight)
    return loss, observation_model,process_regularization,observation_regularization
end 


function gamma_lpdf(x,k,theta)
    log(x)*(k-1)-x/theta
end 


function init_loss_FixedRatio(times,dt_final,data,process_model,observation_model,loss,
                        process_regularization,observation_regularization,k_sigma,theta_sigma)
    
    
    
    
    function loss_function(parameters)
        
        # initialize process model 
        ut = parameters.uhat[:,1]
        dt = times[2] - times[1]
        uhat, r, aux = process_model.predict(ut,dt,parameters.process_model) 
        
        # calcualte loss for first observation 
        yhat = observation_model.link(parameters.uhat[:,1],r,dt,parameters.observation_model)
        L_obs = loss.observation_loss(data[:,1], yhat,parameters.loss)
    
        
        # initalize process loss accumulator 
        L_proc = 0 
        for t in 2:(size(data)[2])
            # calcualte forecasting error 
            ut = parameters.uhat[:,t]
            L_proc += loss.process_loss(ut,uhat,parameters.loss)
            if t < size(data)[2] 
                # calcualte forecast and obervational loss using time between observations
                dt = times[t] - times[t-1]
                uhat, r, aux = process_model.predict(ut,aux,dt,parameters.process_model) 
                yhat = observation_model.link(ut,r,dt,parameters.observation_model)
                L_obs += loss.observation_loss(data[:,t], yhat,parameters.loss)
            else
                # calcualte forecast and obervational loss using final value of delta t
                uhat, r, aux = process_model.predict(ut,aux,dt_final,parameters.process_model) 
                yhat = observation_model.link(ut,r,dt_final,parameters.observation_model)
                L_obs += loss.observation_loss(data[:,t], yhat,parameters.loss) 
            end
        end
        
        # regularization
        L_reg = process_regularization.loss(parameters.process_model,parameters.process_regularization)
        L_reg += observation_regularization.loss(parameters.observation_model,parameters.observation_regularization)
        L_reg += -1*sum(gamma_lpdf.(exp.(-1*parameters.loss.weight),k_sigma,theta_sigma))
        
        return L_obs + L_proc + L_reg
    end
    
end 




mutable struct UDE
    times
    dt_final
    data
    X
    data_frame
    parameters
    loss_function
    process_model
    process_loss 
    observation_model
    observation_loss 
    process_regularization
    observation_regularization
    constructor
end



function FeedForward(data;hidden_units=10,NN_seed = 1,proc_weights=[1.0,0.1], obs_weights = [1.0,1.0], 
        reg_weight = 10^-5,prior_weight = 0.05, q_prior=0.0, dyn_reg = 0.01)
    
    # convert data
    data_frame = data
    times = data.t
    data = transpose(Matrix(data[:,2:size(data)[2]]))
    dim1,dim2 = size(data)
    dt_final = sum(times[2:end] .- times[1:(end-1)])/length(times[2:end])
    
    # init sub models 
    process_model = FeedForwardGrowth(;hidden =hidden_units, seed = NN_seed,dyn_reg = dyn_reg )
    process_loss, observation_model,observation_loss,process_regularization,observation_regularization = init_obs_and_reg(data,proc_weights,obs_weights, reg_weight, prior_weight, q_prior)
    
    
    # parameters
    uhat = zeros(dim1,dim2)
    parameters = (uhat = uhat, process_model = process_model.parameters,process_loss = process_loss.parameters,observation_model = observation_model.parameters,observation_loss = observation_loss.parameters,process_regularization = process_regularization.reg_parameters, observation_regularization = observation_regularization.reg_parameters)
    
    parameters = ComponentArray(parameters)
    
    # loss function
    loss_function = init_loss(times,dt_final,data,process_model,process_loss,observation_model,observation_loss,process_regularization,observation_regularization)

    constructor = data -> FeedForward(data;hidden_units=hidden_units,NN_seed=NN_seed,proc_weights=proc_weights,obs_weights=obs_weights,reg_weight=reg_weight,prior_weight=prior_weight,q_prior=q_prior,dyn_reg=dyn_reg)
    
    
    return UDE(times,dt_final,data,[],data_frame,parameters,loss_function,process_model,process_loss,observation_model,
                observation_loss,process_regularization,observation_regularization,constructor)
end 


function Logistic(data;r0 = 0.5,K0 = 10.0,proc_weights=[1.0,0.1], obs_weights = [1.0,1.0],prior_weight = 0.05, q_prior=0.0, dyn_reg = 0.0)
    
    # convert data
    data_frame = data
    times = data.t
    dt_final = sum(times[2:end] .- times[1:(end-1)])/length(times[2:end])
    
    data = transpose(Matrix(data[:,2:size(data)[2]]))
    dim1,dim2 = size(data)
    
    # init sub models 
    process_model = LogisticGrowth(;r0 = 0.5, K0 = 10.0, dyn_reg=dyn_reg)
    process_loss, observation_model,observation_loss,process_regularization,observation_regularization = init_obs_and_reg(data,proc_weights,obs_weights, 0.0, prior_weight, q_prior)
    process_regularization = Regularization.no_reg()
    
    # parameters
    uhat = zeros(dim1,dim2)
    parameters = (uhat = uhat, process_model = process_model.parameters,process_loss = process_loss.parameters,observation_model = observation_model.parameters,observation_loss = observation_loss.parameters,process_regularization = process_regularization.reg_parameters, observation_regularization = observation_regularization.reg_parameters)
    
    parameters = ComponentArray(parameters)
    
    # loss function
    loss_function = init_loss(times,dt_final,data,process_model,process_loss,observation_model,observation_loss,process_regularization,observation_regularization)

    constructor = data -> Logistic(data;r0=r0,K0=K0,proc_weights=proc_weights,obs_weights=obs_weights,prior_weight=prior_weight,q_prior=q_prior,dyn_reg=dyn_reg)
    
    
    return UDE(times,dt_final,data,[],data_frame,parameters,loss_function,process_model,process_loss,observation_model,
                observation_loss,process_regularization,observation_regularization,constructor)
end 



function LSTM(data;hidden_units=10,NN_seed = 1,proc_weights=[1.0,0.1],obs_weights = [1.0,1.0], reg_weight = 10^-5,prior_weight = 5.0, q_prior=0.0, dyn_reg=0.0,l=0.25)
    
    # convert data
    data_frame = data
    times = data.t
    dt_final = sum(times[2:end] .- times[1:(end-1)])/length(times[2:end])
    
    data = transpose(Matrix(data[:,2:size(data)[2]]))
    dim1,dim2 = size(data)
    # init sub models 
    process_model = LSTM(cell_dim=hidden_units, seed = NN_seed,dyn_reg=dyn_reg,l=l)
    process_loss, observation_model,observation_loss,process_regularization,observation_regularization = init_obs_and_reg(data,proc_weights,obs_weights, reg_weight, prior_weight, q_prior)
    process_regularization = Regularization.L2_LSTM(weight=reg_weight)
    
    
    # parameters
    uhat = zeros(dim1,dim2)
    parameters = (uhat = uhat, process_model = process_model.parameters,process_loss = process_loss.parameters,observation_model = observation_model.parameters,observation_loss = observation_loss.parameters,process_regularization = process_regularization.reg_parameters, observation_regularization = observation_regularization.reg_parameters)
    
    parameters = ComponentArray(parameters)
    
    # loss function
    loss_function = init_loss(times,dt_final,data,process_model,process_loss,observation_model,observation_loss,process_regularization,observation_regularization)

    constructor = data -> LSTM(data;hidden_units=hidden_units,NN_seed=NN_seed,proc_weights=proc_weights,obs_weights=obs_weights,reg_weight=reg_weight,prior_weight=prior_weight,q_prior=q_prior,dyn_reg=dyn_reg,l=l)
    
    
    return UDE(times,dt_final,data,[],data_frame,parameters,loss_function,process_model,process_loss,observation_model,
                observation_loss,process_regularization,observation_regularization,constructor)
    
end 








function LSTMDropOut(data;hidden_units=10,NN_seed = 1,drop_prob=0.1,proc_weights=[1.0,0.1],obs_weights = [1.0,1.0], reg_weight = 10^-5,prior_weight = 5.0, q_prior=0.0)
    
    # convert data
    data_frame = data
    times = data.t
    dt_final = sum(times[2:end] .- times[1:(end-1)])/length(times[2:end])
    
    data = transpose(Matrix(data[:,2:size(data)[2]]))
    dim1,dim2 = size(data)
    # init sub models 
    process_model = LSTMDropOut(cell_dim=hidden_units, seed = NN_seed, drop_prob=drop_prob)
    process_loss, observation_model,observation_loss,process_regularization,observation_regularization = init_obs_and_reg(data,proc_weights,obs_weights, reg_weight, prior_weight, q_prior)
    process_regularization = Regularization.L2_LSTM_drop(weight=reg_weight)
    
    
    # parameters
    uhat = zeros(dim1,dim2)
    parameters = (uhat = uhat, process_model = process_model.parameters,process_loss = process_loss.parameters,observation_model = observation_model.parameters,observation_loss = observation_loss.parameters,process_regularization = process_regularization.reg_parameters, observation_regularization = observation_regularization.reg_parameters)
    
    parameters = ComponentArray(parameters)
    
    # loss function
    loss_function = init_loss(times,dt_final,data,process_model,process_loss,observation_model,observation_loss,process_regularization,observation_regularization)

    constructor = data -> LSTMDropOut(data;hidden_units=hidden_units,NN_seed=NN_seed,drop_prob=drop_prob,proc_weights=proc_weights,obs_weights=obs_weights,reg_weight=reg_weight,prior_weight=prior_weight,q_prior=q_prior)
    
    
    return UDE(times,dt_final,data,[],data_frame,parameters,loss_function,process_model,process_loss,observation_model,
                observation_loss,process_regularization,observation_regularization,constructor)
    
end




function DelayEmbedding(data;lags = 10, hidden_units=10,NN_seed = 1,proc_weights=[1.0,0.1],obs_weights = [1.0,1.0], reg_weight = 10^-5,prior_weight = 5.0, q_prior=0.0,dyn_reg=0.0, l = 0.25)
    
    # convert data
    data_frame = data
    times = data.t
    dt_final = sum(times[2:end] .- times[1:(end-1)])/length(times[2:end])
    
    data = transpose(Matrix(data[:,2:size(data)[2]]))
    dim1,dim2 = size(data)
    # init sub models 
    process_model = DelayEmbedding(lags=lags,hidden = hidden_units, seed = NN_seed, dyn_reg=dyn_reg, l=l)
    process_loss, observation_model,observation_loss,process_regularization,observation_regularization = init_obs_and_reg(data,proc_weights,obs_weights, reg_weight, prior_weight, q_prior)
    
    
    # parameters
    uhat = zeros(dim1,dim2)
    parameters = (uhat = uhat, process_model = process_model.parameters,process_loss = process_loss.parameters,observation_model = observation_model.parameters,observation_loss = observation_loss.parameters,process_regularization = process_regularization.reg_parameters, observation_regularization = observation_regularization.reg_parameters)
    
    parameters = ComponentArray(parameters)
    
    # loss function
    loss_function = init_loss(times,dt_final,data,process_model,process_loss,observation_model,observation_loss,process_regularization,observation_regularization)

    constructor = data -> DelayEmbedding(data;lags=lags,hidden_units=hidden_units,NN_seed=NN_seed,proc_weights=proc_weights,obs_weights=obs_weights,reg_weight=reg_weight,prior_weight=prior_weight,q_prior=q_prior,dyn_reg=dyn_reg)
    
    
    return UDE(times,dt_final,data,[],data_frame,parameters,loss_function,process_model,process_loss,observation_model,
                observation_loss,process_regularization,observation_regularization,constructor)
    
end 

function gradient_decent!(UDE; step_size = 0.05, maxiter = 500, verbos = false)
    
    # set optimization problem 
    target = (x,p) -> UDE.loss_function(x)
    adtype = Optimization.AutoZygote()
    optf = Optimization.OptimizationFunction(target, adtype)
    optprob = Optimization.OptimizationProblem(optf, UDE.parameters)
    
    # print value of loss function at each time step 
    if verbos
        callback = function (p, l; doplot = false)
          print(round(l,digits = 3), " ")
          return false
        end
    else
        callback = function (p, l; doplot = false)
          return false
        end 
    end

    # run optimizer
    sol = Optimization.solve(optprob, OptimizationOptimisers.ADAM(step_size), callback = callback, maxiters = maxiter )
    
    # assign parameters to model 
    UDE.parameters = sol.u
    
    return nothing
end

function BFGS!(UDE; verbos = true, initial_step_norm = 0.01)
    
    if verbos
        callback = function (p, l; doplot = false)
          print(round(l,digits = 3), " ")
          return false
        end
    else
        callback = function (p, l; doplot = false)
          return false
        end 
    end
    
    
    target = (x,p) -> UDE.loss_function(x)
    adtype = Optimization.AutoZygote()
    optf = Optimization.OptimizationFunction(target, adtype)
    optprob = Optimization.OptimizationProblem(optf, UDE.parameters)

    sol = Optimization.solve(optprob, Optim.BFGS(; initial_stepnorm = 0.01);
        callback, allow_f_increases = false)

    # assign parameters to model 
    UDE.parameters = sol.u
    
end 



function plot_state_estiamtes(UDE)
    
    plots = []
    
    # initialize process model 
    obs_hat = zeros(size(UDE.data))
    ut = UDE.parameters.uhat[:,1]
    dt = UDE.times[2] - UDE.times[1]
    uhat, r, aux = UDE.process_model.predict(ut,dt,UDE.parameters.process_model) 

    # calcualte loss for first observation 
    yhat = UDE.observation_model.link(UDE.parameters.uhat[:,1],r,dt,UDE.parameters.observation_model)
    obs_hat[:,1] = yhat
    
    # initalize process loss accumulator 
    L_proc = 0 
    for t in 2:(size(UDE.data)[2])
        # calcualte forecasting error 
        ut = UDE.parameters.uhat[:,t]
        #L_proc += UDE.process_loss.loss(ut,uhat,UDE.parameters.process_loss)
        if t < size(UDE.data)[2] 
            # calcualte forecast and obervational loss using time between observations
            dt = UDE.times[t] -UDE.times[t-1]
            uhat, r, aux = UDE.process_model.predict(ut,aux,dt,UDE.parameters.process_model) 
            yhat = UDE.observation_model.link(ut,r,dt,UDE.parameters.observation_model)
            obs_hat[:,t] = yhat
        else
            # calcualte forecast and obervational loss using final value of delta t
            uhat, r, aux = UDE.process_model.predict(ut,aux,UDE.dt_final,UDE.parameters.process_model) 
            yhat = UDE.observation_model.link(ut,r,UDE.dt_final,UDE.parameters.observation_model)
            obs_hat[:,t] = yhat
        end
    end
    
    for dim in 1:size(UDE.data)[1]
    
        plt=Plots.scatter(UDE.times,UDE.data[dim,:], label = "observations")
        
        Plots.plot!(UDE.times,obs_hat[dim,:], color = "grey", label= "estimated states",
                    xlabel = "time", ylabel = string("x", dim))
       
        push!(plots, plt)
    end 
            
    return plot(plots...)       
end


function predict(UDE)
    
    if UDE.X ==[]
        inits = UDE.parameters.uhat[:,1:(end-1)]
        obs = UDE.parameters.uhat[:,2:end]
        preds = UDE.parameters.uhat[:,2:end]

        # calculate initial prediciton 
        ut = UDE.parameters.uhat[:,1]
        dt = UDE.times[2]-UDE.times[1]
        preds[:,1], r, aux  = UDE.process_model.predict(ut,dt,UDE.parameters.process_model) 
        for t in 2:(size(UDE.data)[2]-1)
            # calcualte loss
            ut = UDE.parameters.uhat[:,t]
            dt = UDE.times[t+1]-UDE.times[t]
            preds[:,t], r, aux = UDE.process_model.predict(ut,aux,dt,UDE.parameters.process_model)
        end

        return inits, obs, preds
    else
        inits = UDE.parameters.uhat[:,1:(end-1)]
        obs = UDE.parameters.uhat[:,2:end]
        preds = UDE.parameters.uhat[:,2:end]

        # calculate initial prediciton 
        ut = UDE.parameters.uhat[:,1]
        dt = UDE.times[2]-UDE.times[1]
        preds[:,1], r, aux  = UDE.process_model.predict(ut,UDE.X[:,1],dt,UDE.parameters.process_model)
        for t in 2:(size(UDE.data)[2]-1)
            # calcualte loss
            ut = UDE.parameters.uhat[:,t]
            dt = UDE.times[t+1]-UDE.times[t]
            preds[:,t], r, aux = UDE.process_model.predict(ut,UDE.X[:,t],aux,dt,UDE.parameters.process_model)
        end

        return inits, obs, preds
    end 
end 


function plot_predictions(UDE)
 
    inits, obs, preds = predict(UDE)
    
    plots = []
    for dim in 1:size(obs)[1]
        difs = obs[dim,:].-inits[dim,:]
        xmin = difs[argmin(difs)]
        xmax = difs[argmax(difs)]
        plt = plot([xmin,xmax],[xmin,xmax],color = "grey", linestyle=:dash, label = "45 degree")
        scatter!(difs,preds[dim,:].-inits[dim,:],color = "white", label = "", 
                                xlabel = string("Observed change ", L"\Delta \hat{u}_t"), 
                                ylabel = string("Predicted change ", L"f(\hat{u}_{t}) - \hat{u}_{t}"))
        push!(plots, plt)
            
    end
        
    return plot(plots...)
end

function plot_fishing_mortality(UDE)
    Plots.plot(UDE.times, UDE.parameters.uhat[2,:], xlabel = "Time", ylabel = "Fishing mortality")
end 

function print_scaling_factor(model)
    print("Scaling factor: ", exp.(model.parameters.observation_model.q))
        
end
    

function forecast_F(UDE,times::AbstractVector{},F::AbstractVector{})
    
    u = UDE.parameters.uhat[1,:]
    umax = u[argmax(u)]
    umin = u[argmin(u)]
    
    # estimated_map = (x,aux,dt) -> UDE.process_model.predict(x,aux,dt,UDE.parameters.process_model)
    estimated_map = (x,aux,dt) -> UDE.process_model.forecast(x,aux,dt,UDE.parameters.process_model,umax,umin)
    
    
    
    dt = UDE.times[2]-UDE.times[1]
    ut, r, aux = UDE.process_model.predict(UDE.parameters.uhat[:,1],dt,UDE.parameters.process_model)

    for t in 2:(size(UDE.data)[2])
        u = UDE.parameters.uhat[:,t]
        if t == size(UDE.data)[2]
            #ut,r, aux = estimated_map(u,aux,UDE.dt_final) 
        else
            dt = UDE.times[t+1]-UDE.times[t]
            ut, r, aux = estimated_map(u,aux,dt) 
        end
    end
   
    x = ut
    T = length(times)
    df = zeros(T,length(x)+1)    
    for t in 1:T
        if t < T
            dt = times[t+1] - times[t]
            xt,r,aux = estimated_map([x[1],F[t]],aux,dt) 
            yhat = UDE.observation_model.link([x[1],F[t]],r,dt,UDE.parameters.observation_model)
            x = xt
        else
            dt = sum(times[2:end].-times[1:(end-1)])/T # assume final dt is equalt o the average time step
            xt,r,aux = estimated_map([x[1],F[t]],aux,dt) 
            yhat = UDE.observation_model.link([x[1],F[t]],r,dt,UDE.parameters.observation_model)
        end
        df[t,:] = vcat([times[t]],yhat)
        
    end 
    
    df = DataFrame(t = times, y = df[:,2], H = df[:,3])
    
    return df
end 




function forecast_F(UDE,T::Int,F::Real)
    dt = sum(UDE.times[2:end] .- UDE.times[1:(end-1)])/(length(UDE.times)-1)
    times = collect(dt:dt:(T*dt))
    times .+= UDE.times[end] 
    F = repeat([F],T)
    return forecast_F(UDE,times,F)

end 

function plot_forecast_F(UDE, T,F)
    df = forecast_F(UDE,T,F)

    plt = plot(df.t,exp.(df.y),color = "grey", linestyle=:dash, label = "forecast",
                    xlabel = "Time", ylabel = string(L"log Biomass"))
    plot!(UDE.times,exp.(UDE.data[1,:]),c=1, label = "data")

    return plt
end 



function forecast_H(UDE,times::AbstractVector{},H::AbstractVector{})
    
    u = UDE.parameters.uhat[1,:]
    umax = u[argmax(u)]
    umin = u[argmin(u)]
    
    # estimated_map = (x,aux,dt) -> UDE.process_model.predict(x,aux,dt,UDE.parameters.process_model)
    estimated_map = (x,aux,dt) -> UDE.process_model.forecast(x,aux,dt,UDE.parameters.process_model,umax,umin)
    
    dt = UDE.times[2]-UDE.times[1]
    ut, r, aux = UDE.process_model.predict(UDE.parameters.uhat[:,1],dt,UDE.parameters.process_model)

    for t in 2:(size(UDE.data)[2])
        u = UDE.parameters.uhat[:,t]
        if t == size(UDE.data)[2]
            #ut,r, aux = estimated_map(u,aux,UDE.dt_final) 
        else
            dt = UDE.times[t+1]-UDE.times[t]
            ut, r, aux = estimated_map(u,aux,dt) 
        end
    end
   
    x = ut
    T = length(times)
    estimated_map = (x,aux,H,dt) -> UDE.process_model.forecast_H(x,aux,H,dt,UDE.parameters.process_model,umax,umin)
    df = zeros(T,length(x)+1)    
    for t in 1:T
        if t < T
            dt = times[t+1] - times[t]
            xt,r,aux = estimated_map([x[1]],aux,H[t],dt) 
            yhat = UDE.observation_model.link([x[1],0.0],r,dt,UDE.parameters.observation_model)
            x = xt
        else
            dt = sum(times[2:end].-times[1:(end-1)])/T # assume final dt is equalt o the average time step
            xt,r,aux = estimated_map([x[1]],aux,H[t],dt) 
            yhat = UDE.observation_model.link([x[1],0.0],r,dt,UDE.parameters.observation_model)
        end
        df[t,:] = vcat([times[t]],[yhat[1],H[t]])
        
    end 
    
    df = DataFrame(t = times, y = df[:,2], H = df[:,3])
    
    return df
end 





function forecast_harvest_loss(UDE,times::AbstractVector{},F::AbstractVector{},harvest)
    
    u = UDE.parameters.uhat[1,:]
    umax = u[argmax(u)]
    umin = u[argmin(u)]
    
    #estimated_map = (x,aux,dt) -> UDE.process_model.predict(x,aux,dt,UDE.parameters.process_model)
    estimated_map = (x,aux,dt) -> UDE.process_model.forecast(x,aux,dt,UDE.parameters.process_model,umax,umin)
    dt = UDE.times[2]-UDE.times[1]
    ut, r, aux = UDE.process_model.predict(UDE.parameters.uhat[:,1],dt,UDE.parameters.process_model)

    for t in 2:(size(UDE.data)[2])
        u = UDE.parameters.uhat[:,t]
        if t == size(UDE.data)[2]
            #ut,r, aux = estimated_map(u,aux,UDE.dt_final) 
        else
            dt = UDE.times[t+1]-UDE.times[t]
            ut, r, aux = estimated_map(u,aux,dt) 
        end
    end
   
    x = ut
    T = length(times)
    df = zeros(T,length(x)+1)  
    L = 0 # initialize accumulator
    for t in 1:T
        if t < T
            dt = times[t+1] - times[t]
            xt,r,aux = estimated_map([x[1],F[t]],aux,dt) 
            yhat = UDE.observation_model.link([x[1],F[t]],r,dt,UDE.parameters.observation_model)
            x = xt
        else
            dt = sum(times[2:end].-times[1:(end-1)])/T # assume final dt is equalt o the average time step
            xt,r,aux = estimated_map([x[1],F[t]],aux,dt) 
            yhat = UDE.observation_model.link([x[1],F[t]],r,dt,UDE.parameters.observation_model)
        end
        L += (harvest[t] - yhat[2])^2
        
    end 
    
    
    return L

end 


function solve_fishing_mortality(UDE,times::AbstractVector{},harvests_::AbstractVector{};verbos = false, step_size = 0.05, maxiters = 200)
    function target(F,p) 
       forecast_harvest_loss(UDE,times,F,harvests_)/length(harvests_)
    end 
    
    #adtype = Optimization.AutoZygote()
    optf = Optimization.OptimizationFunction(target)
    optprob = Optimization.OptimizationProblem(optf, zeros(length(harvests_)))
    
    # print value of loss function at each time step 
    
    if verbos
        callback = function (p, l; doplot = false)
          print(round(l,digits = 3), " ")
          return false
        end
    else
        callback = function (p, l; doplot = false)
          return false
        end 
    end

    # run optimizer
    sol = Optimization.solve(optprob, NelderMead())
    #println("Harvst forecast loss: ", target(sol.u,0) )
    df2 = forecast_F(UDE,times,sol.u)
    return df2, sol.u
end 




function plot_forecast(UDE,times::AbstractVector{},harvest::AbstractVector{})
    
    df,F = solve_fishing_mortality(UDE,times,harvest)
    plt = plot(df.t,exp.(df.y),color = "grey", linestyle=:dash, label = "forecast",
                    xlabel = "Time", ylabel = string(L"log Biomass"))
    plot!(UDE.times,exp.(UDE.data[1,:]),c=1, label = "data")

    
    plt2 = plot(df.t,df.H, label = "forecast",color = "grey", linestyle=:dash)
    plot!(UDE.times,UDE.data[2,:],c=1, label = "data")

    return plt,plt2
end



function plot_forecast!(plt1,plt2,UDE,times::AbstractVector{},harvest::AbstractVector{})
    
    df,F = solve_fishing_mortality(UDE,times,harvest)
    plot!(plt1,df.t,exp.(df.y),color = "grey", linestyle=:dash, label = "",
                    xlabel = "Time", ylabel = string(L"log Biomass"))

    plot(plt2,df.t,df.H, label = "",color = "grey", linestyle=:dash)

    return plt1,plt2
end



function plot_forecast(UDE,T,H)
    
    dt = sum(UDE.times[2:end] .- UDE.times[1:(end-1)])/(length(UDE.times)-1)
    times = collect(dt:dt:(T*dt))
    times .+= UDE.times[end] 
    H = repeat([H],T)
    
    plot_forecast(UDE,times,H)
end 

function crash_index(testing)
    if testing.H[1] > 0.01
        crash = (testing.H .- 0.01) .< 0.0
        if any(crash)
            tcrash = testing.t[crash][argmin(testing.t[crash])]
            tcrash = argmin(abs.(testing.t .- tcrash))
            keep = vcat(repeat([true],tcrash), repeat([false],length(testing.H)-tcrash))
                    
        else  
            keep = repeat([true],length(testing.H))
                
        end

    else     
        keep = repeat([true],length(testing.H))
            
    end
    return keep
end 

function plot_forecast(UDE,testing)
    
    keep = crash_index(testing)
    testing = testing[keep,:]
    #testing = testing[keep,:]
    
    p1,p2=plot_forecast(UDE,testing.t,testing.H)
    plot!(p1,testing.t,exp.(testing.y))
    plot!(p2,testing.t,testing.H)
    plot(p1,p2)
    
end 


function plot_forecast!(plts,UDE,testing)
    
    plt1,plt2=plts
    
    keep = crash_index(testing)
    testing = testing[keep,:]
    
    p1,p2=plot_forecast!(plt1,plt2,UDE,testing.t,testing.H)
    plot!(p1,testing.t,exp.(testing.y), label = "")
    plot!(p2,testing.t,testing.H, label = "")
    return p1, p2
end 

function forecast(UDE,times::AbstractVector{},harvest::AbstractVector{})
    
    df,F = solve_fishing_mortality(UDE,times,harvest)
    
    return df
end

function forecast(UDE,testing)
    forecast(UDE,testing.t,testing.H)
end 


function forecast_MSE(UDE,testing;remove_crash = true, weight_H = 0.0)

    keep = crash_index(testing)

    df = forecast(UDE,testing.t,testing.H)
    N_test = length(testing.t)

    pred_y = exp.(df.y)
    test_y = exp.(testing.y)

    pred_H = df.H
    test_H = testing.H
``
    MSE = (1-weight_H)*abs.(pred_y .- test_y)
    MSE += weight_H*abs.(pred_H .- test_H)

    return MSE, keep
end 

function leave_future_out_cv(model; forecast_length = 10,  forecast_number = 10, spacing = 1, step_size = 0.05, maxiter = 500, step_size2 = 0.01, maxiter2 = 500, verbos = false)
    
    if model.X == []
        # get final time
        data = model.data_frame
        T = length(data.t)
        start1 = T - forecast_length - spacing*(forecast_number-1)
        starts = [start1 + spacing *i for i in 0:(forecast_number-1)]
        training_data = [data[1:t0,:] for t0 in starts]
        testing_data = [data[(t0+1):(t0+forecast_length),:] for t0 in starts]

        standard_errors = [[] for i in 1:Threads.nthreads()]
        predictions = [[] for i in 1:Threads.nthreads()]

        Threads.@threads for i in 1:forecast_number

            model_i = model.constructor(training_data[i])

            gradient_decent!(model_i, step_size = step_size, maxiter = maxiter,verbos = verbos) 
            try 
                BFSG!(model_i,verbos = verbos)
            catch
                gradient_decent!(model_i, step_size = step_size2, maxiter = maxiter2,verbos = verbos) 
            end 
            
            # forecast
            predicted_data,F = solve_fishing_mortality(model_i, testing_data[i].t, testing_data[i].H)

            SE = copy(predicted_data)
            SE[:,2] .= (exp.(predicted_data.y) .- exp.(testing_data[i].y)).^2
            SE[:,3] .= (predicted_data.H .- testing_data[i].H).^2
            
            push!(standard_errors[Threads.threadid()], SE)
            push!(predictions[Threads.threadid()], predicted_data)

        end 

        standard_error = standard_errors[1]
        prediction = predictions[1]
        for i in 2:Threads.nthreads()

            standard_error = vcat(standard_error,standard_errors[i])
            prediction = vcat(prediction,predictions[i])

        end

        return training_data, testing_data, prediction, standard_error
    end
    
end 


function plot_leave_future_out_cv(model; forecast_length = 10,  forecast_number = 10, spacing = 1, step_size = 0.05, maxiter = 500, verbos = false,step_size2 = 0.01, maxiter2 = 500)
    training_data, testing_data, predictions, standard_error=leave_future_out_cv(model; forecast_length = forecast_length,  forecast_number = forecast_number, spacing = spacing, step_size = step_size, maxiter = maxiter, step_size2 = step_size2, maxiter2 = maxiter2, verbos = verbos)
    
    MSE_I = sum([sum(df.y) for df in standard_error ])/(forecast_number*forecast_length)
    println("Index forecast MSE: ", MSE_I)
    
    MSE_H = sum([sum(df.H) for df in standard_error ])/(forecast_number*forecast_length)
    println("Harvest forecast MSE: ", MSE_H)
    
    index_plts = [];harvest_plts = []
    
    plt_I = Plots.scatter(testing_data[1].t, exp.(testing_data[1].y), width = 1.0, label = "data", xlabel = "time", ylabel = "Index")
    Plots.plot!(predictions[1].t, exp.(predictions[1].y), linestyle = :dash, width = 1.0, label = "prediction")
    
    plt_H = Plots.scatter(testing_data[1].t, testing_data[1].H, width = 1.0, label = "data", xlabel = "time", ylabel = "Harvest")
    Plots.plot!(predictions[1].t, predictions[1].H, linestyle = :dash, width = 1.0, label = "prediction")
    push!(index_plts,plt_I)
    push!(harvest_plts,plt_H)
    for i in 2:length(training_data)
        
        plt_I = Plots.scatter(testing_data[i].t, exp.(testing_data[i].y), width = 1.0, label = "", xlabel = "time", ylabel = "Index")
        Plots.plot!(predictions[i].t, exp.(predictions[i].y), linestyle = :dash, width = 1.0, label = "")
        push!(index_plts,plt_I)
        
        plt_H = Plots.scatter(testing_data[i].t, testing_data[i].H, width = 1.0, label = "", xlabel = "time", ylabel = "Harvest")
        Plots.plot!(predictions[i].t, predictions[i].H, linestyle = :dash, width = 1.0, label = "")
        push!(harvest_plts,plt_H)
        
        
    end 
    return plot(index_plts...), plot(harvest_plts...)
end 


function simulation_test(seed,model, simulator; verbos = false, maxiter1 = 500, step_size1 = 0.1, maxiter2 = 100, step_size2 = 0.01)
    training_data, training_X, test_sets, test_X, training_plt = simulator(seed)
    test_model = model.constructor(training_data)
    gradient_decent!(test_model,step_size = step_size1, maxiter = maxiter1,verbos=verbos)
    try
        BFGS!(test_model; verbos = verbos, initial_step_norm = 0.01)
    catch
        print("BFGS failed running gradient decent with smaller step size")
        gradient_decent!(test_model,step_size = step_size2, maxiter = maxiter2, verbos = verbos)
    end 
    

    N = zeros(length(test_sets[1].t))
    MSE = zeros(length(test_sets[1].t))
            
    for test_set in test_sets
        MSE_i, keep = SciMLStockAssessments.forecast_MSE(test_model,test_set) 
        N .+= keep
        MSE .+= MSE_i .* keep
    end
    
    return MSE, N
end 



function plot_simulation_test(seed,model,simulator; verbos = false, maxiter1 = 500, step_size1 = 0.1, maxiter2 = 100, step_size2 = 0.01)
    
    training_data, training_X, test_sets, test_X, training_plt = simulator(seed)
    test_model = model.constructor(training_data)
    
    
    gradient_decent!(test_model,step_size = step_size1, maxiter = maxiter1, verbos = verbos)
    try
        BFGS!(test_model; verbos = verbos, initial_step_norm = 0.01)
    catch
        print("BFGS failed running gradient decent with smaller step size")
        gradient_decent!(test_model,step_size = step_size2, maxiter = maxiter2, verbos = verbos)
    end 
    
    
    N_tests = length(test_sets)
    plts = plot_forecast(test_model,test_sets[1].t,test_sets[1].H) 
    plt1,plt2 = plts
    
    plot!(plt1,test_sets[1].t,exp.(test_sets[1].y), label = "true")
    plot!(plt2,test_sets[1].t,test_sets[1].H, label = "true")
    plts = (plt1,plt2)
   
    for i in 2:length(test_sets)
        plot_forecast!(plts,test_model,test_sets[i])
    end
    
    return plts
    
end 

function simulation_tests(model, simulator, N; seed = 123, maxiter1 = 500, step_size1 = 0.1, maxiter2 = 100, step_size2 = 0.01)

    training_data, training_X, test_sets, test_X, training_plt = simulator(1)
    tsteps = length(test_sets[1].t)
    MSE = [zeros(tsteps) for i in 1:N]
    N_tests = [zeros(tsteps) for i in 1:N]
    
    Threads.@threads for i in 1:N
        
        MSE_i, Ntests_i = simulation_test(seed+i,model,simulator,maxiter1=maxiter1,step_size1=step_size1,maxiter2=maxiter2,step_size2=step_size2)
        MSE[i]  .+= MSE_i ./ (Ntests_i .+ 0.0000001)
        N_tests[i] .+= Ntests_i
    end
    
    return MSE, N_tests
end 

function simulaiton_test_means(MSE,Ntests)
    T = length(MSE[1])
    N = length(MSE)
    mean_ = zeros(T) 
    Nt = zeros(T) 
    for i in 1:N
        mean_ .+= MSE[i] .* (Ntests[i] .+ 0.00001)
        Nt.+= Ntests[i] .+ 0.00001
    end
    return mean_ ./ Nt
end

function simulaiton_test_median(MSE)
    T = length(MSE[1])
    N = length(MSE)
    median_ = zeros(T)
    for t in 1:T
        values = zeros(N)
        for i in 1:N
            values[i] = MSE[i][t]
        end
        median_[t] = median(values)
    end
    return median_ 
end

function plot_simulation_tests(MSE,Ntests;label = "",color = 1)
    mean_ =simulaiton_test_means(MSE,Ntests)
    p1 = Plots.plot(1:length(mean_),mean_, label = label, xlabel = "time", ylabel = "Absolute Error", c = color, width = 2)
    for i in 1:length(MSE)
         Plots.plot!(p1,1:length(MSE[i]),MSE[i], c = color, alpha = 0.5, width = 0.75, label = "")
    end 
    return p1
end 

function plot_simulation_tests!(plt,MSE,Ntests;label = "", color = 2)
    mean_ =simulaiton_test_means(MSE,Ntests)
    Plots.plot!(plt,1:length(mean_),mean_, label = label, xlabel = "time", ylabel = "Absolute Error", c = color, width = 2)
    for i in 1:length(MSE)
         Plots.plot!(plt,1:length(MSE[i]),MSE[i], c = color, alpha = 0.5, width = 0.75, label = "")
    end 
end 



end