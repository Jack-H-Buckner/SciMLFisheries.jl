
function extrapolation!(r,u,umax,umin,extrap_length,extrap_value)
    if u > umax
        w = exp(-0.5/extrap_length^2*((u-umax)/(umax-umin))^2)
        r = w*r -extrap_value*(1-w)
    elseif u < umin
        w = exp(-0.5/extrap_length^2*((u-umin)/(umax-umin))^2)
        r = w*r + extrap_value*(1-w)
    end 
    return r
end 

function init_forecast(predict,extrap_value,extrap_length)
    
    function forecast_F(u,F, aux,dt,parameters,umax,umin)
        upred,r,aux = predict([u,F],aux,dt,parameters) # calcualte growth rate with predict function 
        extrapolation!(r,u,umax,umin,extrap_length,extrap_value) # modify growth rate with extrapolation rule
        # calcualte new biomass 
        x = u .+ dt*r - dt*F
        return x, r, aux
    end 
    
    function forecast_H(u,H, aux,dt,parameters,umax,umin)
        Faprox = H/exp(u) # calcualte F from harvest and biomass 
        upred,r,aux = predict([u,Faprox],aux,dt,parameters) # calcualte growth rate with predict function 
        extrapolation!(r,u,umax,umin,extrap_length,extrap_value) # modify growth rate with extrapolation rule
        # calcualte new biomass 
        x = u .+ dt*r - dt*Faprox
        return x, r, aux
    end 
    
    return forecast_F, forecast_H
    
end     

function FeedForward(;hidden = 10, seed = 1,extrap_value = 0.1,extrap_length=0.25)
    hidden =  round(Int,hidden)
    NN = Lux.Chain(Lux.Dense(1,hidden,tanh), Lux.Dense(hidden,1))
        
    Random.seed!(round(Int,seed)); rng = Random.default_rng() 
    parameters, states = Lux.setup(rng,NN) 
    parameters = (NN = parameters, )
    
    function predict(u, dt,parameters) 
        r = NN([u[1]],parameters.NN,states)[1][1]
        x = u[1] .+ dt*r .- dt*u[2]
        return [x,u[2]], r,  0
    end 

    function predict(u,aux, dt,parameters) 
        r = NN([u[1]],parameters.NN,states)[1][1]
        x = u[1] .+ dt*r .- dt*u[2]
        return [x,u[2]], r,  0
    end 
    
    forecast_F, forecast_H = init_forecast(predict,extrap_value,extrap_length)

    return predict, parameters, forecast_F, forecast_H
    
end 

function LSTM(;cell_dim = 10, seed = 1,extrap_value = 0.1,extrap_length=0.25)
    cell_dim = round(Int,cell_dim)
    LSTM_ = Lux.LSTMCell(1=>cell_dim)
    DenseLayer = Lux.Dense((cell_dim+1)=>1,tanh)
    
    Random.seed!(round(Int,seed)); rng = Random.default_rng() 
    LSTM_parameters, LSTM_states = Lux.setup(rng,LSTM_) 
    dense_parameters, dense_states = Lux.setup(rng, DenseLayer)
    parameters = (Dense = dense_parameters, LSTM = LSTM_parameters, x0 = [0.0])

    function predict(u, dt,parameters)
        x = reshape(u[1:1],1,1)
        (y, c), st_lstm = LSTM_(reshape(parameters.x0,1,1),parameters.LSTM, LSTM_states)
        r, states = DenseLayer(vcat(x,y),parameters.Dense,dense_states)
        x = u[1] + dt*r[1] - dt*u[2]
        return [x,u[2]],r[1],(c, st_lstm, u[1], u[2])
    end 
    
    function predict(u,aux, dt,parameters)
        x = reshape(u[1:1],1,1); c, st_lstm, ut1, ft1 = aux
        rt1 = u[1] - ut1 + ft1
        (y, c), st_lstm = LSTM_((reshape([rt1],1,1),c),parameters.LSTM, st_lstm)
        r, states = DenseLayer(vcat(x,y),parameters.Dense,dense_states);
        x = u[1] + dt*r[1] - dt*u[2] 
        return [x,u[2]], r[1], (c, st_lstm, u[1], u[2])
    end 
    
    forecast_F, forecast_H = init_forecast(predict,extrap_value,extrap_length)
    
    return predict, parameters, forecast_F, forecast_H
    
end 

function LSTMDropOut(;cell_dim = 10, seed = 1, drop_prob = 0.1,extrap_value = 0.1,extrap_length=0.25)
    cell_dim = round(Int,cell_dim)
    LSTM_ = Lux.LSTMCell(2=>cell_dim)
    DenseLayer = Lux.Chain( Lux.Dense( cell_dim +1 => cell_dim), Lux.Dropout(drop_prob),Lux.Dense(cell_dim=>1))
    
    Random.seed!(round(Int,seed)); rng = Random.default_rng() 
    LSTM_parameters, LSTM_states = Lux.setup(rng,LSTM_) 
    dense_parameters, dense_states = Lux.setup(rng, DenseLayer)
    rng = Random.default_rng()  
    parameters = (Dense = dense_parameters,LSTM = LSTM_parameters, x0 = [4.0,0])
    
    function predict(u, dt,parameters)
        x = reshape(u[1:1],1,1)
        (y, c), st_lstm = LSTM_(reshape(parameters.x0,2,1),parameters.LSTM, LSTM_states)
        r, states = DenseLayer(vcat(x,y),parameters.Dense,dense_states)
        x = u[1] + dt*r[1] - dt*u[2]
        return [x,u[2]],r[1],(c, st_lstm, u[1], u[2])
    end 
    
    function predict(u,aux, dt,parameters)
        x = reshape(u[1:1],1,1); c, st_lstm, ut1, ft1 = aux
        (y, c), st_lstm = LSTM_((reshape([ut1,ft1],2,1),c),parameters.LSTM, st_lstm)
        r, states = DenseLayer(vcat(x,y),parameters.Dense,dense_states);
        x = u[1] + dt*r[1] - dt*u[2]
        return [x,u[2]], r[1], (c, st_lstm, u[1], u[2])
    end 
    
    forecast_F, forecast_H = init_forecast(predict,extrap_value,extrap_length)
    
    return predict, parameters, forecast_F, forecast_H
    
end 

function DelayEmbedding(;lags=5,hidden = 10, seed = 1, extrap_value = 0.1, extrap_length=0.25)
    dims_in =  round(Int,1+2*lags)
    hidden =  round(Int,hidden)
    dims_out =  1
    NN = Lux.Chain(Lux.Dense(dims_in,hidden,tanh), Lux.Dense(hidden,dims_out))
 
    Random.seed!(round(Int,seed)); rng = Random.default_rng() 
    NN_parameters, NN_states = Lux.setup(rng,NN) 
    parameters = (NN = NN_parameters, aux0 = zeros(round(Int,2*lags)))
        
    function predict(u,dt,parameters) 
        r = NN(vcat(u[1:1],parameters.aux0),parameters.NN,NN_states)[1][1]
        x = u[1] + dt*r - dt*u[2]
        return [x,u[2]], r, (parameters.aux0,u[1],u[2])
    end 
    
    function predict(u,aux,dt,parameters) 
        lags,ut1,ft1=aux
        lags = vcat([ut1,ft1],lags[1:(end-2)])
        r = NN(vcat(u[1:1],lags),parameters.NN,NN_states)[1][1]
        x = u[1] .+ dt*r - dt*u[2]
        return [x,u[2]],r, (lags,u[1],u[2])
    end 
    
    forecast_F, forecast_H = init_forecast(predict,extrap_value,extrap_length)
    
    return predict, parameters, forecast_F, forecast_H
    
end 

function DelayEmbeddingARD(;lags=5,hidden = 10, seed = 1, extrap_value = 0.1, extrap_length=0.25)
    dims_in =  round(Int,1+2*lags)
    hidden =  round(Int,hidden)
    dims_out =  1
    NN = Lux.Chain(Lux.Dense(dims_in,hidden,tanh), Lux.Dense(hidden,dims_out))
 
    Random.seed!(round(Int,seed)); rng = Random.default_rng() 
    NN_parameters, NN_states = Lux.setup(rng,NN) 
    parameters = (NN = NN_parameters, ARD = ones(dims_in),aux0 = zeros(round(Int,2*lags)))
        
    function predict(u,dt,parameters) 
        input = parameters.ARD .* vcat(u[1:1],parameters.aux0)
        r = NN(input,parameters.NN,NN_states)[1][1]
        x = u[1] + dt*r - dt*u[2]
        return [x,u[2]], r, (parameters.aux0,u[1],u[2])
    end 
    
    function predict(u,aux,dt,parameters) 
        lags,ut1,ft1=aux
        lags = vcat([ut1,ft1],lags[1:(end-2)])
        input = parameters.ARD .* vcat(u[1:1],lags)
        r = NN(input,parameters.NN,NN_states)[1][1]
        x = u[1] .+ dt*r - dt*u[2]
        return [x,u[2]],r, (lags,u[1],u[2])
    end 
    
    forecast_F, forecast_H = init_forecast(predict,extrap_value,extrap_length)
    
    return predict, parameters, forecast_F, forecast_H
    
end 

function DelayEmbeddingDropOut(; lags = 5, hidden = 10, drop_prob = 0.1, seed = 1, extrap_value=0.01, extrap_length = 0.25)
    
    # initial neurla Network
    dims_in = round(Int,1+2*lags)
    hidden = round(Int,hidden)
    NN = Lux.Chain( Lux.Dense(dims_in,hidden,tanh), Lux.Dropout(drop_prob), Lux.Dense(hidden,1))
    Random.seed!(round(Int,seed)) ;rng = Random.default_rng() 
    NN_parameters, NN_states = Lux.setup(rng,NN) 
    parameters = (NN = NN_parameters, aux0 = zeros(round(Int,2*lags)))
        
    function predict(u, dt,parameters) 
        r = NN(vcat(u[1:1],parameters.aux0),parameters.NN,NN_states)[1][1]
        x = u[1] + dt*r - dt*u[2]
        return [x,u[2]], r, (parameters.aux0,u[1],u[2])
    end 
    
    function predict(u,aux,dt,parameters) 
        lags,ut1,ft1=aux
        lags = vcat([ut1,ft1],lags[1:(end-2)])
        r = NN(vcat(u[1:1],lags),parameters.NN,NN_states)[1][1]
        x = u[1] .+ dt*r - dt*u[2]
        return [x,u[2]],r, (lags,u[1],u[2])
    end 
    
    forecast_F, forecast_H = init_forecast(predict,extrap_value,extrap_length)
    
    return predict, parameters, forecast_F, forecast_H
    
end 

function ProductionModel(model,loss,sigma_B,sigma_F,pars)
    
    # Harvest link function 
    predict=x->0;parameters=x->0;forecast_F=x->0;forecast_H=x->0
    if model == "FeedForward"
        predict,parameters,forecast_F,forecast_H = FeedForward(;hidden=pars.hidden,seed=pars.seed,extrap_value=pars.extrap_value,extrap_length=pars.extrap_length)
    elseif model == "LSTM"
        predict,parameters,forecast_F,forecast_H = LSTM(;cell_dim=pars.cell_dim,seed=pars.seed,extrap_value=pars.extrap_value,extrap_length=pars.extrap_length)
    elseif model == "LSTMDropOut"
        predict,parameters,forecast_F,forecast_H = LSTMDropOut(;cell_dim=pars.cell_dim,seed=pars.seed,extrap_value=pars.extrap_value,extrap_length=pars.extrap_length,drop_prob=pars.drop_prob)
    elseif model == "DelayEmbedding"
        predict,parameters,forecast_F,forecast_H = DelayEmbedding(;lags=pars.lags,hidden=pars.hidden,seed=pars.seed,extrap_value=pars.extrap_value,extrap_length=pars.extrap_length)
    elseif model == "DelayEmbeddingARD"
        predict,parameters,forecast_F,forecast_H = DelayEmbeddingARD(;lags=pars.lags,hidden=pars.hidden,seed=pars.seed,extrap_value=pars.extrap_value,extrap_length=pars.extrap_length)
    elseif model == "DelayEmbeddingDropOut"
        predict,parameters,forecast_F,forecast_H = DelayEmbeddingDropOut(;lags=pars.lags,hidden=pars.hidden,seed=pars.seed,extrap_value=pars.extrap_value,extrap_length=pars.extrap_length,drop_prob=pars.drop_prob)
    else
        print("Your choice of production model does not match avaiable options")
        throw(error())   
    end 
    
    # likelihood function 
    loss_function = x -> 0;loss_params = NamedTuple()
    if loss == "FixedVariance"
        loss_function,loss_params = FixedVariance([sigma_B,sigma_F]) 
    elseif loss == "EstimateVariance"
        loss_function,loss_params = EstimateVariance([sigma_B,sigma_F])
    else
        print("Your choice of likelihood does not match avaiable options")
        throw(error()) 
    end
        
    return predict,parameters,forecast_F,forecast_H,loss_function,loss_params
    
end 