
function extrapolation(r,u,umax,umin,extrap_length,extrap_value)
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
        r = extrapolation(r,u,umax,umin,extrap_length,extrap_value) # modify growth rate with extrapolation rule
        # calcualte new biomass 
        x = u .+ dt*r - dt*F
        return x, r, aux
    end 
    
    function forecast_H(u,H, aux,dt,parameters,umax,umin)
        Faprox = H/exp(u) # calcualte F from harvest and biomass 
        upred,r,aux = predict([u,Faprox],aux,dt,parameters) # calcualte growth rate with predict function 
        r = extrapolation(r,u,umax,umin,extrap_length,extrap_value) # modify growth rate with extrapolation rule
        # calcualte new biomass 
        x = u .+ dt*r - dt*Faprox
        return x, r, aux
    end 
    
    return forecast_F, forecast_H
    
end     

function logistic(;n= 2.0, extrap_value = 0.1,extrap_length=0.25)

    parameters = (r = 1.0, K = 3)

    if n == 1.0
        n += 10^-6
    end

    function predict(u, dt,parameters) 
     
        x = u[1]; r = 0
        for i in 1:5
            r = parameters.r/(n - 1)* (1 - (exp(x)/parameters.K)^(n- 1))
            x = x + dt*r/5 - dt*u[2]/5
        end 

        return [x,u[2]], r,  0
    end 

    function predict(u,aux, dt,parameters) 

        x = u[1]; r = 0
        for i in 1:5
            r = parameters.r/(n - 1)* (1 - (exp(x)/parameters.K)^(n- 1))
            x = x + dt*r/5 - dt*u[2]/5
        end 

        return [x,u[2]], r,  0
    end 
    
    forecast_F, forecast_H = init_forecast(predict,extrap_value,extrap_length)

    return predict, parameters, forecast_F, forecast_H
    
end 


function theta_logistic(;n= 2,extrap_value = 0.1,extrap_length=0.25)

    parameters = (r = 1.0, K = 3, n = n)
    function predict(u, dt,parameters) 

        if n == 1.0
            n += 10^-6
        end
        x = u[1]; r = 0
        for i in 1:5
            r = parameters.r/(parameters.n - 1)* (1 - (exp(x)/parameters.K)^(parameters.n - 1))
            x = x + dt*r/5 - dt*u[2]/5
        end 
        return [x,u[2]], r,  0
    end 

    function predict(u,aux, dt,parameters) 

        if n == 1.0
            n += 10^-6
        end

        x = u[1]; r = 0
        for i in 1:5
            r = parameters.r/(parameters.n - 1)* (1 - (exp(x)/parameters.K)^(parameters.n - 1))
            x = x + dt*r/5 - dt*u[2]/5
        end 
        return [x,u[2]], r,  0
    end 
    
    forecast_F, forecast_H = init_forecast(predict,extrap_value,extrap_length)

    return predict, parameters, forecast_F, forecast_H
    
end 

function FeedForward(mu,sd;hidden = 10, seed = 1,extrap_value = 0.1,extrap_length=0.25)
    hidden =  round(Int,hidden)
    NN = Lux.Chain(Lux.Dense(1,hidden,tanh), Lux.Dense(hidden,1))
        
    Random.seed!(round(Int,seed)); rng = Random.default_rng() 
    parameters, states = Lux.setup(rng,NN) 
    parameters = (NN = parameters, )
    
    function predict(u, dt,parameters) 
        r = NN((u[1:1] .- mu)./sd ,parameters.NN,states)[1][1]
        x = u[1] .+ dt*r .- dt*u[2]
        return [x,u[2]], r,  0
    end 

    function predict(u,aux, dt,parameters) 
        r = NN((u[1:1] .- mu)./sd,parameters.NN,states)[1][1]
        x = u[1] .+ dt*r .- dt*u[2]
        return [x,u[2]], r,  0
    end 
    
    forecast_F, forecast_H = init_forecast(predict,extrap_value,extrap_length)

    return predict, parameters, forecast_F, forecast_H
    
end 

function LSTM(mu,sd;cell_dim = 10, seed = 1,extrap_value = 0.1,extrap_length=0.25)
    cell_dim = round(Int,cell_dim)
    LSTM_ = Lux.LSTMCell(1=>cell_dim)
    DenseLayer = Lux.Dense((cell_dim+1)=>1,tanh)
    
    Random.seed!(round(Int,seed)); rng = Random.default_rng() 
    LSTM_parameters, LSTM_states = Lux.setup(rng,LSTM_) 
    dense_parameters, dense_states = Lux.setup(rng, DenseLayer)
    parameters = (Dense = dense_parameters, LSTM = LSTM_parameters, x0 = [0.0])

    function predict(u, dt,parameters)
        x = reshape((u[1:1] .- mu)./sd,1,1)
        (y, c), st_lstm = LSTM_(reshape(parameters.x0,1,1),parameters.LSTM, LSTM_states)
        r, states = DenseLayer(vcat(x,y),parameters.Dense,dense_states)
        x = u[1] + dt*r[1] - dt*u[2]
        return [x,u[2]],r[1],(c, st_lstm, (u[1]-mu)/sd, u[2])
    end 
    
    function predict(u,aux, dt,parameters)
        x = reshape((u[1:1].-mu)./sd,1,1); c, st_lstm, ut1, ft1 = aux
        (y, c), st_lstm = LSTM_((x,c),parameters.LSTM, st_lstm)
        r, states = DenseLayer(vcat(x,y),parameters.Dense,dense_states);
        x = u[1] + dt*r[1] - dt*u[2] 
        return [x,u[2]], r[1], (c, st_lstm, (u[1]-mu)/sd, u[2])
    end 
    
    forecast_F, forecast_H = init_forecast(predict,extrap_value,extrap_length)
    
    return predict, parameters, forecast_F, forecast_H
    
end 


function LSTMDropOut(mu,sd;cell_dim = 10, seed = 1, drop_prob = 0.1,extrap_value = 0.1,extrap_length=0.25)
    cell_dim = round(Int,cell_dim)
    LSTM_ = Lux.LSTMCell(2=>cell_dim)
    DenseLayer = Lux.Chain( Lux.Dense( cell_dim +1 => cell_dim), Lux.Dropout(drop_prob),Lux.Dense(cell_dim=>1))
    
    Random.seed!(round(Int,seed)); rng = Random.default_rng() 
    LSTM_parameters, LSTM_states = Lux.setup(rng,LSTM_) 
    dense_parameters, dense_states = Lux.setup(rng, DenseLayer)
    rng = Random.default_rng()  
    parameters = (Dense = dense_parameters,LSTM = LSTM_parameters, x0 = [4.0,0])
    
    function predict(u, dt,parameters)
        x = reshape((u[1:1].-mu)./sd,1,1)
        (y, c), st_lstm = LSTM_(reshape(parameters.x0,2,1),parameters.LSTM, LSTM_states)
        r, states = DenseLayer(vcat(x,y),parameters.Dense,dense_states)
        x = u[1] + dt*r[1] - dt*u[2]
        return [x,u[2]],r[1],(c, st_lstm, (u[1]-mu)/sd, u[2])
    end 
    
    function predict(u,aux, dt,parameters)
        x = reshape((u[1:1].-mu)./sd,1,1); c, st_lstm, ut1, ft1 = aux
        (y, c), st_lstm = LSTM_((reshape([ut1,ft1],2,1),c),parameters.LSTM, st_lstm)
        r, states = DenseLayer(vcat(x,y),parameters.Dense,dense_states);
        x = u[1] + dt*r[1] - dt*u[2]
        return [x,u[2]], r[1], (c, st_lstm, (u[1]-mu)/sd, u[2])
    end 
    
    forecast_F, forecast_H = init_forecast(predict,extrap_value,extrap_length)
    
    return predict, parameters, forecast_F, forecast_H
    
end 

function DelayEmbedding(mu,sd;lags=5,hidden = 10, seed = 1, extrap_value = 0.1, extrap_length=0.25)
    dims_in =  round(Int,1+2*lags)
    hidden =  round(Int,hidden)
    dims_out =  1
    NN = Lux.Chain(Lux.Dense(dims_in,hidden,tanh), Lux.Dense(hidden,dims_out))
 
    Random.seed!(round(Int,seed)); rng = Random.default_rng() 
    NN_parameters, NN_states = Lux.setup(rng,NN) 
    parameters = (NN = NN_parameters, aux0 = zeros(round(Int,2*lags)))
        
    function predict(u,dt,parameters) 
        r = NN(vcat((u[1:1].-mu)./sd,parameters.aux0),parameters.NN,NN_states)[1][1]
        x = u[1] + dt*r - dt*u[2]
        return [x,u[2]], r, (parameters.aux0,(u[1]-mu)/sd,6*u[2])
    end 
    
    function predict(u,aux,dt,parameters) 
        lags,ut1,ft1=aux
        lags = vcat([ut1,ft1],lags[1:(end-2)])
        r = NN(vcat((u[1:1].-mu)./sd,lags),parameters.NN,NN_states)[1][1]
        x = u[1] .+ dt*r - dt*u[2]
        return [x,u[2]],r, (lags,(u[1].-mu)./sd,6*u[2])
    end 
    
    forecast_F, forecast_H = init_forecast(predict,extrap_value,extrap_length)
    
    return predict, parameters, forecast_F, forecast_H
    
end 



function DelayEmbeddingInputScaling(mu,sd;lags=5,hidden = 10, seed = 1, extrap_value = 0.1, extrap_length=0.25)
    
    dims_in =  round(Int,1+2*lags)
    hidden =  round(Int,hidden)
    dims_out =  1
    NN = Lux.Chain(Lux.Dense(dims_in,hidden,tanh), Lux.Dense(hidden,dims_out))
 
    Random.seed!(round(Int,seed)); rng = Random.default_rng() 
    NN_parameters, NN_states = Lux.setup(rng,NN) 
    parameters = (NN = NN_parameters, scale = zeros(dims_in), aux0 = zeros(round(Int,2*lags)))
        
    function predict(u,dt,parameters) 
        inputs = vcat((u[1:1].-mu)./sd,parameters.aux0)
        r = NN(inputs.*parameters.scale,parameters.NN,NN_states)[1][1]
        x = u[1] + dt*r - dt*u[2]
        return [x,u[2]], r, (parameters.aux0,(u[1]-mu)/sd,6*u[2])
    end 
    
    function predict(u,aux,dt,parameters) 
        lags,ut1,ft1=aux
        lags = vcat([ut1,ft1],lags[1:(end-2)])
        inputs = vcat((u[1:1].-mu)./sd,lags)
        r = NN(inputs.*parameters.scale,parameters.NN,NN_states)[1][1]
        x = u[1] .+ dt*r - dt*u[2]
        return [x,u[2]],r, (lags,(u[1].-mu)./sd,6*u[2])
    end 
    
    forecast_F, forecast_H = init_forecast(predict,extrap_value,extrap_length)
    
    return predict, parameters, forecast_F, forecast_H
    
end 


function LogisticDelayEmbeddingInputScaling(mu,sd;lags=5,hidden = 10, seed = 1, extrap_value = 0.1, extrap_length=0.25)
    
    dims_in =  round(Int,1+lags)
    hidden =  round(Int,hidden)
    dims_out =  1
    NN = Lux.Chain(Lux.Dense(dims_in,hidden,tanh), Lux.Dense(hidden,dims_out))
 
    Random.seed!(round(Int,seed)); rng = Random.default_rng() 
    NN_parameters, NN_states = Lux.setup(rng,NN) 
    parameters = (NN = NN_parameters, scale = zeros(dims_in),r = 1.0,  K = 3.0, aux0 = zeros(round(Int,lags)))
        
    function predict(u,dt,parameters) 
        inputs = vcat((u[1:1].-mu)./sd,parameters.aux0)
        r = NN(inputs.*parameters.scale,parameters.NN,NN_states)[1][1]
        r = r + parameters.r*(1 - exp(u[1])/parameters.K)
        x = u[1] + dt*r - dt*u[2]
        return [x,u[2]], r, (parameters.aux0,(u[1]-mu)/sd)
    end 
    
    function predict(u,aux,dt,parameters) 
        lags,ut1=aux
        lags = vcat([ut1],lags[1:(end-1)])
        inputs = vcat((u[1:1].-mu)./sd,lags)
        r = NN(inputs.*parameters.scale,parameters.NN,NN_states)[1][1]
        r = r + parameters.r*(1 - exp(u[1])/parameters.K)
        x = u[1] .+ dt*r  - dt*u[2]
        return [x,u[2]],r, (lags,(u[1].-mu)./sd)
    end 
    
    forecast_F, forecast_H = init_forecast(predict,extrap_value,extrap_length)
    
    return predict, parameters, forecast_F, forecast_H
    
end 


function BiomassDelayEmbedding(mu,sd;lags=5,hidden = 10, seed = 1, extrap_value = 0.1, extrap_length=0.25)
    dims_in =  round(Int,1+lags)
    hidden =  round(Int,hidden)
    dims_out =  1
    NN = Lux.Chain(Lux.Dense(dims_in,hidden,tanh), Lux.Dense(hidden,dims_out))
 
    Random.seed!(round(Int,seed)); rng = Random.default_rng() 
    NN_parameters, NN_states = Lux.setup(rng,NN) 
    parameters = (NN = NN_parameters, aux0 = zeros(round(Int,lags)))
        
    function predict(u,dt,parameters) 
        r = NN(vcat((u[1:1].-mu)./sd,parameters.aux0),parameters.NN,NN_states)[1][1]
        x = u[1] + dt*r - dt*u[2]
        return [x,u[2]], r, (parameters.aux0,u[1])
    end 
    
    function predict(u,aux,dt,parameters) 
        lags,ut1=aux
        lags = vcat([ut1],lags[1:(end-1)])
        r = NN(vcat((u[1:1].-mu)./sd,lags),parameters.NN,NN_states)[1][1]
        x = u[1] .+ dt*r - dt*u[2]
        return [x,u[2]],r, (lags,(u[1]-mu)/sd)
    end 
    
    forecast_F, forecast_H = init_forecast(predict,extrap_value,extrap_length)
    
    return predict, parameters, forecast_F, forecast_H
    
end 


function LogisticDelayEmbedding(mu,sd;lags=5,hidden = 10,seed = 1,n= 2,extrap_value = 0.1,extrap_length=0.25)
    dims_in =  round(Int,1+lags)
    hidden =  round(Int,hidden)
    dims_out =  1
    NN = Lux.Chain(Lux.Dense(dims_in,hidden,tanh), Lux.Dense(hidden,dims_out))
 
    Random.seed!(round(Int,seed)); rng = Random.default_rng() 
    NN_parameters, NN_states = Lux.setup(rng,NN) 
    parameters = (NN = NN_parameters, ARD = ones(dims_in), aux0 = zeros(round(Int,lags)), r = 1.0, K = 3.0)
        
    function predict(u,dt,parameters) 
        inputs = parameters.ARD .* vcat((u[1:1].-mu)./sd,parameters.aux0)
        nu = NN(inputs,parameters.NN,NN_states)[1][1]

        if n == 1.0
            n += 10^-6
        end
        x = u[1]; r = 0
        for i in 1:5
            r = parameters.r/(n - 1)* (1 - (exp(x)/parameters.K)^(n - 1))
            x = x + dt*(r + nu)/5 - dt*u[2]/5
        end 
        return [x,u[2]], r, (parameters.aux0,(u[1]-mu)/sd)
    end 
    
    function predict(u,aux,dt,parameters) 
        lags,ut1=aux
        lags = vcat([ut1],lags[1:(end-1)])
        inputs = parameters.ARD .*vcat((u[1:1].-mu)./sd,lags)
        nu = NN(inputs,parameters.NN,NN_states)[1][1]

        if n == 1.0
            n += 10^-6
        end
        x = u[1]; r = 0
        for i in 1:5
            r = parameters.r/(n - 1)* (1 - (exp(x)/parameters.K)^(n - 1))
            x = x + dt*(r + nu)/5 - dt*u[2]/5
        end 

        return [x,u[2]],r, (lags,(u[1].-mu)./sd)
    end 
    

    forecast_F, forecast_H = init_forecast(predict,extrap_value,extrap_length)

    return predict, parameters, forecast_F, forecast_H
    
end 


function DelayEmbeddingARD(mu,sd;lags=5,hidden = 10, seed = 1, extrap_value = 0.1, extrap_length=0.25)
    dims_in =  round(Int,1+2*lags)
    hidden =  round(Int,hidden)
    dims_out =  1
    NN = Lux.Chain(Lux.Dense(dims_in,hidden,tanh), Lux.Dense(hidden,dims_out))
 
    Random.seed!(round(Int,seed)); rng = Random.default_rng() 
    NN_parameters, NN_states = Lux.setup(rng,NN) 
    parameters = (NN = NN_parameters, ARD = ones(dims_in),aux0 = zeros(round(Int,2*lags)))
        
    function predict(u,dt,parameters) 
        input = parameters.ARD .* vcat((u[1:1].-mu)./sd,parameters.aux0)
        r = NN(input,parameters.NN,NN_states)[1][1]
        x = u[1] + dt*r - dt*u[2]
        return [x,u[2]], r, (parameters.aux0,(u[1].-mu)./sd,6*u[2])
    end 
    
    function predict(u,aux,dt,parameters) 
        lags,ut1,ft1=aux
        lags = vcat([ut1,ft1],lags[1:(end-2)])
        input = parameters.ARD .* vcat((u[1:1].-mu)./sd,lags)
        r = NN(input,parameters.NN,NN_states)[1][1]
        x = u[1] .+ dt*r - dt*u[2]
        return [x,u[2]],r, (lags,(u[1].-mu)./sd,6*u[2])
    end 
    
    forecast_F, forecast_H = init_forecast(predict,extrap_value,extrap_length)
    
    return predict, parameters, forecast_F, forecast_H
    
end 

function DelayEmbeddingDropOut(mu,sd; lags = 5, hidden = 10, drop_prob = 0.1, seed = 1, extrap_value=0.01, extrap_length = 0.25)
    
    # initial neurla Network
    dims_in = round(Int,1+2*lags)
    hidden = round(Int,hidden)
    NN = Lux.Chain( Lux.Dense(dims_in,hidden,tanh), Lux.Dropout(drop_prob), Lux.Dense(hidden,1))
    Random.seed!(round(Int,seed)) ;rng = Random.default_rng() 
    NN_parameters, NN_states = Lux.setup(rng,NN) 
    parameters = (NN = NN_parameters, aux0 = zeros(round(Int,2*lags)))
        
    function predict(u, dt,parameters) 
        r = NN(vcat((u[1:1].-mu)./sd,parameters.aux0),parameters.NN,NN_states)[1][1]
        x = u[1] + dt*r - dt*u[2]
        return [x,u[2]], r, (parameters.aux0,(u[1]-mu)/sd,6*u[2])
    end 
    
    function predict(u,aux,dt,parameters) 
        lags,ut1,ft1=aux
        lags = vcat([ut1,ft1],lags[1:(end-2)])
        r = NN(vcat((u[1:1].-mu)./sd,lags),parameters.NN,NN_states)[1][1]
        x = u[1] .+ dt*r - dt*u[2]
        return [x,u[2]],r, (lags,(u[1]-mu)/sd,6*u[2])
    end 
    
    forecast_F, forecast_H = init_forecast(predict,extrap_value,extrap_length)
    
    return predict, parameters, forecast_F, forecast_H
    
end 


function init_forecast(predict,extrap_value,extrap_length)
    
    function forecast_F(u,F, aux,dt,parameters,umax,umin)
        upred,r,aux = predict([u,F],aux,dt,parameters) # calcualte growth rate with predict function 
        r = extrapolation(r,u,umax,umin,extrap_length,extrap_value) # modify growth rate with extrapolation rule
        # calcualte new biomass 
        x = u .+ dt*r - dt*F
        return x, r, aux
    end 
    
    function forecast_H(u,H, aux,dt,parameters,umax,umin)
        Faprox = H/exp(u) # calcualte F from harvest and biomass 
        upred,r,aux = predict([u,Faprox],aux,dt,parameters) # calcualte growth rate with predict function 
        r = extrapolation(r,u,umax,umin,extrap_length,extrap_value) # modify growth rate with extrapolation rule
        # calcualte new biomass 
        x = u .+ dt*r - dt*Faprox
        return x, r, aux
    end 
    
    return forecast_F, forecast_H
    
end     




function ProductionModel(model,data,loss,sigma_B,sigma_F,pars,mu,sd)
    
    # Harvest link function 
    predict=x->0;parameters=x->0;forecast_F=x->0;forecast_H=x->0
    if model == "FeedForward"
        predict,parameters,forecast_F,forecast_H = FeedForward(mu,sd;hidden=pars.hidden,seed=pars.seed,extrap_value=pars.extrap_value,extrap_length=pars.extrap_length)
    elseif model == "LSTM"
        predict,parameters,forecast_F,forecast_H = LSTM(mu,sd;cell_dim=pars.cell_dim,seed=pars.seed,extrap_value=pars.extrap_value,extrap_length=pars.extrap_length)
    elseif model == "LSTMDropOut"
        predict,parameters,forecast_F,forecast_H = LSTMDropOut(mu,sd;cell_dim=pars.cell_dim,seed=pars.seed,extrap_value=pars.extrap_value,extrap_length=pars.extrap_length,drop_prob=pars.drop_prob)
    elseif model == "DelayEmbedding"
        predict,parameters,forecast_F,forecast_H = DelayEmbedding(mu,sd;lags=pars.lags,hidden=pars.hidden,seed=pars.seed,extrap_value=pars.extrap_value,extrap_length=pars.extrap_length)
    elseif model == "BiomassDelayEmbedding"
        predict,parameters,forecast_F,forecast_H = BiomassDelayEmbedding(mu,sd;lags=pars.lags,hidden=pars.hidden,seed=pars.seed,extrap_value=pars.extrap_value,extrap_length=pars.extrap_length)
    elseif model == "DelayEmbeddingARD"
        predict,parameters,forecast_F,forecast_H = DelayEmbeddingARD(mu,sd;lags=pars.lags,hidden=pars.hidden,seed=pars.seed,extrap_value=pars.extrap_value,extrap_length=pars.extrap_length)
    elseif model == "DelayEmbeddingInputScaling"
        predict,parameters,forecast_F,forecast_H = DelayEmbeddingInputScaling(mu,sd;lags=pars.lags,hidden=pars.hidden,seed=pars.seed,extrap_value=pars.extrap_value,extrap_length=pars.extrap_length)
    elseif model == "DelayEmbeddingDropOut"
        predict,parameters,forecast_F,forecast_H = DelayEmbeddingDropOut(mu,sd;lags=pars.lags,hidden=pars.hidden,seed=pars.seed,extrap_value=pars.extrap_value,extrap_length=pars.extrap_length,drop_prob=pars.drop_prob)
    elseif model == "ThetaLogistic"
        predict,parameters,forecast_F,forecast_H = theta_logistic(;n = pars.n,extrap_value=pars.extrap_value,extrap_length=pars.extrap_length)
    elseif model == "Logistic"
        predict,parameters,forecast_F,forecast_H = logistic(;n = pars.n,extrap_value=pars.extrap_value,extrap_length=pars.extrap_length)
    elseif model == "LogisticDelayEmbedding"
        predict,parameters,forecast_F,forecast_H = LogisticDelayEmbedding(mu,sd;lags=pars.lags,hidden=pars.hidden,seed=pars.seed,n = pars.n,extrap_value=pars.extrap_value,extrap_length=pars.extrap_length)
    elseif model ==  "LogisticDelayEmbeddingInputScaling"
        predict,parameters,forecast_F,forecast_H = LogisticDelayEmbeddingInputScaling(mu,sd;lags=pars.lags,hidden=pars.hidden,seed=pars.seed,extrap_value=pars.extrap_value,extrap_length=pars.extrap_length)
    elseif model ==  "GPDelayEmbedding"
        predict,parameters,forecast_F,forecast_H = gaussian_process(data; lags=pars.lags, psi = pars.psi, n=pars.n,extrap_value=pars.extrap_value,extrap_length=pars.extrap_length)
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