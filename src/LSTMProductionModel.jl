
# using Optimization, OptimizationOptimisers, OptimizationOptimJL, ComponentArrays, Zygote, LinearAlgebra, DataFrames


soft_plus(x) = x/(1-exp(-x))
function LSTM_process(hidden,cell_dim;steps = 10, seed = 123)
    
    cell_dim = round(Int,cell_dim)
    LSTM_ = Lux.LSTMCell(1=>cell_dim)
    DenseLayer = Lux.Chain(Lux.Dense((cell_dim+1)=>hidden,tanh), Lux.Dense(hidden=>1))
    
    Random.seed!(round(Int,seed)); rng = Random.default_rng() 
    LSTM_parameters, LSTM_states = Lux.setup(rng,LSTM_) 
    dense_parameters, dense_states = Lux.setup(rng, DenseLayer)

    parameters = (log_q = log(0.1), sigma = 0.5,Dense = dense_parameters, LSTM = LSTM_parameters, x0 = [0.0])
   
    function predict(u,dt,parameters)

        (y, c), st_lstm = LSTM_(reshape(parameters.x0,1,1),parameters.LSTM, LSTM_states)
        r, states = DenseLayer(vcat(u[1:1],y),parameters.Dense,dense_states)

        x = u[1]
        for i in 1:steps
            x +=  dt*r[1]/steps - dt*exp(parameters.log_q)*u[2]/exp(x)/steps
        end

        return [x,u[2]],r[1],(c, parameters.x0, st_lstm)

    end 
    
    function predict(u,aux,dt,parameters) 

        c, xlag, st_lstm = aux
        xlag = reshape(xlag,1,1)
        (y, c), st_lstm = LSTM_((xlag,c),parameters.LSTM, st_lstm)
        r, states = DenseLayer(vcat(u[1:1],y),parameters.Dense,dense_states)
        
        x = u[1]
        for i in 1:steps
            x +=  dt*r[1]/steps - dt*exp(parameters.log_q)*u[2]/exp(x)/steps
        end

        return [x,u[2]], 0, (c, xlag, st_lstm)
    end 

    function predict_F(u,aux,F,dt,parameters) 

        c, xlag, st_lstm = aux
        xlag = reshape(xlag,1,1)
        (y, c), st_lstm = LSTM_((xlag,c),parameters.LSTM, st_lstm)
        r, states = DenseLayer(vcat(u[1:1],y),parameters.Dense,dense_states)
        
        x = u[1]

        x +=  dt*r[1] - dt*F

        return [x,u[2]], 0, (c, xlag, st_lstm)

    end 

    return predict, predict_F, parameters
    
end 


function init_loss_LSTM(times,data,predict,ratio, mu_log_q, sigma_log_q, sigma_reg)
    
    function loss_function(parameters, x)
        

        # initialize process model 
        ut = parameters.uhat[:,1]
        dt = times[2] - times[1]
        uhat, r, aux = predict(ut,dt,parameters) 
  

        # calcualte loss for first observation 
        sigma_obs = sqrt(ratio) * parameters.sigma
        sigma_proc = parameters.sigma
        L = 0.5*log.(2*3.14159*sigma_obs^2) + (data[1,1] .- ut[1]).^2 ./ (2*sigma_obs^2)
        L += 0.5*log.(2*3.14159*sigma_obs^2) + (data[2,1] .- ut[2]).^2 ./ (2*sigma_obs^2)


        # initalize process loss accumulator 
        for t in 2:(size(data)[2])
            # calcualte forecasting error 
            ut = parameters.uhat[:,t]
            L += 0.5*log(2*3.14159*sigma_proc^2) + (ut[1] - uhat[1])^2/(2*sigma_proc^2) #  + nugget
            L += 0.5*log(2*3.14159*sigma_proc^2) + (ut[2] - uhat[2])^2/(2*sigma_proc^2)
            
            if t < size(data)[2] 
                # calcualte forecast and obervational loss using time between observations
                dt = times[t] - times[t-1]
                uhat, r, aux = predict(ut,aux,dt,parameters) 
            end

            L += 0.5*log.(2*3.14159*sigma_obs^2) + (data[1,t] .- ut[1])^2 ./ (2*sigma_obs^2)
            L += 0.5*log.(2*3.14159*sigma_obs^2) + (data[2,t] .- ut[2])^2 ./ (2*sigma_obs^2)

        end

        # parametric model priors 
        L += (parameters.log_q - mu_log_q)^2 /(2*sigma_log_q^2)
        L += sum(parameters.Dense.layer_1.weight.^2) / (2*sigma_reg^2)
        L += sum(parameters.Dense.layer_2.weight.^2) / (2*sigma_reg^2)

        return L
    end
    
end 

mutable struct spLSTM_
    data
    times
    predict
    predict_F
    parameters
    priors
    time_column_name
    index_column_name
    harvest_column_name
end



function spLSTM(data, time_column_name, harvest_column_name, index_column_name;
                            hidden = 5,cell_dim=2,
                            ratio = 2.0, k = 2, phi = 0.0025,
                            mu_log_q = 0.0,sigma_log_q = 100.0,
                            sigma_reg = 0.5,
                            step_size = 0.0025, maxiter = 500,
                            steps = 10)

    # organize data sets
    dataframe = data 
    times = data[:,time_column_name]
    T = length(times)
    data = vcat(reshape(data[:,index_column_name],1,T), reshape(data[:,harvest_column_name],1,T))

    # initialize model
    predict = x -> 0; predict_F = x -> 0; loss = (x,p) -> 0; parameters = 0; priors = 0

    predict, predict_F, parameters =LSTM_process(cell_dim,hidden;steps = steps)
    priors = (k=k, phi=phi, mu_log_q=mu_log_q, sigma_log_q=sigma_log_q,sigma_reg=sigma_reg) 
    loss =  init_loss_LSTM(times,data, predict,ratio,  mu_log_q, sigma_log_q, sigma_reg)


    # merge estimates states into parameters
    parameters = merge(parameters, (uhat = deepcopy(data), ))
    parameters = ComponentArray(parameters)

    # set optimization problem 
    adtype = Optimization.AutoZygote()
    optf = Optimization.OptimizationFunction(loss, adtype)
    optprob = Optimization.OptimizationProblem(optf, parameters)

    # # print value of loss function at each time step 
    callback = function (p, l; doplot = false)
        print(round(l,digits = 4), " ")
        return false
    end

    println(loss(parameters, " "))
    # run gradient descent 
    sol = Optimization.solve(optprob, OptimizationOptimisers.ADAM(step_size), callback = callback, maxiters = maxiter)
    println(" ")
    # run BFGS
    parameters = sol.u; optprob = Optimization.OptimizationProblem(optf, parameters)
    sol = Optimization.solve(optprob, OptimizationOptimisers.ADAM(step_size/2), callback = callback, maxiters = maxiter)

    # update prediction function 
    function predict_sol(log_B,H,dt) 
        predict([log_B,H],dt,sol.u)
    end

    function predict_sol(log_B,H,aux,dt)
         predict([log_B,H],aux,dt,sol.u)
    end 

    function predict_F_sol(log_B,F,dt) 
        predict_F([log_B],F,dt,sol.u)
    end

    function predict_F_sol(log_B,F,aux,dt)
        predict_F([log_B],aux,F,dt,sol.u)
    end
 
    return spLSTM_(dataframe, times,predict_sol, predict_F_sol, sol.u, priors, time_column_name, index_column_name, harvest_column_name)

end 

function calculate_final_LSTM_state(model)

        # initialize process model 
        ut = model.parameters.uhat[:,1]
        dt = model.times[2] - model.times[1]
        uhat, r, aux = model.predict(ut[1],ut[2],dt) 
  

        # initalize process loss accumulator 
        for t in 2:(size(model.parameters.uhat)[2])
            # calcualte forecasting error 
            ut = model.parameters.uhat[:,t]
            if t < size(model.parameters.uhat)[2] 
                # calcualte forecast and obervational loss using time between observations
                dt = model.times[t] - model.times[t-1]
                uhat, r, aux = model.predict(ut[1],ut[2],aux,dt) 
            end
        end

        return uhat, aux

end

function forecast(model::spLSTM_,new_data,fishing_column_name,fishing_variable)

    times_forecast = new_data[:,model.time_column_name]
    times = model.data[:,model.time_column_name]
    U = new_data[:,fishing_column_name]
    log_B = model.data[end,model.index_column_name]
    H_T = model.data[end,model.harvest_column_name]
    log_B_values = []

    u0, aux = calculate_final_LSTM_state(model)
    log_B = u0[1]
    if fishing_variable == "Harvest"
        i = 0
        for t in times_forecast
            dt = 0
            i += 1
            if i == 1
                dt = t - times[end]
                u, r, aux = model.predict(log_B,H_T,aux,dt)
            else
                dt = times[i] - times[i-1]
                u,  r, aux = model.predict(log_B,U[i],aux,dt)
            end 
            log_B = u[1]
            push!(log_B_values, log_B)
        end 
    elseif fishing_variable == "Mortality"
        i = 0
        for t in times_forecast
            dt = 0
            i += 1
            if i == 1
                dt = t - times[end]
                u, r, aux = model.predict(log_B,H_T,aux,dt)
            else
                dt = times[i] - times[i-1]
                u,  r, aux = model.predict_F(log_B,U[i],aux,dt)
            end 
            log_B = u[1]
            push!(log_B_values, log_B)
        end 
    else
        throw("Fishing variable is invalid choose between 'Harvest' and 'Mortality' ")
    end 

    return log_B_values
end 


function surplus_production(spLogistic; F = 0.0:0.0001:1.5)

    times = spLogistic.data[:,spLogistic.time_column_name]
    dt = sum(times[2:end] .- times[1:(end-1)])/length(times[1:(end-1)])
    log_B = spLogistic.data[end,spLogistic.index_column_name]
    B_values = []; H_values = []

    for f in vcat(repeat([0],50),F)
        u, r, aux = spLogistic.predict_F(log_B,f,dt)
        log_B = u[1]
        push!(B_values, exp(log_B))
        push!(H_values, exp(log_B)*f)
    end 

    return B_values, H_values
end 


function mapping(x,H,spLogistic;dt=1.0)
    spLogistic.predict(x[1],H,x[2:end],dt)[1][1]
end 


function fishing_mortality(spLogistic)

    B = exp.(spLogistic.parameters.uhat[1,:])
    H = spLogistic.parameters.uhat[2,:]
    q = exp(spLogistic.parameters.log_q)

    return q*H./B
end 