
# using Optimization, OptimizationOptimisers, OptimizationOptimJL, ComponentArrays, Zygote, LinearAlgebra, DataFrames


soft_plus(x) = x/(1-exp(-x))
function logistic_process(data)
    
    inducing_points  = lagged_data(data,E)
    
    parameters = (m = parameters, log_q = 0.0, log_K = log(10), log_r = log(0.25))

    function predict(u,dt,parameters)
        r =  1+dt*exp(parameters.log_r)
        K = exp(parameters.log_K)
        asp = r*u[1]/(1+(r-1)*u[1]/K)
        x = asp - dt*exp(parameters.log_q)*u[2]/exp(u[1])
        return [x,u[2]], asp, []
    end 
    
    function predict(u,aux,dt,parameters) 
        r =  1+dt*exp(parameters.log_r)
        K = exp(parameters.log_K)
        asp = r*u[1]/(1+(r-1)*u[1]/K)
        x = asp - dt*exp(parameters.log_q)*u[2]/exp(u[1])
        return [x,u[2]], asp, []
    end 

    function predict_F(u,aux,F,dt,parameters) 
        r =  1+dt*exp(parameters.log_r)
        K = exp(parameters.log_K)
        asp = r*u[1]/(1+(r-1)*u[1]/K)
        x = asp - dt*F
        return [x], r, []
    end 

    return predict, predict_F, parameters
    
end 


function init_loss(times,data,predict,ratio, mu_log_q, sigma_log_q, mu_log_K, sigma_log_K, mu_log_r, sigma_log_r)
    
    function loss_function(parameters, x)
        

        # initialize process model 
        ut = parameters.uhat[:,1]
        dt = times[2] - times[1]
        uhat,sigma_2, r, aux = predict(ut,dt,parameters) 
  

        # calcualte loss for first observation 
        sigma_obs = sqrt(ratio) * parameters.sigma
        sigma_proc = parameters.sigma
        L = 0.5*log.(2*3.14159*sigma_obs^2) + (data[1,1] .- ut[1]).^2 ./ (2*sigma_obs^2)
        L += 0.5*log.(2*3.14159*sigma_obs^2) + (data[2,1] .- ut[2]).^2 ./ (2*sigma_obs^2)


        # initalize process loss accumulator 
        for t in 2:(size(data)[2])
            # calcualte forecasting error 
            ut = parameters.uhat[:,t]
            L += 0.5*log(2*3.14159*(sigma_2)) + (ut[1] - uhat[1])^2/(2*sigma_2) #  + nugget
            L += 0.5*log(2*3.14159*sigma_proc^2) + (ut[2] - uhat[2])^2/(2*sigma_proc^2)
            
            if t < size(data)[2] 
                # calcualte forecast and obervational loss using time between observations
                dt = times[t] - times[t-1]
                uhat, sigma_2, r, aux = predict(ut,aux,dt,parameters) 
            end

            L += 0.5*log.(2*3.14159*sigma_obs^2) + (data[1,t] .- ut[1])^2 ./ (2*sigma_obs^2)
            L += 0.5*log.(2*3.14159*sigma_obs^2) + (data[2,t] .- ut[2])^2 ./ (2*sigma_obs^2)

        end

        # parametric model priors 
        L += (parameters.log_q - mu_log_q)^2 /(2*sigma_log_q^2)
        L += (parameters.log_K - mu_log_K)^2 /(2*sigma_log_K^2)
        L += (parameters.log_r - mu_log_r)^2 /(2*sigma_log_r^2)

        return L
    end
    
end 

mutable struct spLogistic
    data
    model
    predict
    predict_F
    parameters
    priors
    time_column_name
    index_column_name
    harvest_column_name
end



function spLogistic(data, model, time_column_name, harvest_column_name, index_column_name;
                            ratio = 2.0, theta = 1.0, psi = 3.14159 / 2, k = 2, phi = 0.0025,
                            mu_log_q = 0.0,sigma_log_q = 100.0,
                            mu_log_r = 0.0,sigma_log_r = 100.0,
                            mu_log_K = 0.0,sigma_log_K = 100.0,
                            step_size = 0.0025, maxiter = 500)

    # organize data sets
    dataframe = data 
    times = data[:,time_column_name]
    T = length(times)
    data = vcat(reshape(data[:,index_column_name],1,T), reshape(data[:,harvest_column_name],1,T))

    # initialize model
    predict = x -> 0; predict_F = x -> 0; loss = (x,p) -> 0; parameters = 0; GP = 0; priors = 0

    predict, predict_F, parameters =logistic_process(data)
    priors = (k=k, phi=phi, mu_log_q=mu_log_q, sigma_log_q=sigma_log_q,mu_log_r=mu_log_r,sigma_log_r=sigma_log_r, mu_log_K=mu_log_K, sigma_log_K=sigma_log_K) 
    loss = init_loss(times,data,predict,ratio,  mu_log_q, sigma_log_q, mu_log_K, sigma_log_K, mu_log_r, sigma_log_r)


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
    sol = Optimization.solve(optprob, Optim.BFGS(), callback = callback)

    # update prediction function 
    predict_sol = (log_B,H,aux,dt) -> predict([log_B,H],aux,dt,sol.u)
    predict_F_sol = (log_B,F,aux,dt) -> predict_F([log_B],aux,F,dt,sol.u)


    # final lags
    aux = data[1,(end-E+1):end]
 
    return spLogistic(dataframe, model, predict_sol, predict_F_sol, sol.u, priors, time_column_name, index_column_name, harvest_column_name)

end 


function forecast(spLogistic,new_data,fishing_column_name,fishing_variable)

    times_forecast = new_data[:,spLogistic.time_column_name]
    times = spLogistic.data[:,spLogistic.time_column_name]
    U = new_data[:,fishing_column_name]
    log_B = spLogistic.data[end,spLogistic.index_column_name]
    H_T = spLogistic.data[end,spLogistic.harvest_column_name]
    log_B_values = []
    aux = spLogistic.aux

    if fishing_variable == "Harvest"
        i = 0
        for t in times_forecast
            dt = 0
            i += 1
            if i == 1
                dt = t - times[end]
                u, r, aux = spLogistic.predict(log_B,H_T,aux,dt)
            else
                dt = times[i] - times[i-1]
                u,  r, aux = spLogistic.predict(log_B,U[i],aux,dt)
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
                u, r, aux = spLogistic.predict(log_B,H_T,aux,dt)
            else
                dt = times[i] - times[i-1]
                u,  r, aux = spLogistic.predict_F(log_B,U[i],aux,dt)
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
    aux = spLogistic.aux

    for f in vcat(repeat([0],50),F)
        u, r, aux = spLogistic.predict_F(log_B,f,aux,dt)
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
