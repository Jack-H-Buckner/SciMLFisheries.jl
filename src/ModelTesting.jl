

function plot_state_estiamtes(UDE)
    
    plots = []
    
    # initialize process model 
    obs_hat = zeros(size(UDE.data))
    ut = UDE.parameters.uhat[:,1]
    dt = UDE.times[2] - UDE.times[1]
    uhat, r, aux = UDE.predict(ut,dt,UDE.parameters.predict) 

    # calcualte loss for first observation 
    yhat = UDE.link(UDE.parameters.uhat[:,1],r,dt,UDE.parameters.link)
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
            uhat, r, aux = UDE.predict(ut,aux,dt,UDE.parameters.predict) 
            yhat = UDE.link(ut,r,dt,UDE.parameters.link)
            obs_hat[:,t] = yhat
        else
            # calcualte forecast and obervational loss using final value of delta t
            uhat, r, aux = UDE.predict(ut,aux,UDE.dt_final,UDE.parameters.predict) 
            yhat = UDE.link(ut,r,UDE.dt_final,UDE.parameters.link)
            obs_hat[:,t] = yhat
        end
    end
    
    for dim in 1:size(UDE.data)[1]
        plt=Plots.scatter(UDE.times,UDE.data[dim,:], label = "observations")
        Plots.plot!(UDE.times,obs_hat[dim,:], color = "grey", label= "estimated states",xlabel = "time", ylabel = string("x", dim))
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
        preds[:,1], r, aux  = UDE.predict(ut,dt,UDE.parameters.predict) 
        for t in 2:(size(UDE.data)[2]-1)
            # calcualte loss
            ut = UDE.parameters.uhat[:,t]
            dt = UDE.times[t+1]-UDE.times[t]
            preds[:,t], r, aux = UDE.predict(ut,aux,dt,UDE.parameters.predict)
        end

        return inits, obs, preds
    else
        inits = UDE.parameters.uhat[:,1:(end-1)]
        obs = UDE.parameters.uhat[:,2:end]
        preds = UDE.parameters.uhat[:,2:end]

        # calculate initial prediciton 
        ut = UDE.parameters.uhat[:,1]
        dt = UDE.times[2]-UDE.times[1]
        preds[:,1], r, aux  = UDE.predict(ut,UDE.X[:,1],dt,UDE.parameters.predict)
        for t in 2:(size(UDE.data)[2]-1)
            # calcualte loss
            ut = UDE.parameters.uhat[:,t]
            dt = UDE.times[t+1]-UDE.times[t]
            preds[:,t], r, aux = UDE.predict(ut,UDE.X[:,t],aux,dt,UDE.parameters.predict)
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
    estimated_map = (x,aux,dt) -> UDE.forecast_F(x[1],x[2],aux,dt,UDE.parameters.predict,umax,umin)
    
    
    
    dt = UDE.times[2]-UDE.times[1]
    ut, r, aux = UDE.predict(UDE.parameters.uhat[:,1],dt,UDE.parameters.predict)

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
    df = zeros(T,length(x)+2)    
    for t in 1:T
        if t < T
            dt = times[t+1] - times[t]
            xt,r,aux = estimated_map([x[1],F[t]],aux,dt) 
            yhat = UDE.link([x[1],F[t]],r,dt,UDE.parameters.link)
            x = xt
        else
            dt = sum(times[2:end].-times[1:(end-1)])/T # assume final dt is equalt o the average time step
            xt,r,aux = estimated_map([x[1],F[t]],aux,dt) 
            yhat = UDE.link([x[1],F[t]],r,dt,UDE.parameters.link)
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
    estimated_map = (x,aux,dt) -> UDE.forecast(x,aux,dt,UDE.parameters.predict,umax,umin)
    
    dt = UDE.times[2]-UDE.times[1]
    ut, r, aux = UDE.predict(UDE.parameters.uhat[:,1],dt,UDE.parameters.predict)

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
    estimated_map = (x,aux,H,dt) -> UDE.forecast_H(x,aux,H,dt,UDE.parameters.pedict,umax,umin)
    df = zeros(T,length(x)+1)    
    for t in 1:T
        if t < T
            dt = times[t+1] - times[t]
            xt,r,aux = estimated_map([x[1]],aux,H[t],dt) 
            yhat = UDE.link([x[1],0.0],r,dt,UDE.parameters.link)
            x = xt
        else
            dt = sum(times[2:end].-times[1:(end-1)])/T # assume final dt is equalt o the average time step
            xt,r,aux = estimated_map([x[1]],aux,H[t],dt) 
            yhat = UDE.link([x[1],0.0],r,dt,UDE.parameters.link)
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
    estimated_map = (x,aux,dt) -> UDE.forecast_F(x[1],x[2],aux,dt,UDE.parameters.predict,umax,umin)
    dt = UDE.times[2]-UDE.times[1]
    ut, r, aux = UDE.predict(UDE.parameters.uhat[:,1],dt,UDE.parameters.predict)

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
            yhat = UDE.link([x[1],F[t]],r,dt,UDE.parameters.link)
            x = xt
        else
            dt = sum(times[2:end].-times[1:(end-1)])/T # assume final dt is equalt o the average time step
            xt,r,aux = estimated_map([x[1],F[t]],aux,dt) 
            yhat = UDE.link([x[1],F[t]],r,dt,UDE.parameters.link)
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
    plot!(p1,testing.t,testing.y)
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

    MSE = (1-weight_H)*abs.(pred_y .- test_y)
    MSE += weight_H*abs.(pred_H .- test_H)

    return MSE, keep
end 

function leave_future_out_cv(model; forecast_length = 10,  forecast_number = 10, spacing = 1, step_size = 0.05, maxiter = 500, step_size2 = 0.01, maxiter2 = 500, verbos = false)
    
    if model.X == []
        # get final time
        data = model.dataframe
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
   
    for i in eachindex(test_sets)[2:end]
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
    for i in eachindex(MSE)
         Plots.plot!(p1,1:length(MSE[i]),MSE[i], c = color, alpha = 0.5, width = 0.75, label = "")
    end 
    return p1
end 

function plot_simulation_tests!(plt,MSE,Ntests;label = "", color = 2)
    mean_ =simulaiton_test_means(MSE,Ntests)
    Plots.plot!(plt,1:length(mean_),mean_, label = label, xlabel = "time", ylabel = "Absolute Error", c = color, width = 2)
    for i in 1:eachindex(MSE)
         Plots.plot!(plt,1:length(MSE[i]),MSE[i], c = color, alpha = 0.5, width = 0.75, label = "")
    end 
end 

