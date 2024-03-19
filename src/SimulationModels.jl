
function up_down(t, T, Fmax)
    Tswitch = round(0.5*T)
    Finc = Fmax / Tswitch
    if t < Tswitch
        return t*Finc 
    else
        return Fmax - Finc*(t - Tswitch) #+ (Fmax - Finc*(t - Tswitch))
    end
        
end 

function saw(t, T,Fmax)
    Tswitch1 = round(0.33*T)
    Tswitch2 = round(0.66*T)
    Finc = Fmax / Tswitch1
    if t < Tswitch1
        return t*Finc 
    elseif t < Tswitch2
        return Fmax - 0.5*(t - Tswitch1)*Finc
    else 
        return 0.5*Fmax + 0.5*(t - Tswitch2)*Finc
    end
        
end 

function moratorium(t, T, Fmax)
    Tswitch = round(0.5*T)
    Finc = Fmax / Tswitch
    if t < Tswitch
        return t*Finc 
    else
        return 0.0
    end
        
end 

function up_level(t, T, Fmax)
    Tswitch = round(0.5*T)
    Finc = Fmax / Tswitch
    if t < Tswitch
        return t*Finc #+ t*Finc
    else
        return Fmax 
    end   
end 

function SurplusProduction(;seed = 123,T = 30, p = (r = 0.4, K = 10.0), F = up_down, Fmax = 0.25, q = 1.0,sigma_proc = 0.05, plot = true)
  
    B = p.K; Bls = []; Hls = []
    for t in 1:T
        B += p.r*B*(1-B/p.K) - F(t,T,Fmax)*B + B*rand(Distributions.Normal(0,sigma_proc))
        push!(Bls,B);push!(Hls,F(t,T,Fmax)*B) 
    end 
    
    if plot
        data = DataFrame(t = collect(1:T), B = log.(Bls), H = Hls)
        plt = Plots.plot(data.t,data.B, xlabel = "time", ylabel = "abundance")
        Plots.plot!(data.t,data.H, xlabel = "time", ylabel = "abundance")
        return data, plt
    end 
    
    data = DataFrame(t = collect(1:T), B = log.(Bls), H = Hls)    
    
    return data
end 
    
function SurplusProduction_forecast(;seed = 123,T = 30, p = (r = 0.4, K = 10.0), F = up_down, 
                            Fmax = 0.25, q = 1.0,sigma_proc = 0.05, plot = true, Tforecast = 10, Hvals = [0.5,0.75,1.0,1.25])
    
    B = p.K; Bls = []; Hls = []
    for t in 1:T
        B += p.r*B*(1-B/p.K) - F(t,T,Fmax)*B + B*rand(Distributions.Normal(0,sigma_proc))
        push!(Bls,B);push!(Hls,F(t,T,Fmax)*B) 
    end 
        
    forecasts = []
    for H in Hvals
        B = Bls[end]
        Bls_f = []; Hls_f = []
        for t in 1:Tforecast
            B += p.r*B*(1-B/p.K) - H + B*rand(Distributions.Normal(0,sigma_proc))
            push!(Bls_f,B);push!(Hls_f,H)
        end 
        push!(forecasts, DataFrame(t = T.+ collect(1:Tforecast), B = log.(Bls_f), H = Hls_f))
        
    end 
        
    
        
    if plot
        data = DataFrame(t = collect(1:T), B = log.(Bls), H = Hls)
        plt = Plots.plot(data.t,data.B, xlabel = "time", ylabel = "abundance")
        Plots.plot!(data.t,data.H, xlabel = "time", ylabel = "abundance")
        return data, forecasts, plt
    end 
    
    data = DataFrame(t = collect(1:T), B = log.(Bls), H = Hls)    
    
    return data, forecasts
end 

function simulation(derivs,noise,tspan,u0,tsteps,p,sigma_x,sigma_H)
    
    # generate time series with DifferentialEquations.jl
    prob = SDEProblem(derivs,noise, u0, tspan, p); dt = 0.01
    ode_data = Array(solve(prob, SRIW1(), dt = dt, adaptive = false, saveat = tsteps))
    
    index = ode_data[1,1:(end-1)]
    B0 = index[end]
    index = 4.0 * index ./ B0
    y = log.(index) .+ rand(Distributions.Normal(0.0,sigma_x), length(ode_data[1,2:end]))
        
    H = 4.0 * (ode_data[2,2:end] .- ode_data[2,1:(end-1)])./B0 .+ rand(Distributions.Normal(0.0,sigma_H), length(ode_data[2,2:end]))
    
    X = ode_data[3:end,1:(end-1)]
    
    training_data = DataFrame(t = tsteps[2:end], y = y, H = H)
    training_X = DataFrame(transpose(X), :auto)
    
    return training_data, training_X, ode_data, B0
end 

function test_data_times(Tforecast,tsteps,tspan,Hvals)
    dt = sum(tsteps[2:end].-tsteps[1:(end-1)])/length(tsteps[2:end])
    testing_tspan = (tspan[2]-dt-10^-3,tspan[2]+Tforecast*dt+10^-3)
    testing_tsteps = collect((tspan[2]-dt):dt:(tspan[2]+Tforecast*dt))
    Hvals = Hvals ./ dt
    return testing_tspan, testing_tsteps, Hvals, dt
end

function simulate_test_data(derivs,noise,u0,p,testing_tspan,testing_tsteps,Hvals,sigma_x,B0)
    
    test_sets = []; test_X =[]
    for H in Hvals
        pars = (p,H)
        prob = SDEProblem(derivs,noise, u0, testing_tspan, pars)
        ode_test_data = Array(solve(prob, SRIW1(), dt = 0.01, adaptive = false, tspan = testing_tspan, saveat = testing_tsteps))
        
        index = ode_test_data[1,1:(end-1)]
        index[index .<= 0.0] .= 0.001
        index = 4.0 *index ./ B0
        y = log.(index) .+ rand(Distributions.Normal(0.0,sigma_x), length(ode_test_data[1,2:end]))
            
        H = 4.0*(ode_test_data[2,2:end] .- ode_test_data[2,1:(end-1)])/B0
            
        X = ode_test_data[3:end,1:(end-1)]

        # training data
        test_data = DataFrame(t = testing_tsteps[2:end], y = y, H = H)
        test_x = DataFrame(transpose(X), :auto)
        
        push!(test_sets,test_data);push!(test_X,test_x)
    end
    
    return test_sets, test_X
end

function LogisticLorenz(;plot = true, datasize = 100,T = 30.0,p = (r=1.25,K = 10,sigma = 10, rho = 28, beta = 8/3,link = 0.05), F = up_down, Fmax = 0.25, q = 1.0, Tforecast = 10, Hvals = [0.0,1.0,2.0,3.0], sigma_x = 0.0, sigma_H = 0.0,seed = 123)
    
    Random.seed!(seed)
    # set parameters for data set 
    tspan = (0.0f0, T)
    tsteps = range(tspan[1], tspan[2], length = datasize)

    # model parameters
    u0 = Float32[10.0, 0.0,-10.0, 10.0*(2*rand()-1), 30.0]
    

    function derivs(u, p, t)
        du = zeros(5)
        du[1] = p.r*u[1]*(1-u[1]/p.K) + p.link*u[1]*u[4] - F(t,T,Fmax)*u[1]
        du[2] = F(t,T,Fmax)*u[1]
        du[3] = p.sigma*(u[4]-u[3])
        du[4] = u[3]*(p.rho-u[5]) - u[3]
        du[5] = u[3]*u[4] - p.beta*u[5]
        return du
    end
    
    function noise(u, p, t)
        dW = zeros(5)
        return dW
    end
    
    function derivs_H(u, p, t)
        du = zeros(5)
        du[1] = p[1].r*u[1]*(1-u[1]/p[1].K) + p[1].link*u[1]*u[4] - p[2]
        du[2] = p[2]
        du[3] = p[1].sigma*(u[4]-u[3])
        du[4] = u[3]*(p[1].rho-u[5]) - u[3]
        du[5] = u[3]*u[4] - p[1].beta*u[5]
        if u[1] <= 0.0
            du[1] = 0.0
            du[2] = 0.0
        end
        return du
    end

    # generate time series with DifferentialEquations.jl
    training_data, training_X, ode_data = simulation(derivs,noise,tspan,u0,tsteps,p,sigma_x,sigma_H)
    uT = ode_data[:,end-1]
    
    # training data
    training_plt = 0
    if plot
        training_plt = Plots.plot(training_data.t,training_data.y, xlabel = "time", ylabel = "abundance")
        Plots.plot!(training_data.t,training_data.H, xlabel = "time", ylabel = "abundance")
    end
    
    # testing data
    testing_tspan, testing_tsteps, Hvals, dt = test_data_times(Tforecast,tsteps,tspan,Hvals)
    test_sets, test_X = simulate_test_data(derivs_H,noise,uT,p,testing_tspan,testing_tsteps,Hvals,sigma_x,B0)
    
    return training_data, training_X, test_sets, test_X, training_plt
    
end 


function LogisticPeriodic(;plot = true, datasize = 100,T = 30.0,p = (r=0.3,K = 10,omega = 4,A = 0.4), F = up_down, Fmax = 0.25, q = 1.0, sigma_x = 0.05, sigma_H = 0.05, Tforecast = 10, Hvals = [0.0,1.0,2.0,3.0],seed = 123)
 
    Random.seed!(seed)
    
    # set parameters for data set 
    tspan = (0.0f0, T)
    tsteps = range(tspan[1], tspan[2], length = datasize)

    # model parameters
    u0 = Float32[10,0.0,0.5,0.0]
    

    function derivs(u, p, t)
        du = zeros(4)
        du[1] = p.r*u[1]*(1-u[1]/p.K) - F(t,T,Fmax)*u[1] + p.A*u[1]*u[3]
        du[2] = F(t,T,Fmax)*u[1]
        du[3] = -6.28318*u[4]/p.omega # cos(p.omega)*u[3] - sin(p.omega)*u[4]
        du[4] = 6.28318*u[3]/p.omega # sin(p.omega)*u[3] + cos(p.omega)*u[4]
        return du
    end
    
    function derivs_H(u, p, t)
        du = zeros(4)
        du[1] = p[1].r*u[1]*(1-u[1]/p[1].K) - p[2] + p[1].A*u[1]*u[3] 
        du[2] = p[2]
        du[3] = -6.28318*u[4]/p[1].omega #cos(p[1].omega)*u[3] - sin(p[1].omega)*u[4]
        du[4] = 6.28318*u[3]/p[1].omega #sin(p[1].omega)*u[3] + cos(p[1].omega)*u[4]
        if u[1] < 0.0
            du[1] = 0.0
            du[2] = 0.0
        end
        return du
    end
    
    function noise(u, p, t)
        dW = zeros(4)
        return dW
    end
    
    # generate time series with DifferentialEquations.jl
    training_data, training_X, ode_data = simulation(derivs,noise,tspan,u0,tsteps,p,sigma_x,sigma_H)
    uT = ode_data[:,end-1]
    
    # training data
    training_plt = 0
    if plot
        training_plt = Plots.plot(training_data.t,training_data.y, xlabel = "time", ylabel = "abundance")
        Plots.plot!(training_data.t,training_data.H, xlabel = "time", ylabel = "abundance")
    end
    
    # testing data
    testing_tspan, testing_tsteps, Hvals, dt = test_data_times(Tforecast,tsteps,tspan,Hvals)
    test_sets, test_X = simulate_test_data(derivs_H,noise,uT,p,testing_tspan,testing_tsteps,Hvals,sigma_x)
    
    return training_data, training_X, test_sets, test_X, training_plt
    
end 

function LogisticRedNoise(;plot = true, datasize = 100,T = 30.0,p = (r=0.3,K = 10,rho=-0.5,sigma=0.5),  F = up_down, Fmax = 0.2, q = 1.0, sigma_x = 0.0, sigma_H = 0.0, Tforecast = 10, Hvals = [0.0,1.0,2.0,3.0])
    
    # set parameters for data set 
    tspan = (0.0f0, T)
    tsteps = range(tspan[1], tspan[2], length = datasize)

    # model parameters
    u0 = Float32[10,0.0,0.0]
    

    function derivs(u, p, t)
        du = zeros(3)
        du[1] = p.r*u[1]*(1-u[1]/p.K) - F(t,T,Fmax)*u[1] + u[1]*u[3]
        du[2] = F(t,T,Fmax)*u[1]
        du[3] = p.rho*u[3] 
        return du
    end
    
    function derivs_H(u, p, t)
        du = zeros(3)
        du[1] = p[1].r*u[1]*(1-u[1]/p[1].K) - p[2] + u[1]*u[3]
        du[2] = p[2]
        du[3] = p[1].rho*u[3] 
        if u[1] < 0.0
            du[1] = 0.0
        end
        return du
    end
    
    function noise(u, p, t)
        dW = zeros(3)
        dW[3] = p.sigma
        return dW
    end
    
    function noise_H(u, p, t)
        dW = zeros(3)
        dW[3] = p[1].sigma
        return dW
    end

    # generate time series with DifferentialEquations.jl
    training_data, training_X, ode_data = simulation(derivs,noise,tspan,u0,tsteps,p,sigma_x,sigma_H)
    uT = ode_data[:,end-1]
    
    # training data
    training_plt = 0
    if plot
        training_plt = Plots.plot(training_data.t,training_data.y, xlabel = "time", ylabel = "abundance")
        Plots.plot!(training_data.t,training_data.H, xlabel = "time", ylabel = "abundance")
    end
    
    # testing data
    testing_tspan, testing_tsteps, Hvals, dt = test_data_times(Tforecast,tsteps,tspan,Hvals)
    test_sets, test_X, B0 = simulate_test_data(derivs_H,noise_H,uT,p,testing_tspan,testing_tsteps,Hvals,sigma_x)
    
    return training_data, training_X, test_sets, test_X, training_plt, B0
    
end 

function LogisticPrey(; plot=true, datasize = 100,T = 30.0,p = (r=0.1,K = 20.0,m=0.3,sigma=0.25, r2 = 0.75, k2 = 100, alpha = 0.1,theta = 0.1,N0 = 13.0, P0 = 6.8),  F = up_down, Fmax = 0.2, q = 1.0, sigma_x = 0.0, sigma_H = 0.0, Tforecast = 10, Hvals = [0.0,1.0,2.0,3.0],seed = 123)
    Random.seed!(seed)
    # set parameters for data set 
    tspan = (0.0f0, T)
    tsteps = range(tspan[1], tspan[2], length = datasize)

    # model parameters
    u0 = Float32[p.P0,0.0,p.N0]
    

    function derivs(u, p, t)
        du = zeros(3)
        du[1] = p.r*u[1]*(1-u[1]/p.K) - F(t,T,Fmax)*u[1] + p.theta*p.alpha*u[1]*u[3] - p.m*u[1]
        du[2] = F(t,T,Fmax)*u[1]
        du[3] = p.r2*u[3]*(1-u[3]/p.k2) - p.alpha*u[3]*u[1] #*
        return du
    end
    
    function noise(u, p, t)
        dW = zeros(3)
        dW[1] = p.sigma
        dW[3] = p.sigma
        return dW
    end
    
    function derivs_H(u, p, t)
        du = zeros(3)
        du[1] = p[1].r*u[1]*(1-u[1]/p[1].K) - p[2] + p[1].theta*p[1].alpha*u[1]*u[3] - p[1].m*u[1]
        du[2] = p[2]
        du[3] = p[1].r2*u[3]*(1-u[3]/p[1].k2) - p[1].alpha*u[3]*u[1] 
        if u[1] <= 0 
            du[1] = 0
            du[2] = 0
        end
        return du
    end
    
    function noise_H(u, p, t)
        dW = zeros(3)
        dW[1] = p[1].sigma
        dW[3] = p[1].sigma
        return dW
    end

    # generate time series with DifferentialEquations.jl
    training_data, training_X, ode_data, B0 = simulation(derivs,noise,tspan,u0,tsteps,p,sigma_x,sigma_H)
    uT = ode_data[:,end-1]
    
    # training data
    training_plt = 0
    if plot
        training_plt = Plots.plot(training_data.t,training_data.y, xlabel = "time", ylabel = "abundance")
        Plots.plot!(training_data.t,training_data.H, xlabel = "time", ylabel = "abundance")
    end
    
    # testing data
    testing_tspan, testing_tsteps, Hvals, dt = test_data_times(Tforecast,tsteps,tspan,Hvals)
    test_sets, test_X = simulate_test_data(derivs_H,noise_H,uT,p,testing_tspan,testing_tsteps,Hvals,sigma_x,B0)
    
    return training_data, training_X, test_sets, test_X, training_plt, B0
    
end

function ThreeSpeciesModel(;plot = true, datasize = 100,T = 30.0,p = (r=0.5,K=20.0,C1=0.5,A1=0.2,B1=0.1,D1=0.2,C2=0.5,A2=0.2,B2=0.1,D2=0.2,X0=10.0,Y0=5.0,Z0=1.0,sigma = 0.0),  F = up_down, Fmax = 0.2, q = 1.0, sigma_x = 0.0, sigma_H = 0.0, Tforecast = 10, Hvals = [0.0,1.0,2.0,3.0], seed = 123)
    
    Random.seed!(seed)
    
    # set parameters for data set 
    tspan = (0.0f0, T)
    tsteps = range(tspan[1], tspan[2], length = datasize)

    # model parameters
    u0 = Float32[p.Y0,0.0,p.X0,p.Z0]
    
    function derivs(u, p, t)
        du = zeros(4)
        du[1] = p.C1*p.A1*u[1]*u[3]/(1+p.B1*u[3])-p.A2*u[1]*u[4]/(1+p.B2*u[1]) - p.D1*u[1] - F(t,T,Fmax)*u[1]
        du[2] = F(t,T,Fmax)*u[1]
        du[3] = p.r*u[3]*(1-u[3]/p.K) - p.A1*u[1]*u[3]/(1+p.B1*u[3])
        du[4] = p.C2*p.A2*u[1]*u[4]/(1+p.B2*u[1]) - p.D2*u[4]
        return du
    end
    
    function noise(u, p, t)
        dW = zeros(4)
        dW[1] = u[1]*p.sigma
        return dW
    end
    
    function derivs_H(u, p, t)
        du = zeros(4)
        du[1] = p[1].C1*p[1].A1*u[1]*u[3]/(1+p[1].B1*u[3])-p[1].A2*u[1]*u[4]/(1+p[1].B2*u[1]) - p[1].D1*u[1] - p[2]
        du[2] = p[2]
        du[3] = p[1].r*u[3]*(1-u[3]/p[1].K) - p[1].A1*u[1]*u[3]/(1+p[1].B1*u[3])
        du[4] = p[1].C2*p[1].A2*u[1]*u[4]/(1+p[1].B2*u[1]) - p[1].D2*u[4]
        if u[1] <= 0 
            du[1] = 0
            du[2] = 0
        end
        return du
    end
    
    function noise_H(u, p, t)
        dW = zeros(4)
        dW[1] = u[1]*p[1].sigma
        return dW
    end


    # generate time series with DifferentialEquations.jl
    training_data, training_X, ode_data, B0 = simulation(derivs,noise,tspan,u0,tsteps,p,sigma_x,sigma_H)
    uT = ode_data[:,end-1]
    
    # training data
    training_plt = 0
    if plot
        training_plt = Plots.plot(training_data.t,training_data.y, xlabel = "time", ylabel = "abundance")
        Plots.plot!(training_data.t,training_data.H, xlabel = "time", ylabel = "abundance")
    end
    
    # testing data
    testing_tspan, testing_tsteps, Hvals, dt = test_data_times(Tforecast,tsteps,tspan,Hvals)
    test_sets, test_X = simulate_test_data(derivs_H,noise_H,uT,p,testing_tspan,testing_tsteps,Hvals,sigma_x, B0)
    
    return training_data, training_X, test_sets, test_X, training_plt, B0
    
end

function AgeStuructured(;plot = true, T = 30,
        p = (r = 0.5, b = 0.5, rho = 1.0, m = 0.2, sigma = 1.0, lags = 5, B0 = 30.0), 
        F = up_down, Fmax = 0.2, q = 1.0, sigma_x = 0.0, sigma_H = 0.0, 
        Tforecast = 10, Hvals = [0.0,1.0,2.0,3.0], seed = 123)
    
    Random.seed!(seed)
    function Deriso_Schnute(B, Blags, p, F, t)
        R = p.r * Blags[end]* exp(rand(Distributions.Normal(-0.5*p.sigma^2,p.sigma))) / (1+p.b*Blags[end])
        st = exp( - p.m - F(t))
        st1 = exp( - p.m - F(t-1))
        B_ = (1+p.rho)*st*B - p.rho*st*st1*Blags[1] + R
        Blags = vcat(B,Blags[1:(end-1)])
        H = B*(1-exp(-F(t)))
        return B_, H, Blags, R
    end 
    
    function Deriso_Schnute_H(B, Blags, p, H)
        R = p.r * Blags[end]* exp(rand(Distributions.Normal(-0.5*p.sigma^2,p.sigma))) / (1+p.b*Blags[end])
        nugget = 10^-4
        if H > B
            return nugget,B, Blags, R
        end
        
        if H > Blags[1]
            H = Blags[1] - nugget
        end

        F = -log(1-H/B) 
        Ft1 = -log(1-H/Blags[1]) 
        st = exp( - p.m - F)
        st1 = exp( - p.m - Ft1)
        B_ = (1+p.rho)*st*B - p.rho*st*st1*Blags[1] + R
        Blags = vcat(B,Blags[1:(end-1)])
        if B < 0
            B = nugget 
        end 
        
        return B_, H, Blags, R
    end 
    
    
    
    index = []; Hls = []; Rls = []
    B_ = p.B0; Blags = repeat([p.B0],p.lags)
    for t in 1:T
        B_, H_, Blags, R_ = Deriso_Schnute(B_, Blags, p, t -> F(t,T,Fmax), t)
        push!(index, B_);push!(Hls,H_);push!(Rls,R_)
    end 
    B0 = index[end]; Bend = index[end]; Bend_lags = Blags
    index = 4.0 * index ./ B0
    nugget = 10^-7
    index[index .<= 0] .= nugget
    y = log.(index) .+ rand(Distributions.Normal(0.0,sigma_x), length(T))
        
    H = 4.0 * Hls./B0 .+ Hls.*rand(Distributions.Normal(0.0,sigma_H), length(T))

    training_data = DataFrame(t = 1:T, y = y, H = H)
    training_X = DataFrame(x1 = Rls,)
    
    test_sets = []; test_X = []
    for h in Hvals
        index = []; Hls = []; Rls = []
        B_ = Bend; Blags = Bend_lags
        for t in 1:Tforecast
            B_, H_, Blags, R_ = Deriso_Schnute_H(B_, Blags, p, h)
            push!(index, B_);push!(Hls,H_);push!(Rls,R_)
        end 
        
        index = 4.0 * index ./ B0
        index[index .<= 0] .= nugget
        y = log.(index) .+ rand(Distributions.Normal(0.0,sigma_x), length(T))

        H = 4.0 * Hls./B0 

        test_data = DataFrame(t = (T+1):(T+Tforecast), y = y, H = H)
        test_X_ = DataFrame(x1 = Rls)
        
        push!(test_sets,test_data);push!(test_X,test_X_)
    end 
    
    training_plt = Plots.plot(training_data.t, training_data.y)
    Plots.plot!(training_data.t, training_data.H, xlabel = "Time", ylabel = "Harvest + ")
    
    return training_data, training_X, test_sets, test_X, training_plt, B0
    
end 