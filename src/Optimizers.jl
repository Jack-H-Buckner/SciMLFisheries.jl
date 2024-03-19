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