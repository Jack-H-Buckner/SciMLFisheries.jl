# Link funcitons 
function DiscreteAproxHarvest(u,r,dt)
    return dt*exp(u[1])*u[2]
end 

function LinearAproxHarvest(u,r,dt)
    F = u[2]; B = exp(u[1])
    if (r-F) == 0 
        return dt*F*B
    end
    return F*B*(exp((r-F)*dt) - 1)/(r-F)
end 

function LinearIndex(u,p)
    return u[1] - p.q
end 

function Hyperstability(u,p)
    return p.b*u[1] - p.q
end 

function DataModel(harvest,index,loss,sigma_harvest,sigma_index)
 
    # Harvest link function 
    harvest_model = x -> 0
    if harvest == "DiscreteAprox"
        harvest_model = DiscreteAproxHarvest
    elseif harvest == "LinearAprox"
        harvest_model = LinearAproxHarvest
    else
        print("Your choice of harvest model does not match avaiable options")
        throw(error())   
    end 
    
    # Index link function 
    index_model = x -> 0;link_params = NamedTuple()
    if index == "Linear"
        index_model = LinearIndex
        link_params = (q=0.0,)
    elseif index == "HyperStability"
        index_model = Hyperstability
        link_params = (q=0.0,b=1.0)
    else
        print("Your choice of index model does not match avaiable options")
        throw(error())  
    end 
    
    # likelihood function 
    loss_function = x -> 0;loss_params = NamedTuple()
    if loss == "FixedVariance"
        loss_function,loss_params = FixedVariance([sigma_harvest,sigma_index]) 
    elseif loss == "EstimateVariance"
        loss_function,loss_params = EstimateVariance([sigma_harvest,sigma_index])
    else
        print("Your choice of likelihood does not match avaiable options")
        throw(error()) 
    end
        
        
    function link(u,r,dt,p)
        yt = index_model(u,p)
        H = harvest_model(u,r,dt) 
        return [yt,log(H)]
    end
        
    return link, loss_function, loss_params,link_params
end 