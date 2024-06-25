# Link funcitons 
function DiscreteAproxHarvest(u,r,dt,theta)
    return dt*theta*exp(u[1])*u[2]
end 

function LinearAproxHarvest(u,r,dt,theta)
    F = u[2]; B = exp(u[1])
    if (r-F) == 0 
        return dt*theta*F*B
    end
    return theta*F*B*(exp((r-F/theta)*dt) - 1)/(r-F/theta)
end 

function DiscreteAproxIdentityHarvest(u,r,dt,theta)
    return dt*theta*exp(u[1])*u[2]
end 

function LinearIndex(u,p)
    return u[1] - p.q
end 

function Hyperstability(u,p)
    return p.b*u[1] - p.q
end 


function IdentityIndex(u,p)
    return u[1]
end 



function DataModel(harvest,index,loss,sigma_harvest,sigma_index,theta,sd,mu)
    
    # Harvest link function 
    harvest_model = x -> 0
    
    if harvest == "DiscreteAprox"
        harvest_model = (u,r,dt) -> DiscreteAproxHarvest(u,r,dt,theta)
    elseif harvest == "LinearAprox"
        harvest_model = (u,r,dt) -> LinearAproxHarvest(u,r,dt, theta)
    else
        print("Your choice of harvest model does not match avaiable options")
        throw(error())   
    end
    
    # Index link function 
    index_model = x -> 0;link_params = NamedTuple()
    if index == "Identity"
        index_model = IdentityIndex
        link_params = (q=0.0,)
    elseif index == "Linear"
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
        loss_function,loss_params = FixedVariance([sigma_index,sigma_harvest]) 
    elseif loss == "EstimateVariance"
        loss_function,loss_params = EstimateVariance([sigma_index,sigma_harvest])
    else
        print("Your choice of likelihood does not match avaiable options")
        throw(error()) 
    end

    function link(u,r,dt,p)
        yt = index_model(u,p); H =0
        if index == "Identity"
            H = exp(p.q)*harvest_model(u,r,dt) 
        else
            H = harvest_model(u,r,dt) 
        end
        return [yt,H]
    end

    return link, loss_function, loss_params,link_params
end 
