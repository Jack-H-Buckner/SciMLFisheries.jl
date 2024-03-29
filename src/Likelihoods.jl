function FixedVariance(sigmas)
    parameters = NamedTuple()
    loss = (u,uhat,parameters) -> sum((u .- uhat).^2 ./sigmas.^2 )
    return loss, parameters
end


function EstimateVariance(sigma_0)
    parameters = (sigma = sigma_0,)
    
    function loss(u,uhat,parameters) 
        V = parameters.sigma.^2 .+ 10^-8.0
        Z = 0.5*log.(V)
        ll = 0.5 * (u .- uhat).^2 ./ V
        return sum(ll .+ Z)  
    end
        
    return loss, parameters
end

function gamma_parameters(mean,variance)
    theta = variance/mean
    k = mean^2/variance
    return k, theta
end 

gamma_lpdf(sigma,k,theta) = -1 * ((k-1)*log(sigma) - sigma/theta)

function variance_ratio_prior(sigma_y,sigma_H,sigma_B,sigma_F,k_y,theta_y,k_H,theta_H,k_B,theta_B,k_F,theta_F)
    
    Ly = gamma_lpdf(sigma_y^2,k_y,theta_y)
    LrH = gamma_lpdf(sigma_H^2 / sigma_y^2, k_H, theta_H)
    LrB = gamma_lpdf(sigma_B^2 / sigma_y^2, k_B, theta_B)
    LrF = gamma_lpdf(sigma_F^2 / sigma_y^2, k_F, theta_F)

    return Ly + LrH + LrB + LrF
end

function init_variance_prior(var_y, sigma_y, rH, sigma_rH, rB, sigma_rB, rF, sigma_rF)
    k_y,theta_y = gamma_parameters(var_y, sigma_y)
    k_H,theta_H = gamma_parameters(rH, sigma_rH,)
    k_B,theta_B = gamma_parameters(rB, sigma_rB)
    k_F,theta_F = gamma_parameters(rF, sigma_rF)
    function variance_prior(observation_loss, process_loss)
        sigma_y = observation_loss.sigma[1];sigma_H = observation_loss.sigma[2]
        sigma_B = process_loss.sigma[1];sigma_F = process_loss.sigma[2]
        return variance_ratio_prior(sigma_y,sigma_H,sigma_B,sigma_F,k_y,theta_y,k_H,theta_H,k_B,theta_B,k_F,theta_F)
    end
    return variance_prior
end 

