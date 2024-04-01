
function q_prior(Eq,sigma)
    loss = parameters -> (parameters.q .- Eq).^2/sigma^2
    return loss
end 

function q_and_b_prior(Eq,Eb, sigma_q, sigma_b)
    loss = parameters -> (parameters.q .- Eq).^2/sigma_q^2 + (parameters.b .- Eb).^2/sigma_b^2
    return loss
end 