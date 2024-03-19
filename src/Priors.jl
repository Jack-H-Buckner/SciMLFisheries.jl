
function q_prior(Eq,weight)
    loss = parameters -> weight*(parameters.q .- Eq).^2
    return loss
end 

function q_and_b_prior(Eq,Eb,weight)
    loss = parameters -> (weight.q)*(parameters.q .- Eq).^2 + (weight.b)*(parameters.b .- Eb).^2
    return loss
end 