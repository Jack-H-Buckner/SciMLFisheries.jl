
function distance_matrices(inducing_points)
    N = size(inducing_points)[1]; d = size(inducing_points)[2]
    distances = zeros(N,N,d)
    for i in 1:d
        distances[:,:,i] .= (inducing_points[:,i]' .- inducing_points[:,i])
    end
    return d, N, distances, Matrix(I,N,N)
end 

function ARD_covariance(p, distances,DiagMat)
    nugget = 10^-6
    Sigma = p.alpha^2 * exp.(-1 * sum(p.l.^2 .* distances.^2 , dims=3)[:,:,1]) .+ (p.sigma^2 + nugget) .* DiagMat
end


function cov_function(x1,x2,p) 
    p.alpha^2 * exp(-sum(p.l[1,1,:].^2 .* (x1.- x2).^2))
end 


struct GaussianProcess
    N
    d
    inducing_points
    distances
    diagonal
    psi
    mu 
    mu_values
end 

function init_parameters(GP)
    N = GP.N; d = GP.d
    l = zeros(1,1,d) .+ 1.0
    inducing_values = zeros(N)
    alpha = 0.1
    sigma = 0.1
    return ComponentArray(inducing_values = inducing_values, l = l, alpha = alpha, sigma = sigma)
end

function GaussianProcess(inducing_points)
    d, N, distances, diagonal = distance_matrices(inducing_points)
    psi = 3.14159/2 
    mu = x -> 0
    mu_values = mapslices(mu, inducing_points, dims = 2)[:,1]
    GP = GaussianProcess(N,d,inducing_points,distances,diagonal,psi,mu, mu_values)
    parameters = init_parameters(GP)
    return GP, parameters
end 


function GaussianProcess(inducing_points,psi)
    d, N, distances, diagonal = distance_matrices(inducing_points)
    mu = x -> 0
    mu_values = mapslices(mu, inducing_points, dims = 2)[:,1]
    GP = GaussianProcess(N,d,inducing_points,distances,diagonal,psi, mu, mu_values)
    parameters = init_parameters(GP)
    return GP, parameters
end

function GaussianProcess(inducing_points,psi,mu)
    d, N, distances, diagonal = distance_matrices(inducing_points)
    mu_values = mapslices(mu, inducing_points, dims = 2)[:,1]
    GP = GaussianProcess(N,d,inducing_points,distances,diagonal,psi, mu, mu_values)
    parameters = init_parameters(GP)
    return GP, parameters
end

function likelihood(parameters,GP)
    Sigma = ARD_covariance(parameters, GP.distances, GP.diagonal)
    0.5*( GP.d*log(2*3.14159) + log(det(Sigma)) + (parameters.inducing_values .- GP.mu_values)'*inv(Sigma)*(parameters.inducing_values .- GP.mu_values))
end


function (GP::GaussianProcess)(x,p)
    weights =  broadcast(i->cov_function(GP.inducing_points[i,:],x,p),1:GP.N)
    Sigma = ARD_covariance(p, GP.distances, GP.diagonal)
    Sigma_inv = inv(Sigma)
    y =  GP.mu(x) .+ weights' * Sigma_inv * (p.inducing_values .- GP.mu_values)
    nugget = 10^-6
    sigma  =  p.alpha^2 + p.sigma^2 - weights' * Sigma_inv * weights + nugget
    return y, sigma
end

function lagged_data(data,lags)

    y = data[1,:]
    T = length(y)
    lagged_data = zeros(T-lags,lags+1)

    for i in 1:(T-lags)
        lagged_data[i,:] = data[i:(i+lags)]
    end

    return lagged_data
end 