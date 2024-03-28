using SciMLFisheries, DataFrames
# test default paramters 
data = DataFrame(t = 1:10, y = rand(10), H = rand(10))
model = SurplusProduction(data) 
loss = model.loss_function(model.parameters)

# test production models 
production_models = ["FeedForward","LSTM","LSTMDropOut","DelayEmbedding","DelayEmbeddingARD","DelayEmbeddingDropOut"]
for model in production_models
    model = SurplusProduction(data,production_model=model)
    model.loss_function(model.parameters) 
end

# test harvest models 
harvest_models = ["DiscreteAprox","LinearAprox"]
for model in harvest_models
    model = SurplusProduction(data,harvest_model=model)
    model.loss_function(model.parameters) 
end

# test index models 
index_models = ["Linear","HyperStability"]
for model in index_models
    model = SurplusProduction(data,index_model=model)
    model.loss_function(model.parameters) 
end

# test likelihoods
index_models = ["EstimateVariance","FixedVariance"]
for model in index_models
    model = SurplusProduction(data,likelihood=model)
    model.loss_function(model.parameters) 
end
