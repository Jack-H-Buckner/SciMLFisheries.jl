using SciMLFisheries

data, training_X, test_sets, test_X, training_plt, B0 = SciMLFisheries.AgeStuructured(sigma_x = 0.1, sigma_H = 0.05, )

# Production models + Key word arguments 
model = SurplusProduction(data,production_model = "DelayEmbedding", regularizaiton_type = "L2", produciton_parameters = (lags=5,hidden=10,seed=123,extrap_value=0.0,extrap_length=0.5,regularizaiton_weight=10^-4))
model = SurplusProduction(data,production_model = "DelayEmbeddingDropOut", produciton_parameters = (drop_prob=0.1))
model = SurplusProduction(data,production_model = "DelayEmbeddingARD")

model = SurplusProduction(data,production_model = "LSTM",produciton_parameters = (cell_dim=10))
model = SurplusProduction(data,production_model = "LSTMDropOut", produciton_parameters = (cell_dim = 10, drop_prob = 0.1))
model = SurplusProduction(data,production_model = "FeedForward")

# Observation models
model = SurplusProduction(data,harvest_model = "DiscreteAprox",harvest_parameters = (theta = 1.0,) )
model = SurplusProduction(data,harvest_model = "LinearAprox")

model = SurplusProduction(data,index_model="Linear",index_priors = (q = 0.0, sigma_q = 10.0))
model = SurplusProduction(data,index_model="HyperStability",index_priors = (q = 0.0, sigma_q = 10.0, b = 1.0, sigma_b = 10.0))

model = SurplusProduction(data,likelihood="FixedVariance",variance_priors = (sigma_H=0.1, sigma_y = 0.1, sigma_B = 0.05, sigma_F = 0.2))
model = SurplusProduction(data,likelihood="EstimateVariance",variance_priors = (sigma_y = 0.1, sd_sigma_y=0.05,rH=0.25,sd_rH=0.025,rB=1.0,sd_rB=0.1,rF=5.0,sd_rF=0.2))