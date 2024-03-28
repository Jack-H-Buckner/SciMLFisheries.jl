using UniversalDiffEq

# Production models + Key word arguments 
model = SurplusProduction(data,production_model = "DelayEmbedding", produciton_hyper_parameters = (lags=5,hidden=10,seed=123,extrap_value=0.0,extrap_length=0.5), regularizaiton_type = "L2", regularizaiton_weight = 10^-4) 
model = SurplusProduction(data,production_model = "DelayEmbeddingDropOut", produciton_hyper_parameters = (drop_prob=0.1))
model = SurplusProduction(data, production_model = "DelayEmbeddingARD",regularizaiton_weight = (L1 = 10^-3.5, L2 = 10^-3.5))
model = SurplusProduction(data,production_model = "LSTM", produciton_hyper_parameters = (cell_dim=10))
model = SurplusProduction(data,production_model = "LSTMDropOut", produciton_hyper_parameters = (cell_dim = 10, drop_prob = 0.1))
model = SurplusProduction(data,production_model = "FeedForward",produciton_hyper_parameters = (hidden=10,seed=123,extrap_value=0.0,extrap_length=0.5),regularizaiton_type = "L2", regularizaiton_weight = 10^-4 )

# 