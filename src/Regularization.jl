
function L2(weight)
    
    loss = parameters -> weight * (sum(parameters.NN.layer_1.weight.^2) + sum(parameters.NN.layer_2.weight.^2))
    
    return loss
    
end 

function L1(weight)
    
    loss = parameters -> weight * (sum(abs.(parameters.NN.layer_1.weight))+ sum(abs.(parameters.NN.layer_2.weight)))
    
    return loss
    
end 

function L2_drop(weight)
    
    loss = parameters -> weight * (sum(parameters.NN.layer_1.weight.^2) + sum(parameters.NN.layer_3.weight.^2))
    
    return loss
    
end 

function L1_drop(weight)
    
    loss = parameters -> weight * (sum(abs.(parameters.NN.layer_1.weight))+ sum(abs.(parameters.NN.layer_3.weight)))
    
    return loss
    
end 

function L1_LSTM(weight)

    function loss(parameters)
        L = sum(abs.(parameters.Dense.weight))
            
        L += sum(abs.(parameters.LSTM.weight_i))
                
        L += sum(abs.(parameters.LSTM.weight_h))
            
        return weight * L
    end 

    
    return loss
    
end 

function L2_LSTM(weight)
    
    function loss(parameters)
        L = sum((parameters.Dense.weight).^2)
            
        L += sum((parameters.LSTM.weight_i).^2)
                
        L += sum((parameters.LSTM.weight_h).^2)
            
        return weight * L
    end 

    
    return loss
    
end 

function L2_LSTM_drop(weight)

    function loss(parameters)
        L = sum((parameters.Dense.layer_1.weight).^2)
        
        L += sum((parameters.Dense.layer_3.weight).^2)
        
        L += sum((parameters.LSTM.weight_i).^2)
                
        L += sum((parameters.LSTM.weight_h).^2)
            
        return weight * L
    end 

    
    return loss
    
end 

function L1_LSTM_drop(weight)

    function loss(parameters)
        L = sum(abs.(parameters.Dense.layer_1.weight))
        
        L += sum(abs.(parameters.Dense.layer_3.weight))
        
        L += sum(abs.(parameters.LSTM.weight_i))
                
        L += sum(abs.(parameters.LSTM.weight_h))
            
        return weight * L
    end 
    
    return loss
    
end 

function ARD(weight)
    function loss(parameters)
        L = weight.L1*sum(abs.(parameters.ARD))
        L += weight.L2*(sum(parameters.NN.layer_1.weight.^2) + sum(parameters.NN.layer_2.weight.^2))
    end 
end  

function Regularization(loss,model,weight)
 
    loss_function = params -> 0
    if (loss == "L1") & (model in ["FeedForward","DelayEmbedding", "BiomassDelayEmbedding"])
        loss_function = L1(weight)
    elseif (loss == "L2") & (model in ["FeedForward","DelayEmbedding","BiomassDelayEmbedding"])
        loss_function = L2(weight)
    elseif (loss == "L1") & (model in ["DelayEmbeddingDropOut"])
        loss_function = L1_drop(weight)
    elseif (loss == "L2") & (model in ["DelayEmbeddingDropOut"])
        loss_function = L2_drop(weight)
    elseif (loss == "L1") & (model in ["LSTM"])
        loss_function = L1_LSTM(weight)
    elseif (loss == "L2") & (model in ["LSTM"])
        loss_function = L2_LSTM(weight)
    elseif (loss == "L1") & (model in ["LSTMDropOut"])
        loss_function = L1_LSTM_drop(weight)
    elseif (loss == "L2") & (model in ["LSTMDropOut"])
        loss_function = L2_LSTM_drop(weight)
    elseif model == "DelayEmbeddingARD"
        loss_function = ARD(weight)
    elseif model == "theta_logistic"
        return loss_function 
    elseif model == "logistic"
        return loss_function 
    elseif loss == "none"
        return loss_function 
    else
        print("Your choice of production model does not match avaiable options. No regulairzation will be used.")
    end 
    
    return loss_function
    
end 
    