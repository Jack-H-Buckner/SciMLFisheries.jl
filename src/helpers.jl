
function find_time_alias(nms)
    time_alias = ["T", "t","time", "Time", "times", "Times"] 
    ind = broadcast(nm -> nm in nms, time_alias)
    if any(ind)
        return time_alias[ind][1]   
    end
    print("Cannot find column for time ")
    throw(error()) 
end 

function find_harvest_alias(nms)
    harvest_alias = ["H", "h","Harvest", "harvest", "C", "c", "Catch", "catch"] 
    ind = broadcast(nm -> nm in nms, harvest_alias)
    if any(ind)
        return harvest_alias[ind][1]   
    end
    print("Cannot find column for harvest ")
    throw(error()) 
end 

function find_index_alias(nms)
    index_alias = ["I", "i","Index", "index","y","Y"] 
    ind = broadcast(nm -> nm in nms, index_alias)
    if any(ind)
        return index_alias[ind][1]   
    end
    print("Cannot find column for harvest ")
    throw(error()) 
end 

function process_surplus_production_data(data)
    data_ = deepcopy(data)
    nugget = 10^-4
    # Get column names
    nms = names(data_)
    time_alias = find_time_alias(nms)
    harvest_alias = find_harvest_alias(nms)
    index_alias = find_index_alias(nms)
    
    # Times
    dataframe = sort!(data_,[time_alias])
    times = dataframe[:,time_alias]
    dataframe[:,index_alias] .= log.(dataframe[:,index_alias] .+ nugget)
    data_ = transpose(Matrix(dataframe[:,[index_alias,harvest_alias]]))

    T = length(times)

    return times,data_,dataframe,T
end

