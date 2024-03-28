using SciMLFisheries
using Test

@testset "SciMLFisheries.jl" begin
    include("test_surplus_production.jl")
    include("test_docs_examples.jl")
end
