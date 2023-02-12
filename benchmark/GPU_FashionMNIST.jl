using MLDatasets
include("../FlyHash.jl")
Random.seed!(42)

d = 28^2
dataset = MLDatasets.FashionMNIST()
data = cu(reshape(dataset.features, (d, 60000)))
hashfactor = 32
projM = randproj_GPU(d, Int(d * hashfactor), round(Int, d * 5 / 100))
winners = round(Int, d * hashfactor * 5 / 100)

for _ in 1:11
    tGPU = CUDA.@elapsed flyhash_GPU(data, projM, winners)
    open("resultsGPU_FashionMNIST.txt", "a") do io
        write(io, "$tGPU ")
    end
end
