using MLDatasets
include("../FlyHash.jl")
Random.seed!(42)

d = 32 * 32 * 3
dataset = MLDatasets.CIFAR10()
data = Float32.(reshape(dataset.features, (d, 50000)))
hashfactor = 32
projM = randproj(d, Int(d * hashfactor), round(Int, d * 5 / 100))
winners = round(Int, d * hashfactor * 5 / 100)

for _ in 1:11
    tCPU = CUDA.@elapsed flyhash(data, projM, winners)
    open("resultsCPU_CIFAR10.txt", "a") do io
        write(io, "$tCPU ")
    end
end

