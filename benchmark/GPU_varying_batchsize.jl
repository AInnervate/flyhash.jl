include("../FlyHash.jl")
Random.seed!(42)

batchsize_range = round.(Int, 2 .^ (1:1:15))
hashfactor = 2^7
d = 2^10
winners = round(Int, d * hashfactor * 5 / 100)
for c in batchsize_range
    data = CUDA.rand(d, c)
    projM = randproj_GPU(d, round(Int, d * hashfactor), round(Int, d * 5 / 100))
    for _ in 1:11
        tGPU = CUDA.@elapsed flyhash_GPU(data, projM, winners)
        open("resultsGPU_varying_batchsize.txt", "a") do io
            write(io, "$tGPU ")
        end
    end
    open("resultsGPU_varying_batchsize.txt", "a") do io
        write(io, "\n")
    end
end