using Plots,DelimitedFiles, Statistics, LaTeXStrings

GPU =  readdlm("../benchmark/resultsGPU_varying_batchsize.txt")
CPU =  readdlm("../benchmark/resultsCPU_varying_batchsize.txt")

range=round.(Int,2 .^(1:1:15))

meanGPU = mean(GPU,dims=2)
meanCPU= mean(CPU,dims=2)
stdGPU = std(GPU,dims=2)
stdCPU = std(CPU,dims=2)

plot1=plot(
    range, 
    [meanCPU meanGPU],
    labels=["CPU" "GPU"], 
    c=[1 2],
    lw=2,
    xaxis = :log2
    )

#Add shadow for standard deviations
plot!(range, 
    meanCPU-stdCPU, 
    fillrange = meanCPU+stdCPU, 
    fillalpha = 0.35, 
    c = 1, 
    label = "",
    lw=0,
    xaxis = :log2
    )

plot!(range, 
    meanGPU-stdGPU, 
    fillrange = meanGPU+stdGPU, 
    fillalpha = 0.35, 
    c = 2, 
    label = "",
    lw=0,
    xaxis = :log2, 
    legend=:topleft)

ylabel!("Seconds")
xlabel!(L"log_2(b)")
