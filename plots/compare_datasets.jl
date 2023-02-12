using StatsPlots, Statistics, DelimitedFiles, LaTeXStrings

GPU_cifar = readdlm("../benchmark/resultsGPU_CIFAR10.txt")
CPU_cifar = readdlm("../benchmark/resultsCPU_CIFAR10.txt")

GPU_fM = readdlm("../benchmark/resultsGPU_FashionMNIST.txt")
CPU_fM = readdlm("../benchmark/resultsCPU_FashionMNIST.txt")

meanGPU_cifar = mean(GPU_cifar[:, 2:end])
meanCPU_cifar = mean(CPU_cifar[:, 2:end])
stdGPU_cifar = std(GPU_cifar[:, 2:end])
stdCPU_cifar = std(CPU_cifar[:, 2:end])

meanGPU_fM = mean(GPU_fM[:, 2:end])
meanCPU_fM = mean(CPU_fM[:, 2:end])
stdGPU_fM = std(GPU_fM[:, 2:end])
stdCPU_fM = std(CPU_fM[:, 2:end])

architecture = repeat(["GPU", "CPU"], inner=2)
dataset = ["CIFAR10", "FashionMNIST", "CIFAR10", "FashionMNIST"]

plot1 = groupedbar(
    dataset,
    [meanGPU_cifar, meanGPU_fM, meanCPU_cifar, meanCPU_fM],
    yerr=[stdGPU_cifar, stdGPU_fM, stdCPU_cifar, stdCPU_fM],
    group=architecture,
    ylabel="Seconds",
    xlabel="Datasets",
)

