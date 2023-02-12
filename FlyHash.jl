using CUDA, Random, LinearAlgebra

# CPU random projection matrix
function randproj(inputdim, hashlength, onesperrow)
    M = [ones(Float32, hashlength, onesperrow) zeros(Float32, hashlength, inputdim - onesperrow)]
    for i in 1:hashlength
        M[i, :] = M[i, randperm(inputdim)]
    end
    return M
end
# GPU random projection matrix
randproj_GPU(inputdim, hashlength, onesperrow) = cu(randproj(inputdim, hashlength, onesperrow))
# CPU winner-take-all
function WTA!(x::AbstractVector, nwinners::Int)
    winneridxs = partialsortperm(x, 1:nwinners, rev=true)
    fill!(x, false)
    x[winneridxs] .= true
    return x
end
# CPU FlyHash
function flyhash(X::AbstractMatrix{Float32}, projM::AbstractMatrix{Float32}, nwinners::Int)
    M = projM * X
    for col ∈ collect(eachcol(M))
        WTA!(col, nwinners)
    end
    return M
end

## GPU winner-take-all
function WTA_GPU(X::CuMatrix{Float32}, nwinners::Int)
    nwinners′ = Int32(nwinners)
    lb, ub = minimum(X, dims=1), maximum(X, dims=1)
    @. lb = prevfloat(lb, 2)
    @. ub = nextfloat(ub, 2)
    mid = @. (lb + ub) / 2
    # Preallocations
    tot = similar(mid, Int32)
    isabove = similar(X, Bool)
    ismore = similar(mid, Bool)
    isless = similar(mid, Bool)
    for _ ∈ 1:64
        @. isabove = X > mid
        count!(tot, isabove)
        @. ismore = tot > nwinners′
        @. isless = tot < nwinners′
        @. lb = ifelse(ismore, mid, lb)
        @. ub = ifelse(isless, mid, ub)
        @. mid = (lb + ub) / 2
    end
    return isabove
end

# GPU FlyHash
function flyhash_GPU(X::CuMatrix{Float32}, projM::CuMatrix{Float32}, nwinners::Int)
    M = projM * X
    return WTA_GPU(M, nwinners)
end