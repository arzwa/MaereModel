module MaereModel

using Parameters, Distributions, KernelDensity, StatsBase, AdaptiveMCMC
import Distributions: logpdf

export AgeDistModel, nwgd, Chain, Prior, mh

struct AgeDistModel{T,V}
    λ::T           # duplication rate/time step
    α::Vector{T}   # decay rates
    t::Vector{V}   # WGD timings
    σ::T
    n::Int
    δ::Float64
end

(m::AgeDistModel)(; λ=m.λ, σ=m.σ, α=m.α, t=m.t) = AgeDistModel(λ, α, t, σ, m.n, m.δ)
function (m::AgeDistModel)(x::Vector)
    k = nwgd(m)
    m(λ=x[1], σ=x[2], α=x[3:3+k], t=x[3+k+1:end])
end

nwgd(m::AgeDistModel) = length(m.t)

function nt(m::AgeDistModel)
    ps = [:λ=>m.λ, :σ=>m.σ]
    for i=1:length(m.α)
        push!(ps, Symbol("α$i")=>m.α[i])
    end
    for i=1:length(m.t)
        push!(ps, Symbol("t$i")=>m.t[i])
    end
    (;ps...)
end

function agedensity(m::AgeDistModel)
    @unpack δ = m
    D = simulate_age_dist(m)
    x = vec(sum(D, dims=1))
    x ./= (sum(x) * δ)
    #return poissonsmooth(x, δ)
    return cpdsmooth(x, δ, m.σ)
    #return gammasmooth(x, δ, m.σ)
end

function simulate_ks(m::AgeDistModel{T}, N) where T
    @unpack δ = m
    D = simulate_age_dist(m)
    x = vec(sum(D, dims=1))
    y = rand(Categorical(x ./ sum(x)), N) .- 1
    y = [rand(Poisson(y[i])) for i=1:N] .* δ
end

function logpdf(m::AgeDistModel, x)
    xx = pdf(agedensity(m), x)
    any(xx .< 0) && @warn m
    sum(log.(xx))
end

# must be efficient!
# we take G0 = 1
function simulate_age_dist(m::AgeDistModel{T}) where T
    @unpack λ, α, t, n = m
    #wgds = round.(Int, t .* n)
    wgds = t
    k = length(wgds) + 1
    Dprev = zeros(T, k, n+1)
    for i=1:n
        tot = sum(Dprev)
        D = zeros(T, k, n+1)
        # initialize
        D[1,1] = λ * (tot + 1)
        for j=2:k
            D[j,1] = i == wgds[j-1] ? tot + 1 : 0
        end
        # ages
        for x=2:n+1
            for j=1:k
                D[j,x] = Dprev[j,x-1] * (x/(x-1))^(-α[j])
            end
        end
        Dprev = D
    end
    return Dprev
end

function poissonsmooth(x::Vector{T}, δ) where T
    A = zeros(T, length(x))
    for i=1:length(x)
        for k=1:length(x)
            A[i] += pdf(Poisson(k), i)*x[k]
        end
    end
    A ./= (sum(A) * δ)
    return UnivariateKDE(0:δ:(δ*(length(x)-1)), A)
end

function lnsmooth(x::Vector{T}, δ, σ) where T
    A = zeros(T, length(x))
    for i=1:length(x)
        for k=1:length(x)
            A[i] += pdf(LogNormal(log(k + σ^2/2), σ), i)*x[k]
        end
    end
    A ./= (sum(A) * δ)
    return UnivariateKDE(0:δ:(δ*(length(x)-1)), A)
end

function cpdsmooth(x::Vector{T}, δ, σ) where T
    A = zeros(T, length(x))
    for i=1:length(x)
        for k=1:length(x)
            # rounding seems ad hockery
            #A[i] += pdf(Poisson(σ * k), round(Int, σ * i))*x[k]
            A[i] += pdf(Poisson(σ * k), floor(Int, σ * i))*x[k]
        end
    end
    A ./= (sum(A) * δ)
    return UnivariateKDE(0:δ:(δ*(length(x)-1)), A)
end

function gammasmooth(x::Vector{T}, δ, σ) where T
    A = zeros(T, length(x))
    for i=1:length(x)
        for k=1:length(x)
            β = σ/k
            α = k/β
            A[i] += pdf(Gamma(α, β), i)*x[k]
        end
    end
    A ./= (sum(A) * δ)
    return UnivariateKDE(0:δ:(δ*(length(x)-1)), A)
end

# MCMC
@with_kw mutable struct Chain{D,M,P,Q,V}
    data::D
    model::M
    prior::P
    proposal::Q
    state::V
end

@with_kw struct Prior{T,V,W,U}
    λ::T
    σ::U
    α::V
    t::W
end

function logpdf(m::Prior, model::AgeDistModel)
    l = logpdf(m.λ, model.λ)
    l += logpdf(m.σ, model.σ)
    l += loglikelihood(m.α, model.α)
    l += loglikelihood(m.t, model.t)
end

function mhuv!(chain, f)  # univariate
    @unpack data, model, prior, proposal, state = chain
    q = getfield(proposal, f)
    l, _ = q(getfield(model, f))
    m = model(;[f=>l,]...)
    ℓ = logpdf(m, data) + logpdf(prior, m)
    if log(rand()) < ℓ - state
        chain.state = ℓ
        chain.model = m
        q.accepted += 1
    end
end

function mhmv!(chain, f)  # univariate
    @unpack data, model, prior, proposal, state = chain
    q = getfield(proposal, f)
    x = getfield(model, f)
    y = copy(x)
    for (i, qi) in enumerate(q)
        y[i], _ = qi(y[i])
        m = chain.model(;[f=>y,]...)
        ℓ = logpdf(m, data) + logpdf(prior, m)
        if log(rand()) < ℓ - chain.state
            chain.state = ℓ
            chain.model = m
            qi.accepted += 1
        else
            y[i] = x[i]
        end
    end
end

function mh(chain, n)
    map(1:n) do i
        mhuv!(chain, :λ)
        mhuv!(chain, :σ)
        mhmv!(chain, :α)
        mhmv!(chain, :t)
        i % 100 == 0 && @info (i, chain.state)
        merge(nt(chain.model), (:lp=>chain.state,))
    end
end


end
