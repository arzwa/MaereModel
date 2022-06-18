using MaereModel
using Distributions, Plots, StatsPlots
default(legend=false, grid=false)

m = MaereModel.AgeDistModel(0.05, [0.7, 0.9], [0.6], 50, 0.1)
d = MaereModel.agedensity(m)
x = MaereModel.simulate_ks(m, 5000)
filter!(y->0.0 < y <= 5, x)
histogram(x, norm=:pdf, bins=0:0.1:5, color=:white)
plot!(d, lw=2, color=:black)

# doesn't work
using Turing
@model bmodel(x) = begin
    λ ~ Exponential(0.05)
    α1 ~ Exponential()
    α2 ~ Exponential()
    m = MaereModel.AgeDistModel(λ, [α1, α2], [30])
    d = MaereModel.agedensity(m, 50, 0.1)
    Turing.@addlogprob!(sum(log.(pdf(d, x))))
end

chn = sample(bmodel(x), MH(), 10000)

# Adaptive MH
using AdaptiveMCMC, Parameters

mutable struct ChainStruct{D,M,P,Q,V}
    data::D
    model::M
    prior::P
    proposal::Q
    state::V
end

struct Prior{T,V}
    λ::T
    α::V
end

Distributions.logpdf(m::Prior, x) = logpdf(m.λ, x[1]) + loglikelihood(m.α, x[2:end])

function mhstep!(chain)
    @unpack data, model, prior, proposal, state = chain
    for (i, q) in enumerate(proposals)
        x_ = copy(state)
        x_[i+1], _ = q(state[i+1])
        M = model(λ=x_[2], α=x_[3:end])
        l = MaereModel.logpdf(M, data)
        p = logpdf(prior, x_[2:end])
        # accept/reject
        a = l + p - state[1]
        if log(rand()) < a
            state[i+1] = x_[i+1]
            state[1] = l + p
            q.accepted += 1
        end
    end
    return state
end

function mh(chain, n)
    map(1:n) do i
        state = mhstep!(chain)
        i % 100 == 0 && @info (i, state[1])
        copy(state)
    end
end

proposals = [AdaptiveUvProposal(bounds=(0.,1.)) for i=1:3]
proposals[1].bounds = (0.0, 0.1)
chn = ChainStruct(x, m, 
                  Prior(Exponential(0.1), Exponential()), 
                  proposals, [-Inf, 0.1, 0.5, 0.5])
out = mh(chn, 5000)

p = out[end]
histogram(x, bin=0:0.1:5, norm=true, color=:white)
plot!(MaereModel.agedensity(m(λ=p[2], α=p[3:4])), color=:black, lw=2)


# infer age
struct Prior2{T,V,W}
    λ::T
    α::V
    t::W
end

function Distributions.logpdf(m::Prior2, x)
    nwgd = (length(x) - 2) ÷ 2
    l = logpdf(m.λ, x[1])
    l += loglikelihood(m.α, x[2:2+nwgd])
    l += loglikelihood(m.t, x[2+nwgd+1:end])
end

function mhstep!(chain)
    @unpack data, model, prior, proposal, state = chain
    nw = nwgd(model)
    for (i, q) in enumerate(proposals)
        x_ = copy(state)
        x_[i+1], _ = q(state[i+1])
        M = model(λ=x_[2], α=x_[3:3+nw], t=x_[3+nw+1:end])
        l = MaereModel.logpdf(M, data)
        p = logpdf(prior, x_[2:end])
        # accept/reject
        a = l + p - state[1]
        if log(rand()) < a
            state[i+1] = x_[i+1]
            state[1] = l + p
            q.accepted += 1
        end
    end
    return state
end

function mh(chain, n)
    map(1:n) do i
        state = mhstep!(chain)
        i % 100 == 0 && @info (i, state[1])
        copy(state)
    end
end

proposals = [AdaptiveUvProposal(bounds=(0.,1.)) for i=1:4]
chn = ChainStruct(x, m, 
                  Prior2(Exponential(0.1), Exponential(), Beta()), 
                  proposals, [-Inf, 0.1, 0.5, 0.5, 0.5])
out = mh(chn, 5000)

P = histogram(x, bin=0:0.1:5, norm=true, color=:white)
for i=100:10:length(out)
    p = out[i][2:end]
    plot!(P, MaereModel.agedensity(m(p)), color=:black, lw=1, alpha=0.1)
end
plot(P)



# Vitis
using CSV, DataFrames
df = CSV.read("/home/arzwa/research/ksdistributions/example/data/vvi.cds.fasta.ks.tsv", DataFrame)

y = combine(groupby(df, [:Family, :Node]), :Ks=>mean)[:,end]
y = Array{Float64}(filter!(x->!ismissing(x) && 0.05 < x < 5, y))
histogram(y)

m = MaereModel.AgeDistModel(0.05, [1., 1.], [0.5], 100, 0.05)
proposals = [AdaptiveUvProposal(bounds=(0.,1.)) for i=1:4]
proposals[1].bounds = (0.0, 10.)
chn = ChainStruct(y, m, 
                  Prior2(Exponential(), Exponential(), Beta()), 
                  proposals, [-Inf, 0.1, 1., 1., 0.5])
out = mh(chn, 10000)

post = DataFrame(permutedims(hcat(out...)))

plot(post[:,5])

P = histogram(y, bin=0:0.1:5, norm=true, color=:white)
for i=100:10:length(out)
    p = out[i][2:end]
    plot!(P, MaereModel.agedensity(m(p)), color=:black, lw=1, alpha=0.2)
end
plot(P)


# nicer implementation ?
proposals = (λ=AdaptiveUvProposal(bounds=(0,Inf)),
             α=[AdaptiveUvProposal(bounds=(0,Inf)) for i=1:2],
             t=AdaptiveUvProposal(bounds=(0.,1.)))
θ = (λ=0.1, α=[0.5,0.5], t=0.5)

struct _Prior{T,V,W}
    λ::T
    α::V
    t::W
end

function Distributions.logpdf(m::_Prior, θ) 
    ℓ = 0
    for f in keys(θ)
        ℓ += loglikelihood(getfield(m, f), getfield(θ, f))
    end
    return ℓ
end

function mhstep!(chain)
    @unpack data, model, prior, proposal, state = chain
    newstate = Dict(pairs(state))
    for f in keys(proposals)
        x_ = copy(getfield(state, f))
        q = getfield(proposal, f)
        ℓ, x_ = mhinner!(x_, model, prior, q, f)
        # accept/reject
        a = l + p - newstate[:ℓ]
        if log(rand()) < a
            newstate[f] = x_
            newstate[:ℓ] = ℓ
            q.accepted += 1
        end
    end
    return (;newstate...)
end

function mhinner!(x::Vector, model, prior, q, f)
    i = rand(1:length(x))
    x[i], _ = q[i](x[i])
    M = model(; (;f=>x)...)
end

function mh(chain, n)
    map(1:n) do i
        state = mhstep!(chain)
        i % 100 == 0 && @info (i, state[1])
        copy(state)
    end
end

function mhstep!(chain)
    @unpack data, model, prior, proposal, state = chain
    newstate = Dict(pairs(state))
    mhlambda!(newstate, chain)
    mhalpha!(newstate, chain)
    mht!(newstate, chain)
    return (;newstate...)
end



# smoothing tests
m = MaereModel.AgeDistModel(0.05, [0.7, 0.8], [0.7], 1., 50, 0.1)
A = MaereModel.simulate_age_dist(m)
x = vec(sum(A, dims=1))
p = plot(0:m.δ:(m.δ*(length(x)-1)), x ./ m.δ, fill=true, 
         color=:lightgray, linetype=:steppost)
z = MaereModel.poissonsmooth(x, m.δ)
plot!(z, color=:black, lw=2, linetype=:steppost, 
      xlabel="age \$x\$", ylabel="probability density")
for σ = [1, 10., 30., 100.]
    #plot!(MaereModel.lnsmooth(x, m.δ, σ))
    plot!(MaereModel.gammasmooth(x, m.δ, σ))
end
plot(p)

y = MaereModel.lnsmooth(x, m.δ, 0.5)


