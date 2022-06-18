using Pkg; Pkg.activate(@__DIR__)
using MaereModel
using Distributions, Plots, StatsPlots, AdaptiveMCMC, DataFrames
default(legend=false, grid=false, titlefont=9, title_loc=:left)

# Simulation example
# ==================
# We simulate data from the model
m = MaereModel.AgeDistModel(0.05, [0.7, 0.9], [30], 1., 50, 0.1)
d = MaereModel.agedensity(m)
x = MaereModel.simulate_ks(m, 5000)
filter!(y->0.0 < y <= 5, x)
histogram(x, norm=:pdf, bins=0:0.1:5, color=:white)
plot!(d, lw=2, color=:black)

# ... and conduct Bayesian inference for it, assuming the smoothed and
# normalized predicted age distribution as a probability density function for
# the the observed Ks distribution.

# Set the proposals
proposals = (λ=AdaptiveUvProposal(bounds=(0,Inf)), 
             σ=DiscreteRwProposal(bounds=(1,100)),
             α=[AdaptiveUvProposal(bounds=(0,Inf)) for i=1:2],
             t=[DiscreteRwProposal(bounds=(1,50))])

# ... and the prior
prior = Prior(λ=Exponential(), 
              σ=Poisson(1),
              α=Exponential(),
              t=DiscreteUniform(0,50))

# ... and construct a Chain object
chn = Chain(data=x, model=m, prior=prior, proposal=proposals, state=-Inf)

# ... then run the MCMC sampler
out = mh(chn, 5000)

# We can collect the posterior sample in a data frame and plot traces etc.
post = DataFrame(out)
plot([plot(post[500:end,i]) for i=1:6]...)

# ... and conduct posterior predictive simulations
P = histogram(x, bin=0:0.1:5, norm=true, color=:white)
for i=100:10:length(out)
    model = m(λ=out[i].λ, σ=out[i].σ, t=[out[i].t1], α=[out[i].α1,out[i].α2])
    plot!(P, MaereModel.agedensity(model), color=:black, lw=1, alpha=0.1)
end
plot(P)


# Vitis example
# =============
using CSV, DataFrames
df = CSV.read("test/vvi.ks.tsv", DataFrame)
y = combine(groupby(df, [:Family, :Node]), :Ks=>mean)[:,end]
y = Array{Float64}(filter!(x->!ismissing(x) && 0.01 < x < 5, y))
histogram(y, bins=0:0.1:5)

δ = 0.05
T = 100
props() = (λ=AdaptiveUvProposal(bounds=(0,Inf)), 
           σ=DiscreteRwProposal(bounds=(1,100)),
           α=[AdaptiveUvProposal(bounds=(0,Inf)) for i=1:2],
           t=[DiscreteRwProposal(bounds=(1,T))])
prior = Prior(λ=Exponential(), 
              σ=Poisson(1),
              α=Exponential(),
              t=DiscreteUniform(0,T))
m = MaereModel.AgeDistModel(0.05, [0.7, 0.9], [80], 5., T, δ)
chn = Chain(data=y, model=m, prior=prior, proposal=proposals, state=-Inf)

chn1 = Chain(y, m, prior, props(), -Inf)
out1 = mh(chn1, 1100)
post1 = DataFrame(out1)

chn2 = Chain(y, m, prior, props(), -Inf)
out2 = mh(chn2, 11000)
post2 = DataFrame(out2)

map(1:6) do i
    p = plot(post1[500:end,i])
    plot!(post2[500:end,i])
end |> x->plot(x..., size=(800,300))

pp = map(out1[1000:10:end]) do x
    model = m(λ=x.λ, σ=x.σ, t=[x.t1], α=[x.α1,x.α2])
    MaereModel.agedensity(model)
end
pp = mapreduce(x->x.density, hcat, pp)
ts = (1 .- post1[1000:end,:t1]/T) .* 5
t1, t2 = extrema(ts)

mn = mean(pp, dims=2)
q1 = mapslices(x->quantile(x, 0.025), pp, dims=2)
q2 = mapslices(x->quantile(x, 0.975), pp, dims=2)
xs = 0:m.δ:5
P = stephist(y, bin=0:m.δ:5, norm=true, color=:lightgray, fill=true)
plot!(xs, mn, ribbon=(mn .- q1, q2 .- mn), color=:black)
_, ymx = ylims(P)
#plot!(P, Shape([(t1,0), (t2,0), (t2,ymx), (t1, ymx)]), color=:salmon, alpha=0.5)
bar!(proportionmap(ts), color=:salmon, alpha=0.5, linecolor=:salmon)
plot!(P, title="(A)",
     xlabel="\$K_\\mathrm{S}\$", 
     ylabel="probability density")

using StatsBase
nf = length(unique(df[:,:Family]))
h = fit(Histogram, y, 0:δ:5)
hy = h.weights ./ nf;
P2 = plot(0:δ:(δ*(T-1)), hy, linetype=:steppost, fill=true, color=:lightgray) 
xs = mean(Matrix(post1[1000:end,:]), dims=1)
model = m(λ=xs[1], σ=xs[2], t=[round(xs[5])], α=xs[3:4])
A = MaereModel.simulate_age_dist(model)
plot!(P2, 0:δ:5, vec(sum(A, dims=1)), color=:black, lw=2, linetype=:steppost,
      xlabel="\$K_\\mathrm{S}\$", ylabel="duplicate genes/family", title="(B)")

plot(P, P2, size=(700,250), guidefont=9, bottom_margin=3mm)

savefig("img/vitis.pdf")

d = 0.01
h = fit(Histogram, y, 0:d:1)
hy = h.weights ./ 9805;
plot(0:d:(1-d), log.(hy))
plot!(x->log(0.014/0.01) -6.75 *x)

