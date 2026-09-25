# Gate the build BEFORE involving GB-25: is the library loadable, does it see the
# four MI300A agents, and does a trivial kernel run -- plain and 4-way sharded?
# This is what CI cannot tell you today: there, a GB-25 failure is indistinguishable
# from a bad build.
#
# NB: `stage("name") do ... end` passes the do-block as the FIRST argument, so the
# function parameter must come first.
function stage(f, name)
    try
        r = f(); printstyled("  ok   ", name, "\n"; color=:green); return r
    catch e
        printstyled("  FAIL ", name, "\n"; color=:red)
        showerror(stdout, e, catch_backtrace()); println(); exit(1)
    end
end

println("=== build check ===")
stage("using Reactant") do
    @eval Main using Reactant
end
devs = stage("Reactant.devices()") do
    collect(Reactant.devices())
end
println("  devices: ", length(devs))
for d in devs
    println("    ", d)
end
stage("4 GPU agents present") do
    length(devs) == 4 || error("expected 4 GPU agents, got $(length(devs))")
end
stage("trivial kernel (@jit sum)") do
    a = Reactant.to_rarray(ones(Float64, 8, 8))
    r = @jit sum(a)
    isapprox(Float64(r), 64.0) || error("wrong result: $r")
end
stage("sharded kernel (4-way @jit sum)") do
    m = Reactant.Sharding.Mesh(reshape(collect(devs), 4), (:x,))
    s = Reactant.Sharding.NamedSharding(m, (:x, nothing))
    a = Reactant.to_rarray(ones(Float64, 16, 8); sharding = s)
    r = @jit sum(a)
    isapprox(Float64(r), 128.0) || error("wrong sharded result: $r")
end
printstyled("=== build OK ===\n"; color=:green)
