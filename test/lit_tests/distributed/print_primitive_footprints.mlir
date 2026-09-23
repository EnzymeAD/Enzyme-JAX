// RUN: enzymexlamlir-opt %s --pass-pipeline='builtin.module(distributed-print-primitive-footprints{kind=all-reduce payload-bytes=300000 extents=4 axes=0})' 2>&1 >/dev/null | FileCheck %s --check-prefix=ALL_REDUCE_HALVING_DOUBLING
// RUN: enzymexlamlir-opt %s --pass-pipeline='builtin.module(distributed-print-primitive-footprints{kind=all-reduce payload-bytes=800 extents=8 axes=0})' 2>&1 >/dev/null | FileCheck %s --check-prefix=ALL_REDUCE_RECURSIVE_DOUBLING
// RUN: enzymexlamlir-opt %s --pass-pipeline='builtin.module(distributed-print-primitive-footprints{kind=all-reduce payload-bytes=500000 extents=8 axes=0})' 2>&1 >/dev/null | FileCheck %s --check-prefix=ALL_REDUCE_HALVING_DOUBLING_LARGE
// RUN: enzymexlamlir-opt %s --pass-pipeline='builtin.module(distributed-print-primitive-footprints{kind=all-reduce payload-bytes=300000 extents=2,2 axes=0,0})' 2>&1 >/dev/null | FileCheck %s --check-prefix=ALL_REDUCE_TWO_ATOMS_ONE_AXIS
// RUN: enzymexlamlir-opt %s --pass-pipeline='builtin.module(distributed-print-primitive-footprints{kind=reduce-scatter payload-bytes=10000 extents=4 axes=0 bandwidths=1,1})' 2>&1 >/dev/null | FileCheck %s --check-prefix=REDUCE_SCATTER
// RUN: enzymexlamlir-opt %s --pass-pipeline='builtin.module(distributed-print-primitive-footprints{kind=all-gather payload-bytes=120000 extents=4 axes=0})' 2>&1 >/dev/null | FileCheck %s --check-prefix=ALL_GATHER
// RUN: enzymexlamlir-opt %s --pass-pipeline='builtin.module(distributed-print-primitive-footprints{kind=all-to-all payload-bytes=100 extents=8 axes=0})' 2>&1 >/dev/null | FileCheck %s --check-prefix=ALL_TO_ALL_BRUCK
// RUN: enzymexlamlir-opt %s --pass-pipeline='builtin.module(distributed-print-primitive-footprints{kind=all-to-all payload-bytes=500000 extents=8 axes=0})' 2>&1 >/dev/null | FileCheck %s --check-prefix=ALL_TO_ALL_PAIRWISE
// RUN: enzymexlamlir-opt %s --pass-pipeline='builtin.module(distributed-print-primitive-footprints{kind=all-to-all payload-bytes=200000 extents=3 axes=0})' 2>&1 >/dev/null | FileCheck %s --check-prefix=ALL_TO_ALL_NON_POWER_OF_TWO
// RUN: enzymexlamlir-opt %s --pass-pipeline='builtin.module(distributed-print-primitive-footprints{kind=collective-permute payload-bytes=10000 extents=2,2 axes=0,1 bandwidths=1,2 changed-fractions=0.5,0.5})' 2>&1 >/dev/null | FileCheck %s --check-prefix=PERMUTE_TWO_AXES
// RUN: enzymexlamlir-opt %s --pass-pipeline='builtin.module(distributed-print-primitive-footprints{kind=all-reduce payload-bytes=100 extents=1 axes=0})' 2>&1 >/dev/null | FileCheck %s --check-prefix=GROUP_OF_ONE
// RUN: enzymexlamlir-opt %s --pass-pipeline='builtin.module(distributed-print-primitive-footprints{kind=local-slice payload-bytes=100 extents=2 axes=0})' 2>&1 >/dev/null | FileCheck %s --check-prefix=LOCAL_SLICE
// RUN: not enzymexlamlir-opt %s --pass-pipeline='builtin.module(distributed-print-primitive-footprints{kind=all-reduce payload-bytes=800 extents=2,4 axes=0,1})' 2>&1 | FileCheck %s --check-prefix=MULTIAXIS
// RUN: not enzymexlamlir-opt %s --pass-pipeline='builtin.module(distributed-print-primitive-footprints{kind=all-reduce payload-bytes=800 extents=0 axes=0})' 2>&1 | FileCheck %s --check-prefix=BAD_EXTENT

// distributed-print-primitive-footprints prints one primitive's closed-form
// footprint as a remark on the module (remarks go to stderr). Atoms are given
// by parallel `extents` and `axes` lists; `bandwidths` is per physical axis in
// bytes/ns (default MeshCostParams::kDefaultBandwidth, 50); `latency` is the
// per-round latency on every axis in nanoseconds (default 1000); the launch
// latency is fixed at its default, 5000ns. Every duration below is an integer
// nanosecond count, rounded once from the underlying float volume/bandwidth
// division (CollectiveCost.h's Units note). The network assumptions N1..N9
// cited below are listed in CollectiveCost.h. V[a] is per-port bytes,
// rho[a] = (V[a]/BW[a]) / max_b(V[b]/BW[b]), k = ceil(log2 n).

// n=4, k=ceil(log2 4)=2, S=300000, BW=50, round latency 1000, launch 5000 (N1, N2, N6).
// halving+doubling: 2k=4 rounds, V=2*S*(n-1)/n=450000, L=5000+4000=9000, duration 9000+450000/50 = 18000
// recursive doubling: k=2 rounds, V=S*k=600000, L=5000+2000=7000, duration 7000+600000/50 = 19000
// min is halving+doubling (N9).
// ALL_REDUCE_HALVING_DOUBLING: all-reduce
// ALL_REDUCE_HALVING_DOUBLING-NEXT: atoms: axis0.0(x4)
// ALL_REDUCE_HALVING_DOUBLING-NEXT: payload: 300000 -> 300000
// ALL_REDUCE_HALVING_DOUBLING-NEXT: latency: 9000
// ALL_REDUCE_HALVING_DOUBLING-NEXT: V: [450000]
// ALL_REDUCE_HALVING_DOUBLING-NEXT: rho: [1]
// ALL_REDUCE_HALVING_DOUBLING-NEXT: duration: 18000

// n=8, k=3, S=800, small payload so latency dominates (N2, N6, N7).
// halving+doubling: 6 rounds, V=2*800*7/8=1400, L=5000+6000=11000, duration 11000+1400/50 = 11028
// recursive doubling: 3 rounds, V=S*k=2400, L=5000+3000=8000, duration 8000+2400/50 = 8048
// min is recursive doubling (N9).
// ALL_REDUCE_RECURSIVE_DOUBLING: all-reduce
// ALL_REDUCE_RECURSIVE_DOUBLING-NEXT: atoms: axis0.0(x8)
// ALL_REDUCE_RECURSIVE_DOUBLING-NEXT: payload: 800 -> 800
// ALL_REDUCE_RECURSIVE_DOUBLING-NEXT: latency: 8000
// ALL_REDUCE_RECURSIVE_DOUBLING-NEXT: V: [2400]
// ALL_REDUCE_RECURSIVE_DOUBLING-NEXT: rho: [1]
// ALL_REDUCE_RECURSIVE_DOUBLING-NEXT: duration: 8048

// n=8, k=3, S=500000, large payload so volume dominates.
// halving+doubling: 6 rounds, V=2*500000*7/8=875000, L=11000, duration 11000+875000/50 = 28500
// recursive doubling: 3 rounds, V=500000*3=1500000, L=8000, duration 8000+1500000/50 = 38000
// min is halving+doubling (N9).
// ALL_REDUCE_HALVING_DOUBLING_LARGE: all-reduce
// ALL_REDUCE_HALVING_DOUBLING_LARGE-NEXT: atoms: axis0.0(x8)
// ALL_REDUCE_HALVING_DOUBLING_LARGE-NEXT: payload: 500000 -> 500000
// ALL_REDUCE_HALVING_DOUBLING_LARGE-NEXT: latency: 11000
// ALL_REDUCE_HALVING_DOUBLING_LARGE-NEXT: V: [875000]
// ALL_REDUCE_HALVING_DOUBLING_LARGE-NEXT: rho: [1]
// ALL_REDUCE_HALVING_DOUBLING_LARGE-NEXT: duration: 28500

// Two atoms on the same axis form one group of n=4 (N1: strides do not matter); same numbers as the first all-reduce case: V=450000, L=9000, duration 18000.
// ALL_REDUCE_TWO_ATOMS_ONE_AXIS: all-reduce
// ALL_REDUCE_TWO_ATOMS_ONE_AXIS-NEXT: atoms: axis0.0(x2) axis0.1(x2)
// ALL_REDUCE_TWO_ATOMS_ONE_AXIS-NEXT: payload: 300000 -> 300000
// ALL_REDUCE_TWO_ATOMS_ONE_AXIS-NEXT: latency: 9000
// ALL_REDUCE_TWO_ATOMS_ONE_AXIS-NEXT: V: [450000]
// ALL_REDUCE_TWO_ATOMS_ONE_AXIS-NEXT: rho: [1]
// ALL_REDUCE_TWO_ATOMS_ONE_AXIS-NEXT: duration: 18000

// n=4, k=2, S=10000 (input), bandwidth 1 on every axis (an explicit slow-link
// override, independent of the default): V=S*(n-1)/n=7500, payloadOut=S/n=2500,
// L=5000+2000=7000, duration 7000+7500/1 = 14500 (N1, N2, N3, N6)
// axis 1 is untouched (N4): V[1]=0 and rho[1]=0
// REDUCE_SCATTER: reduce-scatter
// REDUCE_SCATTER-NEXT: atoms: axis0.0(x4)
// REDUCE_SCATTER-NEXT: payload: 10000 -> 2500
// REDUCE_SCATTER-NEXT: latency: 7000
// REDUCE_SCATTER-NEXT: V: [7500, 0]
// REDUCE_SCATTER-NEXT: rho: [1, 0]
// REDUCE_SCATTER-NEXT: duration: 14500

// n=4, k=2, input shard 120000: V=payloadIn*(n-1)=360000 (= payloadOut*(n-1)/n), payloadOut=n*120000=480000, L=5000+2000=7000, duration 7000+360000/50 = 14200 (N1, N2, N3)
// ALL_GATHER: all-gather
// ALL_GATHER-NEXT: atoms: axis0.0(x4)
// ALL_GATHER-NEXT: payload: 120000 -> 480000
// ALL_GATHER-NEXT: latency: 7000
// ALL_GATHER-NEXT: V: [360000]
// ALL_GATHER-NEXT: rho: [1]
// ALL_GATHER-NEXT: duration: 14200

// n=8, k=3, S=100, small payload so latency dominates (N2, N6, N7).
// pairwise: n-1=7 rounds, V=S*(n-1)/n=87.5, L=5000+7000=12000, duration 12000+87.5/50 = 12002 (rounds to 12002)
// Bruck: k=3 rounds, V=(S/2)*k=150, L=5000+3000=8000, duration 8000+150/50 = 8003
// min is Bruck (N9).
// ALL_TO_ALL_BRUCK: all-to-all
// ALL_TO_ALL_BRUCK-NEXT: atoms: axis0.0(x8)
// ALL_TO_ALL_BRUCK-NEXT: payload: 100 -> 100
// ALL_TO_ALL_BRUCK-NEXT: latency: 8000
// ALL_TO_ALL_BRUCK-NEXT: V: [150]
// ALL_TO_ALL_BRUCK-NEXT: rho: [1]
// ALL_TO_ALL_BRUCK-NEXT: duration: 8003

// n=8, k=3, S=500000, large payload.
// pairwise: 7 rounds, V=500000*7/8=437500, L=12000, duration 12000+437500/50 = 20750
// Bruck: 3 rounds, V=250000*3=750000, L=8000, duration 8000+750000/50 = 23000
// min is pairwise (N9).
// ALL_TO_ALL_PAIRWISE: all-to-all
// ALL_TO_ALL_PAIRWISE-NEXT: atoms: axis0.0(x8)
// ALL_TO_ALL_PAIRWISE-NEXT: payload: 500000 -> 500000
// ALL_TO_ALL_PAIRWISE-NEXT: latency: 12000
// ALL_TO_ALL_PAIRWISE-NEXT: V: [437500]
// ALL_TO_ALL_PAIRWISE-NEXT: rho: [1]
// ALL_TO_ALL_PAIRWISE-NEXT: duration: 20750

// n=3, k=ceil(log2 3)=2 (N2), S=200000: both variants take the same 2 rounds
// (L=5000+2000=7000 either way), so the tie goes entirely to volume.
// pairwise: V=200000*2/3=133333, duration 7000+133333/50 = 9667
// Bruck: V=100000*2=200000, duration 7000+200000/50 = 11000; min is pairwise.
// ALL_TO_ALL_NON_POWER_OF_TWO: all-to-all
// ALL_TO_ALL_NON_POWER_OF_TWO-NEXT: atoms: axis0.0(x3)
// ALL_TO_ALL_NON_POWER_OF_TWO-NEXT: payload: 200000 -> 200000
// ALL_TO_ALL_NON_POWER_OF_TWO-NEXT: latency: 7000
// ALL_TO_ALL_NON_POWER_OF_TWO-NEXT: V: [133333]
// ALL_TO_ALL_NON_POWER_OF_TWO-NEXT: rho: [1]
// ALL_TO_ALL_NON_POWER_OF_TWO-NEXT: duration: 9667

// S=10000, half the devices change digit on each atom, so each axis has a
// nonzero change fraction: V=[10000, 10000] (N3, a moving device sends its
// whole payload, and the step lasts as long as its slowest device); time=
// [10000/1, 10000/2]=[10000, 5000] (N1); rho=[1, 0.5]
// one round crossing both axes (N5): L = launch 5000 + round latency 1000 + round latency 1000 = 7000 (summed, not max); duration 7000 + 10000 = 17000
// PERMUTE_TWO_AXES: permute
// PERMUTE_TWO_AXES-NEXT: atoms: axis0.0(x2) axis1.1(x2)
// PERMUTE_TWO_AXES-NEXT: payload: 10000 -> 10000
// PERMUTE_TWO_AXES-NEXT: latency: 7000
// PERMUTE_TWO_AXES-NEXT: V: [10000, 10000]
// PERMUTE_TWO_AXES-NEXT: rho: [1, 0.5]
// PERMUTE_TWO_AXES-NEXT: duration: 17000

// n=1, k=0: nothing moves and nothing launches: V=0, L=0, rho=0, duration 0 (both all-reduce candidates tie at 0)
// GROUP_OF_ONE: all-reduce
// GROUP_OF_ONE-NEXT: atoms: axis0.0(x1)
// GROUP_OF_ONE-NEXT: payload: 100 -> 100
// GROUP_OF_ONE-NEXT: latency: 0
// GROUP_OF_ONE-NEXT: V: [0]
// GROUP_OF_ONE-NEXT: rho: [0]
// GROUP_OF_ONE-NEXT: duration: 0

// A local slice is free: no volume, no latency, rho=0 (the option pass reports payloadOut = payloadIn)
// LOCAL_SLICE: local-slice
// LOCAL_SLICE-NEXT: atoms: axis0.0(x2)
// LOCAL_SLICE-NEXT: payload: 100 -> 100
// LOCAL_SLICE-NEXT: latency: 0
// LOCAL_SLICE-NEXT: V: [0]
// LOCAL_SLICE-NEXT: rho: [0]
// LOCAL_SLICE-NEXT: duration: 0

// A multi-axis all-reduce is not a single step (N1, N4): the collective is a
// chain of single-axis steps composed elsewhere, so atoms on two axes are
// rejected.
// MULTIAXIS: 'all-reduce' atoms must share one axis

module {
}

// Options are user input: a non-positive extent is rejected instead of costed.
// BAD_EXTENT: axes must be non-negative and extents positive
