// RUN: enzymexlamlir-opt %s --pass-pipeline='builtin.module(distributed-print-primitive-footprints{kind=all-reduce payload-bytes=1200 extents=4 axes=0})' 2>&1 >/dev/null | FileCheck %s --check-prefix=ALL_REDUCE_HALVING_DOUBLING
// RUN: enzymexlamlir-opt %s --pass-pipeline='builtin.module(distributed-print-primitive-footprints{kind=all-reduce payload-bytes=1 extents=8 axes=0})' 2>&1 >/dev/null | FileCheck %s --check-prefix=ALL_REDUCE_RECURSIVE_DOUBLING
// RUN: enzymexlamlir-opt %s --pass-pipeline='builtin.module(distributed-print-primitive-footprints{kind=all-reduce payload-bytes=800 extents=8 axes=0})' 2>&1 >/dev/null | FileCheck %s --check-prefix=ALL_REDUCE_HALVING_DOUBLING_LARGE
// RUN: enzymexlamlir-opt %s --pass-pipeline='builtin.module(distributed-print-primitive-footprints{kind=all-reduce payload-bytes=1200 extents=2,2 axes=0,0})' 2>&1 >/dev/null | FileCheck %s --check-prefix=ALL_REDUCE_TWO_ATOMS_ONE_AXIS
// RUN: enzymexlamlir-opt %s --pass-pipeline='builtin.module(distributed-print-primitive-footprints{kind=reduce-scatter payload-bytes=1200 extents=4 axes=0 bandwidths=1,1})' 2>&1 >/dev/null | FileCheck %s --check-prefix=REDUCE_SCATTER
// RUN: enzymexlamlir-opt %s --pass-pipeline='builtin.module(distributed-print-primitive-footprints{kind=all-gather payload-bytes=300 extents=4 axes=0})' 2>&1 >/dev/null | FileCheck %s --check-prefix=ALL_GATHER
// RUN: enzymexlamlir-opt %s --pass-pipeline='builtin.module(distributed-print-primitive-footprints{kind=all-to-all payload-bytes=2 extents=8 axes=0})' 2>&1 >/dev/null | FileCheck %s --check-prefix=ALL_TO_ALL_BRUCK
// RUN: enzymexlamlir-opt %s --pass-pipeline='builtin.module(distributed-print-primitive-footprints{kind=all-to-all payload-bytes=800 extents=8 axes=0})' 2>&1 >/dev/null | FileCheck %s --check-prefix=ALL_TO_ALL_PAIRWISE
// RUN: enzymexlamlir-opt %s --pass-pipeline='builtin.module(distributed-print-primitive-footprints{kind=all-to-all payload-bytes=1200 extents=3 axes=0})' 2>&1 >/dev/null | FileCheck %s --check-prefix=ALL_TO_ALL_NON_POWER_OF_TWO
// RUN: enzymexlamlir-opt %s --pass-pipeline='builtin.module(distributed-print-primitive-footprints{kind=collective-permute payload-bytes=1200 extents=2,2 axes=0,1 bandwidths=1,2 changed-fractions=0.5,0.5})' 2>&1 >/dev/null | FileCheck %s --check-prefix=PERMUTE_TWO_AXES
// RUN: enzymexlamlir-opt %s --pass-pipeline='builtin.module(distributed-print-primitive-footprints{kind=all-reduce payload-bytes=100 extents=1 axes=0})' 2>&1 >/dev/null | FileCheck %s --check-prefix=GROUP_OF_ONE
// RUN: enzymexlamlir-opt %s --pass-pipeline='builtin.module(distributed-print-primitive-footprints{kind=local-slice payload-bytes=100 extents=2 axes=0})' 2>&1 >/dev/null | FileCheck %s --check-prefix=LOCAL_SLICE
// RUN: not enzymexlamlir-opt %s --pass-pipeline='builtin.module(distributed-print-primitive-footprints{kind=all-reduce payload-bytes=800 extents=2,4 axes=0,1})' 2>&1 | FileCheck %s --check-prefix=MULTIAXIS
// RUN: not enzymexlamlir-opt %s --pass-pipeline='builtin.module(distributed-print-primitive-footprints{kind=all-reduce payload-bytes=800 extents=0 axes=0})' 2>&1 | FileCheck %s --check-prefix=BAD_EXTENT

// distributed-print-primitive-footprints prints one primitive's closed-form
// footprint as a remark on the module (remarks go to stderr). Atoms are given
// by parallel `extents` and `axes` lists; `bandwidths` is per physical axis
// (default 1); `latency` is the per-round latency on every axis (default 1);
// the launch latency is fixed at its default 0.1. The network assumptions
// N1..N9 cited below are listed in CollectiveCost.h. V[a] is per-port bytes,
// rho[a] = (V[a]/BW[a]) / max_b(V[b]/BW[b]), k = ceil(log2 n).

// n=4, k=ceil(log2 4)=2, S=1200, BW=1, round latency 1, launch 0.1 (N1, N2, N6).
// halving+doubling: 2k=4 rounds, V=2*S*(n-1)/n=1800, duration 0.1+4+1800 = 1804.1
// recursive doubling: k=2 rounds, V=S*k=2400, duration 0.1+2+2400 = 2402.1
// min is halving+doubling (N9).
// ALL_REDUCE_HALVING_DOUBLING: all-reduce
// ALL_REDUCE_HALVING_DOUBLING-NEXT: atoms: axis0.0(x4)
// ALL_REDUCE_HALVING_DOUBLING-NEXT: payload: 1200 -> 1200
// ALL_REDUCE_HALVING_DOUBLING-NEXT: latency: 4.1
// ALL_REDUCE_HALVING_DOUBLING-NEXT: V: [1800]
// ALL_REDUCE_HALVING_DOUBLING-NEXT: rho: [1]
// ALL_REDUCE_HALVING_DOUBLING-NEXT: duration: 1804.1

// n=8, k=3, S=1, small payload so latency dominates (N2, N6, N7).
// halving+doubling: 6 rounds, V=2*1*7/8=1.75, duration 0.1+6+1.75 = 7.85
// recursive doubling: 3 rounds, V=S*k=3, duration 0.1+3+3 = 6.1
// min is recursive doubling (N9); crossover is 3 = 1.25*S, i.e. S=2.4.
// ALL_REDUCE_RECURSIVE_DOUBLING: all-reduce
// ALL_REDUCE_RECURSIVE_DOUBLING-NEXT: atoms: axis0.0(x8)
// ALL_REDUCE_RECURSIVE_DOUBLING-NEXT: payload: 1 -> 1
// ALL_REDUCE_RECURSIVE_DOUBLING-NEXT: latency: 3.1
// ALL_REDUCE_RECURSIVE_DOUBLING-NEXT: V: [3]
// ALL_REDUCE_RECURSIVE_DOUBLING-NEXT: rho: [1]
// ALL_REDUCE_RECURSIVE_DOUBLING-NEXT: duration: 6.1

// n=8, k=3, S=800, large payload so volume dominates.
// halving+doubling: 6 rounds, V=2*800*7/8=1400, duration 0.1+6+1400 = 1406.1
// recursive doubling: 3 rounds, V=800*3=2400, duration 0.1+3+2400 = 2403.1
// min is halving+doubling (N9).
// ALL_REDUCE_HALVING_DOUBLING_LARGE: all-reduce
// ALL_REDUCE_HALVING_DOUBLING_LARGE-NEXT: atoms: axis0.0(x8)
// ALL_REDUCE_HALVING_DOUBLING_LARGE-NEXT: payload: 800 -> 800
// ALL_REDUCE_HALVING_DOUBLING_LARGE-NEXT: latency: 6.1
// ALL_REDUCE_HALVING_DOUBLING_LARGE-NEXT: V: [1400]
// ALL_REDUCE_HALVING_DOUBLING_LARGE-NEXT: rho: [1]
// ALL_REDUCE_HALVING_DOUBLING_LARGE-NEXT: duration: 1406.1

// Two atoms on the same axis form one group of n=4 (N1: strides do not matter); same numbers as the first all-reduce case: V=1800, L=0.1+4=4.1, duration 1804.1.
// ALL_REDUCE_TWO_ATOMS_ONE_AXIS: all-reduce
// ALL_REDUCE_TWO_ATOMS_ONE_AXIS-NEXT: atoms: axis0.0(x2) axis0.1(x2)
// ALL_REDUCE_TWO_ATOMS_ONE_AXIS-NEXT: payload: 1200 -> 1200
// ALL_REDUCE_TWO_ATOMS_ONE_AXIS-NEXT: latency: 4.1
// ALL_REDUCE_TWO_ATOMS_ONE_AXIS-NEXT: V: [1800]
// ALL_REDUCE_TWO_ATOMS_ONE_AXIS-NEXT: rho: [1]
// ALL_REDUCE_TWO_ATOMS_ONE_AXIS-NEXT: duration: 1804.1

// n=4, k=2, S=1200 (input): V=S*(n-1)/n=900, payloadOut=S/n=300, rounds k=2, L=0.1+2=2.1, duration 902.1 (N1, N2, N3, N6)
// axis 1 is untouched (N4): V[1]=0 and rho[1]=0
// REDUCE_SCATTER: reduce-scatter
// REDUCE_SCATTER-NEXT: atoms: axis0.0(x4)
// REDUCE_SCATTER-NEXT: payload: 1200 -> 300
// REDUCE_SCATTER-NEXT: latency: 2.1
// REDUCE_SCATTER-NEXT: V: [900, 0]
// REDUCE_SCATTER-NEXT: rho: [1, 0]
// REDUCE_SCATTER-NEXT: duration: 902.1

// n=4, k=2, input shard 300: V=payloadIn*(n-1)=900 (= payloadOut*(n-1)/n), payloadOut=n*300=1200, L=0.1+2=2.1, duration 902.1 (N1, N2, N3)
// ALL_GATHER: all-gather
// ALL_GATHER-NEXT: atoms: axis0.0(x4)
// ALL_GATHER-NEXT: payload: 300 -> 1200
// ALL_GATHER-NEXT: latency: 2.1
// ALL_GATHER-NEXT: V: [900]
// ALL_GATHER-NEXT: rho: [1]
// ALL_GATHER-NEXT: duration: 902.1

// n=8, k=3, S=2, small payload (N2, N6, N7).
// pairwise: n-1=7 rounds, V=S*(n-1)/n=1.75, duration 0.1+7+1.75 = 8.85
// Bruck: k=3 rounds, V=(S/2)*k=3, duration 0.1+3+3 = 6.1
// min is Bruck (N9); crossover is 4 = 0.625*S in volume, i.e. S=6.4.
// ALL_TO_ALL_BRUCK: all-to-all
// ALL_TO_ALL_BRUCK-NEXT: atoms: axis0.0(x8)
// ALL_TO_ALL_BRUCK-NEXT: payload: 2 -> 2
// ALL_TO_ALL_BRUCK-NEXT: latency: 3.1
// ALL_TO_ALL_BRUCK-NEXT: V: [3]
// ALL_TO_ALL_BRUCK-NEXT: rho: [1]
// ALL_TO_ALL_BRUCK-NEXT: duration: 6.1

// n=8, k=3, S=800, large payload.
// pairwise: 7 rounds, V=800*7/8=700, duration 0.1+7+700 = 707.1
// Bruck: 3 rounds, V=400*3=1200, duration 0.1+3+1200 = 1203.1
// min is pairwise (N9).
// ALL_TO_ALL_PAIRWISE: all-to-all
// ALL_TO_ALL_PAIRWISE-NEXT: atoms: axis0.0(x8)
// ALL_TO_ALL_PAIRWISE-NEXT: payload: 800 -> 800
// ALL_TO_ALL_PAIRWISE-NEXT: latency: 7.1
// ALL_TO_ALL_PAIRWISE-NEXT: V: [700]
// ALL_TO_ALL_PAIRWISE-NEXT: rho: [1]
// ALL_TO_ALL_PAIRWISE-NEXT: duration: 707.1

// n=3, k=ceil(log2 3)=2 (N2), S=1200.
// pairwise: n-1=2 rounds, V=1200*2/3=800, duration 0.1+2+800 = 802.1
// Bruck: 2 rounds, V=600*2=1200, duration 1202.1; min is pairwise.
// ALL_TO_ALL_NON_POWER_OF_TWO: all-to-all
// ALL_TO_ALL_NON_POWER_OF_TWO-NEXT: atoms: axis0.0(x3)
// ALL_TO_ALL_NON_POWER_OF_TWO-NEXT: payload: 1200 -> 1200
// ALL_TO_ALL_NON_POWER_OF_TWO-NEXT: latency: 2.1
// ALL_TO_ALL_NON_POWER_OF_TWO-NEXT: V: [800]
// ALL_TO_ALL_NON_POWER_OF_TWO-NEXT: rho: [1]
// ALL_TO_ALL_NON_POWER_OF_TWO-NEXT: duration: 802.1

// S=1200, half the devices change digit on each atom: V=[600, 600] (N3); time=[600/1, 600/2]=[600, 300] (N1); rho=[1, 0.5]
// one round crossing both axes (N5): L = launch 0.1 + round latency 1 + round latency 1 = 2.1 (summed, not max); duration 2.1 + 600 = 602.1
// PERMUTE_TWO_AXES: permute
// PERMUTE_TWO_AXES-NEXT: atoms: axis0.0(x2) axis1.1(x2)
// PERMUTE_TWO_AXES-NEXT: payload: 1200 -> 1200
// PERMUTE_TWO_AXES-NEXT: latency: 2.1
// PERMUTE_TWO_AXES-NEXT: V: [600, 600]
// PERMUTE_TWO_AXES-NEXT: rho: [1, 0.5]
// PERMUTE_TWO_AXES-NEXT: duration: 602.1

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
