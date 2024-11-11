# RUN: python %s | FileCheck %s

import numpy as np
import jax
import jax.numpy as jnp
from enzyme_ad.jax import cpp_call

def lower(fn, vals, parameters=None):
    if hasattr(fn, "trace"):
        if parameters is not None:
            return fn.trace(*vals).lower(_private_parameters=parameters)
        else:
            return fn.trace(*vals).lower()
    else:
        if parameters is not None:
            return fn.lower(*vals, _experimental_lowering_parameters=parameters)
        else:
            return fn.lower(*vals)

argv = ("-I/usr/include/c++/11", "-I/usr/include/x86_64-linux-gnu/c++/11")

def do_something(ones, twos):
    shape = jax.core.ShapedArray(tuple(3 * s for s in ones.shape), ones.dtype)
    shape2 = jax.core.ShapedArray(tuple(2 * s for s in ones.shape), ones.dtype)
    a, b = cpp_call(
        ones,
        twos,
        out_shapes=[shape, shape2],
        source="""
    template<std::size_t N3, std::size_t M3, std::size_t N, std::size_t M, std::size_t N2, std::size_t M2, std::size_t N4, std::size_t M4>
    void myfn(enzyme::tensor<float, N, M>& out0, enzyme::tensor<float, N2, M2>& out1, const enzyme::tensor<float, N3, M3>& in0, const enzyme::tensor<float, N4, M4>& in1) {
        for (int j=0; j<N; j++) {
        for (int k=0; k<M; k++) {
            out0[j][k] = in0[0][0] + 42;
        }
        }
        for (int j=0; j<2; j++) {
        for (int k=0; k<2; k++) {
            out1[j][k] = in0[j][k] * in0[j][k];
        }
        }
    }
    """, argv=argv,
        fn="myfn",
    )
    return a, b

dev = jax.devices("cpu")[0]
ones = jnp.asarray(np.ones((2, 3), jnp.float32), device=dev)
twos = jnp.asarray(np.ones((5, 7), jnp.float32), device=dev)


def fwdmode(a, b, c, d):
    return jax.jvp(do_something, (a, b), (c, d))


print(lower(jax.jit(fwdmode, backend='cpu'), (ones, twos, ones, twos)).compiler_ir(dialect="stablehlo"))

# CHECK: module @jit_fwdmode attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
# CHECK-NEXT:   func.func public @main(%arg0: tensor<2x3xf32> {mhlo.layout_mode = "default", mhlo.sharding = "{replicated}"}, %arg1: tensor<5x7xf32> {mhlo.layout_mode = "default", mhlo.sharding = "{replicated}"}, %arg2: tensor<2x3xf32> {mhlo.layout_mode = "default", mhlo.sharding = "{replicated}"}, %arg3: tensor<5x7xf32> {mhlo.layout_mode = "default", mhlo.sharding = "{replicated}"}) -> (tensor<6x9xf32> {jax.result_info = "[0][0]", mhlo.layout_mode = "default", mhlo.sharding = "{replicated}"}, tensor<4x6xf32> {jax.result_info = "[0][1]", mhlo.layout_mode = "default", mhlo.sharding = "{replicated}"}, tensor<6x9xf32> {jax.result_info = "[1][0]", mhlo.layout_mode = "default", mhlo.sharding = "{replicated}"}, tensor<4x6xf32> {jax.result_info = "[1][1]", mhlo.layout_mode = "default", mhlo.sharding = "{replicated}"})
# CHECK-NEXT:     %[[i0:.+]] = stablehlo.constant dense<1> : tensor<1xi64>
# CHECK-NEXT:     %[[i1:.+]]:4 = stablehlo.custom_call @jaxzyme.fwd(%[[i0]], %arg0, %arg2, %arg1, %arg3) : (tensor<1xi64>, tensor<2x3xf32>, tensor<2x3xf32>, tensor<5x7xf32>, tensor<5x7xf32>) -> (tensor<6x9xf32>, tensor<6x9xf32>, tensor<4x6xf32>, tensor<4x6xf32>)
# CHECK-NEXT:     return %[[i1]]#0, %[[i1]]#2, %[[i1]]#1, %[[i1]]#3 : tensor<6x9xf32>, tensor<4x6xf32>, tensor<6x9xf32>, tensor<4x6xf32>
# CHECK-NEXT:   }
# CHECK-NEXT: }


def f(a, b):
    return jax.vjp(do_something, a, b)


print(lower(jax.jit(f, backend='cpu'), (ones, twos)).compiler_ir(dialect="stablehlo"))

# CHECK: module @jit_f attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
# CHECK-NEXT:  func.func public @main
# CHECK-NEXT:     %[[i0:.+]] = stablehlo.constant dense<2> : tensor<1xi64>
# CHECK-NEXT:     %[[i1:.+]]:3 = stablehlo.custom_call @jaxzyme.aug(%[[i0]], %arg0, %arg1) : (tensor<1xi64>, tensor<2x3xf32>, tensor<5x7xf32>) -> (tensor<6x9xf32>, tensor<4x6xf32>, tensor<16xi8>)
# CHECK-NEXT:     return %[[i1]]#0, %[[i1]]#1, %[[i1]]#2, %arg0, %arg1 : tensor<6x9xf32>, tensor<4x6xf32>, tensor<16xi8>, tensor<2x3xf32>, tensor<5x7xf32>
# CHECK-NEXT:   }
# CHECK-NEXT: }

x = jnp.asarray(np.ones((6, 9), jnp.float32), device=dev)
y = jnp.asarray(np.ones((4, 6), jnp.float32), device=dev)


def g(a, b, x, y):
    primals, f_vjp = jax.vjp(do_something, a, b)
    return primals, f_vjp((x, y))

# CHECK: module @jit_g attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
# CHECK-NEXT: func.func public @main
# CHECK-NEXT:     %[[i0:.+]] = stablehlo.constant dense<3> : tensor<1xi64>
# CHECK-NEXT:     %[[i1:.+]]:3 = stablehlo.custom_call @jaxzyme.aug(%[[i0]], %arg0, %arg1) : (tensor<1xi64>, tensor<2x3xf32>, tensor<5x7xf32>) -> (tensor<6x9xf32>, tensor<4x6xf32>, tensor<16xi8>)
# CHECK-NEXT:     %[[i2:.+]] = stablehlo.constant dense<4> : tensor<1xi64>
# CHECK-NEXT:     %[[i3:.+]]:2 = stablehlo.custom_call @jaxzyme.rev(%[[i2]], %[[i1]]#2, %arg2, %arg3) : (tensor<1xi64>, tensor<16xi8>, tensor<6x9xf32>, tensor<4x6xf32>) -> (tensor<2x3xf32>, tensor<5x7xf32>)
# CHECK-NEXT:     return %[[i1]]#0, %[[i1]]#1, %[[i3]]#0, %[[i3]]#1 : tensor<6x9xf32>, tensor<4x6xf32>, tensor<2x3xf32>, tensor<5x7xf32>
# CHECK-NEXT:   }
# CHECK-NEXT: }

print(lower(jax.jit(g, backend='cpu'), (ones, twos, x, y)).compiler_ir(dialect="stablehlo"))

primals, f_vjp = jax.vjp(jax.jit(do_something, backend='cpu'), ones, twos)

print(lower(jax.jit(f_vjp, backend='cpu'), ((x, y),)).compiler_ir(dialect="stablehlo"))
# CHECK: module @jit__unnamed_wrapped_function_ attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
# CHECK-NEXT:   func.func public @main
# CHECK-NEXT:     %[[i0:.+]] = stablehlo.constant dense<[0, 0, -128, 63, 0, 0, -128, 63, 0, 0, -128, 63, 0, 0, -128, 63]> : tensor<16xi8>
# CHECK-NEXT:     %[[i1:.+]]:2 = call @do_something(%[[i0]], %arg0, %arg1) : (tensor<16xi8>, tensor<6x9xf32>, tensor<4x6xf32>) -> (tensor<2x3xf32>, tensor<5x7xf32>)
# CHECK-NEXT:     return %[[i1]]#0, %[[i1]]#1 : tensor<2x3xf32>, tensor<5x7xf32>
# CHECK-NEXT:   }
# CHECK:   func.func private @do_something
# CHECK-NEXT:     %[[shard1:.+]] = stablehlo.custom_call @Sharding(%arg1) {backend_config = "", mhlo.sharding = "{replicated}"} : (tensor<6x9xf32>) -> tensor<6x9xf32>
# CHECK-NEXT:     %[[shard2:.+]] = stablehlo.custom_call @Sharding(%arg2) {backend_config = "", mhlo.sharding = "{replicated}"} : (tensor<4x6xf32>) -> tensor<4x6xf32>
# CHECK-NEXT:     %[[i0:.+]] = stablehlo.constant dense<6> : tensor<1xi64>
# CHECK-NEXT:     %[[i1:.+]]:2 = stablehlo.custom_call @jaxzyme.rev(%[[i0]], %arg0, %[[shard1]], %[[shard2]]) : (tensor<1xi64>, tensor<16xi8>, tensor<6x9xf32>, tensor<4x6xf32>) -> (tensor<2x3xf32>, tensor<5x7xf32>)
# CHECK-NEXT:     %[[res1:.+]] = stablehlo.custom_call @Sharding(%[[i1]]#0) {backend_config = "", mhlo.sharding = "{replicated}"} : (tensor<2x3xf32>) -> tensor<2x3xf32>
# CHECK-NEXT:     %[[res2:.+]] = stablehlo.custom_call @Sharding(%[[i1]]#1) {backend_config = "", mhlo.sharding = "{replicated}"} : (tensor<5x7xf32>) -> tensor<5x7xf32>
# CHECK-NEXT:     return %[[res1]], %[[res2]] : tensor<2x3xf32>, tensor<5x7xf32>
# CHECK-NEXT:   }
# CHECK-NEXT: }
