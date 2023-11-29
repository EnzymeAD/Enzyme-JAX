# RUN: python %s | FileCheck %s

import jax
import jax.numpy as jnp
from enzyme_jax import cpp_call

def do_something(ones, twos):
    shape = jax.core.ShapedArray(tuple(3 * s for s in ones.shape), ones.dtype)
    shape2 = jax.core.ShapedArray(tuple(2 * s for s in ones.shape), ones.dtype)
    a, b = cpp_call(ones, twos, out_shapes=[shape, shape2], source="""
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
    """, fn="myfn")
    return a, b

ones = jnp.ones((2, 3), jnp.float32)
twos = jnp.ones((5, 7), jnp.float32)

@jax.jit
def fwdmode(a, b, c, d):
    return jax.jvp(do_something, (a, b), (c,d))

print(fwdmode.lower(ones, twos, ones, twos).compiler_ir(dialect="mhlo"))

# CHECK: module @jit_fwdmode attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
# CHECK-NEXT:   func.func public @main(%arg0: tensor<2x3xf32> {mhlo.sharding = "{replicated}"}, %arg1: tensor<5x7xf32> {mhlo.sharding = "{replicated}"}, %arg2: tensor<2x3xf32> {mhlo.sharding = "{replicated}"}, %arg3: tensor<5x7xf32> {mhlo.sharding = "{replicated}"}) -> (tensor<6x9xf32> {jax.result_info = "[0][0]"}, tensor<4x6xf32> {jax.result_info = "[0][1]"}, tensor<6x9xf32> {jax.result_info = "[1][0]"}, tensor<4x6xf32> {jax.result_info = "[1][1]"}) {
# CHECK-NEXT:     %0 = mhlo.constant dense<1> : tensor<1xi64>
# CHECK-NEXT:     %1:4 = mhlo.custom_call @jaxzyme.fwd(%0, %arg0, %arg2, %arg1, %arg3) {backend_config = ""} : (tensor<1xi64>, tensor<2x3xf32>, tensor<2x3xf32>, tensor<5x7xf32>, tensor<5x7xf32>) -> (tensor<6x9xf32>, tensor<6x9xf32>, tensor<4x6xf32>, tensor<4x6xf32>)
# CHECK-NEXT:     return %1#0, %1#2, %1#1, %1#3 : tensor<6x9xf32>, tensor<4x6xf32>, tensor<6x9xf32>, tensor<4x6xf32>
# CHECK-NEXT:   }
# CHECK-NEXT: }

@jax.jit
def f(a, b):
    return jax.vjp(do_something, a, b)

print(f.lower(ones, twos).compiler_ir(dialect="mhlo"))

# CHECK: module @jit_f attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
# CHECK-NEXT:   func.func public @main(%arg0: tensor<2x3xf32> {mhlo.sharding = "{replicated}"}, %arg1: tensor<5x7xf32> {mhlo.sharding = "{replicated}"}) -> (tensor<6x9xf32> {jax.result_info = "[0][0]"}, tensor<4x6xf32> {jax.result_info = "[0][1]"}, tensor<16xi8> {jax.result_info = "[1][<flat index 0>][0][<flat index 0>][0][0]"}, tensor<2x3xf32> {jax.result_info = "[1][<flat index 0>][0][<flat index 0>][0][1]"}, tensor<5x7xf32> {jax.result_info = "[1][<flat index 0>][0][<flat index 0>][0][2]"}) {
# CHECK-NEXT:     %0 = mhlo.constant dense<2> : tensor<1xi64>
# CHECK-NEXT:     %1:3 = mhlo.custom_call @jaxzyme.aug(%0, %arg0, %arg1) {backend_config = ""} : (tensor<1xi64>, tensor<2x3xf32>, tensor<5x7xf32>) -> (tensor<6x9xf32>, tensor<4x6xf32>, tensor<16xi8>)
# CHECK-NEXT:     return %1#0, %1#1, %1#2, %arg0, %arg1 : tensor<6x9xf32>, tensor<4x6xf32>, tensor<16xi8>, tensor<2x3xf32>, tensor<5x7xf32>
# CHECK-NEXT:   }
# CHECK-NEXT: }

x = jnp.ones((6, 9), jnp.float32)
y = jnp.ones((4, 6), jnp.float32)

@jax.jit
def g(a, b, x, y):
    primals, f_vjp = jax.vjp(do_something, a, b)
    return primals, f_vjp((x, y))

# CHECK: module @jit_g attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
# CHECK-NEXT:   func.func public @main(%arg0: tensor<2x3xf32> {mhlo.sharding = "{replicated}"}, %arg1: tensor<5x7xf32> {mhlo.sharding = "{replicated}"}, %arg2: tensor<6x9xf32> {mhlo.sharding = "{replicated}"}, %arg3: tensor<4x6xf32> {mhlo.sharding = "{replicated}"}) -> (tensor<6x9xf32> {jax.result_info = "[0][0]"}, tensor<4x6xf32> {jax.result_info = "[0][1]"}, tensor<2x3xf32> {jax.result_info = "[1][0]"}, tensor<5x7xf32> {jax.result_info = "[1][1]"}) {
# CHECK-NEXT:     %0 = mhlo.constant dense<3> : tensor<1xi64>
# CHECK-NEXT:     %1:3 = mhlo.custom_call @jaxzyme.aug(%0, %arg0, %arg1) {backend_config = ""} : (tensor<1xi64>, tensor<2x3xf32>, tensor<5x7xf32>) -> (tensor<6x9xf32>, tensor<4x6xf32>, tensor<16xi8>)
# CHECK-NEXT:     %2 = mhlo.constant dense<4> : tensor<1xi64>
# CHECK-NEXT:     %3:2 = mhlo.custom_call @jaxzyme.rev(%2, %1#2, %arg2, %arg3) {backend_config = ""} : (tensor<1xi64>, tensor<16xi8>, tensor<6x9xf32>, tensor<4x6xf32>) -> (tensor<2x3xf32>, tensor<5x7xf32>)
# CHECK-NEXT:     return %1#0, %1#1, %3#0, %3#1 : tensor<6x9xf32>, tensor<4x6xf32>, tensor<2x3xf32>, tensor<5x7xf32>
# CHECK-NEXT:   }
# CHECK-NEXT: }

print(g.lower(ones, twos, x, y).compiler_ir(dialect="mhlo"))

primals, f_vjp = jax.vjp(jax.jit(do_something), ones, twos)

print(jax.jit(f_vjp).lower((x, y)).compiler_ir(dialect="mhlo"))
# CHECK: module @jit__unnamed_wrapped_function_ attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
# CHECK-NEXT:   func.func public @main(%arg0: tensor<6x9xf32> {mhlo.sharding = "{replicated}"}, %arg1: tensor<4x6xf32> {mhlo.sharding = "{replicated}"}) -> (tensor<2x3xf32>, tensor<5x7xf32>) {
# CHECK-NEXT:     %0 = mhlo.constant dense<[0, 0, -128, 63, 0, 0, -128, 63, 0, 0, -128, 63, 0, 0, -128, 63]> : tensor<16xi8>
# CHECK-NEXT:     %1:2 = call @do_something(%0, %arg0, %arg1) : (tensor<16xi8>, tensor<6x9xf32>, tensor<4x6xf32>) -> (tensor<2x3xf32>, tensor<5x7xf32>)
# CHECK-NEXT:     return %1#0, %1#1 : tensor<2x3xf32>, tensor<5x7xf32>
# CHECK-NEXT:   }
# CHECK-NEXT:   func.func private @do_something(%arg0: tensor<16xi8>, %arg1: tensor<6x9xf32>, %arg2: tensor<4x6xf32>) -> (tensor<2x3xf32>, tensor<5x7xf32>) {
# CHECK-NEXT:     %0 = mhlo.constant dense<6> : tensor<1xi64>
# CHECK-NEXT:     %1:2 = mhlo.custom_call @jaxzyme.rev(%0, %arg0, %arg1, %arg2) {backend_config = ""} : (tensor<1xi64>, tensor<16xi8>, tensor<6x9xf32>, tensor<4x6xf32>) -> (tensor<2x3xf32>, tensor<5x7xf32>)
# CHECK-NEXT:     return %1#0, %1#1 : tensor<2x3xf32>, tensor<5x7xf32>
# CHECK-NEXT:   }
# CHECK-NEXT: }