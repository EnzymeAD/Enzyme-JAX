// RUN: enzymexlamlir-opt --canonicalize-loops %s | FileCheck %s

#map31 = affine_map<(d0, d1) -> (d0 + d1 * 16 - (d0 floordiv 16) * 16 + 2625)>  
#map32 = affine_map<(d0, d1) -> (d0 + d1 * 16 - (d0 floordiv 16) * 16 + 2251)>
#map33 = affine_map<(d0, d1) -> (d0 + d1 * 16 - (d0 floordiv 16) * 16 + 47131)>
#map34 = affine_map<(d0, d1) -> (d0 + d1 * 16 - (d0 floordiv 16) * 16 + 47505)>

func.func private @bar(%arg1 : index, %arg2 : index, %arg3 : index, %arg4 : index, %arg5 : index, %arg6 : index)

// CHECK-LABEL: foo
func.func private @foo(%arg0: !llvm.ptr<1>) {
    %c4_i32 = arith.constant 4 : i32
    %c4_i64 = arith.constant 4 : i64
    %c1_i64 = arith.constant 1 : i64
    %c-361_i64 = arith.constant -361 : i64
    %c-360_i64 = arith.constant -360 : i64
    affine.parallel (%arg1, %arg2, %arg3, %arg4, %arg5, %arg6) = (0, 0, 0, 0, 0, 0) to (23, 1, 1, 256, 1, 1) {
      func.call @bar(%arg1, %arg2, %arg3, %arg4, %arg5, %arg6) : (index, index, index, index, index, index) -> ()
    }
    // CHECK: affine.parallel (%arg1, %arg2) = (0, 0) to (23, 256)
    // CHECK: func.call @bar(%arg1, %c0, %c0, %arg2, %c0, %c0)
    return
  }
