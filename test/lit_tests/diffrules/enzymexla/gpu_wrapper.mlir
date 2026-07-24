// RUN: enzymexlamlir-opt %s --enzyme --canonicalize --remove-unnecessary-enzyme-ops --enzyme-simplify-math --split-input-file | FileCheck %s
// RUN: enzymexlamlir-opt %s --enzyme=dataflow --canonicalize --remove-unnecessary-enzyme-ops --enzyme-simplify-math --split-input-file | FileCheck %s --check-prefix=DF

func.func private @gpu_wrapper(%ptr: !llvm.ptr) {
  %c1 = arith.constant 1 : index
  %c120 = arith.constant 120 : index
  %code = "enzymexla.gpu_wrapper"(%c1, %c1, %c1, %c120, %c120, %c1) ({
    affine.parallel (%iv, %jv) = (0, 0) to (120, 120) {
      %mem = "enzymexla.pointer2memref"(%ptr) : (!llvm.ptr) -> memref<?xf32>
      %val = affine.load %mem[%iv * 120 + %jv] : memref<?xf32>
      %sq = arith.mulf %val, %val : f32
      affine.store %sq, %mem[%iv * 120 + %jv] : memref<?xf32>
    }
    "enzymexla.polygeist_yield"() : () -> ()
  }) : (index, index, index, index, index, index) -> index
  return
}

func.func @dgpu_wrapper() {
  %alloc = gpu.alloc() : memref<120x120xf32, 1>
  %dalloc = gpu.alloc() : memref<120x120xf32, 1>
  %ptr = "enzymexla.memref2pointer"(%alloc) : (memref<120x120xf32, 1>) -> !llvm.ptr
  %dptr = "enzymexla.memref2pointer"(%dalloc) : (memref<120x120xf32, 1>) -> !llvm.ptr
  enzyme.autodiff @gpu_wrapper(%ptr, %dptr) {
    activity = [#enzyme<activity enzyme_dup>],
    ret_activity = []
  } : (!llvm.ptr, !llvm.ptr) -> ()
  return
}

// CHECK: #[[$ATTR_0:.+]] = affine_map<(d0, d1) -> (d0 * 120 + d1)>
// CHECK-LABEL:   func.func private @diffegpu_wrapper(
// CHECK-SAME:      %[[ARG0:.*]]: !llvm.ptr,
// CHECK-SAME:      %[[ARG1:.*]]: !llvm.ptr) {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 120 : index
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant 1 : index
// CHECK:           %[[CONSTANT_2:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:           %[[VAL_0:.*]] = "enzymexla.pointer2memref"(%[[ARG1]]) : (!llvm.ptr) -> memref<?xf32>
// CHECK:           %[[ALLOC_0:.*]] = gpu.alloc  () : memref<120x120xf32, 1>
// CHECK:           %[[VAL_1:.*]] = "enzymexla.gpu_wrapper"(%[[CONSTANT_1]], %[[CONSTANT_1]], %[[CONSTANT_1]], %[[CONSTANT_0]], %[[CONSTANT_0]], %[[CONSTANT_1]]) ({
// CHECK:             %[[VAL_2:.*]] = "enzymexla.pointer2memref"(%[[ARG0]]) : (!llvm.ptr) -> memref<?xf32>
// CHECK:             affine.parallel (%[[VAL_3:.*]], %[[VAL_4:.*]]) = (0, 0) to (120, 120) {
// CHECK:               %[[LOAD_0:.*]] = affine.load %[[VAL_2]]{{\[}}%[[VAL_3]] * 120 + %[[VAL_4]]] : memref<?xf32>
// CHECK:               memref.store %[[LOAD_0]], %[[ALLOC_0]]{{\[}}%[[VAL_3]], %[[VAL_4]]] : memref<120x120xf32, 1>
// CHECK:               %[[MULF_0:.*]] = arith.mulf %[[LOAD_0]], %[[LOAD_0]] : f32
// CHECK:               affine.store %[[MULF_0]], %[[VAL_2]]{{\[}}%[[VAL_3]] * 120 + %[[VAL_4]]] : memref<?xf32>
// CHECK:             }
// CHECK:             "enzymexla.polygeist_yield"() : () -> ()
// CHECK:           }) : (index, index, index, index, index, index) -> index
// CHECK:           %[[VAL_5:.*]] = "enzymexla.gpu_wrapper"(%[[CONSTANT_1]], %[[CONSTANT_1]], %[[CONSTANT_1]], %[[CONSTANT_0]], %[[CONSTANT_0]], %[[CONSTANT_1]]) ({
// CHECK:             affine.parallel (%[[VAL_6:.*]], %[[VAL_7:.*]]) = (0, 0) to (120, 120) {
// CHECK:               %[[LOAD_1:.*]] = memref.load %[[ALLOC_0]]{{\[}}%[[VAL_6]], %[[VAL_7]]] : memref<120x120xf32, 1>
// CHECK:               %[[APPLY_0:.*]] = affine.apply #[[$ATTR_0]](%[[VAL_6]], %[[VAL_7]])
// CHECK:               %[[LOAD_2:.*]] = memref.load %[[VAL_0]]{{\[}}%[[APPLY_0]]] : memref<?xf32>
// CHECK:               %[[APPLY_1:.*]] = affine.apply #[[$ATTR_0]](%[[VAL_6]], %[[VAL_7]])
// CHECK:               memref.store %[[CONSTANT_2]], %[[VAL_0]]{{\[}}%[[APPLY_1]]] : memref<?xf32>
// CHECK:               %[[MULF_1:.*]] = arith.mulf %[[LOAD_2]], %[[LOAD_1]] : f32
// CHECK:               %[[MULF_2:.*]] = arith.mulf %[[LOAD_2]], %[[LOAD_1]] : f32
// CHECK:               %[[ADDF_0:.*]] = arith.addf %[[MULF_1]], %[[MULF_2]] : f32
// CHECK:               %[[APPLY_2:.*]] = affine.apply #[[$ATTR_0]](%[[VAL_6]], %[[VAL_7]])
// CHECK:               %[[ATOMIC_RMW_0:.*]] = enzyme.atomic_rmw addf %[[ADDF_0]], %[[VAL_0]]{{\[}}%[[APPLY_2]]] monotonic : (f32, memref<?xf32>) -> f32
// CHECK:             }
// CHECK:             "enzymexla.polygeist_yield"() : () -> ()
// CHECK:           }) : (index, index, index, index, index, index) -> index
// CHECK:           gpu.dealloc  %[[ALLOC_0]] : memref<120x120xf32, 1>
// CHECK:           return
// CHECK:         }

// DF: #[[$ATTR_0:.+]] = affine_map<(d0, d1) -> (d0 * 120 + d1)>
// DF-LABEL:   func.func private @diffegpu_wrapper(
// DF-SAME:      %[[ARG0:.*]]: !llvm.ptr,
// DF-SAME:      %[[ARG1:.*]]: !llvm.ptr) {
// DF:           %[[CONSTANT_0:.*]] = arith.constant 120 : index
// DF:           %[[CONSTANT_1:.*]] = arith.constant 1 : index
// DF:           %[[CONSTANT_2:.*]] = arith.constant 0.000000e+00 : f32
// DF:           %[[VAL_0:.*]] = "enzymexla.pointer2memref"(%[[ARG1]]) : (!llvm.ptr) -> memref<?xf32>
// DF:           %[[ALLOC_0:.*]] = gpu.alloc  () : memref<120x120xf32, 1>
// DF:           %[[VAL_1:.*]] = "enzymexla.gpu_wrapper"(%[[CONSTANT_1]], %[[CONSTANT_1]], %[[CONSTANT_1]], %[[CONSTANT_0]], %[[CONSTANT_0]], %[[CONSTANT_1]]) ({
// DF:             %[[VAL_2:.*]] = "enzymexla.pointer2memref"(%[[ARG0]]) : (!llvm.ptr) -> memref<?xf32>
// DF:             affine.parallel (%[[VAL_3:.*]], %[[VAL_4:.*]]) = (0, 0) to (120, 120) {
// DF:               %[[LOAD_0:.*]] = affine.load %[[VAL_2]]{{\[}}%[[VAL_3]] * 120 + %[[VAL_4]]] : memref<?xf32>
// DF:               memref.store %[[LOAD_0]], %[[ALLOC_0]]{{\[}}%[[VAL_3]], %[[VAL_4]]] : memref<120x120xf32, 1>
// DF:               %[[MULF_0:.*]] = arith.mulf %[[LOAD_0]], %[[LOAD_0]] : f32
// DF:               affine.store %[[MULF_0]], %[[VAL_2]]{{\[}}%[[VAL_3]] * 120 + %[[VAL_4]]] : memref<?xf32>
// DF:             }
// DF:             "enzymexla.polygeist_yield"() : () -> ()
// DF:           }) : (index, index, index, index, index, index) -> index
// DF:           %[[VAL_5:.*]] = "enzymexla.gpu_wrapper"(%[[CONSTANT_1]], %[[CONSTANT_1]], %[[CONSTANT_1]], %[[CONSTANT_0]], %[[CONSTANT_0]], %[[CONSTANT_1]]) ({
// DF:             affine.parallel (%[[VAL_6:.*]], %[[VAL_7:.*]]) = (0, 0) to (120, 120) {
// DF:               %[[LOAD_1:.*]] = memref.load %[[ALLOC_0]]{{\[}}%[[VAL_6]], %[[VAL_7]]] : memref<120x120xf32, 1>
// DF:               %[[APPLY_0:.*]] = affine.apply #[[$ATTR_0]](%[[VAL_6]], %[[VAL_7]])
// DF:               %[[LOAD_2:.*]] = memref.load %[[VAL_0]]{{\[}}%[[APPLY_0]]] : memref<?xf32>
// DF:               %[[APPLY_1:.*]] = affine.apply #[[$ATTR_0]](%[[VAL_6]], %[[VAL_7]])
// DF:               memref.store %[[CONSTANT_2]], %[[VAL_0]]{{\[}}%[[APPLY_1]]] : memref<?xf32>
// DF:               %[[MULF_1:.*]] = arith.mulf %[[LOAD_2]], %[[LOAD_1]] : f32
// DF:               %[[MULF_2:.*]] = arith.mulf %[[LOAD_2]], %[[LOAD_1]] : f32
// DF:               %[[ADDF_0:.*]] = arith.addf %[[MULF_1]], %[[MULF_2]] : f32
// DF:               %[[APPLY_2:.*]] = affine.apply #[[$ATTR_0]](%[[VAL_6]], %[[VAL_7]])
// DF:               %[[ATOMIC_RMW_0:.*]] = enzyme.atomic_rmw addf %[[ADDF_0]], %[[VAL_0]]{{\[}}%[[APPLY_2]]] monotonic : (f32, memref<?xf32>) -> f32
// DF:             }
// DF:             "enzymexla.polygeist_yield"() : () -> ()
// DF:           }) : (index, index, index, index, index, index) -> index
// DF:           gpu.dealloc  %[[ALLOC_0]] : memref<120x120xf32, 1>
// DF:           return
// DF:         }

// -----


func.func private @gpu_wrapper_partial_inactive(%ptr: !llvm.ptr, %inactive: !llvm.ptr) {
  %c1 = arith.constant 1 : index
  %c120 = arith.constant 120 : index
  %code = "enzymexla.gpu_wrapper"(%c1, %c1, %c1, %c120, %c120, %c1) ({
    affine.parallel (%iv, %jv) = (0, 0) to (120, 120) {
      %mem = "enzymexla.pointer2memref"(%ptr) : (!llvm.ptr) -> memref<?xf32>
      %imem = "enzymexla.pointer2memref"(%inactive) : (!llvm.ptr) -> memref<?xf32>
      %val = affine.load %mem[%iv * 120 + %jv] : memref<?xf32>
      %load = affine.load %imem[%iv * 120 + %jv] : memref<?xf32>
      %sin = math.sin %load : f32
      %sq = arith.mulf %val, %sin : f32
      affine.store %sq, %mem[%iv * 120 + %jv] : memref<?xf32>
    }
    "enzymexla.polygeist_yield"() : () -> ()
  }) : (index, index, index, index, index, index) -> index
  return
}

func.func @dgpu_wrapper_part_inactive() {
  %alloc = gpu.alloc() : memref<120x120xf32, 1>
  %dalloc = gpu.alloc() : memref<120x120xf32, 1>
  %inactive = gpu.alloc() : memref<120x120xf32, 1>
  %ptr = "enzymexla.memref2pointer"(%alloc) : (memref<120x120xf32, 1>) -> !llvm.ptr
  %dptr = "enzymexla.memref2pointer"(%dalloc) : (memref<120x120xf32, 1>) -> !llvm.ptr
  %i_ptr = "enzymexla.memref2pointer"(%inactive) : (memref<120x120xf32, 1>) -> !llvm.ptr
  enzyme.autodiff @gpu_wrapper_partial_inactive(%ptr, %dptr, %i_ptr) {
    activity = [#enzyme<activity enzyme_dup>, #enzyme<activity enzyme_const>],
    ret_activity = []
  } : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
  return
}

// CHECK: #[[$ATTR_0:.+]] = affine_map<(d0, d1) -> (d0 * 120 + d1)>
// CHECK-LABEL:   func.func private @diffegpu_wrapper_partial_inactive(
// CHECK-SAME:      %[[ARG0:.*]]: !llvm.ptr, %[[ARG1:.*]]: !llvm.ptr, %[[ARG2:.*]]: !llvm.ptr) {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 120 : index
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant 1 : index
// CHECK:           %[[CONSTANT_2:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:           %[[VAL_0:.*]] = "enzymexla.pointer2memref"(%[[ARG1]]) : (!llvm.ptr) -> memref<?xf32>
// CHECK:           %[[ALLOC_0:.*]] = gpu.alloc  () : memref<120x120xf32, 1>
// CHECK:           %[[VAL_1:.*]] = "enzymexla.gpu_wrapper"(%[[CONSTANT_1]], %[[CONSTANT_1]], %[[CONSTANT_1]], %[[CONSTANT_0]], %[[CONSTANT_0]], %[[CONSTANT_1]]) ({
// CHECK:             %[[VAL_2:.*]] = "enzymexla.pointer2memref"(%[[ARG0]]) : (!llvm.ptr) -> memref<?xf32>
// CHECK:             %[[VAL_3:.*]] = "enzymexla.pointer2memref"(%[[ARG2]]) : (!llvm.ptr) -> memref<?xf32>
// CHECK:             affine.parallel (%[[VAL_4:.*]], %[[VAL_5:.*]]) = (0, 0) to (120, 120) {
// CHECK:               %[[LOAD_0:.*]] = affine.load %[[VAL_2]]{{\[}}%[[VAL_4]] * 120 + %[[VAL_5]]] : memref<?xf32>
// CHECK:               %[[LOAD_1:.*]] = affine.load %[[VAL_3]]{{\[}}%[[VAL_4]] * 120 + %[[VAL_5]]] : memref<?xf32>
// CHECK:               memref.store %[[LOAD_1]], %[[ALLOC_0]]{{\[}}%[[VAL_4]], %[[VAL_5]]] : memref<120x120xf32, 1>
// CHECK:               %[[SIN_0:.*]] = math.sin %[[LOAD_1]] : f32
// CHECK:               %[[MULF_0:.*]] = arith.mulf %[[LOAD_0]], %[[SIN_0]] : f32
// CHECK:               affine.store %[[MULF_0]], %[[VAL_2]]{{\[}}%[[VAL_4]] * 120 + %[[VAL_5]]] : memref<?xf32>
// CHECK:             }
// CHECK:             "enzymexla.polygeist_yield"() : () -> ()
// CHECK:           }) : (index, index, index, index, index, index) -> index
// CHECK:           %[[VAL_6:.*]] = "enzymexla.gpu_wrapper"(%[[CONSTANT_1]], %[[CONSTANT_1]], %[[CONSTANT_1]], %[[CONSTANT_0]], %[[CONSTANT_0]], %[[CONSTANT_1]]) ({
// CHECK:             affine.parallel (%[[VAL_7:.*]], %[[VAL_8:.*]]) = (0, 0) to (120, 120) {
// CHECK:               %[[LOAD_2:.*]] = memref.load %[[ALLOC_0]]{{\[}}%[[VAL_7]], %[[VAL_8]]] : memref<120x120xf32, 1>
// CHECK:               %[[SIN_1:.*]] = math.sin %[[LOAD_2]] : f32
// CHECK:               %[[APPLY_0:.*]] = affine.apply #[[$ATTR_0]](%[[VAL_7]], %[[VAL_8]])
// CHECK:               %[[LOAD_3:.*]] = memref.load %[[VAL_0]]{{\[}}%[[APPLY_0]]] : memref<?xf32>
// CHECK:               %[[APPLY_1:.*]] = affine.apply #[[$ATTR_0]](%[[VAL_7]], %[[VAL_8]])
// CHECK:               memref.store %[[CONSTANT_2]], %[[VAL_0]]{{\[}}%[[APPLY_1]]] : memref<?xf32>
// CHECK:               %[[MULF_1:.*]] = arith.mulf %[[LOAD_3]], %[[SIN_1]] : f32
// CHECK:               %[[APPLY_2:.*]] = affine.apply #[[$ATTR_0]](%[[VAL_7]], %[[VAL_8]])
// CHECK:               %[[ATOMIC_RMW_0:.*]] = enzyme.atomic_rmw addf %[[MULF_1]], %[[VAL_0]]{{\[}}%[[APPLY_2]]] monotonic : (f32, memref<?xf32>) -> f32
// CHECK:             }
// CHECK:             "enzymexla.polygeist_yield"() : () -> ()
// CHECK:           }) : (index, index, index, index, index, index) -> index
// CHECK:           gpu.dealloc  %[[ALLOC_0]] : memref<120x120xf32, 1>
// CHECK:           return
// CHECK:         }

// DF: #[[$ATTR_0:.+]] = affine_map<(d0, d1) -> (d0 * 120 + d1)>
// DF-LABEL:   func.func private @diffegpu_wrapper_partial_inactive(
// DF-SAME:      %[[ARG0:.*]]: !llvm.ptr, %[[ARG1:.*]]: !llvm.ptr, %[[ARG2:.*]]: !llvm.ptr) {
// DF:           %[[CONSTANT_0:.*]] = arith.constant 120 : index
// DF:           %[[CONSTANT_1:.*]] = arith.constant 1 : index
// DF:           %[[CONSTANT_2:.*]] = arith.constant 0.000000e+00 : f32
// DF:           %[[VAL_0:.*]] = "enzymexla.pointer2memref"(%[[ARG1]]) : (!llvm.ptr) -> memref<?xf32>
// DF:           %[[ALLOC_0:.*]] = gpu.alloc  () : memref<120x120xf32, 1>
// DF:           %[[VAL_1:.*]] = "enzymexla.gpu_wrapper"(%[[CONSTANT_1]], %[[CONSTANT_1]], %[[CONSTANT_1]], %[[CONSTANT_0]], %[[CONSTANT_0]], %[[CONSTANT_1]]) ({
// DF:             %[[VAL_2:.*]] = "enzymexla.pointer2memref"(%[[ARG0]]) : (!llvm.ptr) -> memref<?xf32>
// DF:             %[[VAL_3:.*]] = "enzymexla.pointer2memref"(%[[ARG2]]) : (!llvm.ptr) -> memref<?xf32>
// DF:             affine.parallel (%[[VAL_4:.*]], %[[VAL_5:.*]]) = (0, 0) to (120, 120) {
// DF:               %[[LOAD_0:.*]] = affine.load %[[VAL_2]]{{\[}}%[[VAL_4]] * 120 + %[[VAL_5]]] : memref<?xf32>
// DF:               %[[LOAD_1:.*]] = affine.load %[[VAL_3]]{{\[}}%[[VAL_4]] * 120 + %[[VAL_5]]] : memref<?xf32>
// DF:               memref.store %[[LOAD_1]], %[[ALLOC_0]]{{\[}}%[[VAL_4]], %[[VAL_5]]] : memref<120x120xf32, 1>
// DF:               %[[SIN_0:.*]] = math.sin %[[LOAD_1]] : f32
// DF:               %[[MULF_0:.*]] = arith.mulf %[[LOAD_0]], %[[SIN_0]] : f32
// DF:               affine.store %[[MULF_0]], %[[VAL_2]]{{\[}}%[[VAL_4]] * 120 + %[[VAL_5]]] : memref<?xf32>
// DF:             }
// DF:             "enzymexla.polygeist_yield"() : () -> ()
// DF:           }) : (index, index, index, index, index, index) -> index
// DF:           %[[VAL_6:.*]] = "enzymexla.gpu_wrapper"(%[[CONSTANT_1]], %[[CONSTANT_1]], %[[CONSTANT_1]], %[[CONSTANT_0]], %[[CONSTANT_0]], %[[CONSTANT_1]]) ({
// DF:             affine.parallel (%[[VAL_7:.*]], %[[VAL_8:.*]]) = (0, 0) to (120, 120) {
// DF:               %[[LOAD_2:.*]] = memref.load %[[ALLOC_0]]{{\[}}%[[VAL_7]], %[[VAL_8]]] : memref<120x120xf32, 1>
// DF:               %[[SIN_1:.*]] = math.sin %[[LOAD_2]] : f32
// DF:               %[[APPLY_0:.*]] = affine.apply #[[$ATTR_0]](%[[VAL_7]], %[[VAL_8]])
// DF:               %[[LOAD_3:.*]] = memref.load %[[VAL_0]]{{\[}}%[[APPLY_0]]] : memref<?xf32>
// DF:               %[[APPLY_1:.*]] = affine.apply #[[$ATTR_0]](%[[VAL_7]], %[[VAL_8]])
// DF:               memref.store %[[CONSTANT_2]], %[[VAL_0]]{{\[}}%[[APPLY_1]]] : memref<?xf32>
// DF:               %[[MULF_1:.*]] = arith.mulf %[[LOAD_3]], %[[SIN_1]] : f32
// DF:               %[[APPLY_2:.*]] = affine.apply #[[$ATTR_0]](%[[VAL_7]], %[[VAL_8]])
// DF:               %[[ATOMIC_RMW_0:.*]] = enzyme.atomic_rmw addf %[[MULF_1]], %[[VAL_0]]{{\[}}%[[APPLY_2]]] monotonic : (f32, memref<?xf32>) -> f32
// DF:             }
// DF:             "enzymexla.polygeist_yield"() : () -> ()
// DF:           }) : (index, index, index, index, index, index) -> index
// DF:           gpu.dealloc  %[[ALLOC_0]] : memref<120x120xf32, 1>
// DF:           return
// DF:         }
