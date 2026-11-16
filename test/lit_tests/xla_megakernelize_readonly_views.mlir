// RUN: enzymexlamlir-opt %s --split-input-file --xla-megakernelize --symbol-dce | FileCheck %s

// The byte view comes first. Keep its representation as the shared state.
// Convert to floats for the writer, then return the updated bytes.
module {
  llvm.func @bytes_first(%p: !llvm.ptr) {
    %bytes = "enzymexla.pointer2memref"(%p) : (!llvm.ptr) -> memref<?xi8>
    %floats = "enzymexla.pointer2memref"(%p) : (!llvm.ptr) -> memref<?xf32>
    enzymexla.xla_wrapper @read (%bytes, %floats) : (memref<?xi8>, memref<?xf32>) -> ()
    enzymexla.xla_wrapper @write (%floats) : (memref<?xf32>) -> ()
    llvm.return
  }
  func.func private @read(%bytes: tensor<?xi8>, %floats: tensor<?xf32>) -> (tensor<?xi8>, tensor<?xf32>) {
    return %bytes, %floats : tensor<?xi8>, tensor<?xf32>
  }
  func.func private @write(%floats: tensor<?xf32>) -> tensor<?xf32> {
    %updated = stablehlo.negate %floats : tensor<?xf32>
    return %updated : tensor<?xf32>
  }
}
// CHECK-LABEL: llvm.func @bytes_first
// CHECK: %[[BYTES:.*]] = "enzymexla.pointer2memref"{{.*}}memref<?xi8>
// CHECK: enzymexla.xla_wrapper @read (%[[BYTES]]) : (memref<?xi8>) -> ()
// CHECK-NOT: enzymexla.xla_wrapper
// CHECK: llvm.return
// CHECK-LABEL: func.func private @read(
// CHECK-SAME: %[[BYTES:[^:]+]]: tensor<?xi8>) -> tensor<?xi8>
// CHECK: %[[SPLIT:.*]] = stablehlo.dynamic_reshape %[[BYTES]], {{.*}} -> tensor<?x4xi8>
// CHECK: %[[FLOATS:.*]] = stablehlo.bitcast_convert %[[SPLIT]] : (tensor<?x4xi8>) -> tensor<?xf32>
// CHECK: %[[UPDATED:.*]] = stablehlo.negate %[[FLOATS]]
// CHECK: %[[UPDATED_BYTES:.*]] = stablehlo.bitcast_convert %[[UPDATED]] : (tensor<?xf32>) -> tensor<?x4xi8>
// CHECK: %[[FLAT:.*]] = stablehlo.dynamic_reshape %[[UPDATED_BYTES]], {{.*}} -> tensor<?xi8>
// CHECK: return %[[FLAT]] : tensor<?xi8>

// -----

// One alias changes while the other returns its old value. Do not choose a
// result for this buffer, even when the other wrapper has no duplicate inputs.
module {
  llvm.func @conflicting_alias_results(%p: !llvm.ptr) {
    %floats = "enzymexla.pointer2memref"(%p) : (!llvm.ptr) -> memref<?xf32>
    %bytes = "enzymexla.pointer2memref"(%p) : (!llvm.ptr) -> memref<?xi8>
    enzymexla.xla_wrapper @conflict (%floats, %bytes) : (memref<?xf32>, memref<?xi8>) -> ()
    enzymexla.xla_wrapper @write (%floats) : (memref<?xf32>) -> ()
    enzymexla.xla_wrapper @conflict (%floats, %bytes) : (memref<?xf32>, memref<?xi8>) -> ()
    llvm.return
  }
  func.func private @conflict(%floats: tensor<?xf32>, %bytes: tensor<?xi8>) -> (tensor<?xf32>, tensor<?xi8>) {
    %flag = stablehlo.slice %bytes [0:1] : (tensor<?xi8>) -> tensor<1xi8>
    %flag_float = stablehlo.convert %flag : (tensor<1xi8>) -> tensor<1xf32>
    %n = stablehlo.get_dimension_size %floats, dim = 0 : (tensor<?xf32>) -> tensor<i32>
    %shape = stablehlo.reshape %n : (tensor<i32>) -> tensor<1xi32>
    %flags = stablehlo.dynamic_broadcast_in_dim %flag_float, %shape, dims = [0] : (tensor<1xf32>, tensor<1xi32>) -> tensor<?xf32>
    %updated = stablehlo.add %floats, %flags : tensor<?xf32>
    return %updated, %bytes : tensor<?xf32>, tensor<?xi8>
  }
  func.func private @write(%floats: tensor<?xf32>) -> tensor<?xf32> {
    %updated = stablehlo.negate %floats : tensor<?xf32>
    return %updated : tensor<?xf32>
  }
}
// CHECK-LABEL: llvm.func @conflicting_alias_results
// CHECK: enzymexla.xla_wrapper @conflict
// CHECK-NEXT: enzymexla.xla_wrapper @write
// CHECK-NEXT: enzymexla.xla_wrapper @conflict
// CHECK-NEXT: llvm.return

// -----

// The same address does not imply the same extent. These static views cover
// four bytes and sixteen bytes. Keep both wrappers.
module {
  llvm.func @different_extents(%p: !llvm.ptr) {
    %bytes = "enzymexla.pointer2memref"(%p) : (!llvm.ptr) -> memref<4xi8>
    %floats = "enzymexla.pointer2memref"(%p) : (!llvm.ptr) -> memref<4xf32>
    enzymexla.xla_wrapper @write_bytes (%bytes) : (memref<4xi8>) -> ()
    enzymexla.xla_wrapper @write_floats (%floats) : (memref<4xf32>) -> ()
    llvm.return
  }
  func.func private @write_bytes(%bytes: tensor<4xi8>) -> tensor<4xi8> {
    %updated = stablehlo.negate %bytes : tensor<4xi8>
    return %updated : tensor<4xi8>
  }
  func.func private @write_floats(%floats: tensor<4xf32>) -> tensor<4xf32> {
    %updated = stablehlo.negate %floats : tensor<4xf32>
    return %updated : tensor<4xf32>
  }
}
// CHECK-LABEL: llvm.func @different_extents
// CHECK: enzymexla.xla_wrapper @write_bytes
// CHECK-NEXT: enzymexla.xla_wrapper @write_floats
// CHECK-NEXT: llvm.return
