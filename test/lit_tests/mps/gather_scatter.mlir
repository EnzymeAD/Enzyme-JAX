// RUN: enzymexlamlir-opt --split-input-file %s | enzymexlamlir-opt --split-input-file | FileCheck %s

// CHECK-LABEL: func.func @gather(
func.func @gather(%x: tensor<4x3xf32>, %i: tensor<2xi32>) {
  // CHECK: %{{.+}} = mps.gather %{{.+}}, %{{.+}} <axis = 0, batch_dimensions = 0> : (tensor<4x3xf32>, tensor<2xi32>) -> tensor<2x3xf32>
  %0 = mps.gather %x, %i <axis = 0, batch_dimensions = 0> : (tensor<4x3xf32>, tensor<2xi32>) -> tensor<2x3xf32>
  return
}

// -----

// CHECK-LABEL: func.func @gather_nd(
func.func @gather_nd(%x: tensor<4x3xf32>, %ind: tensor<2x1xi32>) {
  // CHECK: %{{.+}} = mps.gather_nd %{{.+}}, %{{.+}} <batch_dimensions = 0> : (tensor<4x3xf32>, tensor<2x1xi32>) -> tensor<2x3xf32>
  %0 = mps.gather_nd %x, %ind <batch_dimensions = 0> : (tensor<4x3xf32>, tensor<2x1xi32>) -> tensor<2x3xf32>
  return
}

// -----

// CHECK-LABEL: func.func @gather_along_axis(
func.func @gather_along_axis(%x: tensor<4x3xf32>, %ia: tensor<2x3xi32>) {
  // CHECK: %{{.+}} = mps.gather_along_axis %{{.+}}, %{{.+}} <axis = 0> : (tensor<4x3xf32>, tensor<2x3xi32>) -> tensor<2x3xf32>
  %0 = mps.gather_along_axis %x, %ia <axis = 0> : (tensor<4x3xf32>, tensor<2x3xi32>) -> tensor<2x3xf32>
  return
}

// -----

// CHECK-LABEL: func.func @scatter(
func.func @scatter(%u: tensor<2x3xf32>, %i: tensor<2xi32>) {
  // CHECK: %{{.+}} = mps.scatter %{{.+}}, %{{.+}} <axis = 0, mode = add> : (tensor<2x3xf32>, tensor<2xi32>) -> tensor<4x3xf32>
  %0 = mps.scatter %u, %i <axis = 0, mode = add> : (tensor<2x3xf32>, tensor<2xi32>) -> tensor<4x3xf32>
  return
}

// -----

// CHECK-LABEL: func.func @scatter_with_data(
func.func @scatter_with_data(%data: tensor<4x3xf32>, %u: tensor<2x3xf32>, %i: tensor<2xi32>) {
  // CHECK: %{{.+}} = mps.scatter_with_data %{{.+}}, %{{.+}}, %{{.+}} <axis = 0, mode = set> : (tensor<4x3xf32>, tensor<2x3xf32>, tensor<2xi32>) -> tensor<4x3xf32>
  %0 = mps.scatter_with_data %data, %u, %i <axis = 0, mode = set> : (tensor<4x3xf32>, tensor<2x3xf32>, tensor<2xi32>) -> tensor<4x3xf32>
  return
}

// -----

// CHECK-LABEL: func.func @scatter_nd(
func.func @scatter_nd(%u: tensor<2x3xf32>, %ind: tensor<2x1xi32>) {
  // CHECK: %{{.+}} = mps.scatter_nd %{{.+}}, %{{.+}} <batch_dimensions = 0, mode = sub> : (tensor<2x3xf32>, tensor<2x1xi32>) -> tensor<4x3xf32>
  %0 = mps.scatter_nd %u, %ind <batch_dimensions = 0, mode = sub> : (tensor<2x3xf32>, tensor<2x1xi32>) -> tensor<4x3xf32>
  return
}

// -----

// CHECK-LABEL: func.func @scatter_nd_with_data(
func.func @scatter_nd_with_data(%data: tensor<4x3xf32>, %u: tensor<2x3xf32>, %ind: tensor<2x1xi32>) {
  // CHECK: %{{.+}} = mps.scatter_nd_with_data %{{.+}}, %{{.+}}, %{{.+}} <batch_dimensions = 0, mode = mul> : (tensor<4x3xf32>, tensor<2x3xf32>, tensor<2x1xi32>) -> tensor<4x3xf32>
  %0 = mps.scatter_nd_with_data %data, %u, %ind <batch_dimensions = 0, mode = mul> : (tensor<4x3xf32>, tensor<2x3xf32>, tensor<2x1xi32>) -> tensor<4x3xf32>
  return
}

// -----

// CHECK-LABEL: func.func @scatter_along_axis(
func.func @scatter_along_axis(%u: tensor<2x3xf32>, %ia: tensor<2x3xi32>) {
  // CHECK: %{{.+}} = mps.scatter_along_axis %{{.+}}, %{{.+}} <axis = 0, mode = div> : (tensor<2x3xf32>, tensor<2x3xi32>) -> tensor<4x3xf32>
  %0 = mps.scatter_along_axis %u, %ia <axis = 0, mode = div> : (tensor<2x3xf32>, tensor<2x3xi32>) -> tensor<4x3xf32>
  return
}

// -----

// CHECK-LABEL: func.func @scatter_along_axis_with_data(
func.func @scatter_along_axis_with_data(%data: tensor<4x3xf32>, %u: tensor<2x3xf32>, %ia: tensor<2x3xi32>) {
  // CHECK: %{{.+}} = mps.scatter_along_axis_with_data %{{.+}}, %{{.+}}, %{{.+}} <axis = 0, mode = max> : (tensor<4x3xf32>, tensor<2x3xf32>, tensor<2x3xi32>) -> tensor<4x3xf32>
  %0 = mps.scatter_along_axis_with_data %data, %u, %ia <axis = 0, mode = max> : (tensor<4x3xf32>, tensor<2x3xf32>, tensor<2x3xi32>) -> tensor<4x3xf32>
  return
}
