// RUN: enzymexlamlir-opt --pass-pipeline="any(enzyme-hlo-generate-td{patterns=concat_insert_dim_elementwise},transform-interpreter,enzyme-hlo-remove-transform)"

module {
  func.func @mapped_sub(%arg0: tensor<3x5x10xf32>, %arg1: tensor<3x5x10xf32>) -> (tensor<5x3x10xf32>, tensor<3x5x10xf32>, tensor<3x5x10xf32>) {
    %0 = stablehlo.transpose %arg0, dims = [2, 1, 0] : (tensor<3x5x10xf32>) -> tensor<10x5x3xf32>
    %1 = stablehlo.transpose %arg1, dims = [2, 1, 0] : (tensor<3x5x10xf32>) -> tensor<10x5x3xf32>
    %2 = stablehlo.slice %0 [0:10, 0:1, 0:3] : (tensor<10x5x3xf32>) -> tensor<10x1x3xf32>
    %3 = stablehlo.transpose %2, dims = [2, 1, 0] : (tensor<10x1x3xf32>) -> tensor<3x1x10xf32>
    %4 = stablehlo.reshape %3 : (tensor<3x1x10xf32>) -> tensor<3x10xf32>
    %5 = stablehlo.transpose %4, dims = [1, 0] : (tensor<3x10xf32>) -> tensor<10x3xf32>
    %6 = stablehlo.convert %5 : tensor<10x3xf32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [0, 1] : (tensor<10x3xf32>) -> tensor<10x3xf32>
    %8 = stablehlo.broadcast_in_dim %7, dims = [0, 1] : (tensor<10x3xf32>) -> tensor<10x3xf32>
    %9 = stablehlo.slice %1 [0:10, 0:1, 0:3] : (tensor<10x5x3xf32>) -> tensor<10x1x3xf32>
    %10 = stablehlo.transpose %9, dims = [2, 1, 0] : (tensor<10x1x3xf32>) -> tensor<3x1x10xf32>
    %11 = stablehlo.reshape %10 : (tensor<3x1x10xf32>) -> tensor<3x10xf32>
    %12 = stablehlo.transpose %11, dims = [1, 0] : (tensor<3x10xf32>) -> tensor<10x3xf32>
    %13 = stablehlo.convert %12 : tensor<10x3xf32>
    %14 = stablehlo.broadcast_in_dim %13, dims = [0, 1] : (tensor<10x3xf32>) -> tensor<10x3xf32>
    %15 = stablehlo.broadcast_in_dim %14, dims = [0, 1] : (tensor<10x3xf32>) -> tensor<10x3xf32>
    %16 = stablehlo.subtract %8, %15 : tensor<10x3xf32>
    %17 = stablehlo.slice %0 [0:10, 1:2, 0:3] : (tensor<10x5x3xf32>) -> tensor<10x1x3xf32>
    %18 = stablehlo.transpose %17, dims = [2, 1, 0] : (tensor<10x1x3xf32>) -> tensor<3x1x10xf32>
    %19 = stablehlo.reshape %18 : (tensor<3x1x10xf32>) -> tensor<3x10xf32>
    %20 = stablehlo.transpose %19, dims = [1, 0] : (tensor<3x10xf32>) -> tensor<10x3xf32>
    %21 = stablehlo.convert %20 : tensor<10x3xf32>
    %22 = stablehlo.broadcast_in_dim %21, dims = [0, 1] : (tensor<10x3xf32>) -> tensor<10x3xf32>
    %23 = stablehlo.broadcast_in_dim %22, dims = [0, 1] : (tensor<10x3xf32>) -> tensor<10x3xf32>
    %24 = stablehlo.slice %1 [0:10, 1:2, 0:3] : (tensor<10x5x3xf32>) -> tensor<10x1x3xf32>
    %25 = stablehlo.transpose %24, dims = [2, 1, 0] : (tensor<10x1x3xf32>) -> tensor<3x1x10xf32>
    %26 = stablehlo.reshape %25 : (tensor<3x1x10xf32>) -> tensor<3x10xf32>
    %27 = stablehlo.transpose %26, dims = [1, 0] : (tensor<3x10xf32>) -> tensor<10x3xf32>
    %28 = stablehlo.convert %27 : tensor<10x3xf32>
    %29 = stablehlo.broadcast_in_dim %28, dims = [0, 1] : (tensor<10x3xf32>) -> tensor<10x3xf32>
    %30 = stablehlo.broadcast_in_dim %29, dims = [0, 1] : (tensor<10x3xf32>) -> tensor<10x3xf32>
    %31 = stablehlo.subtract %23, %30 : tensor<10x3xf32>
    %32 = stablehlo.slice %0 [0:10, 2:3, 0:3] : (tensor<10x5x3xf32>) -> tensor<10x1x3xf32>
    %33 = stablehlo.transpose %32, dims = [2, 1, 0] : (tensor<10x1x3xf32>) -> tensor<3x1x10xf32>
    %34 = stablehlo.reshape %33 : (tensor<3x1x10xf32>) -> tensor<3x10xf32>
    %35 = stablehlo.transpose %34, dims = [1, 0] : (tensor<3x10xf32>) -> tensor<10x3xf32>
    %36 = stablehlo.convert %35 : tensor<10x3xf32>
    %37 = stablehlo.broadcast_in_dim %36, dims = [0, 1] : (tensor<10x3xf32>) -> tensor<10x3xf32>
    %38 = stablehlo.broadcast_in_dim %37, dims = [0, 1] : (tensor<10x3xf32>) -> tensor<10x3xf32>
    %39 = stablehlo.slice %1 [0:10, 2:3, 0:3] : (tensor<10x5x3xf32>) -> tensor<10x1x3xf32>
    %40 = stablehlo.transpose %39, dims = [2, 1, 0] : (tensor<10x1x3xf32>) -> tensor<3x1x10xf32>
    %41 = stablehlo.reshape %40 : (tensor<3x1x10xf32>) -> tensor<3x10xf32>
    %42 = stablehlo.transpose %41, dims = [1, 0] : (tensor<3x10xf32>) -> tensor<10x3xf32>
    %43 = stablehlo.convert %42 : tensor<10x3xf32>
    %44 = stablehlo.broadcast_in_dim %43, dims = [0, 1] : (tensor<10x3xf32>) -> tensor<10x3xf32>
    %45 = stablehlo.broadcast_in_dim %44, dims = [0, 1] : (tensor<10x3xf32>) -> tensor<10x3xf32>
    %46 = stablehlo.subtract %38, %45 : tensor<10x3xf32>
    %47 = stablehlo.slice %0 [0:10, 3:4, 0:3] : (tensor<10x5x3xf32>) -> tensor<10x1x3xf32>
    %48 = stablehlo.transpose %47, dims = [2, 1, 0] : (tensor<10x1x3xf32>) -> tensor<3x1x10xf32>
    %49 = stablehlo.reshape %48 : (tensor<3x1x10xf32>) -> tensor<3x10xf32>
    %50 = stablehlo.transpose %49, dims = [1, 0] : (tensor<3x10xf32>) -> tensor<10x3xf32>
    %51 = stablehlo.convert %50 : tensor<10x3xf32>
    %52 = stablehlo.broadcast_in_dim %51, dims = [0, 1] : (tensor<10x3xf32>) -> tensor<10x3xf32>
    %53 = stablehlo.broadcast_in_dim %52, dims = [0, 1] : (tensor<10x3xf32>) -> tensor<10x3xf32>
    %54 = stablehlo.slice %1 [0:10, 3:4, 0:3] : (tensor<10x5x3xf32>) -> tensor<10x1x3xf32>
    %55 = stablehlo.transpose %54, dims = [2, 1, 0] : (tensor<10x1x3xf32>) -> tensor<3x1x10xf32>
    %56 = stablehlo.reshape %55 : (tensor<3x1x10xf32>) -> tensor<3x10xf32>
    %57 = stablehlo.transpose %56, dims = [1, 0] : (tensor<3x10xf32>) -> tensor<10x3xf32>
    %58 = stablehlo.convert %57 : tensor<10x3xf32>
    %59 = stablehlo.broadcast_in_dim %58, dims = [0, 1] : (tensor<10x3xf32>) -> tensor<10x3xf32>
    %60 = stablehlo.broadcast_in_dim %59, dims = [0, 1] : (tensor<10x3xf32>) -> tensor<10x3xf32>
    %61 = stablehlo.subtract %53, %60 : tensor<10x3xf32>
    %62 = stablehlo.slice %0 [0:10, 4:5, 0:3] : (tensor<10x5x3xf32>) -> tensor<10x1x3xf32>
    %63 = stablehlo.transpose %62, dims = [2, 1, 0] : (tensor<10x1x3xf32>) -> tensor<3x1x10xf32>
    %64 = stablehlo.reshape %63 : (tensor<3x1x10xf32>) -> tensor<3x10xf32>
    %65 = stablehlo.transpose %64, dims = [1, 0] : (tensor<3x10xf32>) -> tensor<10x3xf32>
    %66 = stablehlo.convert %65 : tensor<10x3xf32>
    %67 = stablehlo.broadcast_in_dim %66, dims = [0, 1] : (tensor<10x3xf32>) -> tensor<10x3xf32>
    %68 = stablehlo.broadcast_in_dim %67, dims = [0, 1] : (tensor<10x3xf32>) -> tensor<10x3xf32>
    %69 = stablehlo.slice %1 [0:10, 4:5, 0:3] : (tensor<10x5x3xf32>) -> tensor<10x1x3xf32>
    %70 = stablehlo.transpose %69, dims = [2, 1, 0] : (tensor<10x1x3xf32>) -> tensor<3x1x10xf32>
    %71 = stablehlo.reshape %70 : (tensor<3x1x10xf32>) -> tensor<3x10xf32>
    %72 = stablehlo.transpose %71, dims = [1, 0] : (tensor<3x10xf32>) -> tensor<10x3xf32>
    %73 = stablehlo.convert %72 : tensor<10x3xf32>
    %74 = stablehlo.broadcast_in_dim %73, dims = [0, 1] : (tensor<10x3xf32>) -> tensor<10x3xf32>
    %75 = stablehlo.broadcast_in_dim %74, dims = [0, 1] : (tensor<10x3xf32>) -> tensor<10x3xf32>
    %76 = stablehlo.subtract %68, %75 : tensor<10x3xf32>
    %77 = stablehlo.broadcast_in_dim %16, dims = [2, 1] : (tensor<10x3xf32>) -> tensor<1x3x10xf32>
    %78 = stablehlo.broadcast_in_dim %31, dims = [2, 1] : (tensor<10x3xf32>) -> tensor<1x3x10xf32>
    %79 = stablehlo.broadcast_in_dim %46, dims = [2, 1] : (tensor<10x3xf32>) -> tensor<1x3x10xf32>
    %80 = stablehlo.broadcast_in_dim %61, dims = [2, 1] : (tensor<10x3xf32>) -> tensor<1x3x10xf32>
    %81 = stablehlo.broadcast_in_dim %76, dims = [2, 1] : (tensor<10x3xf32>) -> tensor<1x3x10xf32>
    %82 = stablehlo.concatenate %77, %78, %79, %80, %81, dim = 0 : (tensor<1x3x10xf32>, tensor<1x3x10xf32>, tensor<1x3x10xf32>, tensor<1x3x10xf32>, tensor<1x3x10xf32>) -> tensor<5x3x10xf32>
    // CHECK: stablehlo.concatenate
    %83 = stablehlo.transpose %0, dims = [2, 1, 0] : (tensor<10x5x3xf32>) -> tensor<3x5x10xf32>
    %84 = stablehlo.transpose %1, dims = [2, 1, 0] : (tensor<10x5x3xf32>) -> tensor<3x5x10xf32>
    return %82, %83, %84 : tensor<5x3x10xf32>, tensor<3x5x10xf32>, tensor<3x5x10xf32>
  }
}
