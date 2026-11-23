// RUN: enzymexlamlir-opt --lower-enzymexla-math %s | FileCheck %s --check-prefixes=CHECK,KEEP-F64,KEEP-F32,DECOMP-F16,DECOMP-BF16
// RUN: enzymexlamlir-opt --lower-enzymexla-math="backend=tpu" %s | FileCheck %s --check-prefixes=CHECK,DECOMP-F64,DECOMP-F32,DECOMP-F16,DECOMP-BF16
// RUN: enzymexlamlir-opt --lower-enzymexla-ml="backend=tpu" %s | FileCheck %s --check-prefixes=CHECK,DECOMP-F64,DECOMP-F32,DECOMP-F16,DECOMP-BF16
// RUN: enzymexlamlir-opt --lower-enzymexla-math="backend=tpu decompose-tpu-fma=false" %s | FileCheck %s --check-prefixes=CHECK,DECOMP-F64,KEEP-F32,KEEP-F16,KEEP-BF16
// RUN: enzymexlamlir-opt --lower-enzymexla-math="backend=cuda" %s | FileCheck %s --check-prefixes=CHECK,KEEP-F64,KEEP-F32,KEEP-F16,KEEP-BF16
// RUN: enzymexlamlir-opt --lower-enzymexla-math="backend=cuda_sm80" %s | FileCheck %s --check-prefixes=CHECK,KEEP-F64,KEEP-F32,KEEP-F16,KEEP-BF16
// RUN: enzymexlamlir-opt --lower-enzymexla-math="backend=cuda_sm70" %s | FileCheck %s --check-prefixes=CHECK,KEEP-F64,KEEP-F32,KEEP-F16,DECOMP-BF16
// RUN: enzymexlamlir-opt --lower-enzymexla-math="backend=cuda_sm53" %s | FileCheck %s --check-prefixes=CHECK,KEEP-F64,KEEP-F32,KEEP-F16,DECOMP-BF16
// RUN: enzymexlamlir-opt --lower-enzymexla-math="backend=cuda_sm50" %s | FileCheck %s --check-prefixes=CHECK,KEEP-F64,KEEP-F32,DECOMP-F16,DECOMP-BF16
// RUN: enzymexlamlir-opt --lower-enzymexla-math="backend=rocm" %s | FileCheck %s --check-prefixes=CHECK,KEEP-F64,KEEP-F32,KEEP-F16,DECOMP-BF16
// RUN: enzymexlamlir-opt --lower-enzymexla-math="backend=neuron" %s | FileCheck %s --check-prefixes=CHECK,DECOMP-F64,DECOMP-F32,DECOMP-F16,DECOMP-BF16

// CHECK-LABEL: func.func @fma_f64(
// CHECK-SAME:    %[[A:.+]]: tensor<4xf64>, %[[B:.+]]: tensor<4xf64>, %[[C:.+]]: tensor<4xf64>
func.func @fma_f64(%a: tensor<4xf64>, %b: tensor<4xf64>, %c: tensor<4xf64>) -> tensor<4xf64> {
  // KEEP-F64-NEXT:   %[[R:.+]] = stablehlo.fma %[[A]], %[[B]], %[[C]] : tensor<4xf64>
  // KEEP-F64-NEXT:   return %[[R]] : tensor<4xf64>
  // DECOMP-F64-NEXT: %[[M:.+]] = stablehlo.multiply %[[A]], %[[B]] : tensor<4xf64>
  // DECOMP-F64-NEXT: %[[R:.+]] = stablehlo.add %[[M]], %[[C]] : tensor<4xf64>
  // DECOMP-F64-NEXT: return %[[R]] : tensor<4xf64>
  %0 = stablehlo.fma %a, %b, %c : tensor<4xf64>
  return %0 : tensor<4xf64>
}

// CHECK-LABEL: func.func @fma_f32(
// CHECK-SAME:    %[[A:.+]]: tensor<4xf32>, %[[B:.+]]: tensor<4xf32>, %[[C:.+]]: tensor<4xf32>
func.func @fma_f32(%a: tensor<4xf32>, %b: tensor<4xf32>, %c: tensor<4xf32>) -> tensor<4xf32> {
  // KEEP-F32-NEXT:   %[[R:.+]] = stablehlo.fma %[[A]], %[[B]], %[[C]] : tensor<4xf32>
  // KEEP-F32-NEXT:   return %[[R]] : tensor<4xf32>
  // DECOMP-F32-NEXT: %[[M:.+]] = stablehlo.multiply %[[A]], %[[B]] : tensor<4xf32>
  // DECOMP-F32-NEXT: %[[R:.+]] = stablehlo.add %[[M]], %[[C]] : tensor<4xf32>
  // DECOMP-F32-NEXT: return %[[R]] : tensor<4xf32>
  %0 = stablehlo.fma %a, %b, %c : tensor<4xf32>
  return %0 : tensor<4xf32>
}

// CHECK-LABEL: func.func @fma_f16(
// CHECK-SAME:    %[[A:.+]]: tensor<4xf16>, %[[B:.+]]: tensor<4xf16>, %[[C:.+]]: tensor<4xf16>
func.func @fma_f16(%a: tensor<4xf16>, %b: tensor<4xf16>, %c: tensor<4xf16>) -> tensor<4xf16> {
  // KEEP-F16-NEXT:   %[[R:.+]] = stablehlo.fma %[[A]], %[[B]], %[[C]] : tensor<4xf16>
  // KEEP-F16-NEXT:   return %[[R]] : tensor<4xf16>
  // DECOMP-F16-NEXT: %[[M:.+]] = stablehlo.multiply %[[A]], %[[B]] : tensor<4xf16>
  // DECOMP-F16-NEXT: %[[R:.+]] = stablehlo.add %[[M]], %[[C]] : tensor<4xf16>
  // DECOMP-F16-NEXT: return %[[R]] : tensor<4xf16>
  %0 = stablehlo.fma %a, %b, %c : tensor<4xf16>
  return %0 : tensor<4xf16>
}

// CHECK-LABEL: func.func @fma_bf16(
// CHECK-SAME:    %[[A:.+]]: tensor<4xbf16>, %[[B:.+]]: tensor<4xbf16>, %[[C:.+]]: tensor<4xbf16>
func.func @fma_bf16(%a: tensor<4xbf16>, %b: tensor<4xbf16>, %c: tensor<4xbf16>) -> tensor<4xbf16> {
  // KEEP-BF16-NEXT:   %[[R:.+]] = stablehlo.fma %[[A]], %[[B]], %[[C]] : tensor<4xbf16>
  // KEEP-BF16-NEXT:   return %[[R]] : tensor<4xbf16>
  // DECOMP-BF16-NEXT: %[[M:.+]] = stablehlo.multiply %[[A]], %[[B]] : tensor<4xbf16>
  // DECOMP-BF16-NEXT: %[[R:.+]] = stablehlo.add %[[M]], %[[C]] : tensor<4xbf16>
  // DECOMP-BF16-NEXT: return %[[R]] : tensor<4xbf16>
  %0 = stablehlo.fma %a, %b, %c : tensor<4xbf16>
  return %0 : tensor<4xbf16>
}

// CHECK-LABEL: func.func @fma_f8(
// CHECK-SAME:    %[[A:.+]]: tensor<4xf8E4M3FN>, %[[B:.+]]: tensor<4xf8E4M3FN>, %[[C:.+]]: tensor<4xf8E4M3FN>
func.func @fma_f8(%a: tensor<4xf8E4M3FN>, %b: tensor<4xf8E4M3FN>, %c: tensor<4xf8E4M3FN>) -> tensor<4xf8E4M3FN> {
  // CHECK-NEXT: %[[M:.+]] = stablehlo.multiply %[[A]], %[[B]] : tensor<4xf8E4M3FN>
  // CHECK-NEXT: %[[R:.+]] = stablehlo.add %[[M]], %[[C]] : tensor<4xf8E4M3FN>
  // CHECK-NEXT: return %[[R]] : tensor<4xf8E4M3FN>
  %0 = stablehlo.fma %a, %b, %c : tensor<4xf8E4M3FN>
  return %0 : tensor<4xf8E4M3FN>
}

// CHECK-LABEL: func.func @fma_dynamic(
// CHECK-SAME:    %[[A:.+]]: tensor<?xf64>, %[[B:.+]]: tensor<?xf64>, %[[C:.+]]: tensor<?xf64>
func.func @fma_dynamic(%a: tensor<?xf64>, %b: tensor<?xf64>, %c: tensor<?xf64>) -> tensor<?xf64> {
  // KEEP-F64-NEXT:   %[[R:.+]] = stablehlo.fma %[[A]], %[[B]], %[[C]] : tensor<?xf64>
  // KEEP-F64-NEXT:   return %[[R]] : tensor<?xf64>
  // DECOMP-F64-NEXT: %[[M:.+]] = stablehlo.multiply %[[A]], %[[B]] : tensor<?xf64>
  // DECOMP-F64-NEXT: %[[R:.+]] = stablehlo.add %[[M]], %[[C]] : tensor<?xf64>
  // DECOMP-F64-NEXT: return %[[R]] : tensor<?xf64>
  %0 = stablehlo.fma %a, %b, %c : tensor<?xf64>
  return %0 : tensor<?xf64>
}
