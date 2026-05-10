// RUN: enzymexlamlir-opt %s --arith-raise | stablehlo-translate - --interpret

module {
  func.func @main() {
    %c1 = arith.constant dense<1> : tensor<1xi64>
    %c_minus_1 = arith.constant dense<-1> : tensor<1xi64>

    // Unsigned comparison: -1 > 1 (as bits, 0xFF...FF > 0x00...01)
    %res1 = arith.cmpi ugt, %c_minus_1, %c1 : tensor<1xi64>
    
    // Signed comparison: -1 < 1
    %res2 = arith.cmpi sgt, %c_minus_1, %c1 : tensor<1xi64>

    %true = arith.constant dense<true> : tensor<1xi1>
    %false = arith.constant dense<false> : tensor<1xi1>

    %res1_f = stablehlo.convert %res1 : (tensor<1xi1>) -> tensor<1xf64>
    %true_f = stablehlo.convert %true : (tensor<1xi1>) -> tensor<1xf64>
    "check.expect_close"(%res1_f, %true_f) {max_ulp_difference = 0 : ui64} : (tensor<1xf64>, tensor<1xf64>) -> ()

    %res2_f = stablehlo.convert %res2 : (tensor<1xi1>) -> tensor<1xf64>
    %false_f = stablehlo.convert %false : (tensor<1xi1>) -> tensor<1xf64>
    "check.expect_close"(%res2_f, %false_f) {max_ulp_difference = 0 : ui64} : (tensor<1xf64>, tensor<1xf64>) -> ()

    return
  }
}
