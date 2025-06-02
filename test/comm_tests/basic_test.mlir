// 
// Simple test that contains the IR constructs. Not really meant
// to test optimizations, just that the dialect parses and verifies
// correctly.

module {

  func.func @main(%a : tensor<2x2xf32>) -> tensor<i32> {

    comm.split {
      %msg = comm.token tensor<i32>
      comm.branch [1, 4] {
        %start = stablehlo.constant dense<0> : tensor<i32>
        comm.send %msg %start : tensor<i32>
      }
      comm.branch [2] {
        %start = comm.recv %msg : tensor<i32>
        %lim = stablehlo.constant dense<5> : tensor<i32>
        %step = stablehlo.constant dense<1> : tensor<i32>
        %w:2 = stablehlo.while(%iterArg = %a, %iterArg_0 = %start) : tensor<2x2xf32>, tensor<i32>
        cond {
          %9737 = stablehlo.compare  LT, %iterArg_0, %lim,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
          stablehlo.return %9737 : tensor<i1>
        } do {
          %next = stablehlo.add %iterArg, %iterArg : tensor<2x2xf32>
          %ni = stablehlo.add %iterArg_0, %step : tensor<i32>
          stablehlo.return %next, %ni : tensor<2x2xf32>, tensor<i32>
        }
      }
    }
    %1 = stablehlo.constant dense<0> : tensor<i32>
    return %1 : tensor<i32>
  }
}