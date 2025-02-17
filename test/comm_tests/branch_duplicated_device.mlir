module {

  func.func @main(%a : tensor<2x2xf32>) -> tensor<2x2xf32> {

    comm.split {
      %msg = comm.simple_msg tensor<2x2xf32>
      comm.branch [1, 4] {
        ^start:
        comm.split { 
          comm.branch [1] {
            comm.join
          }
          comm.branch [4] {
            comm.join
          }
        }
        comm.join
      }
      comm.branch [2, 4] {
        ^start:
        comm.join
      }
    }
    
    
    %start = stablehlo.constant dense<0> : tensor<i32>
    
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
    return %w#0 : tensor<2x2xf32>
  }
}