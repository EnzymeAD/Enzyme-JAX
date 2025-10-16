module {
    perfify.cost "arith.mul" 3
//   Perfify.assumptions { // operation in the dialect

//      # maybe the individual cost should be an interface?
//      Perfify.cost “arith.mul” 3 // op
//      Perfify.cost “func.return” 0
//      Perfify.cost “scf.yield” 0
//      Perfify.cost “scf.if” “recursive”


//      Perfify.conditions @foo() verify=true pre { 
//         %b0 = perfify.arg 0 // op
//         %c0 = arith.constant 0 
//         %cmp = arith.cmpi eq %c0, %b0
//         Perfify.assume %cmp
//      } post {
//         %cost = perfify.fn_cost : perfify.cost
//         %c9 = perfify.constant_cost 9 : perfify.cost // then our cost is 9

//         %cmp = arith.cmpi eq %cost, %c9
//         Perfify.assume %cmp
//      }


//      Perfify.conditions @foo() verify=true pre { //op
//         %b0 = perfify.arg 0 // if the argument to func is 0
//         %c0 = arith.constant  // constant 0
//         %cmp = arith.cmpi ne %c0, %b0
//         Perfify.assume %cmp
//      } post {
//         %cost = perfify.fn_cost : perfify.cost
//         %c9 = perfify.constant_cost 9 : perfify.cost // then our cost is 9

//         %cmp = arith.cmpi eq %cost, %c9
//         Perfify.assume %cmp
//      } 
//    }

//   func.func @foo(%b0, %a0) {
//      %res = scf.if b0 == 0 { // take this branch, and we assume no extra overhead for taking this if statement
//        %a1 = arith.mul %a0, a0 // take 3 ops in our cost model
//        %a2 = arith.mul %a1, a1 // 3 ops
//        %a3 = arith.mul %a2, a2
//        Scf.yield %a3 // total of 9 ops by this point
//      } else {
//         scf.yield a0
// 	}
//      func.return %res 
//   }
}