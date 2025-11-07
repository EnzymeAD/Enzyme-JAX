module {
    perfify.assumptions { // operation in the dialect
     perfify.cost "arith.mul" 3 // op
     perfify.cost "func.return" 0
     perfify.cost "scf.yield" 0
    func.func @foo() -> () { return }

     perfify.conditions @foo true pre { 
        %b0 = perfify.arg 0 // op
        %c0 = arith.constant 0 
        %cmp = arith.cmpi eq, %c0, %b0 : i64
        perfify.assume %cmp
     } post {
        // %cost = perfify.fn_cost : perfify.cost
        // %c9 = perfify.constant_cost 9 : perfify.cost // then our cost is 9

        // %cmp = arith.cmpi eq, %cost, %c9
        // perfify.assume %cmp
     }
    
    }
}