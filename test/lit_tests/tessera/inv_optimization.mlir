// RUN: enzymexlamlir-opt %s -tessera-pdl | FileCheck %s

module {
    module @patterns {
        pdl.pattern @simplify_inv : benefit(1) {
            %resultType = pdl.type
            %x = pdl.operand
            %inv_str = pdl.attribute = @inv
            %inner_op = pdl.operation "tessera.call"(%x : !pdl.value) {"callee" = %inv_str} -> (%resultType : !pdl.type)
            %inner_result = pdl.result 0 of %inner_op
            %outer_op = pdl.operation "tessera.call"(%inner_result : !pdl.value) {"callee" = %inv_str} -> (%resultType : !pdl.type)
            
            pdl.rewrite %outer_op {
                pdl.replace %outer_op with (%x : !pdl.value)
            }
        }
    }

    module @ir {
        tessera.define @inv(%arg0 : f32) -> f32 {
            tessera.return %arg0 : f32
        }

        // CHECK-LABEL: func.func @main
        func.func @main(%x : f32) -> f32 {
            // CHECK: return %arg0
            %0 = tessera.call @inv(%x) : (f32) -> f32
            %1 = tessera.call @inv(%0) : (f32) -> f32
            return %1 : f32
        }
    }
}
