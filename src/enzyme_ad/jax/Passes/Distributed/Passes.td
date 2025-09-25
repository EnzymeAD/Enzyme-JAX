#ifndef ENZYMEXLA_DISTRIBUTED_PASSES
#define ENZYMEXLA_DISTRIBUTED_PASSES

include "mlir/Pass/PassBase.td"

def EliminateConstantCommunicationPass : Pass<"eliminate-constant-communication"> {
    let summary = "Replaces communicated constants with local constants";
    let description = [{
      This pass identifies send instructions with constant operands and replaces
      the corresponding receive instructions with local constants.
    }];
    let dependentDialects = ["enzyme::distributed::DistributedDialect"];
}

#endif // ENZYMEXLA_DISTRIBUTED_PASSES