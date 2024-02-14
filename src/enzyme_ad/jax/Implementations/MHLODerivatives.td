include "Common.td"

def : MLIRDerivative<"mhlo", "AddOp", (Op $x, $y),
                    [
                      (DiffeRet),
                      (DiffeRet),
                    ]
                  >;
