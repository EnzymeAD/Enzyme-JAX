include "src/enzyme_ad/jax/Implementations/Common.td"

class HLODerivative<string opName_, dag patternToMatch, list<dag> resultOps, dag forwardOps=(ForwardFromSummedReverse)> : MLIRDerivative<"stablehlo", opName_, patternToMatch, resultOps, forwardOps>;

class HLOInst<string m, string postopt=""> : Inst<m, "stablehlo", postopt>;

class HLOMemoryIdentityOp<string opName_, list<int> ptrargs_, list<int> storedargs_ = [], dag patternToMatch=(Unimplemented), list<dag> reverse_ = []>  : MemoryIdentityOp<"stablehlo", opName_, ptrargs_, storedargs_, patternToMatch, reverse_>;

class HLOReadOnlyIdentityOp<string opName_, list<int> ptrargs_ = [0], dag patternToMatch=(Unimplemented), list<dag> reverse_ = []> : ReadOnlyIdentityOp<"stablehlo", opName_, ptrargs_, patternToMatch, reverse_>;

class HLOControlFlowOp<string opName_, string impl_> : ControlFlowOp<"stablehlo", opName_, impl_>;

class HLOConstantFP<string m> : ConstantFP<m, "stablehlo", "ConstantOp", "mlir::ElementsAttr">;

class HLORegionTerminatorOp<string m> : RegionTerminatorOp<"stablehlo", m>;

class HLOInactiveOp<string m> : InactiveOp<"stablehlo", m>;

include "HLODerivatives.td"

def : HLODerivative<"EinsumOp", (Op $lhs, $rhs),
                    [
                        // TODO add support for complex numbers by conjugating `$rhs` and `$lhs` (only if complex tensors)
                        (Einsum (TypeOf $lhs), (DiffeRet), $rhs, (GradEinsumConfigLhs)),
                        (Einsum (TypeOf $rhs), (DiffeRet), $lhs, (GradEinsumConfigRhs)),
                    ],
                    (Add (SelectIfActive $lhs, (Einsum (ResultTypes), (Shadow $lhs), $rhs, (EinsumConfig)), (HLOConstantFP<"0">)), (SelectIfActive $rhs, (Einsum (ResultTypes), $lhs, (Shadow $rhs), (EinsumConfig)), (HLOConstantFP<"0">)))
                  >;

def : HLODerivative<"UnaryEinsumOp", (Op $x),
                    [
                        // TODO maybe replace `UnaryEinsum` for `Reshape` + `BroadcastInDim`
                        (UnaryEinsum (TypeOf $x), (DiffeRet), (GradUnaryEinsumConfig)),
                    ],
                    (UnaryEinsum (ResultTypes), (Shadow $x), (EinsumConfig))
                  >;

def : ControlFlowOp<"stablehlo", "CaseOp", [{
  Operation *createWithShadows(Operation *op, OpBuilder &builder,
                               MGradientUtils *gutils, Operation *original,
                               ValueRange remappedOperands,
                               TypeRange rettys) const {
    return builder.create<stablehlo::CaseOp>(original->getLoc(), rettys,
                                        remappedOperands, original->getAttrs(), cast<stablehlo::CaseOp>(op).getBranches().size());
  }
}]>;


// ConvolutionOp

def ConvWindowStrides : GlobalExpr</*needsprimal*/0, /*needsshadow*/0, [{
  op.getWindowStridesAttr();
}]>;

def ConvPadding : GlobalExpr</*needsprimal*/0, /*needsshadow*/0, [{
  op.getPaddingAttr();
}]>;

def ConvLhsDilation : GlobalExpr</*needsprimal*/0, /*needsshadow*/0, [{
  op.getLhsDilationAttr();
}]>;

def ConvRhsDilation : GlobalExpr</*needsprimal*/0, /*needsshadow*/0, [{
  op.getRhsDilationAttr();
}]>;

def ConvWindowReversal : GlobalExpr</*needsprimal*/0, /*needsshadow*/0, [{
  op.getWindowReversalAttr();
}]>;

def ConvDimensionNumbers : GlobalExpr</*needsprimal*/0, /*needsshadow*/0, [{
  op.getDimensionNumbersAttr();
}]>;

def ConvFeatureGroupCount : GlobalExpr</*needsprimal*/0, /*needsshadow*/0, [{
  op.getFeatureGroupCountAttr();
}]>;

def ConvBatchGroupCount : GlobalExpr</*needsprimal*/0, /*needsshadow*/0, [{
  op.getBatchGroupCountAttr();
}]>;

// GradData

def GradDataConvWindowStrides : GlobalExpr</*needsprimal*/0, /*needsshadow*/0, [{
  int64_t N = op.getType().getShape().size() - 2;
  llvm::SmallVector<int64_t> windowStrides(N, 1);
  auto windowStridesAttr = builder.getDenseI64ArrayAttr(windowStrides);
  windowStridesAttr;
}]>;

def GradDataConvPadding : GlobalExpr</*needsprimal*/0, /*needsshadow*/0, [{
  auto dimensionNumbers = op.getDimensionNumbers();
  int64_t N = dimensionNumbers.getKernelSpatialDimensions().size();

  SmallVector<int64_t> newPaddingValues(2 * N, 0);

  auto initialPadding = op.getPadding();
  if (initialPadding.has_value()) {
    newPaddingValues.assign(initialPadding.value().value_begin<int64_t>(),
                            initialPadding.value().value_end<int64_t>());
  }

  for (int i = 0; i < N; ++i) {
    auto weightDim = dimensionNumbers.getKernelSpatialDimensions()[i];
    auto dataDim = dimensionNumbers.getInputSpatialDimensions()[i];
    auto outputDim = dimensionNumbers.getOutputSpatialDimensions()[i];

    auto padBefore = newPaddingValues[2 * i];
    auto padAfter = newPaddingValues[2 * i + 1];

    auto rhsShape = op.getRhs().getType().getShape()[weightDim];
    auto lhsShape = op.getLhs().getType().getShape()[dataDim];
    auto outShape = op.getType().getShape()[outputDim];

    auto newPadBefore = rhsShape - padBefore - 1;
    newPaddingValues[2 * i] = newPadBefore;
    newPaddingValues[2 * i + 1] = lhsShape + rhsShape - 1 - outShape - newPadBefore;
  }

  auto paddingType = mlir::RankedTensorType::get({N, 2}, builder.getI64Type());
  auto newPaddingAttr = mlir::DenseIntElementsAttr::get(paddingType, newPaddingValues);

  newPaddingAttr;
}]>;

def GradDataConvLhsDilation : GlobalExpr</*needsprimal*/0, /*needsshadow*/0, [{
  op.getWindowStridesAttr();
}]>;

def GradDataConvRhsDilation : GlobalExpr</*needsprimal*/0, /*needsshadow*/0, [{
  op.getRhsDilationAttr();
}]>;

def GradDataConvWindowReversal : GlobalExpr</*needsprimal*/0, /*needsshadow*/0, [{
  auto dimensionNumbers = op.getDimensionNumbers();
  int64_t N = dimensionNumbers.getKernelSpatialDimensions().size();
  llvm::SmallVector<bool> newWindowReversalValues(N, true);
  auto windowReversal = op.getWindowReversal();

  if (windowReversal.has_value()) {
    for (int i = 0; i < N; ++i) {
      newWindowReversalValues[i] = !windowReversal.value()[i];
    }
  }

  auto newWindowReversalAttr = builder.getDenseBoolArrayAttr(newWindowReversalValues);
  newWindowReversalAttr;
}]>;

def GradDataConvDimensionNumbers : GlobalExpr</*needsprimal*/0, /*needsshadow*/0, [{
  auto dimensionNumbers = op.getDimensionNumbers();
  auto newDimensionNumbers = mlir::stablehlo::ConvDimensionNumbersAttr::get(
    op.getContext(),
    dimensionNumbers.getOutputBatchDimension(),
    dimensionNumbers.getOutputFeatureDimension(),
    dimensionNumbers.getOutputSpatialDimensions(),
    dimensionNumbers.getKernelOutputFeatureDimension(),
    dimensionNumbers.getKernelInputFeatureDimension(),
    dimensionNumbers.getKernelSpatialDimensions(),
    dimensionNumbers.getInputBatchDimension(),
    dimensionNumbers.getInputFeatureDimension(),
    dimensionNumbers.getInputSpatialDimensions()
  );
  newDimensionNumbers;
}]>;

def GradDataConvFeatureGroupCount : GlobalExpr</*needsprimal*/0, /*needsshadow*/0, [{
  op.getFeatureGroupCountAttr();
}]>;

def GradDataConvBatchGroupCount : GlobalExpr</*needsprimal*/0, /*needsshadow*/0, [{
  op.getBatchGroupCountAttr();
}]>;

// GradFilter

def GradFilterConvWindowStrides : GlobalExpr</*needsprimal*/0, /*needsshadow*/0, [{
  op.getRhsDilationAttr();
}]>;

def GradFilterConvPadding : GlobalExpr</*needsprimal*/0, /*needsshadow*/0, [{
  auto dimensionNumbers = op.getDimensionNumbers();
  int64_t N = dimensionNumbers.getKernelSpatialDimensions().size();

  SmallVector<int64_t> newPaddingValues(2 * N, 0);

  auto initialPadding = op.getPadding();
  if (initialPadding.has_value()) {
    newPaddingValues.assign(initialPadding.value().value_begin<int64_t>(),
                            initialPadding.value().value_end<int64_t>());
  }

  for (int i = 0; i < N; ++i) {
    auto weightDim = dimensionNumbers.getKernelSpatialDimensions()[i];
    auto dataDim = dimensionNumbers.getInputSpatialDimensions()[i];
    auto outputDim = dimensionNumbers.getOutputSpatialDimensions()[i];

    auto padBefore = newPaddingValues[2 * i];
    auto padAfter = newPaddingValues[2 * i + 1];

    auto rhsShape = op.getRhs().getType().getShape()[weightDim];
    auto lhsShape = op.getLhs().getType().getShape()[dataDim];
    auto outShape = op.getType().getShape()[outputDim];

    newPaddingValues[2 * i] = padBefore;
    newPaddingValues[2 * i + 1] = outShape - lhsShape + rhsShape - padBefore - 1;
  }

  auto paddingType = mlir::RankedTensorType::get({N, 2}, builder.getI64Type());
  auto newPaddingAttr = mlir::DenseIntElementsAttr::get(paddingType, newPaddingValues);

  newPaddingAttr;
}]>;

def GradFilterConvLhsDilation : GlobalExpr</*needsprimal*/0, /*needsshadow*/0, [{
  op.getLhsDilationAttr();
}]>;

def GradFilterConvRhsDilation : GlobalExpr</*needsprimal*/0, /*needsshadow*/0, [{
  op.getWindowStridesAttr();
}]>;

def GradFilterConvWindowReversal : GlobalExpr</*needsprimal*/0, /*needsshadow*/0, [{
  op.getWindowReversalAttr();
}]>;

def GradFilterConvDimensionNumbers : GlobalExpr</*needsprimal*/0, /*needsshadow*/0, [{
  auto dimensionNumbers = op.getDimensionNumbers();
  auto newDimensionNumbers = mlir::stablehlo::ConvDimensionNumbersAttr::get(
    op.getContext(),
    dimensionNumbers.getInputFeatureDimension(),
    dimensionNumbers.getInputBatchDimension(),
    dimensionNumbers.getInputSpatialDimensions(),
    dimensionNumbers.getOutputBatchDimension(),
    dimensionNumbers.getOutputFeatureDimension(),
    dimensionNumbers.getOutputSpatialDimensions(),
    dimensionNumbers.getKernelInputFeatureDimension(),
    dimensionNumbers.getKernelOutputFeatureDimension(),
    dimensionNumbers.getKernelSpatialDimensions()
  );
  newDimensionNumbers;
}]>;

def GradFilterConvFeatureGroupCount : GlobalExpr</*needsprimal*/0, /*needsshadow*/0, [{
  unsigned int newFeatureGroupCount = 1;
  newFeatureGroupCount;
}]>;

def GradFilterConvBatchGroupCount : GlobalExpr</*needsprimal*/0, /*needsshadow*/0, [{
  auto featureGroupCount = op.getFeatureGroupCount();
  unsigned int newBatchGroupCount = featureGroupCount > 1 ? featureGroupCount : 1;
  newBatchGroupCount;
}]>;


def : HLODerivative<"ConvolutionOp", (Op $lhs, $rhs),
                    [
                        (Convolution
                          (TypeOf $lhs),
                          (DiffeRet),
                          $rhs,
                          (GradDataConvWindowStrides),
                          (GradDataConvPadding),
                          (GradDataConvLhsDilation),
                          (GradDataConvRhsDilation),
                          (GradDataConvWindowReversal),
                          (GradDataConvDimensionNumbers),
                          (GradDataConvFeatureGroupCount),
                          (GradDataConvBatchGroupCount),
                          (ResultDotPrec)
                        ),
                        (Convolution
                          (TypeOf $rhs),
                          $lhs,
                          (DiffeRet),
                          (GradFilterConvWindowStrides),
                          (GradFilterConvPadding),
                          (GradFilterConvLhsDilation),
                          (GradFilterConvRhsDilation),
                          (GradFilterConvWindowReversal),
                          (GradFilterConvDimensionNumbers),
                          (GradFilterConvFeatureGroupCount),
                          (GradFilterConvBatchGroupCount),
                          (ResultDotPrec)
                        )
                    ],
                    (Add
                      (SelectIfActive $lhs,
                        (Convolution
                          (ResultTypes),
                          (Shadow $lhs),
                          $rhs,
                          (ConvWindowStrides),
                          (ConvPadding),
                          (ConvLhsDilation),
                          (ConvRhsDilation),
                          (ConvWindowReversal),
                          (ConvDimensionNumbers),
                          (ConvFeatureGroupCount),
                          (ConvBatchGroupCount),
                          (ResultDotPrec)
                        ),
                        (HLOConstantFP<"0">)
                      ),
                      (SelectIfActive $rhs,
                        (Convolution
                          (ResultTypes),
                          $lhs,
                          (Shadow $rhs),
                          (ConvWindowStrides),
                          (ConvPadding),
                          (ConvLhsDilation),
                          (ConvRhsDilation),
                          (ConvWindowReversal),
                          (ConvDimensionNumbers),
                          (ConvFeatureGroupCount),
                          (ConvBatchGroupCount),
                          (ResultDotPrec)
                        ),
                        (HLOConstantFP<"0">)
                      )
                    )
                  >;

