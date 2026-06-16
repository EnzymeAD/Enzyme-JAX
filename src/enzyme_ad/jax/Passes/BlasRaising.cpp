//===- ArithRaising.cpp - Raise to Arith dialect --------------------------- //
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------------------===//
//
// This file implements a pass to raise operations to arith dialect.
//===---------------------------------------------------------------------===//

#include "Enzyme/MLIR/Dialect/Dialect.h"
#include "Enzyme/MLIR/Dialect/Ops.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "src/enzyme_ad/jax/Dialect/Dialect.h"
#include "src/enzyme_ad/jax/Dialect/Ops.h"
#include "src/enzyme_ad/jax/Passes/Passes.h"
#include "xla/mlir_hlo/mhlo/IR/hlo_ops.h"

#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/IR/AffineValueMap.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Affine/Utils.h"

#include "stablehlo/dialect/ChloOps.h"
#include "stablehlo/dialect/StablehloOps.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include <cstdint>
#include <string>

namespace mlir {
namespace enzyme {
#define GEN_PASS_DEF_BLASRAISINGPASS
#include "src/enzyme_ad/jax/Passes/Passes.h.inc"
} // namespace enzyme
} // namespace mlir

using namespace mlir;
using namespace mlir::enzyme;

#define CUBLAS_OP_N (0)
#define CUBLAS_OP_T (1)
#define CUBLAS_OP_C (2)

namespace {
struct BlasRaisingPass
    : public enzyme::impl::BlasRaisingPassBase<BlasRaisingPass> {
  using BlasRaisingPassBase::BlasRaisingPassBase;

  llvm::DenseMap<llvm::StringRef, func::FuncOp> transformedCublasFunctions;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<mlir::func::FuncDialect>();
    registry.insert<mlir::LLVM::LLVMDialect>();
    registry.insert<mlir::stablehlo::StablehloDialect>();
    registry.insert<mlir::memref::MemRefDialect>();
    registry.insert<mlir::gpu::GPUDialect>();
  }

  using CublasConstructor = std::function<func::FuncOp(
      LLVM::CallOp, SmallVector<Value>, func::FuncOp)>;
  using CublasOperandTypeFn = std::function<SmallVector<Type>(MLIRContext *)>;
  using CublasOperandShapeFn = std::function<std::map<int, SmallVector<int>>()>;

  SmallVector<Value> transformOperands(LLVM::CallOp call,
                                       SmallVector<Type> targetTypes) {
    OpBuilder builder(call);
    Location loc = call.getLoc();

    SmallVector<Value> newOperands;
    int idx = 0;
    for (auto it = std::next(call.getOperands().begin());
         it != call.getOperands().end(); ++it, ++idx) {
      Value arg = *it;
      Type desiredType = targetTypes[idx];

      Attribute attr;
      // Largely copied from AffineToStableHLORaising.cpp
      // Is tensor, just convert to memref
      if (auto tensorType = dyn_cast<TensorType>(desiredType)) {
        auto MT = MemRefType::get({ShapedType::kDynamic},
                                  tensorType.getElementType());
        newOperands.push_back(
            enzymexla::Pointer2MemrefOp::create(builder, loc, MT, arg));
        continue;
      }

      // Not a tensor, must check if you have to load the ptr
      if (isa<LLVM::LLVMPointerType>(arg.getType())) {
        arg = LLVM::LoadOp::create(builder, loc, desiredType, arg);
      }
      // convert scalar value into appropriate memref
      auto MT0 = MemRefType::get({}, arg.getType(), MemRefLayoutAttrInterface{},
                                 builder.getI64IntegerAttr(0));
      auto MT = MemRefType::get({}, arg.getType(), MemRefLayoutAttrInterface{},
                                builder.getI64IntegerAttr(1));

      auto res = gpu::AllocOp::create(builder, loc, MT, (mlir::Type) nullptr,
                                      ValueRange(), ValueRange(), ValueRange())
                     ->getResult(0);

      auto res0 = memref::AllocaOp::create(builder, loc, MT0);
      affine::AffineStoreOp::create(builder, loc, arg, res0,
                                    builder.getMultiDimIdentityMap(0),
                                    ValueRange());
      // TODO: add check for size of datatype, and set 4 to instead be that size
      auto c1 = arith::ConstantIndexOp::create(builder, loc, 4);
      enzymexla::MemcpyOp::create(builder, loc, (mlir::Type) nullptr,
                                  ValueRange(), res, res0, c1);

      builder.setInsertionPointAfter(call);
      gpu::DeallocOp::create(builder, loc, (mlir::Type) nullptr, ValueRange(),
                             res);
      builder.setInsertionPoint(call);
      newOperands.push_back(res);
    }
    return newOperands;
  }

  Type getElemType(Value tensor) {
    Type inputTy = tensor.getType();
    auto vecInputTy = cast<TensorType>(inputTy);
    return vecInputTy.getElementType();
  }

  int64_t getConstantValue(Value constant) {
    auto dim0C = constant.getDefiningOp<stablehlo::ConstantOp>();
    auto dim0Dense = cast<DenseElementsAttr>(dim0C.getValue());
    return dim0Dense.getSplatValue<IntegerAttr>().getInt();
  }

  Value getIsEnum(OpBuilder &builder, Location &loc, Value tensorVal,
                  SmallVector<int> enums) {
    auto i32Tensor = RankedTensorType::get({}, builder.getI32Type());
    auto cmp = [&](Value a, Value b) {
      return stablehlo::CompareOp::create(builder, loc, a, b,
                                          stablehlo::ComparisonDirection::EQ);
    };

    auto enumTensor = stablehlo::ConstantOp::create(
        builder, loc, DenseIntElementsAttr::get(i32Tensor, enums[0]));
    Value valCmp = cmp(tensorVal, enumTensor);
    for (int i = 1; i < enums.size(); i++) {
      enumTensor = stablehlo::ConstantOp::create(
          builder, loc, DenseIntElementsAttr::get(i32Tensor, enums[i]));
      Value newValCmp = cmp(tensorVal, enumTensor);
      valCmp = builder.create<stablehlo::OrOp>(loc, valCmp, newValCmp);
    }
    return valCmp;
  }

  Value makePairFromScalars(OpBuilder &builder, Location &loc, Value a,
                            Value b) {
    auto i64 = builder.getI64Type();
    auto a64 = stablehlo::ConvertOp::create(builder, loc,
                                            RankedTensorType::get({}, i64), a);
    auto b64 = stablehlo::ConvertOp::create(builder, loc,
                                            RankedTensorType::get({}, i64), b);

    auto aTensor = stablehlo::ReshapeOp::create(
        builder, loc, RankedTensorType::get({1}, i64), a64);
    auto bTensor = stablehlo::ReshapeOp::create(
        builder, loc, RankedTensorType::get({1}, i64), b64);
    return stablehlo::ConcatenateOp::create(
        builder, loc, RankedTensorType::get({2}, i64),
        ValueRange{aTensor, bTensor}, builder.getI64IntegerAttr(0));
  }

  // The input tensor has dimensions lda x cols if not transposed, and lda x
  // rows if transposed (in column major). We want to return it cropped and
  // reshaped to row-major rows x cols if not transposed, and cols x rows (then
  // transposed to rows x cols) if transposed. other_dim to cols

  // Also, the name rows vs cols is somewhat arbitrary, due to the mixing and
  // transposing between row and col major. The true meaning is that the shape
  // is passed to the JIT shape propagation pass as (ldim, rows, cols).
  Value make2DTensor(OpBuilder &builder, Location loc, Value tensor,
                     int64_t ldim, int64_t rows, int64_t cols,
                     int64_t sourceIdx, bool isTransposed) {
    auto elemTy = getElemType(tensor);
    auto ctx = builder.getContext();

    // Common attributes
    auto rowAttr = builder.getStringAttr("row");
    auto colAttr = builder.getStringAttr("col");
    auto ldimAttr = builder.getStringAttr("ldim");
    auto rowLdimAttr = builder.getStringAttr("ldim.row");
    auto colLdimAttr = builder.getStringAttr("ldim.col");
    auto sourceIdxAttr = builder.getI64IntegerAttr(sourceIdx);
    auto transposedAttr = builder.getBoolAttr(isTransposed);

    // --- Step 0: initial slice ---
    int64_t outer = isTransposed ? rows : cols;
    auto flatSliceTy = RankedTensorType::get({outer * ldim}, elemTy);

    Value flatSliced = builder.create<stablehlo::SliceOp>(
        loc, flatSliceTy, tensor, DenseI64ArrayAttr::get(ctx, {0}),
        DenseI64ArrayAttr::get(ctx, {outer * ldim}),
        DenseI64ArrayAttr::get(ctx, {1}));
    flatSliced.getDefiningOp()->setAttr("dim.0", isTransposed ? rowLdimAttr
                                                              : colLdimAttr);
    flatSliced.getDefiningOp()->setAttr("transposed", transposedAttr);
    flatSliced.getDefiningOp()->setAttr("sourceArgIdx", sourceIdxAttr);

    // --- Step 1: reshape ---
    auto reshapeTy = RankedTensorType::get({outer, ldim}, elemTy);

    Value reshaped =
        builder.create<stablehlo::ReshapeOp>(loc, reshapeTy, flatSliced);

    reshaped.getDefiningOp()->setAttr("dim.0",
                                      isTransposed ? rowAttr : colAttr);
    reshaped.getDefiningOp()->setAttr("dim.1", ldimAttr);
    reshaped.getDefiningOp()->setAttr("transposed", transposedAttr);
    reshaped.getDefiningOp()->setAttr("sourceArgIdx", sourceIdxAttr);

    // --- Step 2: transpose to make it row-major ---
    int64_t transInner = isTransposed ? rows : cols;
    auto transTy = RankedTensorType::get({ldim, transInner}, elemTy);

    Value transposed = builder.create<stablehlo::TransposeOp>(
        loc, transTy, reshaped, ArrayRef<int64_t>{1, 0});
    transposed.getDefiningOp()->setAttr("transposed", transposedAttr);
    transposed.getDefiningOp()->setAttr("sourceArgIdx", sourceIdxAttr);

    // --- Step 3: slice to desired logical shape ---
    int64_t sliceRows = isTransposed ? cols : rows;
    int64_t sliceCols = isTransposed ? rows : cols;

    auto sliceTy = RankedTensorType::get({sliceRows, sliceCols}, elemTy);

    Value sliced = builder.create<stablehlo::SliceOp>(
        loc, sliceTy, transposed, DenseI64ArrayAttr::get(ctx, {0, 0}),
        DenseI64ArrayAttr::get(ctx, {sliceRows, sliceCols}),
        DenseI64ArrayAttr::get(ctx, {1, 1}));

    sliced.getDefiningOp()->setAttr("dim.0", isTransposed ? colAttr : rowAttr);
    sliced.getDefiningOp()->setAttr("dim.1", isTransposed ? rowAttr : colAttr);
    sliced.getDefiningOp()->setAttr("transposed", transposedAttr);
    sliced.getDefiningOp()->setAttr("sourceArgIdx", sourceIdxAttr);

    if (isTransposed) {
      Value slicedTransposed = builder.create<stablehlo::TransposeOp>(
          loc, RankedTensorType::get({rows, cols}, elemTy), sliced,
          ArrayRef<int64_t>{1, 0});
      slicedTransposed.getDefiningOp()->setAttr("transposed", transposedAttr);
      slicedTransposed.getDefiningOp()->setAttr("sourceArgIdx", sourceIdxAttr);
      return slicedTransposed;
    }

    return sliced;
  }

  Value writebackTo1DTensor(OpBuilder &builder, Location loc, Value orig,
                            Value update, int64_t ldim, int64_t rows,
                            int64_t cols, int64_t sourceIdx) {
    Type elemTy = getElemType(orig);

    // auto startIndices = DenseI64ArrayAttr::get(ctx, {0, 0});
    mlir::Attribute colAttr =
        mlir::StringAttr::get(builder.getContext(), "col");
    mlir::Attribute ldimAttr =
        mlir::StringAttr::get(builder.getContext(), "ldim");
    // mlir::Attribute ldimSizeAttr =
    // mlir::StringAttr::get(builder.getContext(), "alongLdim");
    mlir::Attribute ldimColAttr =
        mlir::StringAttr::get(builder.getContext(), "ldim.col");
    auto sourceIdxAttr = builder.getI64IntegerAttr(sourceIdx);

    auto reshaped_orig = stablehlo::ReshapeOp::create(
        builder, loc, RankedTensorType::get({cols, ldim}, elemTy), orig);
    reshaped_orig->setAttr("dim.0", colAttr);
    reshaped_orig->setAttr("dim.1", ldimAttr);
    reshaped_orig->setAttr("sourceArgIdx", sourceIdxAttr);

    auto transposed_update = stablehlo::TransposeOp::create(
        builder, loc, RankedTensorType::get({cols, rows}, elemTy), update,
        SmallVector<int64_t>{1, 0});

    Value zero_tensor = stablehlo::ConstantOp::create(
        builder, loc,
        DenseIntElementsAttr::get(
            RankedTensorType::get({}, builder.getI64Type()), {(int64_t)0}));
    auto updated_2D_orig = stablehlo::DynamicUpdateSliceOp::create(
        builder, loc, reshaped_orig.getType(),
        ValueRange{reshaped_orig, transposed_update, zero_tensor, zero_tensor});

    auto tmp = stablehlo::ReshapeOp::create(
        builder, loc, RankedTensorType::get({cols * ldim}, elemTy),
        updated_2D_orig);
    tmp->setAttr("dim.0", ldimColAttr);
    tmp->setAttr("sourceArgIdx", sourceIdxAttr);

    Value size_tensor = stablehlo::ConstantOp::create(
        builder, loc,
        DenseIntElementsAttr::get(
            RankedTensorType::get({1}, builder.getI64Type()), {cols * ldim}));

    // this being a dynamic reshape op is a sneaky workaround to the fact that
    // we need to return a <?xdatatype> tensor, but we use placeholder shapes
    // that aren't dynamic and we can't cast from <nxdatatype>. Annotate this so
    // we can delete it in the JIT pass.
    mlir::Attribute reshapeAttr =
        mlir::BoolAttr::get(builder.getContext(), true);
    auto reshape = stablehlo::DynamicReshapeOp::create(
        builder, loc, RankedTensorType::get({ShapedType::kDynamic}, elemTy),
        tmp, size_tensor);
    reshape->setAttr("replaceWithStaticReshape", reshapeAttr);
    reshape->setAttr("sourceArgIdx", sourceIdxAttr);
    return reshape;
  }

  Value broadcastToInputSize(OpBuilder &builder, Location loc, Value scalar,
                             int64_t rows, int64_t cols, int64_t sourceIdx) {
    mlir::Attribute colAttr =
        mlir::StringAttr::get(builder.getContext(), "col");
    mlir::Attribute rowAttr =
        mlir::StringAttr::get(builder.getContext(), "row");
    mlir::Attribute sourceIdxAttr = builder.getI64IntegerAttr(sourceIdx);

    Type elemTy = cast<ShapedType>(scalar.getType()).getElementType();
    auto reshaped = stablehlo::ReshapeOp::create(
        builder, loc, RankedTensorType::get({1, 1}, elemTy), scalar);

    auto resultType = RankedTensorType::get({rows, cols}, elemTy);
    auto broadcast = stablehlo::BroadcastInDimOp::create(
        builder, loc, resultType, reshaped,
        builder.getDenseI64ArrayAttr({(int64_t)0, (int64_t)1}));

    broadcast->setAttr("dim.0", rowAttr);
    broadcast->setAttr("dim.1", colAttr);
    broadcast->setAttr("sourceArgIdx", sourceIdxAttr);
    return broadcast;
  }

  Value createSelectStatement(OpBuilder &builder, Location loc, Value ifVal,
                              Value elseVal, int64_t sourceIdx) {
    mlir::Attribute sourceIdxAttr = builder.getI64IntegerAttr(sourceIdx);
    mlir::Attribute IsTransposeSelectAttr = builder.getBoolAttr(true);

    auto predType = RankedTensorType::get({}, builder.getI1Type());
    auto trueAttr = DenseElementsAttr::get(predType, builder.getBoolAttr(true));
    Value pred = builder.create<stablehlo::ConstantOp>(loc, predType, trueAttr);

    auto selOp = stablehlo::SelectOp::create(builder, loc,
                                             ifVal.getType(), // result type
                                             pred, ifVal, elseVal);

    selOp->setAttr("sourceArgIdx", sourceIdxAttr);
    selOp->setAttr("isTransposeSelect", IsTransposeSelectAttr);

    return selOp;
  }

  func::FuncOp
  buildFunctionSignature(LLVM::CallOp call, SmallVector<Value> operands,
                         std::map<int, SmallVector<int>> operandShapes,
                         StringRef name) {
    MLIRContext *ctx = call.getContext();
    OpBuilder Builder(ctx);
    auto loc = call.getLoc();
    auto module = call->getParentOfType<ModuleOp>();

    // Construct new function type
    SmallVector<Type> newInputs;
    for (auto &value_wrapper : operands) {
      auto argTy = cast<MemRefType>(value_wrapper.getType());
      newInputs.push_back(
          RankedTensorType::get(argTy.getShape(), argTy.getElementType()));
    }
    SmallVector<Type> results(newInputs);
    auto newFuncType = mlir::FunctionType::get(ctx, newInputs, results);

    // Construct new function
    func::FuncOp fn = func::FuncOp::create(loc, name, newFuncType);
    fn.setPrivate();
    module.push_back(fn);

    for (const auto &pair : operandShapes) {
      int idx = pair.first;
      const llvm::SmallVector<int> &shapeDims = pair.second;
      fn.setArgAttr(idx, "shape.ld", Builder.getI32IntegerAttr(shapeDims[1]));
      fn.setArgAttr(idx, "shape.transpose",
                    Builder.getI32IntegerAttr(shapeDims[0]));
      for (int shapeDimIdx = 2; shapeDimIdx < shapeDims.size(); shapeDimIdx++) {
        fn.setArgAttr(idx, "shape." + std::to_string(shapeDimIdx - 2),
                      Builder.getI32IntegerAttr(shapeDims[shapeDimIdx]));
      }
    }

    return fn;
  }

  // Shape info is presented as (leading dimension, dim0, dim1, ...)
  std::map<int, SmallVector<int>> getShapeInfoCublasSGemm_v2(MLIRContext *ctx) {
    std::map<int, SmallVector<int>> shapeMap;
    shapeMap[6] = SmallVector<int>({0, 7, 2, 4});
    shapeMap[8] = SmallVector<int>({1, 9, 4, 3});
    shapeMap[11] = SmallVector<int>({-1, 12, 2, 3});
    return shapeMap;
  }

  SmallVector<Type> getOperandTypesCublasSGemm_v2(MLIRContext *ctx) {
    Type i32 = IntegerType::get(ctx, 32);
    Type f32 = Float32Type::get(ctx);
    Type f32DynTensor2D = RankedTensorType::get(
        {ShapedType::kDynamic, ShapedType::kDynamic}, f32);

    SmallVector<Type> types;
    // cublasSgemm_v2
    types.push_back(i32);            // transA
    types.push_back(i32);            // transB
    types.push_back(i32);            // m
    types.push_back(i32);            // n
    types.push_back(i32);            // k
    types.push_back(f32);            // alpha
    types.push_back(f32DynTensor2D); // A
    types.push_back(i32);            // lda
    types.push_back(f32DynTensor2D); // B
    types.push_back(i32);            // ldb
    types.push_back(f32);            // beta
    types.push_back(f32DynTensor2D); // C
    types.push_back(i32);            // ldc
    return types;
  }

  void replaceCublasSGemm_v2(LLVM::CallOp call) {
    llvm::StringRef name = "CublasSGemm_v2";
    MLIRContext *ctx = call.getContext();

    SmallVector<Value> operands =
        transformOperands(call, getOperandTypesCublasSGemm_v2(ctx));

    func::FuncOp f;
    llvm::errs() << "name: " << name << ", count: " << transformedCublasFunctions.count(name) << "\n";
    if (transformedCublasFunctions.count(name) > 0) {
      f = transformedCublasFunctions[name];
    } else {
      func::FuncOp fn = buildFunctionSignature(
          call, operands, getShapeInfoCublasSGemm_v2(ctx), name);
      f = constructCublasSGemm_v2(fn);
      transformedCublasFunctions[name] = f;
    }

    // create call to new function
    OpBuilder builder(call);
    enzymexla::XLAWrapperOp::create(
        builder, call->getLoc(), SymbolRefAttr::get(f),
        llvm::to_vector(operands), nullptr, nullptr);
  }

  func::FuncOp constructCublasSGemm_v2(func::FuncOp fn) {
    Block *entry = fn.addEntryBlock();
    OpBuilder bodyBuilder(entry, entry->begin());
    SmallVector<Value> operands(fn.getArguments().begin(),
                                fn.getArguments().end());

    // auto module = call->getParentOfType<ModuleOp>();
    auto loc = fn.getLoc();

    // Extract arguments
    int i = 0;
    // Value transAenum = operands[i++];
    // Value transBenum = operands[i++];
    i += 4;
    // Value m = operands[i++];
    // Value n = operands[i++];
    // Value k = operands[i++];
    i++;
    Value alpha = operands[i++];
    Value A_flat = operands[i++];
    // Value lda = operands[i++];
    i++;
    Value B_flat = operands[i++];
    // Value ldb = operands[i++];
    i++;
    Value beta = operands[i++];
    Value C_flat = operands[i++];
    // Value ldc = operands[i++];
    i++;

    Type elemTy = getElemType(A_flat);

    // int64_t m_const = getConstantValue(m);
    int64_t m_const = 1;
    // int64_t n_const = getConstantValue(n);
    int64_t n_const = 1;
    // int64_t k_const = getConstantValue(k);
    int64_t k_const = 1;
    // int64_t lda_const = getConstantValue(lda);
    int64_t lda_const = 1;
    // int64_t ldb_const = getConstantValue(ldb);
    int64_t ldb_const = 1;
    // int64_t ldc_const = getConstantValue(ldc);
    int64_t ldc_const = 1;

    // Value transA = buildEnumMatch(bodyBuilder, loc, transAenum, {1, 2});
    // Value transB = buildEnumMatch(bodyBuilder, loc, transBenum, {1, 2});

    // If transA or transB matches any of these enums, take the transpose
    // SmallVector<int64_t> transposeEnums = {1, 2};
    // bool transA = llvm::is_contained(transposeEnums, transAenum_const);
    // bool transB = llvm::is_contained(transposeEnums, transBenum_const);

    Value A_IsTransposed = make2DTensor(bodyBuilder, loc, A_flat, lda_const,
                                        m_const, k_const, 6, true);
    Value A_NotTransposed = make2DTensor(bodyBuilder, loc, A_flat, lda_const,
                                         m_const, k_const, 6, false);

    Value B_IsTransposed = make2DTensor(bodyBuilder, loc, B_flat, ldb_const,
                                        k_const, n_const, 8, true);
    Value B_NotTransposed = make2DTensor(bodyBuilder, loc, B_flat, ldb_const,
                                         k_const, n_const, 8, false);

    Value A_eff = createSelectStatement(bodyBuilder, loc, A_IsTransposed,
                                        A_NotTransposed, 6);
    Value B_eff = createSelectStatement(bodyBuilder, loc, B_IsTransposed,
                                        B_NotTransposed, 8);
    // stablehlo::IfOp B_if = createIfStatement(bodyBuilder, loc,
    // B_IsTransposed,
    //  B_NotTransposed, transB);

    // STEP 2: Dot general: A_eff [m,k], B_eff [k,n] => [m,n]
    // Mixed batch dims are empty; contracting dimension is {1}.
    // auto resultType = UnrankedTensorType::get(f32);

    // cuBLAS uses column-major, StableHLO uses row-major
    // Column-major: C(m,n) = A(m,k) * B(k,n)
    // Row-major: C_row(n,m) = B_row(n,k) * A_row(k,m)
    // A_eff is now [k, m] and B_eff is now [n, k]
    // Compute B_eff[n,k] * A_eff[k,m] = [n,m]
    // This [n,m] row-major result corresponds to [m,n] column-major output
    auto dotDimNumbers =
        stablehlo::DotDimensionNumbersAttr::get(bodyBuilder.getContext(),
                                                /*lhsBatchingDims=*/{},
                                                /*rhsBatchingDims=*/{},
                                                /*lhsContractingDims=*/{1},
                                                /*rhsContractingDims=*/{0});

    Value dot = stablehlo::DotGeneralOp::create(
        bodyBuilder, loc, RankedTensorType::get({m_const, n_const}, elemTy),
        A_eff, B_eff, dotDimNumbers, nullptr, nullptr);

    // STEP 3: alpha * dot + beta * C
    // All operations are in row-major space
    // Scale dot
    Value alphaBroadcast =
        broadcastToInputSize(bodyBuilder, loc, alpha, m_const, n_const, 11);
    Value scaledDot = bodyBuilder.create<stablehlo::MulOp>(
        loc, alphaBroadcast.getType(), dot, alphaBroadcast);

    auto C_sliced = make2DTensor(bodyBuilder, loc, C_flat, ldc_const, m_const,
                                 n_const, 11, false);
    // Scale C
    Value betaBroadcast =
        broadcastToInputSize(bodyBuilder, loc, beta, m_const, n_const, 11);
    Value scaledC = bodyBuilder.create<stablehlo::MulOp>(
        loc, C_sliced.getType(), C_sliced, betaBroadcast);

    // Add: out = scaledDot + scaledC
    Value update = bodyBuilder.create<stablehlo::AddOp>(
        loc, scaledDot.getType(), scaledDot, scaledC);

    // Write back
    // Value outFlat = stablehlo::ReshapeOp::create(bodyBuilder, loc,
    //   RankedTensorType::get({n_const * m_const}, elemTy), update);
    // Value zero_i64 = stablehlo::ConstantOp::create(bodyBuilder, loc,
    //   DenseIntElementsAttr::get(RankedTensorType::get({},
    //   bodyBuilder.getI64Type()), {(int64_t)0}));
    // outFlat = stablehlo::DynamicUpdateSliceOp::create(bodyBuilder, loc,
    //   C_flat.getType(), ValueRange{C_flat, outFlat, zero_i64});
    Value outFlat = writebackTo1DTensor(bodyBuilder, loc, C_flat, update,
                                        ldc_const, m_const, n_const, 11);

    operands[11] = outFlat;
    SmallVector<Value> result;
    for (int idx = 0; idx < operands.size(); idx++) {
      result.push_back(operands[idx]);
    }
    func::ReturnOp::create(bodyBuilder, loc, ValueRange{result});
    return fn;
  }

  void runOnOperation() override {
    auto op = getOperation();
    llvm::errs() << "=== BlasRaisingPass running ===\n";
    op->dump();

    // op->walk([&](LLVM::LLVMFuncOp callOp) {
    //   auto calleeName = callOp.getName();
    //   if (calleeName == "cublasSgemm_v2") {
    //     getCublasSGemm_v2(callOp);
    //   }
    // });
    SmallVector<LLVM::CallOp, 4> cublasCalls;

    op->walk([&](LLVM::CallOp callOp) {
      auto calleeName = callOp.getCallee().value_or("");
      if (calleeName == "cublasSgemm_v2") {
        replaceCublasSGemm_v2(callOp);
        cublasCalls.push_back(callOp);
      }
    });

    for (auto call : cublasCalls) {
      OpBuilder builder(call);
      Value zero = LLVM::ConstantOp::create(builder, call->getLoc(),
                                            builder.getI32Type(), 0);

      for (auto result : call.getResults()) {
        result.replaceAllUsesWith(zero);
      }
      call.erase();
    }
    llvm::errs() << "=== BlasRaisingPass done ===\n";
    op->dump();
    // llvm::errs().flush();
    
  }
};

} // end anonymous namespace
