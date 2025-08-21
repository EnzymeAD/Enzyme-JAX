//===- LLVMToControlFlow.cpp - ControlFlow to LLVM dialect conversion -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to convert MLIR standard and builtin dialects
// into the LLVM IR dialect.
//
//===----------------------------------------------------------------------===//

#include "src/enzyme_ad/jax/Dialect/Ops.h"
#include "src/enzyme_ad/jax/Passes/Passes.h"

#include "mlir/Analysis/DataLayoutAnalysis.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/GPUCommon/GPUCommonPass.h"
#include "mlir/Conversion/GPUToNVVM/GPUToNVVMPass.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/MathToLibm/MathToLibm.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/OpenMPToLLVM/ConvertOpenMPToLLVM.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/UBToLLVM/UBToLLVM.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/Passes.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/FunctionCallUtils.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/NVGPU/IR/NVGPUDialect.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Target/LLVMIR/Import.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"

#include "mlir/Transforms/DialectConversion.h"

#include "mlir/Dialect/Async/IR/Async.h"

#include "xla/mlir/utils/type_util.h"

#include "RuntimeWrapperUtils.h"

#include <fstream>
#include <limits>
#include <map>
#include <numeric>

#define DEBUG_TYPE "convert-enzymexla-to-llvm"
#define DBGS() ::llvm::dbgs() << "[" DEBUG_TYPE ":" << PATTERN << "] "

namespace mlir {
namespace enzyme {
#define GEN_PASS_DEF_CONVERTPOLYGEISTTOLLVM
#include "src/enzyme_ad/jax/Passes/Passes.h.inc"
} // namespace enzyme
} // namespace mlir

using namespace mlir;
using namespace mlir::enzymexla;

mlir::LLVM::LLVMFuncOp GetOrCreateFreeFunction(ModuleOp module);

Type convertMemrefElementTypeForLLVMPointer(
    MemRefType type, const LLVMTypeConverter &converter) {
  Type converted = converter.convertType(type.getElementType());
  if (!converted)
    return Type();

  if (type.getRank() == 0) {
    return converted;
  }

  // Only the leading dimension can be dynamic.
  if (llvm::any_of(type.getShape().drop_front(), ShapedType::isDynamic))
    return Type();

  // Only identity layout is supported.
  // TODO: detect the strided layout that is equivalent to identity
  // given the static part of the shape.
  if (!type.getLayout().isIdentity())
    return Type();

  if (type.getRank() > 0) {
    for (int64_t size : llvm::reverse(type.getShape().drop_front()))
      converted = LLVM::LLVMArrayType::get(converted, size);
  }
  return converted;
}

static Value insertXLAInitDeinit(mlir::ModuleOp moduleOp, StringRef backend,
                                 OpBuilder &rewriter) {
  auto loc = moduleOp.getLoc();
  // TODO is it okay to be using OpBuilder's in op rewriter?
  // OpBuilder moduleBuilder(moduleOp.getBodyRegion());
  SmallString<128> ctorNameBuffer("__reactant_xla_init");
  LLVM::LLVMFuncOp ctor = dyn_cast_or_null<LLVM::LLVMFuncOp>(
      SymbolTable::lookupSymbolIn(moduleOp, ctorNameBuffer));
  SmallString<128> dtorNameBuffer("__reactant_xla_deinit");
  LLVM::LLVMFuncOp dtor = dyn_cast_or_null<LLVM::LLVMFuncOp>(
      SymbolTable::lookupSymbolIn(moduleOp, dtorNameBuffer));

  SmallString<128> dataNameBuffer("__reactant_xla_data");
  LLVM::GlobalOp data = dyn_cast_or_null<LLVM::GlobalOp>(
      SymbolTable::lookupSymbolIn(moduleOp, dataNameBuffer));

  auto ptrty = LLVM::LLVMPointerType::get(moduleOp->getContext());

  if (ctor) {
    assert(dtor && "xla module constructor does not exist but destructor does");
    assert(data && "xla module constructor does not exist but data does");
    return rewriter.create<LLVM::AddressOfOp>(loc, ptrty,
                                              data.getSymNameAttr());
  }

  {
    PatternRewriter::InsertionGuard B(rewriter);
    rewriter.setInsertionPointToEnd(moduleOp.getBody());
    ctor = rewriter.create<LLVM::LLVMFuncOp>(
        loc, ctorNameBuffer,
        LLVM::LLVMFunctionType::get(
            LLVM::LLVMVoidType::get(moduleOp.getContext()), {}),
        LLVM::Linkage::Private);
    dtor = rewriter.create<LLVM::LLVMFuncOp>(
        loc, dtorNameBuffer,
        LLVM::LLVMFunctionType::get(
            LLVM::LLVMVoidType::get(moduleOp.getContext()), {}),
        LLVM::Linkage::Private);

    auto ctorSymbol = FlatSymbolRefAttr::get(ctor);
    rewriter.create<LLVM::GlobalCtorsOp>(
        loc, rewriter.getArrayAttr({std::move(ctorSymbol)}),
        rewriter.getI32ArrayAttr({65535}),
        rewriter.getArrayAttr({LLVM::ZeroAttr::get(rewriter.getContext())}));

    auto dtorSymbol = FlatSymbolRefAttr::get(dtor);
    rewriter.create<LLVM::GlobalDtorsOp>(
        loc, rewriter.getArrayAttr({std::move(dtorSymbol)}),
        rewriter.getI32ArrayAttr({65535}),
        rewriter.getArrayAttr({LLVM::ZeroAttr::get(rewriter.getContext())}));

    data = rewriter.create<LLVM::GlobalOp>(
        loc, ptrty, /*constant*/ false, LLVM::Linkage::Internal, dataNameBuffer,
        /* initValue */ mlir::Attribute(),
        /* alignment */ 8, /* addrSpace */ 0);
  }

  // device id, ptr
  Type tys[] = {ptrty, ptrty};

  Type tys2[] = {ptrty};

  auto xlaInitFn =
      LLVM::lookupOrCreateFn(rewriter, moduleOp, "reactantXLAInit", tys,
                             LLVM::LLVMVoidType::get(moduleOp->getContext()));
  if (failed(xlaInitFn)) {
    llvm::errs() << " xlaExec already exists with different types\n";
    return nullptr;
  }

  auto xlaDeInitFn =
      LLVM::lookupOrCreateFn(rewriter, moduleOp, "reactantXLADeInit", tys2,
                             LLVM::LLVMVoidType::get(moduleOp->getContext()));
  if (failed(xlaInitFn)) {
    llvm::errs() << " xlaExec already exists with different types\n";
    return nullptr;
  }

  {
    PatternRewriter::InsertionGuard B(rewriter);
    rewriter.setInsertionPointToEnd(ctor.addEntryBlock(rewriter));

    std::string bstr;
    llvm::raw_string_ostream stream(bstr);
    stream << backend << '\0';
    auto stringval = mlir::LLVM::createGlobalString(
        loc, rewriter, "xlabackend", bstr, LLVM::Linkage::Internal);

    auto glob =
        rewriter.create<LLVM::AddressOfOp>(loc, ptrty, data.getSymNameAttr());
    Value args[] = {glob, stringval};
    rewriter.create<LLVM::CallOp>(loc, xlaInitFn.value(), args);
    rewriter.create<LLVM::ReturnOp>(loc, ValueRange());
  }

  {
    PatternRewriter::InsertionGuard B(rewriter);
    rewriter.setInsertionPointToEnd(dtor.addEntryBlock(rewriter));

    auto glob =
        rewriter.create<LLVM::AddressOfOp>(loc, ptrty, data.getSymNameAttr());
    Value args[] = {glob};
    rewriter.create<LLVM::CallOp>(loc, xlaDeInitFn.value(), args);
    rewriter.create<LLVM::ReturnOp>(loc, ValueRange());
  }

  return rewriter.create<LLVM::AddressOfOp>(loc, ptrty, data.getSymNameAttr());
}

struct Stream2TokenOpLowering : public ConvertOpToLLVMPattern<StreamToTokenOp> {
  using ConvertOpToLLVMPattern<StreamToTokenOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(StreamToTokenOp op, OpAdaptor transformed,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOp(op, {transformed.getSource()});
    return success();
  }
};

struct Memref2PointerOpLowering
    : public ConvertOpToLLVMPattern<Memref2PointerOp> {
  using ConvertOpToLLVMPattern<Memref2PointerOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(Memref2PointerOp op, OpAdaptor transformed,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    auto LPT = cast<LLVM::LLVMPointerType>(op.getType());
    auto space0 = op.getSource().getType().getMemorySpaceAsInt();
    if (isa<LLVM::LLVMPointerType>(transformed.getSource().getType())) {
      mlir::Value ptr = transformed.getSource();
      if (space0 != LPT.getAddressSpace())
        ptr = rewriter.create<LLVM::AddrSpaceCastOp>(loc, LPT, ptr);
      rewriter.replaceOp(op, {ptr});
      return success();
    }

    // MemRefDescriptor sourceMemRef(operands.front());
    MemRefDescriptor targetMemRef(
        transformed.getSource()); // MemRefDescriptor::undef(rewriter, loc,
                                  // targetDescTy);

    // Offset.
    Value baseOffset = targetMemRef.offset(rewriter, loc);
    Value ptr = targetMemRef.alignedPtr(rewriter, loc);
    Value idxs[] = {baseOffset};
    ptr = rewriter.create<LLVM::GEPOp>(loc, ptr.getType(), rewriter.getI8Type(),
                                       ptr, idxs);
    if (space0 != LPT.getAddressSpace())
      ptr = rewriter.create<LLVM::AddrSpaceCastOp>(loc, LPT, ptr);

    rewriter.replaceOp(op, {ptr});
    return success();
  }
};

struct Pointer2MemrefOpLowering
    : public ConvertOpToLLVMPattern<Pointer2MemrefOp> {
  using ConvertOpToLLVMPattern<Pointer2MemrefOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(Pointer2MemrefOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    // MemRefDescriptor sourceMemRef(operands.front());
    auto convertedType = getTypeConverter()->convertType(op.getType());
    assert(convertedType && "unexpected failure in memref type conversion");
    auto space1 = op.getType().getMemorySpaceAsInt();
    if (auto PT = dyn_cast<LLVM::LLVMPointerType>(convertedType)) {
      mlir::Value ptr = adaptor.getSource();
      if (space1 != cast<LLVM::LLVMPointerType>(op.getOperand().getType())
                        .getAddressSpace())
        ptr = rewriter.create<LLVM::AddrSpaceCastOp>(loc, PT, ptr);
      rewriter.replaceOp(op, {ptr});
      return success();
    }

    auto descr = MemRefDescriptor::poison(rewriter, loc, convertedType);
    Value ptr = adaptor.getSource();

    if (space1 != cast<LLVM::LLVMPointerType>(op.getOperand().getType())
                      .getAddressSpace())
      ptr = rewriter.create<LLVM::AddrSpaceCastOp>(
          loc, descr.getElementPtrType(), ptr);

    // Extract all strides and offsets and verify they are static.
    int64_t offset;
    SmallVector<int64_t, 4> strides;
    auto result = op.getType().getStridesAndOffset(strides, offset);
    (void)result;
    assert(succeeded(result) && "unexpected failure in stride computation");
    assert(offset != ShapedType::kDynamic && "expected static offset");

    bool first = true;
    assert(!llvm::any_of(strides, [&](int64_t stride) {
      if (first) {
        first = false;
        return false;
      }
      return stride == ShapedType::kDynamic;
    }) && "expected static strides except first element");
    (void)first;

    descr.setAllocatedPtr(rewriter, loc, ptr);
    descr.setAlignedPtr(rewriter, loc, ptr);
    descr.setConstantOffset(rewriter, loc, offset);

    // Fill in sizes and strides
    for (unsigned i = 0, e = op.getType().getRank(); i != e; ++i) {
      descr.setConstantSize(rewriter, loc, i, op.getType().getDimSize(i));
      descr.setConstantStride(rewriter, loc, i, strides[i]);
    }

    rewriter.replaceOp(op, {descr});
    return success();
  }
};

void populatePolygeistToLLVMConversionPatterns(LLVMTypeConverter &converter,
                                               RewritePatternSet &patterns) {
  // clang-format off
  //patterns.add<TypeSizeOpLowering>(converter);
  //patterns.add<TypeAlignOpLowering>(converter);
  //patterns.add<UndefLowering>(converter);
  //patterns.add<SubIndexOpLowering>(converter);
  patterns.add<Stream2TokenOpLowering>(converter);
  patterns.add<Memref2PointerOpLowering>(converter);
  patterns.add<Pointer2MemrefOpLowering>(converter);
  // clang-format on
}

namespace {
struct LLVMOpLowering : public ConversionPattern {
  explicit LLVMOpLowering(LLVMTypeConverter &converter)
      : ConversionPattern(converter, Pattern::MatchAnyOpTypeTag(), 1,
                          &converter.getContext()) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    const TypeConverter *converter = getTypeConverter();

    SmallVector<Type> convertedResultTypes;
    if (failed(converter->convertTypes(op->getResultTypes(),
                                       convertedResultTypes))) {
      return failure();
    }
    SmallVector<Type> convertedOperandTypes;
    if (failed(converter->convertTypes(op->getOperandTypes(),
                                       convertedOperandTypes))) {
      return failure();
    }

    bool typeAttrsConverted = true;
    for (auto &attr : op->getAttrs())
      if (auto tyAttr = dyn_cast<TypeAttr>(attr.getValue()))
        if (converter->convertType(tyAttr.getValue()) != tyAttr.getValue())
          typeAttrsConverted = false;

    if (convertedResultTypes == op->getResultTypes() &&
        convertedOperandTypes == op->getOperandTypes() && typeAttrsConverted) {
      return failure();
    }
    if (isa<UnrealizedConversionCastOp>(op))
      return failure();

    SmallVector<NamedAttribute> convertedAttrs;
    for (auto &attr : op->getAttrs()) {
      NamedAttribute convertedAttr = attr;
      if (auto tyAttr = dyn_cast<TypeAttr>(attr.getValue())) {
        Type convertedTy = converter->convertType(tyAttr.getValue());
        if (!convertedTy)
          return failure();
        convertedAttr.setValue(TypeAttr::get(convertedTy));
      }
      convertedAttrs.push_back(convertedAttr);
    }

    OperationState state(op->getLoc(), op->getName());
    state.addOperands(operands);
    state.addTypes(convertedResultTypes);
    state.addAttributes(convertedAttrs);
    state.addSuccessors(op->getSuccessors());
    for (unsigned i = 0, e = op->getNumRegions(); i < e; ++i)
      state.addRegion();

    Operation *rewritten = rewriter.create(state);

    for (unsigned i = 0, e = op->getNumRegions(); i < e; ++i)
      rewriter.inlineRegionBefore(op->getRegion(i), rewritten->getRegion(i),
                                  rewritten->getRegion(i).begin());

    rewriter.replaceOp(op, rewritten->getResults());

    return success();
  }
};

struct URLLVMOpLowering
    : public ConvertOpToLLVMPattern<UnrealizedConversionCastOp> {
  using ConvertOpToLLVMPattern<
      UnrealizedConversionCastOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(UnrealizedConversionCastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (op.use_empty()) {
      rewriter.eraseOp(op);
      return success();
    }
    if (op->getResult(0).getType() != op->getOperand(0).getType())
      return failure();

    rewriter.replaceOp(op, op->getOperands());
    return success();
  }
};

struct GlobalOpTypeConversion : public OpConversionPattern<LLVM::GlobalOp> {
  explicit GlobalOpTypeConversion(LLVMTypeConverter &converter)
      : OpConversionPattern<LLVM::GlobalOp>(converter,
                                            &converter.getContext()) {}

  LogicalResult
  matchAndRewrite(LLVM::GlobalOp op, LLVM::GlobalOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    const TypeConverter *converter = getTypeConverter();
    Type globalType = adaptor.getGlobalType();
    Type convertedType = converter->convertType(globalType);
    if (!convertedType)
      return failure();
    if (convertedType == globalType)
      return failure();

    rewriter.modifyOpInPlace(
        op, [&]() { op.setGlobalTypeAttr(TypeAttr::get(convertedType)); });
    return success();
  }
};

struct ReturnOpTypeConversion : public ConvertOpToLLVMPattern<LLVM::ReturnOp> {
  using ConvertOpToLLVMPattern<LLVM::ReturnOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(LLVM::ReturnOp op, LLVM::ReturnOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto replacement =
        rewriter.replaceOpWithNewOp<LLVM::ReturnOp>(op, adaptor.getArg());
    replacement->setAttrs(adaptor.getAttributes());
    return success();
  }
};
} // namespace

//===-----------------------------------------------------------------------===/
// Patterns for C-compatible MemRef lowering.
//===-----------------------------------------------------------------------===/
// Additional patterns for converting MLIR ops from MemRef and Func dialects
// to the LLVM dialect using the C-compatible type conversion for memrefs.
// Specifically, a memref such as memref<A x B x C x type> is converted into
// a pointer to an array of arrays such as !llvm.ptr<array<B x array<C x type>>
// with additional conversion of the element type. This approach is only
// applicable to memrefs with static shapes in all dimensions but the outermost,
// which coincides with the nested array constructs allowed in C (except VLA).
// This also matches the type produced by Clang for such array constructs,
// removing the need for ABI compatibility layers.
//===-----------------------------------------------------------------------===/

namespace {
/// Pattern for allocation-like operations.
template <typename OpTy>
struct AllocLikeOpLowering : public ConvertOpToLLVMPattern<OpTy> {
public:
  using ConvertOpToLLVMPattern<OpTy>::ConvertOpToLLVMPattern;

protected:
  /// Returns the value containing the outermost dimension of the memref to be
  /// allocated, or 1 if the memref has rank zero.
  Value getOuterSize(OpTy original,
                     typename ConvertOpToLLVMPattern<OpTy>::OpAdaptor adaptor,
                     ConversionPatternRewriter &rewriter) const {
    if (!adaptor.getDynamicSizes().empty())
      return adaptor.getDynamicSizes().front();

    Type indexType = rewriter.getIndexType();
    return this->createIndexAttrConstant(
        rewriter, original->getLoc(), indexType,
        original.getType().getRank() == 0 ? 1
                                          : original.getType().getDimSize(0));
  }
};

/// Pattern for lowering automatic stack allocations.
struct CAllocaOpLowering : public AllocLikeOpLowering<memref::AllocaOp> {
public:
  using AllocLikeOpLowering<memref::AllocaOp>::AllocLikeOpLowering;

  LogicalResult
  matchAndRewrite(memref::AllocaOp allocaOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = allocaOp.getLoc();
    MemRefType originalType = allocaOp.getType();
    auto convertedType = dyn_cast_or_null<LLVM::LLVMPointerType>(
        getTypeConverter()->convertType(originalType));
    auto elTy = convertMemrefElementTypeForLLVMPointer(
        originalType, *this->getTypeConverter());
    if (!convertedType || !elTy)
      return rewriter.notifyMatchFailure(loc, "unsupported memref type");

    assert(adaptor.getDynamicSizes().size() <= 1 &&
           "expected at most one dynamic size");

    Value outerSize = getOuterSize(allocaOp, adaptor, rewriter);
    rewriter.replaceOpWithNewOp<LLVM::AllocaOp>(
        allocaOp, convertedType, elTy, outerSize,
        adaptor.getAlignment().value_or(0));
    return success();
  }
};

/// Pattern for lowering heap allocations via malloc.
struct CAllocOpLowering : public AllocLikeOpLowering<memref::AllocOp> {
public:
  using AllocLikeOpLowering<memref::AllocOp>::AllocLikeOpLowering;

  LogicalResult
  matchAndRewrite(memref::AllocOp allocOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto module = allocOp->getParentOfType<ModuleOp>();
    Location loc = allocOp.getLoc();
    MemRefType originalType = allocOp.getType();
    auto convertedType = dyn_cast_or_null<LLVM::LLVMPointerType>(
        getTypeConverter()->convertType(originalType));

    if (!convertedType)
      return rewriter.notifyMatchFailure(loc, "unsupported memref type");
    if (adaptor.getAlignment() && adaptor.getAlignment().value() != 0)
      return rewriter.notifyMatchFailure(loc, "unsupported alignment");

    Value outerSize = getOuterSize(allocOp, adaptor, rewriter);
    Value totalSize = outerSize;
    if (originalType.getRank() > 1) {
      int64_t innerSizes = 1;
      for (int64_t size : originalType.getShape().drop_front())
        innerSizes *= size;
      totalSize = rewriter.createOrFold<LLVM::MulOp>(
          loc, outerSize,
          createIndexAttrConstant(rewriter, loc, rewriter.getIndexType(),
                                  innerSizes));
    }
    assert(0 && "todo alloc lower");
    Value elementSize;
    // = rewriter.create<enzymexla::TypeSizeOp>(
    //     loc, rewriter.getIndexType(),
    //     mlir::TypeAttr::get(originalType.getElementType()));
    Value size = rewriter.create<LLVM::MulOp>(loc, totalSize, elementSize);

    if (auto F = module.lookupSymbol<mlir::func::FuncOp>("malloc")) {
      Value allocated =
          rewriter.create<func::CallOp>(loc, F, size).getResult(0);
      rewriter.replaceOpWithNewOp<enzymexla::Memref2PointerOp>(
          allocOp, convertedType, allocated);
    } else {
      FailureOr<LLVM::LLVMFuncOp> mallocFunc =
          getTypeConverter()->getOptions().useGenericFunctions
              ? LLVM::lookupOrCreateGenericAllocFn(rewriter, module,
                                                   getIndexType())
              : LLVM::lookupOrCreateMallocFn(rewriter, module, getIndexType());
      if (failed(mallocFunc))
        return failure();
      Value allocated =
          rewriter.create<LLVM::CallOp>(loc, mallocFunc.value(), size)
              .getResult();
      rewriter.replaceOpWithNewOp<LLVM::BitcastOp>(allocOp, convertedType,
                                                   allocated);
    }
    return success();
  }
};

/// Pattern for lowering heap deallocations via free.
struct CDeallocOpLowering : public ConvertOpToLLVMPattern<memref::DeallocOp> {
public:
  using ConvertOpToLLVMPattern<memref::DeallocOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(memref::DeallocOp deallocOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto module = deallocOp->getParentOfType<ModuleOp>();
    if (auto F = module.lookupSymbol<mlir::func::FuncOp>("free")) {
      Value casted = rewriter.create<enzymexla::Pointer2MemrefOp>(
          deallocOp->getLoc(), MemRefType::get({-1}, rewriter.getI8Type()),
          adaptor.getMemref());
      rewriter.replaceOpWithNewOp<func::CallOp>(deallocOp, F, casted);
    } else {
      FailureOr<LLVM::LLVMFuncOp> freeFunc =
          getTypeConverter()->getOptions().useGenericFunctions
              ? LLVM::lookupOrCreateGenericFreeFn(rewriter, module)
              : LLVM::lookupOrCreateFreeFn(rewriter, module);
      if (failed(freeFunc))
        return failure();
      rewriter.replaceOpWithNewOp<LLVM::CallOp>(deallocOp, freeFunc.value(),
                                                adaptor.getMemref());
    }
    return success();
  }
};

/// Converts the given memref type into the LLVM type that can be used for a
/// global. The memref type must have all dimensions statically known. The
/// provided type converter is used to convert the elemental type.
static Type convertGlobalMemRefTypeToLLVM(MemRefType type,
                                          const TypeConverter &typeConverter) {
  if (!type.hasStaticShape() || !type.getLayout().isIdentity())
    return nullptr;

  Type convertedType = typeConverter.convertType(type.getElementType());
  if (!convertedType)
    return nullptr;

  for (int64_t size : llvm::reverse(type.getShape()))
    convertedType = LLVM::LLVMArrayType::get(convertedType, size);
  return convertedType;
}

/// Pattern for lowering global memref declarations.
struct GlobalOpLowering : public ConvertOpToLLVMPattern<memref::GlobalOp> {
public:
  using ConvertOpToLLVMPattern<memref::GlobalOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(memref::GlobalOp globalOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    MemRefType originalType = globalOp.getType();
    if (!originalType.hasStaticShape() ||
        !originalType.getLayout().isIdentity()) {
      return rewriter.notifyMatchFailure(globalOp->getLoc(),
                                         "unsupported type");
    }

    Type convertedType =
        convertGlobalMemRefTypeToLLVM(originalType, *typeConverter);
    LLVM::Linkage linkage =
        globalOp.isPublic() ? LLVM::Linkage::External : LLVM::Linkage::Private;
    if (!convertedType) {
      return rewriter.notifyMatchFailure(globalOp->getLoc(),
                                         "failed to convert memref type");
    }

    Attribute initialValue = nullptr;
    if (!globalOp.isExternal() && !globalOp.isUninitialized()) {
      auto elementsAttr = cast<ElementsAttr>(*globalOp.getInitialValue());
      initialValue = elementsAttr;

      // For scalar memrefs, the global variable created is of the element type,
      // so unpack the elements attribute to extract the value.
      if (originalType.getRank() == 0)
        initialValue = elementsAttr.getSplatValue<Attribute>();
    }

    unsigned alignment = globalOp.getAlignment() ? *globalOp.getAlignment() : 0;
    bool dso_local = globalOp->getAttr("enzymexla.cuda_device") ||
                     globalOp->getAttr("enzymexla.cuda_constant");
    bool thread_local_ = false;
    unsigned addr = originalType.getMemorySpaceAsInt();
    auto newGlobal = rewriter.replaceOpWithNewOp<LLVM::GlobalOp>(
        globalOp, convertedType, globalOp.getConstant(), linkage,
        globalOp.getSymName(), initialValue, alignment, addr, dso_local,
        thread_local_);
    if (!globalOp.isExternal() && globalOp.isUninitialized()) {
      Block *block =
          rewriter.createBlock(&newGlobal.getInitializerRegion(),
                               newGlobal.getInitializerRegion().begin());
      rewriter.setInsertionPointToStart(block);
      Value undef =
          rewriter.create<LLVM::UndefOp>(globalOp->getLoc(), convertedType);
      rewriter.create<LLVM::ReturnOp>(globalOp->getLoc(), undef);
    }
    return success();
  }
};

/// Pattern for lowering operations taking the address of a global memref.
struct GetGlobalOpLowering
    : public ConvertOpToLLVMPattern<memref::GetGlobalOp> {
public:
  using ConvertOpToLLVMPattern<memref::GetGlobalOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(memref::GetGlobalOp getGlobalOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    MemRefType originalType = getGlobalOp.getType();
    Type convertedType = getTypeConverter()->convertType(originalType);
    Value wholeAddress = rewriter.create<LLVM::AddressOfOp>(
        getGlobalOp->getLoc(), convertedType, getGlobalOp.getName());

    rewriter.replaceOp(getGlobalOp, wholeAddress);
    return success();
  }
};

/// Base class for patterns lowering memory access operations.
template <typename OpTy>
struct CLoadStoreOpLowering : public ConvertOpToLLVMPattern<OpTy> {
protected:
  using ConvertOpToLLVMPattern<OpTy>::ConvertOpToLLVMPattern;

  /// Emits the IR that computes the address of the memory being accessed.
  Value getAddress(OpTy op,
                   typename ConvertOpToLLVMPattern<OpTy>::OpAdaptor adaptor,
                   ConversionPatternRewriter &rewriter) const {
    Location loc = op.getLoc();
    MemRefType originalType = op.getMemRefType();
    auto convertedType = dyn_cast_or_null<LLVM::LLVMPointerType>(
        this->getTypeConverter()->convertType(originalType));
    if (!convertedType) {
      (void)rewriter.notifyMatchFailure(loc, "unsupported memref type");
      return nullptr;
    }

    SmallVector<LLVM::GEPArg> args = llvm::to_vector(llvm::map_range(
        adaptor.getIndices(), [](Value v) { return LLVM::GEPArg(v); }));
    auto elTy = convertMemrefElementTypeForLLVMPointer(
        originalType, *this->getTypeConverter());
    if (!elTy) {
      (void)rewriter.notifyMatchFailure(loc, "unsupported memref type");
      return nullptr;
    }
    return rewriter.create<LLVM::GEPOp>(
        loc,
        LLVM::LLVMPointerType::get(op.getContext(),
                                   originalType.getMemorySpaceAsInt()),
        elTy, adaptor.getMemref(), args);
  }
};

/// Pattern for lowering a memory load.
struct CLoadOpLowering : public CLoadStoreOpLowering<memref::LoadOp> {
public:
  using CLoadStoreOpLowering<memref::LoadOp>::CLoadStoreOpLowering;

  LogicalResult
  matchAndRewrite(memref::LoadOp loadOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value address = getAddress(loadOp, adaptor, rewriter);
    if (!address)
      return failure();

    rewriter.replaceOpWithNewOp<LLVM::LoadOp>(
        loadOp,
        typeConverter->convertType(loadOp.getMemRefType().getElementType()),
        address);
    return success();
  }
};

/// Try to match the kind of a memref.atomic_rmw to determine whether to use a
/// lowering to llvm.atomicrmw or fallback to llvm.cmpxchg.
static std::optional<LLVM::AtomicBinOp>
matchSimpleAtomicOp(memref::AtomicRMWOp atomicOp) {
  switch (atomicOp.getKind()) {
  case arith::AtomicRMWKind::addf:
    return LLVM::AtomicBinOp::fadd;
  case arith::AtomicRMWKind::addi:
    return LLVM::AtomicBinOp::add;
  case arith::AtomicRMWKind::assign:
    return LLVM::AtomicBinOp::xchg;
  case arith::AtomicRMWKind::maximumf:
    return LLVM::AtomicBinOp::fmax;
  case arith::AtomicRMWKind::maxs:
    return LLVM::AtomicBinOp::max;
  case arith::AtomicRMWKind::maxu:
    return LLVM::AtomicBinOp::umax;
  case arith::AtomicRMWKind::minimumf:
    return LLVM::AtomicBinOp::fmin;
  case arith::AtomicRMWKind::mins:
    return LLVM::AtomicBinOp::min;
  case arith::AtomicRMWKind::minu:
    return LLVM::AtomicBinOp::umin;
  case arith::AtomicRMWKind::ori:
    return LLVM::AtomicBinOp::_or;
  case arith::AtomicRMWKind::andi:
    return LLVM::AtomicBinOp::_and;
  default:
    return std::nullopt;
  }
  llvm_unreachable("Invalid AtomicRMWKind");
}

struct CAtomicRMWOpLowering : public CLoadStoreOpLowering<memref::AtomicRMWOp> {
  using CLoadStoreOpLowering<memref::AtomicRMWOp>::CLoadStoreOpLowering;

  LogicalResult
  matchAndRewrite(memref::AtomicRMWOp atomicOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto maybeKind = matchSimpleAtomicOp(atomicOp);
    if (!maybeKind)
      return failure();
    auto dataPtr = getAddress(atomicOp, adaptor, rewriter);
    if (!dataPtr)
      return failure();
    rewriter.replaceOpWithNewOp<LLVM::AtomicRMWOp>(
        atomicOp, *maybeKind, dataPtr, adaptor.getValue(),
        LLVM::AtomicOrdering::acq_rel);
    return success();
  }
};

/// Pattern for lowering a memory store.
struct CStoreOpLowering : public CLoadStoreOpLowering<memref::StoreOp> {
public:
  using CLoadStoreOpLowering<memref::StoreOp>::CLoadStoreOpLowering;

  LogicalResult
  matchAndRewrite(memref::StoreOp storeOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value address = getAddress(storeOp, adaptor, rewriter);
    if (!address)
      return failure();

    rewriter.replaceOpWithNewOp<LLVM::StoreOp>(storeOp, adaptor.getValue(),
                                               address);
    return success();
  }
};

struct CMemcpyOpLowering : public CLoadStoreOpLowering<enzymexla::MemcpyOp> {
public:
  StringRef backend;

  CMemcpyOpLowering(LLVMTypeConverter &typeConverter, StringRef backend)
      : CLoadStoreOpLowering<enzymexla::MemcpyOp>(typeConverter),
        backend(backend) {}

  LogicalResult
  matchAndRewrite(enzymexla::MemcpyOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    MemRefType dstType = op.getTarget().getType();
    auto convertedDstType = dyn_cast_or_null<LLVM::LLVMPointerType>(
        this->getTypeConverter()->convertType(dstType));
    if (!convertedDstType) {
      (void)rewriter.notifyMatchFailure(loc, "unsupported memref type");
      return failure();
    }

    MemRefType srcType = op.getSource().getType();
    auto convertedSrcType = dyn_cast_or_null<LLVM::LLVMPointerType>(
        this->getTypeConverter()->convertType(srcType));
    if (!convertedSrcType) {
      (void)rewriter.notifyMatchFailure(loc, "unsupported memref type");
      return failure();
    }
    auto elTyDst = convertMemrefElementTypeForLLVMPointer(
        dstType, *this->getTypeConverter());
    auto elTySrc = convertMemrefElementTypeForLLVMPointer(
        srcType, *this->getTypeConverter());
    if (!elTyDst) {
      (void)rewriter.notifyMatchFailure(loc, "unsupported memref type");
      return failure();
    }
    if (!elTySrc) {
      (void)rewriter.notifyMatchFailure(loc, "unsupported memref type");
      return failure();
    }

    Value dst = adaptor.getTarget();
    Value src = adaptor.getSource();

    Value size = adaptor.getSize();

    if (dstType.getMemorySpaceAsInt() == 0 &&
        srcType.getMemorySpaceAsInt() == 0) {
      rewriter.create<LLVM::MemcpyOp>(op.getLoc(), dst, src, size, false);
      rewriter.eraseOp(op);
      return success();
    }
    if (backend == "cpu") {
      dst = rewriter.create<LLVM::AddrSpaceCastOp>(
          op.getLoc(), LLVM::LLVMPointerType::get(op.getContext()), dst);
      src = rewriter.create<LLVM::AddrSpaceCastOp>(
          op.getLoc(), LLVM::LLVMPointerType::get(op.getContext()), src);
      rewriter.create<LLVM::MemcpyOp>(op.getLoc(), dst, src, size, false);
      rewriter.eraseOp(op);
      return success();
    }

    int direction = 0;
    if (dstType.getMemorySpaceAsInt() == 0 &&
        srcType.getMemorySpaceAsInt() == 0) {
      direction = 0;
    } else if (dstType.getMemorySpaceAsInt() == 1 &&
               srcType.getMemorySpaceAsInt() == 0) {
      direction = 1;
    } else if (dstType.getMemorySpaceAsInt() == 0 &&
               srcType.getMemorySpaceAsInt() == 1) {
      direction = 2;
    } else if (dstType.getMemorySpaceAsInt() == 1 &&
               srcType.getMemorySpaceAsInt() == 1) {
      direction = 3;
    } else {
      return failure();
    }

    auto moduleOp = op->getParentOfType<ModuleOp>();

    auto ptrty = LLVM::LLVMPointerType::get(op.getContext());

    SmallVector<Type> tys = {ptrty, ptrty, size.getType(),
                             rewriter.getIntegerType(32)};
    if (backend.starts_with("xla")) {
      tys.insert(tys.begin(), ptrty);
    }
    auto i32 = rewriter.getIntegerType(32);
    bool xla = backend.starts_with("xla");

    auto cudaMemcpyFn = LLVM::lookupOrCreateFn(
        rewriter, moduleOp, xla ? "reactantXLAMemcpy" : "cudaMemcpy", tys,
        xla ? (mlir::Type)LLVM::LLVMVoidType::get(rewriter.getContext())
            : (mlir::Type)i32);
    if (failed(cudaMemcpyFn))
      return failure();

    SmallVector<Value> args = {dst, src, size,
                               rewriter.create<LLVM::ConstantOp>(
                                   op.getLoc(), tys[3 + xla], direction)};
    for (int i = 0; i < 2; i++)
      if (args[i].getType() != tys[i])
        args[i] = rewriter.create<LLVM::AddrSpaceCastOp>(op.getLoc(),
                                                         tys[i + xla], args[i]);

    if (backend.starts_with("xla")) {
      auto xdata = insertXLAInitDeinit(moduleOp, backend, rewriter);
      args.insert(args.begin(), xdata);
    }

    rewriter.create<LLVM::CallOp>(op.getLoc(), cudaMemcpyFn.value(), args);
    rewriter.eraseOp(op);
    return success();
  }
};
} // namespace

/// Only retain those attributes that are not constructed by
/// `LLVMFuncOp::build`. If `filterArgAttrs` is set, also filter out argument
/// attributes.
static void filterFuncAttributes(func::FuncOp func, bool filterArgAndResAttrs,
                                 SmallVectorImpl<NamedAttribute> &result) {
  for (const NamedAttribute &attr : func->getAttrs()) {
    if (attr.getName() == SymbolTable::getSymbolAttrName() ||
        attr.getName() == func.getFunctionTypeAttrName() ||
        attr.getName() == "func.varargs" ||
        (filterArgAndResAttrs &&
         (attr.getName() == func.getArgAttrsAttrName() ||
          attr.getName() == func.getResAttrsAttrName())))
      continue;
    result.push_back(attr);
  }
}

static constexpr llvm::StringLiteral kLLVMLinkageAttrName = "llvm.linkage";

/// Convert function argument, operation and result attributes to the LLVM
/// dialect. This identifies attributes known to contain types and converts
/// those types using the converter provided. This also accounts for the calling
/// convention of packing multiple values returned from a function into an
/// anonymous struct. Adapted from upstream MLIR.
static SmallVector<NamedAttribute> convertFuncAttributes(
    func::FuncOp funcOp, const TypeConverter &typeConverter,
    const TypeConverter::SignatureConversion &signatureConversion,
    OpBuilder &rewriter) {
  // Propagate argument/result attributes to all converted arguments/result
  // obtained after converting a given original argument/result.
  SmallVector<NamedAttribute> attributes;
  filterFuncAttributes(funcOp, /*filterArgAndResAttrs=*/true, attributes);
  if (ArrayAttr resAttrDicts = funcOp.getAllResultAttrs()) {
    assert(!resAttrDicts.empty() && "expected array to be non-empty");
    auto newResAttrDicts =
        (funcOp.getNumResults() == 1)
            ? resAttrDicts
            : rewriter.getArrayAttr(rewriter.getDictionaryAttr({}));
    attributes.push_back(
        rewriter.getNamedAttr(funcOp.getResAttrsAttrName(), newResAttrDicts));
  }
  if (ArrayAttr argAttrDicts = funcOp.getAllArgAttrs()) {
    SmallVector<Attribute> newArgAttrs(funcOp.getNumArguments());
    for (unsigned i = 0, e = funcOp.getNumArguments(); i < e; ++i) {
      // Some LLVM IR attribute have a type attached to them. During FuncOp ->
      // LLVMFuncOp conversion these types may have changed. Account for that
      // change by converting attributes' types as well.
      SmallVector<NamedAttribute, 4> convertedAttrs;
      auto attrsDict = cast<DictionaryAttr>(argAttrDicts[i]);
      convertedAttrs.reserve(attrsDict.size());
      for (const NamedAttribute &attr : attrsDict) {
        const auto convert = [&](const NamedAttribute &attr) {
          return TypeAttr::get(typeConverter.convertType(
              cast<TypeAttr>(attr.getValue()).getValue()));
        };
        if (attr.getName().getValue() ==
            LLVM::LLVMDialect::getByValAttrName()) {
          convertedAttrs.push_back(rewriter.getNamedAttr(
              LLVM::LLVMDialect::getByValAttrName(), convert(attr)));
        } else if (attr.getName().getValue() ==
                   LLVM::LLVMDialect::getByRefAttrName()) {
          convertedAttrs.push_back(rewriter.getNamedAttr(
              LLVM::LLVMDialect::getByRefAttrName(), convert(attr)));
        } else if (attr.getName().getValue() ==
                   LLVM::LLVMDialect::getStructRetAttrName()) {
          convertedAttrs.push_back(rewriter.getNamedAttr(
              LLVM::LLVMDialect::getStructRetAttrName(), convert(attr)));
        } else if (attr.getName().getValue() ==
                   LLVM::LLVMDialect::getInAllocaAttrName()) {
          convertedAttrs.push_back(rewriter.getNamedAttr(
              LLVM::LLVMDialect::getInAllocaAttrName(), convert(attr)));
        } else {
          convertedAttrs.push_back(attr);
        }
      }
      auto mapping = signatureConversion.getInputMapping(i);
      assert(mapping && "unexpected deletion of function argument");
      for (size_t j = 0; j < mapping->size; ++j)
        newArgAttrs[mapping->inputNo + j] =
            DictionaryAttr::get(rewriter.getContext(), convertedAttrs);
    }
    attributes.push_back(rewriter.getNamedAttr(
        funcOp.getArgAttrsAttrName(), rewriter.getArrayAttr(newArgAttrs)));
  }
  for (const auto &pair : llvm::enumerate(attributes)) {
    if (pair.value().getName() == kLLVMLinkageAttrName) {
      attributes.erase(attributes.begin() + pair.index());
      break;
    }
  }

  return attributes;
}

/// Returns the LLVM dialect type suitable for constructing the LLVM function
/// type that has the same results as the given type. If multiple results are to
/// be returned, packs them into an anonymous LLVM dialect structure type.
static Type
convertAndPackFunctionResultType(FunctionType type,
                                 const TypeConverter &typeConverter) {
  SmallVector<Type> convertedResultTypes;
  if (failed(
          typeConverter.convertTypes(type.getResults(), convertedResultTypes)))
    return nullptr;

  if (convertedResultTypes.empty())
    return LLVM::LLVMVoidType::get(type.getContext());
  if (convertedResultTypes.size() == 1)
    return convertedResultTypes[0];
  return LLVM::LLVMStructType::getLiteral(type.getContext(),
                                          convertedResultTypes);
}

/// Attempts to convert the function type representing the signature of the
/// given function to the LLVM dialect equivalent type. On success, returns the
/// converted type and the signature conversion object that can be used to
/// update the arguments of the function's entry block.
template <typename FuncOpType>
static std::optional<
    std::pair<LLVM::LLVMFunctionType, TypeConverter::SignatureConversion>>
convertFunctionType(FuncOpType funcOp, const TypeConverter &typeConverter) {
  TypeConverter::SignatureConversion signatureConversion(
      funcOp.getNumArguments());
  for (const auto &[index, type] : llvm::enumerate(funcOp.getArgumentTypes())) {
    Type converted = typeConverter.convertType(type);
    if (!converted)
      return std::nullopt;

    signatureConversion.addInputs(index, converted);
  }

  Type resultType =
      convertAndPackFunctionResultType(funcOp.getFunctionType(), typeConverter);
  if (!resultType)
    return std::nullopt;

  auto varargsAttr = funcOp->template getAttrOfType<BoolAttr>("func.varargs");
  auto convertedType = LLVM::LLVMFunctionType::get(
      resultType, signatureConversion.getConvertedTypes(),
      varargsAttr && varargsAttr.getValue());

  return std::make_pair(convertedType, signatureConversion);
}

static constexpr const char *kGpuBinaryStorageSuffix = "_gpubin_cst";
static constexpr const char *kGpuModuleCtorSuffix = "_gpubin_ctor";
static constexpr const char *kGpuModuleDtorSuffix = "_gpubin_dtor";

// Generates an LLVM IR dialect global that contains the name of the given
// kernel function as a C string, and returns a pointer to its beginning.
// The code is essentially:
//
// llvm.global constant @kernel_name("function_name\00")
// func(...) {
//   %0 = llvm.addressof @kernel_name
//   %1 = llvm.constant (0 : index)
//   %2 = llvm.getelementptr %0[%1, %1] : !llvm<"i8*">
// }
static Value generateKernelNameConstant(StringRef moduleName, StringRef name,
                                        Location loc, OpBuilder &builder) {
  // Make sure the trailing zero is included in the constant.
  std::vector<char> kernelName(name.begin(), name.end());
  kernelName.push_back('\0');

  std::string globalName =
      std::string(llvm::formatv("{0}_{1}_kernel_name", moduleName, name));
  return LLVM::createGlobalString(
      loc, builder, globalName, StringRef(kernelName.data(), kernelName.size()),
      LLVM::Linkage::Internal);
}

static std::string getFuncStubName(StringRef moduleName, StringRef name) {
  return std::string(
      llvm::formatv("__polygeist_{0}_{1}_device_stub", moduleName, name));
};

class ConvertLaunchFuncOpToGpuRuntimeCallPattern
    : public ConvertOpToGpuRuntimeCallPattern<gpu::LaunchFuncOp> {
public:
  ConvertLaunchFuncOpToGpuRuntimeCallPattern(LLVMTypeConverter &typeConverter,
                                             StringRef gpuBinaryAnnotation,
                                             std::string gpuTarget)
      : ConvertOpToGpuRuntimeCallPattern<gpu::LaunchFuncOp>(typeConverter),
        gpuBinaryAnnotation(gpuBinaryAnnotation), gpuTarget(gpuTarget) {}

private:
  Value generateParamsArray(gpu::LaunchFuncOp launchOp, OpAdaptor adaptor,
                            OpBuilder &builder, Block *allocaBlock) const;

  LogicalResult
  matchAndRewrite(gpu::LaunchFuncOp launchOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;

  llvm::SmallString<32> gpuBinaryAnnotation;
  std::string gpuTarget;
};

class ConvertGPUModuleOp
    : public ConvertOpToGpuRuntimeCallPattern<gpu::GPUModuleOp> {
public:
  ConvertGPUModuleOp(LLVMTypeConverter &typeConverter,
                     StringRef gpuBinaryAnnotation, std::string gpuTarget)
      : ConvertOpToGpuRuntimeCallPattern<gpu::GPUModuleOp>(typeConverter),
        gpuBinaryAnnotation(gpuBinaryAnnotation), gpuTarget(gpuTarget) {}

private:
  LogicalResult
  matchAndRewrite(gpu::GPUModuleOp launchOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;

  llvm::SmallString<32> gpuBinaryAnnotation;
  std::string gpuTarget;
};

// tuple helpers
template <typename Tuple> constexpr auto pop_front(Tuple tuple) {
  static_assert(std::tuple_size<Tuple>::value > 0,
                "Cannot pop from an empty tuple");
  return std::apply([](auto, auto... rest) { return std::make_tuple(rest...); },
                    tuple);
}
template <typename Stream, class Tuple, std::size_t N> struct TuplePrinter {
  static void print(Stream &stream, const Tuple &t) {
    TuplePrinter<Stream, Tuple, N - 1>::print(stream, t);
    stream << ", " << std::get<N - 1>(t);
  }
};
template <typename Stream, class Tuple> struct TuplePrinter<Stream, Tuple, 1> {
  static void print(Stream &stream, const Tuple &t) {
    stream << std::get<0>(t);
  }
};
template <typename Stream, typename... Args,
          std::enable_if_t<sizeof...(Args) != 0, int> = 0>
void print(Stream &stream, const std::tuple<Args...> &t) {
  TuplePrinter<Stream, decltype(t), sizeof...(Args)>::print(stream, t);
}

#if 0
struct LowerGPUAlternativesOp
    : public OpRewritePattern<polygeist::AlternativesOp>,
      public GpuRuntimeCallBuilders {
  using OpRewritePattern<polygeist::AlternativesOp>::OpRewritePattern;
  const char *PATTERN = "lower-gpu-alternatives";

  LogicalResult matchAndRewrite(polygeist::AlternativesOp gao,
                                PatternRewriter &rewriter) const override {

    if (gao->getAttrOfType<StringAttr>("alternatives.type").getValue() !=
        "gpu_kernel")
      return failure();

    Location loc = gao->getLoc();
    std::string locStr =
        gao->getAttrOfType<StringAttr>("polygeist.altop.id").data();

    auto descs = gao->getAttrOfType<ArrayAttr>("alternatives.descs");

    // TODO each region in the alternatives op should containt only a single
    // block - write a verifier for that

    typedef std::tuple<Region *, int, int, int, int, int, int, int, int, int,
                       int, int, int>
        kernelInfoTy;
    std::vector<kernelInfoTy> infos;

    auto printInfos = [&](auto &strm, std::vector<kernelInfoTy> infos) {
      int i = 0;
      for (auto tup : infos) {
        strm << "polygeistKernelInfo: " << locStr << "," << i << "," << descs[i]
             << ",";
        auto _tup = pop_front(tup);
        print(strm, _tup);
        strm << "\n";
        i++;
      }
    };

    auto gatherInfos = [&]() {
      typedef std::tuple<int, int, int, int, int, int> kernelLLVMInfoTy;
      auto gatherLLVMInfos = [&](Operation *gpuFunc) -> kernelLLVMInfoTy {
        int ops = 0, floatOps = 0, intOps = 0, loads = 0, stores = 0,
            branches = 0;

        // TODO This should use the GPU data layout and not the Host one
        DataLayout DLI(gao->getParentOfType<ModuleOp>());
        gpuFunc->walk([&](Operation *op) {
          ops++;
          if (isa<LLVM::BrOp>(op)) {
            branches++;
          } else if (isa<LLVM::FAddOp>(op) || isa<LLVM::FMulOp>(op) ||
                     isa<LLVM::FDivOp>(op) || isa<LLVM::FSubOp>(op) ||
                     isa<LLVM::FRemOp>(op)) {
            int width =
                dyn_cast<FloatType>(op->getOperand(0).getType()).getWidth();
            // TODO these are pretty random atm
            if (width == 16) {
              floatOps++;
            } else if (width == 32) {
              floatOps += 2;
            } else if (width == 64) {
              floatOps += 4;
            }
          } else if (isa<LLVM::AddOp>(op) || isa<LLVM::SubOp>(op) ||
                     isa<LLVM::MulOp>(op) || isa<LLVM::UDivOp>(op) ||
                     isa<LLVM::SDivOp>(op)) {
            intOps++;
          } else if (auto load = dyn_cast<LLVM::LoadOp>(op)) {
            int bytes = DLI.getTypeSize(load.getRes().getType());
            loads += bytes;
          } else if (auto store = dyn_cast<LLVM::StoreOp>(op)) {
            int bytes = DLI.getTypeSize(store->getOperand(0).getType());
            stores += bytes;
          }
        });
        return {
            ops, floatOps, intOps, loads, stores, branches,
        };
      };

#if POLYGEIST_ENABLE_CUDA
      if (gpuTarget == "cuda") {
        char cuErrorBuffer[4096] = {0};

        // TODO implement a version that does this at runtime for when we dont
        // have block sizes or shared mem

        RETURN_ON_CUDA_ERROR(cuInit(0));
        // For whatever reason we need a device context
        CUdevice device;
        RETURN_ON_CUDA_ERROR(cuDeviceGet(&device, 0));
        CUcontext context;
        RETURN_ON_CUDA_ERROR(cuCtxCreate(&context, 0, device));

        for (auto &region : gao->getRegions()) {
          gpu::LaunchFuncOp launchOp = nullptr;
          region.walk([&](gpu::LaunchFuncOp l) {
            launchOp = l;
            return WalkResult::interrupt();
          });
          assert(launchOp);

          auto gpuFunc = launchOp->getParentOfType<ModuleOp>().lookupSymbol(
              launchOp.getKernel());
          assert(gpuFunc);
          auto gpuModule = gpuFunc->getParentOfType<gpu::GPUModuleOp>();
          assert(gpuModule);
          const char *blob =
              gpuModule->getAttrOfType<StringAttr>(gpuBinaryAnnotation).data();

          CUmodule cuModule;
          CUfunction cuFunction;
          RETURN_ON_CUDA_ERROR(cuModuleLoadData(&cuModule, blob));
          RETURN_ON_CUDA_ERROR(cuModuleGetFunction(
              &cuFunction, cuModule, launchOp.getKernelName().data()));

          int maxThreadsPerBlock, sharedMemSize, constMemSize,
              /* stack frame size */ localMemSize, numRegs;
          // TODO we dont seem to be able to get spilled stores/loads count from
          // here but ptxas outputs it? should we parse the ptxas output and add
          // an attribute for those values
          RETURN_ON_CUDA_ERROR(cuFuncGetAttribute(
              &maxThreadsPerBlock, CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
              cuFunction));
          RETURN_ON_CUDA_ERROR(cuFuncGetAttribute(
              &sharedMemSize, CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, cuFunction));
          RETURN_ON_CUDA_ERROR(cuFuncGetAttribute(
              &constMemSize, CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES, cuFunction));
          RETURN_ON_CUDA_ERROR(cuFuncGetAttribute(
              &localMemSize, CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, cuFunction));
          RETURN_ON_CUDA_ERROR(cuFuncGetAttribute(
              &numRegs, CU_FUNC_ATTRIBUTE_NUM_REGS, cuFunction));

          int blockSize = 1;
          gpu::KernelDim3 blockDims = launchOp.getBlockSizeOperandValues();
          for (auto dim : {blockDims.x, blockDims.y, blockDims.z}) {
            if (auto cstint = dyn_cast_or_null<arith::ConstantIntOp>(
                    dim.getDefiningOp())) {
              blockSize *= cstint.value();
            } else if (auto cstindex = dyn_cast_or_null<arith::ConstantIndexOp>(
                           dim.getDefiningOp())) {
              blockSize *= cstindex.value();
            } else {
              blockSize = 0;
              break;
            }
          }

          // in the current state, only kernels with no shared memory should use
          // the alternatives op, thus assume 0 TODO check it
          size_t dynamicSharedMemSize = 0;

          int occupancyNumBlocks;
          if (blockSize > 0) {
            RETURN_ON_CUDA_ERROR(cuOccupancyMaxActiveBlocksPerMultiprocessor(
                &occupancyNumBlocks, cuFunction, blockSize,
                dynamicSharedMemSize));
          } else {
            occupancyNumBlocks = 0;
          }

          RETURN_ON_CUDA_ERROR(cuModuleUnload(cuModule));

          auto kernelLLVMInfo = gatherLLVMInfos(gpuFunc);

          assert(maxThreadsPerBlock >= blockSize);
          // int activeThreads = occupancyNumBlocks * blockSize;
          infos.push_back(std::tuple_cat(
              std::make_tuple(&region),
              std::make_tuple(localMemSize, occupancyNumBlocks, numRegs,
                              blockSize, sharedMemSize, constMemSize),
              kernelLLVMInfo));
        }
      }
#endif
#if POLYGEIST_ENABLE_ROCM
      if (gpuTarget == "rocm") {
        char hipErrorBuffer[4096] = {0};

        // TODO implement a version that does this at runtime for when we dont
        // have block sizes or shared mem

        RETURN_ON_HIP_ERROR(hipInit(0));
        // For whatever reason we need a device context
        hipDevice_t device;
        RETURN_ON_HIP_ERROR(hipDeviceGet(&device, 0));

        for (auto &region : gao->getRegions()) {
          gpu::LaunchFuncOp launchOp = nullptr;
          region.walk([&](gpu::LaunchFuncOp l) {
            launchOp = l;
            return WalkResult::interrupt();
          });
          assert(launchOp);

          auto gpuFunc = launchOp->getParentOfType<ModuleOp>().lookupSymbol(
              launchOp.getKernel());
          assert(gpuFunc);
          auto gpuModule = gpuFunc->getParentOfType<gpu::GPUModuleOp>();
          assert(gpuModule);
          const char *blob =
              gpuModule->getAttrOfType<StringAttr>(gpuBinaryAnnotation).data();

          hipModule_t hipModule;
          hipFunction_t hipFunction;
          RETURN_ON_HIP_ERROR(hipModuleLoadData(&hipModule, blob));
          RETURN_ON_HIP_ERROR(hipModuleGetFunction(
              &hipFunction, hipModule, launchOp.getKernelName().data()));

          int maxThreadsPerBlock, sharedMemSize, constMemSize,
              /* stack frame size */ localMemSize, numRegs;
          // TODO we dont seem to be able to get spilled stores/loads count from
          // here but ptxas outputs it? should we parse the ptxas output and add
          // an attribute for those values
          RETURN_ON_HIP_ERROR(hipFuncGetAttribute(
              &maxThreadsPerBlock, HIP_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
              hipFunction));
          RETURN_ON_HIP_ERROR(hipFuncGetAttribute(
              &sharedMemSize, HIP_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES,
              hipFunction));
          RETURN_ON_HIP_ERROR(hipFuncGetAttribute(
              &constMemSize, HIP_FUNC_ATTRIBUTE_CONST_SIZE_BYTES, hipFunction));
          RETURN_ON_HIP_ERROR(hipFuncGetAttribute(
              &localMemSize, HIP_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, hipFunction));
          RETURN_ON_HIP_ERROR(hipFuncGetAttribute(
              &numRegs, HIP_FUNC_ATTRIBUTE_NUM_REGS, hipFunction));

          int blockSize = 1;
          gpu::KernelDim3 blockDims = launchOp.getBlockSizeOperandValues();
          for (auto dim : {blockDims.x, blockDims.y, blockDims.z}) {
            if (auto cstint = dyn_cast_or_null<arith::ConstantIntOp>(
                    dim.getDefiningOp())) {
              blockSize *= cstint.value();
            } else if (auto cstindex = dyn_cast_or_null<arith::ConstantIndexOp>(
                           dim.getDefiningOp())) {
              blockSize *= cstindex.value();
            } else {
              blockSize = 0;
              break;
            }
          }

          // in the current state, only kernels with no shared memory should use
          // the alternatives op, thus assume 0 TODO check it
          size_t dynamicSharedMemSize = 0;

          int occupancyNumBlocks;
          if (blockSize > 0) {
            auto succeeded =
                [&]() {
                  RETURN_ON_HIP_ERROR(
                      hipOccupancyMaxActiveBlocksPerMultiprocessor(
                          &occupancyNumBlocks, hipFunction, blockSize,
                          dynamicSharedMemSize));
                  return success();
                }()
                    .succeeded();

            if (!succeeded) {
              llvm::errs() << "Why does this fail with block size " << blockSize
                           << " and dynamic shared mem size "
                           << dynamicSharedMemSize << " \n";
              occupancyNumBlocks = 0;
            }
          } else {
            occupancyNumBlocks = 0;
          }

          RETURN_ON_HIP_ERROR(hipModuleUnload(hipModule));

          auto kernelLLVMInfo = gatherLLVMInfos(gpuFunc);

          assert(maxThreadsPerBlock >= blockSize);
          // int activeThreads = occupancyNumBlocks * blockSize;
          infos.push_back(std::tuple_cat(
              std::make_tuple(&region),
              std::make_tuple(localMemSize, occupancyNumBlocks, numRegs,
                              blockSize, sharedMemSize, constMemSize),
              kernelLLVMInfo));
        }
      }
#endif
      return success();
    };

    auto sortInfos = [&]() {
      auto getCost = [](auto a) -> double {
        std::vector<float> coefficients = {4, -2, -0.1, -0.01};
        return coefficients[0] * std::get<0>(a) +
               coefficients[1] * std::get<1>(a) +
               coefficients[2] * std::get<2>(a) +
               coefficients[3] * std::get<3>(a) + 0 * std::get<4>(a) +
               0 * std::get<5>(a);
      };
      std::stable_sort(infos.begin(), infos.end(), [&](auto a, auto b) {
        auto _a = pop_front(a);
        auto _b = pop_front(b);
        return getCost(_a) < getCost(_b);
      });
    };

    bool shouldPrintInfo = getenv("POLYGEIST_GPU_ALTERNATIVES_PRINT_INFO");
    if (shouldPrintInfo || PolygeistAlternativesMode == PAM_Static) {
      if (gatherInfos().failed())
        return failure();
      LLVM_DEBUG(DBGS() << "GPU Alternatives theoretical infos unsorted:\n");
      LLVM_DEBUG(printInfos(DBGS(), infos));
    }
    if (shouldPrintInfo)
      printInfos(llvm::errs(), infos);

    if (PolygeistAlternativesMode == PAM_Static) {
      Block *block = nullptr;
      sortInfos();
      LLVM_DEBUG(DBGS() << "GPU Alternatives theoretical infos sorted:\n");
      LLVM_DEBUG(printInfos(DBGS(), infos));
      LLVM_DEBUG(DBGS() << "Choosing top option\n");

      block = &*gao->getRegions()[0].begin();
      if (!infos.empty())
        block = &*std::get<0>(infos[0])->begin();

      rewriter.eraseOp(block->getTerminator());
      rewriter.inlineBlockBefore(block, gao);
      rewriter.eraseOp(gao);

      return success();

    } else if (PolygeistAlternativesMode == PAM_PGO_Profile) {
      rewriter.setInsertionPoint(gao);
      static int num = 0;
      // Append `\0` to follow C style string given that
      // LLVM::createGlobalString() won't handle this directly for us.
      SmallString<16> nullTermLocStr(locStr.begin(), locStr.end());
      nullTermLocStr.push_back('\0');
      auto kernelId = LLVM::createGlobalString(
          loc, rewriter, std::string("kernelId.") + std::to_string(num++),
          nullTermLocStr, LLVM::Linkage::Internal, /*opaquePointers*/ true);
      auto totalAlternatives = rewriter.create<LLVM::ConstantOp>(
          loc, llvmInt32Type, gao->getNumRegions());
      auto alternative =
          rtPGOGetAlternativeCallBuilder
              .create(loc, rewriter, {kernelId, totalAlternatives})
              ->getResult(0);

      int i = 0;
      for (auto &region : gao->getRegions()) {
        auto cmpOp = rewriter.create<arith::CmpIOp>(
            loc, arith::CmpIPredicate::eq, alternative,
            rewriter.create<arith::ConstantIntOp>(loc, i, 32));
        auto ifOp = rewriter.create<scf::IfOp>(loc, cmpOp, /* hasElse */ true);
        auto block = &region.front();
        rewriter.eraseOp(block->getTerminator());
        rewriter.inlineBlockBefore(
            block, ifOp.getThenRegion().front().getTerminator());

        // Timing
        rewriter.setInsertionPointToStart(&ifOp.getThenRegion().front());
        rtPGOStartCallBuilder.create(loc, rewriter,
                                     {kernelId, totalAlternatives});
        rewriter.setInsertionPoint(
            ifOp.getThenRegion().front().getTerminator());
        rtPGOEndCallBuilder.create(loc, rewriter,
                                   {kernelId, totalAlternatives});

        rewriter.setInsertionPointToStart(&ifOp.getElseRegion().front());
        i++;
      }

      rewriter.eraseOp(gao);
      return success();
    } else if (PolygeistAlternativesMode == PAM_PGO_Opt) {
      std::string dirname = []() {
        if (char *d = getenv(POLYGEIST_PGO_DATA_DIR_ENV_VAR)) {
          return std::string(d);
        } else {
          return std::string(POLYGEIST_PGO_DEFAULT_DATA_DIR);
        }
      }();
      // TODO error handling
      std::ifstream ifile;
      int numAlternatives = gao->getNumRegions();
      std::vector<std::vector<double>> timings;
      for (int i = 0; i < numAlternatives; i++) {
        timings.push_back({});
      }
      ifile.open(std::string(dirname) + "/" + locStr, std::ios::in);
      while (ifile) {
        int alt;
        double time;
        ifile >> alt >> time;
        if (alt >= 0 && alt < numAlternatives) {
          timings[alt].push_back(time);
        } else {
          llvm::errs() << "Invalid alternative data";
          assert(0);
        }
      }
      std::vector<double> avgs;
      for (int i = 0; i < numAlternatives; i++) {
        if (timings[i].size() == 0) {
          llvm::errs() << "No data for alternative " << i << "," << descs[i]
                       << " of " << locStr << "\n";
          assert(0);
          avgs.push_back(std::numeric_limits<double>::infinity());
        } else {
          // TODO might get some round off errors here, maybe use a better alg
          // or median
          avgs.push_back(
              std::accumulate(timings[i].begin(), timings[i].end(), 0.0f) /
              timings[i].size());
          llvm::errs() << "Alternative " << i << "," << descs[i] << " is "
                       << avgs[i] << "\n";
        }
      }

      int bestAlt = std::distance(avgs.begin(),
                                  std::min_element(avgs.begin(), avgs.end()));
      llvm::errs() << "Picking " << bestAlt << "," << descs[bestAlt] << "\n";

      auto block = &*gao->getRegions()[bestAlt].begin();

      rewriter.eraseOp(block->getTerminator());
      rewriter.inlineBlockBefore(block, gao);
      rewriter.eraseOp(gao);

      return success();
    } else {
      llvm_unreachable("Invalid enum");
    }
  }

  LowerGPUAlternativesOp(MLIRContext *context, LLVMTypeConverter &typeConverter,
                         StringRef gpuBinaryAnnotation, StringRef gpuTarget)
      : OpRewritePattern<polygeist::AlternativesOp>(context),
        GpuRuntimeCallBuilders(context, typeConverter),
        gpuBinaryAnnotation(gpuBinaryAnnotation), gpuTarget(gpuTarget) {}

  llvm::SmallString<32> gpuBinaryAnnotation;
  llvm::SmallString<4> gpuTarget;
};
#endif

// Creates a struct containing all kernel parameters on the stack and returns
// an array of type-erased pointers to the fields of the struct. The array can
// then be passed to the CUDA / ROCm (HIP) kernel launch calls.
// The generated code is essentially as follows:
//
// %struct = alloca(sizeof(struct { Parameters... }))
// %array = alloca(NumParameters * sizeof(void *))
// for (i : [0, NumParameters))
//   %fieldPtr = llvm.getelementptr %struct[0, i]
//   llvm.store parameters[i], %fieldPtr
//   %elementPtr = llvm.getelementptr %array[i]
//   llvm.store %fieldPtr, %elementPtr
// return %array
Value ConvertLaunchFuncOpToGpuRuntimeCallPattern::generateParamsArray(
    gpu::LaunchFuncOp launchOp, OpAdaptor adaptor, OpBuilder &builder,
    Block *allocaBlock) const {
  auto loc = launchOp.getLoc();
  auto numKernelOperands = launchOp.getNumKernelOperands();
  SmallVector<Value, 4> arguments =
      adaptor.getOperands().take_back(numKernelOperands);
  auto numArguments = arguments.size();
  SmallVector<Type, 4> argumentTypes;
  argumentTypes.reserve(numArguments);
  for (auto argument : arguments)
    argumentTypes.push_back(argument.getType());
  auto structType = LLVM::LLVMStructType::getNewIdentified(context, StringRef(),
                                                           argumentTypes);
  Value structPtr, arrayPtr;
  {
    PatternRewriter::InsertionGuard B(builder);
    builder.setInsertionPointToStart(allocaBlock);
    auto one = builder.create<LLVM::ConstantOp>(loc, llvmInt32Type, 1);
    structPtr = builder.create<LLVM::AllocaOp>(
        loc, LLVM::LLVMPointerType::get(builder.getContext()), structType, one,
        /*alignment=*/0);
    auto arraySize =
        builder.create<LLVM::ConstantOp>(loc, llvmInt32Type, numArguments);
    arrayPtr = builder.create<LLVM::AllocaOp>(loc, llvmPointerPointerType,
                                              llvmPointerType, arraySize,
                                              /*alignment=*/0);
  }
  for (const auto &en : llvm::enumerate(arguments)) {
    auto fieldPtr = builder.create<LLVM::GEPOp>(
        loc, LLVM::LLVMPointerType::get(builder.getContext()), structType,
        structPtr, ArrayRef<LLVM::GEPArg>{0, en.index()});
    builder.create<LLVM::StoreOp>(loc, en.value(), fieldPtr);
    auto elementPtr = builder.create<LLVM::GEPOp>(
        loc, llvmPointerType, llvmPointerPointerType, arrayPtr,
        ArrayRef<LLVM::GEPArg>{en.index()});
    auto casted =
        builder.create<LLVM::BitcastOp>(loc, llvmPointerType, fieldPtr);
    builder.create<LLVM::StoreOp>(loc, casted, elementPtr);
  }
  return arrayPtr;
}

// Returns whether all operands are of LLVM type.
static LogicalResult areAllLLVMTypes(Operation *op, ValueRange operands,
                                     ConversionPatternRewriter &rewriter) {
  if (!llvm::all_of(operands, [](Value value) {
        return LLVM::isCompatibleType(value.getType());
      }))
    return rewriter.notifyMatchFailure(
        op, "Cannot convert if operands aren't of LLVM type.");
  return success();
}

LogicalResult
ConvertGPUModuleOp::matchAndRewrite(gpu::GPUModuleOp kernelModule,
                                    OpAdaptor adaptor,
                                    ConversionPatternRewriter &rewriter) const {

  if (kernelModule->hasAttr("polygeist_stubs"))
    return failure();

  ModuleOp moduleOp = kernelModule->getParentOfType<ModuleOp>();

  auto loc = kernelModule.getLoc();
  rewriter.modifyOpInPlace(kernelModule, [&]() {
    kernelModule->setAttr("polygeist_stubs", rewriter.getUnitAttr());
  });
  // TODO is it okay to be using OpBuilder's in op rewriter?
  // OpBuilder moduleBuilder(moduleOp.getBodyRegion());
  SmallString<128> ctorNameBuffer(kernelModule.getName());
  ctorNameBuffer.append(kGpuModuleCtorSuffix);
  SmallString<128> dtorNameBuffer(kernelModule.getName());
  dtorNameBuffer.append(kGpuModuleDtorSuffix);
  LLVM::LLVMFuncOp ctor;
  LLVM::LLVMFuncOp dtor;
  {
    {
      PatternRewriter::InsertionGuard B(rewriter);
      rewriter.setInsertionPointToEnd(moduleOp.getBody());
      ctor = rewriter.create<LLVM::LLVMFuncOp>(
          loc, ctorNameBuffer,
          LLVM::LLVMFunctionType::get(
              LLVM::LLVMVoidType::get(moduleOp.getContext()), {}),
          LLVM::Linkage::Private);
      dtor = rewriter.create<LLVM::LLVMFuncOp>(
          loc, dtorNameBuffer,
          LLVM::LLVMFunctionType::get(
              LLVM::LLVMVoidType::get(moduleOp.getContext()), {}),
          LLVM::Linkage::Private);
    }
    /*
  auto binaryAttr =
      kernelModule->getAttrOfType<StringAttr>(gpuBinaryAnnotation);
  if (!binaryAttr) {
    kernelModule.emitOpError()
        << "missing " << gpuBinaryAnnotation << " attribute";
    return failure();
  }
*/

    auto moduleName = kernelModule.getName();
    LLVM::GlobalOp moduleGlobal = nullptr;

    {
      PatternRewriter::InsertionGuard B(rewriter);
      OpBuilder &ctorBuilder = rewriter;
      ctorBuilder.setInsertionPointToEnd(ctor.addEntryBlock(ctorBuilder));

      SmallString<128> nameBuffer(kernelModule.getName());
      nameBuffer.append(kGpuBinaryStorageSuffix);

      const char *fatbinConstantName;
      const char *fatbinSectionName;
      const char *moduleIDSectionName;
      StringRef moduleIDPrefix;
      unsigned fatMagic;
      constexpr unsigned CudaFatMagic = 0x466243b1;
      constexpr unsigned HIPFatMagic = 0x48495046; // "HIPF"
      if (gpuTarget == "cuda") {
        fatbinConstantName = // CGM.getTriple().isMacOSX() ?
                             // "__NV_CUDA,__nv_fatbin" :
            ".nv_fatbin";
        // NVIDIA's cuobjdump looks for fatbins in this section.
        fatbinSectionName = // CGM.getTriple().isMacOSX() ?
                            // "__NV_CUDA,__fatbin"
                            // :
            ".nvFatBinSegment";
        moduleIDSectionName = // CGM.getTriple().isMacOSX() ?
                              // "__NV_CUDA,__nv_module_id" :
            "__nv_module_id";
        moduleIDPrefix = "__nv_";
        fatMagic = CudaFatMagic;
      } else {
        fatbinConstantName = ".hip_fatbin";
        fatbinSectionName = ".hipFatBinSegment";
        moduleIDSectionName = "__hip_module_id";
        moduleIDPrefix = "__hip_";
        fatMagic = HIPFatMagic;
      }
      (void)fatbinConstantName;
      (void)moduleIDSectionName;

      // Register modules and functions like clang
      // (clang/CodeGen/CGCUDANV.cpp)

      // Create and initialize the fatbin wrapper struct
      auto fatBinWrapperType = mlir::LLVM::LLVMStructType::getLiteral(
          moduleOp->getContext(),
          {llvmInt32Type, llvmInt32Type, llvmPointerType, llvmPointerType});
      LLVM::GlobalOp fatBinWrapper;

      {
        PatternRewriter::InsertionGuard B(rewriter);
        rewriter.setInsertionPointToEnd(moduleOp.getBody());
        fatBinWrapper = rewriter.create<LLVM::GlobalOp>(
            loc, fatBinWrapperType, /*constant*/ true, LLVM::Linkage::Internal,
            std::string(
                llvm::formatv("__polygeist_{0}_fatbin_wrapper", moduleName)),
            /* initValue */ mlir::Attribute(),
            /* alignment */ 8, /* addrSpace */ 0);
        fatBinWrapper.setSectionAttr(rewriter.getStringAttr(fatbinSectionName));
      }

      OpBuilder globalBuilder(moduleOp->getContext());
      fatBinWrapper.getRegion().push_back(new Block);
      globalBuilder.setInsertionPointToStart(fatBinWrapper.getBody());
      auto fatbinMagicVal =
          globalBuilder.create<LLVM::ConstantOp>(loc, llvmInt32Type, fatMagic);
      auto fatbinVersionVal =
          globalBuilder.create<LLVM::ConstantOp>(loc, llvmInt32Type, 1);
      auto nullPtr = globalBuilder.create<LLVM::ZeroOp>(loc, llvmPointerType);
      Value constructedStruct =
          globalBuilder.create<LLVM::UndefOp>(loc, fatBinWrapperType);
      {
        int i = 0;
        constructedStruct = globalBuilder.create<LLVM::InsertValueOp>(
            loc, fatBinWrapperType, constructedStruct, fatbinMagicVal,
            globalBuilder.getDenseI64ArrayAttr(i++));
        constructedStruct = globalBuilder.create<LLVM::InsertValueOp>(
            loc, fatBinWrapperType, constructedStruct, fatbinVersionVal,
            globalBuilder.getDenseI64ArrayAttr(i++));
        // TODO do we need to specify the section name here...?
        // data.setSectionAttr(moduleBuilder.getStringAttr(fatbinSectionName));
        Value data = LLVM::createGlobalString(
            loc, globalBuilder, nameBuffer.str(), "binaryAttr",
            // loc, globalBuilder, nameBuffer.str(), binaryAttr.getValue(),
            LLVM::Linkage::Internal);
        constructedStruct = globalBuilder.create<LLVM::InsertValueOp>(
            loc, fatBinWrapperType, constructedStruct, data,
            globalBuilder.getDenseI64ArrayAttr(i++));
        constructedStruct = globalBuilder.create<LLVM::InsertValueOp>(
            loc, fatBinWrapperType, constructedStruct, nullPtr,
            globalBuilder.getDenseI64ArrayAttr(i++));
      }
      globalBuilder.create<LLVM::ReturnOp>(loc, constructedStruct);

      auto addressOfWrapper =
          ctorBuilder.create<LLVM::AddressOfOp>(loc, fatBinWrapper);
      auto bitcastOfWrapper = ctorBuilder.create<LLVM::AddrSpaceCastOp>(
          loc, llvmPointerType, addressOfWrapper);

      auto cudaRegisterFatbinFn =
          LLVM::lookupOrCreateFn(rewriter, moduleOp, "__cudaRegisterFatBinary",
                                 llvmPointerType, llvmPointerType);
      if (failed(cudaRegisterFatbinFn)) {
        llvm::errs() << " cudamalloc already exists with different types\n";
        return failure();
      }

      auto module = rewriter.create<LLVM::CallOp>(
          loc, cudaRegisterFatbinFn.value(), ValueRange(bitcastOfWrapper));

      auto moduleGlobalName =
          std::string(llvm::formatv("polygeist_{0}_module_ptr", moduleName));
      {
        PatternRewriter::InsertionGuard B(rewriter);
        rewriter.setInsertionPointToEnd(moduleOp.getBody());
        moduleGlobal = rewriter.create<LLVM::GlobalOp>(
            loc, llvmPointerPointerType, /* isConstant */ false,
            LLVM::Linkage::Internal, moduleGlobalName,
            /* initValue */ mlir::Attribute(),
            /* alignment */ 8, /* addrSpace */ 0);
      }
      auto aoo = ctorBuilder.create<LLVM::AddressOfOp>(loc, moduleGlobal);
      ctorBuilder.create<LLVM::StoreOp>(loc, module->getResult(0),
                                        aoo->getResult(0));
      for (Operation &op : kernelModule->getRegion(0).front()) {
        if (auto f = dyn_cast<FunctionOpInterface>(op)) {
          if (!f->getAttr("gpu.kernel"))
            continue;
          auto kernelName = generateKernelNameConstant(
              kernelModule.getName(), f.getName(), loc, ctorBuilder);

          auto nullPtr = ctorBuilder.create<LLVM::ZeroOp>(loc, llvmPointerType);
          // TODO second param should be ptr to the the original function stub
          // here like clang does it: e.g. kernel_name_device_stub
          //
          // TODO We should probably always generate the original kernel as
          // well and register it too (in addition to the lowered to parallel
          // and re-outlined version that we generate) in case the pointer to
          // the stub is captured somewhere and it is called through
          // cudaLaunchKernel
          LLVM::LLVMFuncOp stub;
          {
            PatternRewriter::InsertionGuard B(rewriter);
            rewriter.setInsertionPointToEnd(moduleOp.getBody());
            stub = rewriter.create<LLVM::LLVMFuncOp>(
                loc, getFuncStubName(moduleName, f.getName()),
                LLVM::LLVMFunctionType::get(llvmVoidType, {}));
          }
          {
            OpBuilder::InsertionGuard guard(rewriter);
            rewriter.setInsertionPointToEnd(stub.addEntryBlock(rewriter));
            rewriter.create<LLVM::ReturnOp>(loc, ValueRange());
          }
          auto aoo = ctorBuilder.create<LLVM::AddressOfOp>(loc, stub);
          auto bitcast = ctorBuilder.create<LLVM::AddrSpaceCastOp>(
              loc, llvmPointerType, aoo);

          Type tys[] = {llvmPointerType, llvmPointerType, llvmPointerType,
                        llvmPointerType, llvmInt32Type,   llvmPointerType,
                        llvmPointerType, llvmPointerType, llvmPointerType,
                        llvmPointerType};
          auto cudaRegisterFn = LLVM::lookupOrCreateFn(
              rewriter, moduleOp, "__cudaRegisterFunction", tys, llvmInt32Type);
          if (failed(cudaRegisterFn)) {
            llvm::errs() << " cudamalloc already exists with different types\n";
            return failure();
          }
          Value args[] = {
              module.getResult(),
              bitcast,
              kernelName,
              kernelName,
              ctorBuilder.create<LLVM::ConstantOp>(loc, llvmInt32Type, -1),
              nullPtr,
              nullPtr,
              nullPtr,
              nullPtr,
              nullPtr};

          rewriter.create<LLVM::CallOp>(loc, cudaRegisterFn.value(), args);
        } else if (LLVM::GlobalOp g = dyn_cast<LLVM::GlobalOp>(op)) {
          int addrSpace = g.getAddrSpace();
          if (addrSpace != 1 /* device */ && addrSpace != 4 /* constant */)
            continue;
          auto symbolName = [&]() {
            auto name = g.getName();
            std::vector<char> sname(name.begin(), name.end());
            sname.push_back('\0');

            std::string globalName = std::string(llvm::formatv(
                "__polygeist_{0}_{1}_global_name", moduleName, name));

            return LLVM::createGlobalString(
                loc, ctorBuilder, globalName,
                StringRef(sname.data(), sname.size()), LLVM::Linkage::Internal);
          }();
          // TODO could this be a memref global op?
          auto stub = moduleOp.lookupSymbol<LLVM::GlobalOp>(g.getName());
          assert(stub);
          auto aoo = ctorBuilder.create<LLVM::AddressOfOp>(loc, stub);
          auto bitcast = ctorBuilder.create<LLVM::AddrSpaceCastOp>(
              loc, llvmPointerType, aoo);
          auto globalTy = stub.getGlobalType();
          // TODO This should actually be the GPUModuleOp's data layout I
          // believe, there were problems with assigning the data layout to
          // the gpumodule because MLIR didnt like the nested data layout, and
          // that's why it doesnt have its own, try to fix that or find a way
          // to pass the GPU DL in here
          DataLayout DLI(moduleOp);
          auto size = DLI.getTypeSize(globalTy);
          rtRegisterVarCallBuilder.create(
              loc, ctorBuilder,
              {module.getResult(), bitcast, symbolName, symbolName,
               /*isExtern*/
               ctorBuilder.create<LLVM::ConstantOp>(loc, llvmInt32Type,
                                                    /* TODO */ 0),
               /*varSize*/
               ctorBuilder.create<LLVM::ConstantOp>(loc, llvmIntPtrType, size),
               /*isConstant*/
               ctorBuilder.create<LLVM::ConstantOp>(loc, llvmInt32Type,
                                                    /* TODO */ 0),
               /* just a 0? */
               ctorBuilder.create<LLVM::ConstantOp>(loc, llvmInt32Type, 0)});
        }
      }
      // TODO this has to happen only for some CUDA versions
      if (gpuTarget == "cuda") {
        auto cudaRegisterFatbinFn = LLVM::lookupOrCreateFn(
            rewriter, moduleOp, "__cudaRegisterFatBinaryEnd", llvmPointerType,
            llvmVoidType);
        if (failed(cudaRegisterFatbinFn)) {
          llvm::errs() << " cudamalloc already exists with different types\n";
          return failure();
        }

        rewriter.create<LLVM::CallOp>(loc, cudaRegisterFatbinFn.value(),
                                      ValueRange(module->getResult(0)));
      }
      ctorBuilder.create<LLVM::ReturnOp>(loc, ValueRange());
    }
    auto ctorSymbol = FlatSymbolRefAttr::get(ctor);
    {
      PatternRewriter::InsertionGuard B(rewriter);
      rewriter.setInsertionPointToEnd(moduleOp.getBody());
      rewriter.create<LLVM::GlobalCtorsOp>(
          loc, rewriter.getArrayAttr({std::move(ctorSymbol)}),
          rewriter.getI32ArrayAttr({65535}),
          rewriter.getArrayAttr({LLVM::ZeroAttr::get(rewriter.getContext())}));
    }
    {
      PatternRewriter::InsertionGuard B(rewriter);
      OpBuilder &dtorBuilder = rewriter;
      dtorBuilder.setInsertionPointToEnd(dtor.addEntryBlock(dtorBuilder));
      auto aoo = dtorBuilder.create<LLVM::AddressOfOp>(loc, moduleGlobal);
      auto module = dtorBuilder.create<LLVM::LoadOp>(
          loc, llvmPointerPointerType, aoo->getResult(0));

      auto cudaUnRegisterFatbinFn = LLVM::lookupOrCreateFn(
          rewriter, moduleOp, "__cudaUnregisterFatBinary", llvmPointerType,
          llvmVoidType);
      if (failed(cudaUnRegisterFatbinFn)) {
        llvm::errs() << " cudamalloc already exists with different types\n";
        return failure();
      }

      rewriter.create<LLVM::CallOp>(loc, cudaUnRegisterFatbinFn.value(),
                                    ValueRange(module));
      dtorBuilder.create<LLVM::ReturnOp>(loc, ValueRange());
      auto dtorSymbol = FlatSymbolRefAttr::get(dtor);
      {
        PatternRewriter::InsertionGuard B(rewriter);
        rewriter.setInsertionPointToEnd(moduleOp.getBody());
        Attribute attrs[] = {LLVM::ZeroAttr::get(rewriter.getContext())};
        rewriter.create<LLVM::GlobalDtorsOp>(
            loc, rewriter.getArrayAttr({std::move(dtorSymbol)}),
            rewriter.getI32ArrayAttr({65535}), rewriter.getArrayAttr(attrs));
      }
    }
  }
  return success();
}

// Emits LLVM IR to launch a kernel function. Expects the module that contains
// the compiled kernel function as a cubin in the 'nvvm.cubin' attribute, or a
// hsaco in the 'rocdl.hsaco' attribute of the kernel function in the IR.
//
// %0 = call %binarygetter
// %1 = call %moduleLoad(%0)
// %2 = <see generateKernelNameConstant>
// %3 = call %moduleGetFunction(%1, %2)
// %4 = call %streamCreate()
// %5 = <see generateParamsArray>
// call %launchKernel(%3, <launchOp operands 0..5>, 0, %4, %5, nullptr)
// call %streamSynchronize(%4)
// call %streamDestroy(%4)
// call %moduleUnload(%1)
//
// If the op is async, the stream corresponds to the (single) async dependency
// as well as the async token the op produces.
LogicalResult ConvertLaunchFuncOpToGpuRuntimeCallPattern::matchAndRewrite(
    gpu::LaunchFuncOp launchOp, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  if (failed(areAllLLVMTypes(launchOp, adaptor.getOperands(), rewriter)))
    return failure();

  if (launchOp.getAsyncDependencies().size() > 1)
    return rewriter.notifyMatchFailure(
        launchOp, "Cannot convert with more than one async dependency.");

  Block *allocaBlock = nullptr;
  {
    Operation *currentOp = launchOp;
    while (Operation *parentOp = currentOp->getParentOp()) {
      if (parentOp->mightHaveTrait<OpTrait::IsIsolatedFromAbove>() ||
          parentOp->mightHaveTrait<OpTrait::AutomaticAllocationScope>()) {
        allocaBlock = &currentOp->getParentRegion()->front();
        break;
      }
      currentOp = parentOp;
    }
  }

  ModuleOp moduleOp = launchOp->getParentOfType<ModuleOp>();

  Location loc = launchOp.getLoc();

  GPUErrorOp errOp = dyn_cast<GPUErrorOp>(launchOp->getParentOp());

  // Create an LLVM global with CUBIN extracted from the kernel annotation and
  // obtain a pointer to the first byte in it.
  auto kernelModule = SymbolTable::lookupNearestSymbolFrom<gpu::GPUModuleOp>(
      launchOp, launchOp.getKernelModuleName());
  if (!kernelModule)
    return rewriter.notifyMatchFailure(launchOp, "Expected a kernel module");

  // auto getFuncGlobalName = [](StringRef moduleName, StringRef name) {
  //   return std::string(
  //       llvm::formatv("__polygeist_{0}_{1}_fun_ptr", moduleName, name));
  // };

  // Build module constructor and destructor
  std::string funcStubName =
      getFuncStubName(launchOp.getKernelModuleName().getValue(),
                      launchOp.getKernelName().getValue());

  auto bitcast =
      rewriter.create<LLVM::AddressOfOp>(loc, llvmPointerType, funcStubName);

  Value zero = rewriter.create<LLVM::ConstantOp>(loc, llvmInt32Type, 0);
  auto nullpointer = rewriter.create<LLVM::ZeroOp>(loc, llvmPointerType);
  Value stream = adaptor.getAsyncDependencies().empty()
                     ? nullpointer
                     : adaptor.getAsyncDependencies().front();

  // Create array of pointers to kernel arguments.
  auto kernelParams =
      generateParamsArray(launchOp, adaptor, rewriter, allocaBlock);
  Value dynamicSharedMemorySize = launchOp.getDynamicSharedMemorySize()
                                      ? launchOp.getDynamicSharedMemorySize()
                                      : zero;

  SmallVector<Value> args;
  args.push_back(bitcast);
  auto i32 = rewriter.getIntegerType(32);
  auto i64 = rewriter.getIntegerType(64);
  auto dim3 = [&](Value x, Value y, Value z) {
    x = rewriter.create<LLVM::TruncOp>(x.getLoc(), i32, x);
    y = rewriter.create<LLVM::TruncOp>(y.getLoc(), i32, y);
    z = rewriter.create<LLVM::TruncOp>(z.getLoc(), i32, z);

    x = rewriter.create<LLVM::ZExtOp>(x.getLoc(), i64, x);
    y = rewriter.create<LLVM::ZExtOp>(y.getLoc(), i64, y);

    y = rewriter.create<LLVM::ShlOp>(
        y.getLoc(), y, rewriter.create<LLVM::ConstantOp>(y.getLoc(), i64, 32));
    args.push_back(rewriter.create<LLVM::OrOp>(x.getLoc(), x, y));
    args.push_back(z);
  };
  dim3(adaptor.getGridSizeX(), adaptor.getGridSizeY(), adaptor.getGridSizeZ());
  dim3(adaptor.getBlockSizeX(), adaptor.getBlockSizeY(),
       adaptor.getBlockSizeZ());

  args.push_back(kernelParams);
  args.push_back(
      rewriter.create<LLVM::ZExtOp>(loc, i64, dynamicSharedMemorySize));
  args.push_back(stream);

  auto ptrty = LLVM::LLVMPointerType::get(rewriter.getContext());
  Type tys[] = {ptrty, i64, i32, i64, i32, ptrty, i64, ptrty};

  auto launchCall = rewriter.create<LLVM::CallOp>(
      loc, TypeRange(i32), "cudaLaunchKernel", args); // FlatSymbolRefAttr::get(rewriter.getStringAttr("cudaLaunchKernel")),
                                                      // args);
  if (launchOp.getAsyncToken()) {
    // Async launch: make dependent ops use the same stream.
    rewriter.replaceOp(launchOp, {stream});
  } else {
    rewriter.eraseOp(launchOp);
  }

  if (errOp) {
    rewriter.setInsertionPoint(errOp);
    auto reg = rewriter.create<scf::ExecuteRegionOp>(
        errOp.getLoc(), launchCall->getResultTypes()[0]);
    rewriter.inlineRegionBefore(errOp.getRegion(), reg.getRegion(),
                                reg.getRegion().begin());
    rewriter.createBlock(&errOp.getRegion());

    rewriter.setInsertionPointToStart(allocaBlock);

    auto ptrty = LLVM::LLVMPointerType::get(rewriter.getContext());

    auto one = rewriter.create<LLVM::ConstantOp>(loc, i64,
                                                 rewriter.getI64IntegerAttr(1));

    auto alloca = rewriter.create<LLVM::AllocaOp>(
        launchOp.getLoc(), ptrty, launchCall->getResultTypes()[0], one);
    auto zero = rewriter.create<arith::ConstantIntOp>(
        loc, launchCall->getResultTypes()[0], 0);

    rewriter.setInsertionPoint(errOp);
    rewriter.create<LLVM::StoreOp>(launchOp.getLoc(), zero, alloca);

    rewriter.setInsertionPointAfter(launchCall);
    rewriter.create<LLVM::StoreOp>(launchOp.getLoc(), launchCall->getResult(0),
                                   alloca);

    for (auto &block : reg.getRegion()) {
      if (auto terminator =
              dyn_cast<enzymexla::PolygeistYieldOp>(block.getTerminator())) {
        rewriter.setInsertionPointToEnd(&block);
        auto load = rewriter.create<LLVM::LoadOp>(
            launchOp.getLoc(), launchCall->getResultTypes()[0], alloca);
        rewriter.replaceOpWithNewOp<scf::YieldOp>(terminator,
                                                  load->getResults());
      }
    }

    rewriter.setInsertionPointAfter(errOp);
    auto cast = rewriter.create<arith::IndexCastOp>(
        loc, rewriter.getIndexType(), reg->getResult(0));
    rewriter.replaceOp(errOp, cast->getResults());
  }

  return success();
}

/// A rewrite patter to legalize gpu.launch_func with LLVM types.
class LegalizeLaunchFuncOpPattern
    : public ConvertOpToGpuRuntimeCallPattern<gpu::LaunchFuncOp> {
public:
  LegalizeLaunchFuncOpPattern(LLVMTypeConverter &typeConverter,
                              bool kernelBarePtrCallConv,
                              bool kernelIntersperseSizeCallConv)
      : ConvertOpToGpuRuntimeCallPattern<gpu::LaunchFuncOp>(typeConverter),
        kernelBarePtrCallConv(kernelBarePtrCallConv),
        kernelIntersperseSizeCallConv(kernelIntersperseSizeCallConv) {}

private:
  LogicalResult
  matchAndRewrite(gpu::LaunchFuncOp launchOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;

  bool kernelBarePtrCallConv;
  bool kernelIntersperseSizeCallConv;
};

// Legalize the op's operands.
LogicalResult LegalizeLaunchFuncOpPattern::matchAndRewrite(
    gpu::LaunchFuncOp launchOp, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  if (failed(areAllLLVMTypes(launchOp, adaptor.getOperands(), rewriter)))
    return failure();

  if (launchOp.getAsyncDependencies().size() > 1)
    return rewriter.notifyMatchFailure(
        launchOp, "Cannot convert with more than one async dependency.");

  // Fail when the synchronous version of the op has async dependencies. The
  // lowering destroys the stream, and we do not want to check that there is no
  // use of the stream after this op.
  if (!launchOp.getAsyncToken() && !launchOp.getAsyncDependencies().empty())
    return rewriter.notifyMatchFailure(
        launchOp, "Cannot convert non-async op with async dependencies.");

  Location loc = launchOp.getLoc();

  Value stream = Value();
  if (!adaptor.getAsyncDependencies().empty())
    stream = adaptor.getAsyncDependencies().front();
  // If the async keyword is present and there are no dependencies, then a
  // stream must be created to pass to subsequent operations.
  else if (launchOp.getAsyncToken())
    stream = streamCreateCallBuilder.create(loc, rewriter, {}).getResult();

  // Lower the kernel operands to match kernel parameters.
  // Note: If `useBarePtrCallConv` is set in the type converter's options,
  // the value of `kernelBarePtrCallConv` will be ignored.
  OperandRange origArguments = launchOp.getKernelOperands();
  SmallVector<Value, 8> llvmArguments =
      llvm::to_vector(adaptor.getKernelOperands());
  for (auto arg : llvmArguments) {
    llvm::errs() << " arg: " << arg << "\n";
  }
  // getTypeConverter()->promoteOperands(
  //     loc, origArguments, adaptor.getKernelOperands(), rewriter,
  //    /*useBarePtrCallConv=*/kernelBarePtrCallConv);
  SmallVector<Value, 8> llvmArgumentsWithSizes;

  // Intersperse size information if requested.
  if (kernelIntersperseSizeCallConv) {
    if (origArguments.size() != llvmArguments.size()) {
      // This shouldn't happen if the bare-pointer calling convention is used.
      return rewriter.notifyMatchFailure(
          launchOp,
          "Cannot add sizes to arguments with one-to-many LLVM IR expansion.");
    }

    llvmArgumentsWithSizes.reserve(llvmArguments.size() * 2);
    for (auto [llvmArg, origArg] : zip_equal(llvmArguments, origArguments)) {
      auto memrefTy = dyn_cast<MemRefType>(origArg.getType());
      if (!memrefTy) {
        return rewriter.notifyMatchFailure(
            launchOp, "Operand to launch op is not a memref.");
      }

      if (!memrefTy.hasStaticShape() ||
          !memrefTy.getElementType().isIntOrFloat()) {
        return rewriter.notifyMatchFailure(
            launchOp, "Operand to launch op is not a memref with a static "
                      "shape and an integer or float element type.");
      }

      unsigned bitwidth = memrefTy.getElementTypeBitWidth();
      if (bitwidth % 8 != 0) {
        return rewriter.notifyMatchFailure(
            launchOp, "Operand to launch op is not a memref with a "
                      "byte-aligned element type.");
      }

      uint64_t staticSize = static_cast<uint64_t>(bitwidth / 8) *
                            static_cast<uint64_t>(memrefTy.getNumElements());

      Value sizeArg = rewriter.create<LLVM::ConstantOp>(
          loc, getIndexType(), rewriter.getIndexAttr(staticSize));
      llvmArgumentsWithSizes.push_back(llvmArg); // Presumably a bare pointer.
      llvmArgumentsWithSizes.push_back(sizeArg);
    }
  }

  std::optional<gpu::KernelDim3> clusterSize = std::nullopt;
  if (launchOp.hasClusterSize()) {
    clusterSize =
        gpu::KernelDim3{adaptor.getClusterSizeX(), adaptor.getClusterSizeY(),
                        adaptor.getClusterSizeZ()};
  }
  rewriter.create<gpu::LaunchFuncOp>(
      launchOp.getLoc(), launchOp.getKernelAttr(),
      gpu::KernelDim3{adaptor.getGridSizeX(), adaptor.getGridSizeY(),
                      adaptor.getGridSizeZ()},
      gpu::KernelDim3{adaptor.getBlockSizeX(), adaptor.getBlockSizeY(),
                      adaptor.getBlockSizeZ()},
      adaptor.getDynamicSharedMemorySize(),
      llvmArgumentsWithSizes.empty() ? llvmArguments : llvmArgumentsWithSizes,
      stream, clusterSize);
  if (launchOp.getAsyncToken())
    rewriter.replaceOp(launchOp, {stream});
  else
    rewriter.eraseOp(launchOp);
  return success();
}

static LogicalResult
isAsyncWithOneDependency(ConversionPatternRewriter &rewriter,
                         gpu::AsyncOpInterface op) {
  if (op.getAsyncDependencies().size() != 1)
    return rewriter.notifyMatchFailure(
        op, "Can only convert with exactly one async dependency.");

  if (!op.getAsyncToken())
    return rewriter.notifyMatchFailure(op, "Can convert only async version.");

  return success();
}

static LogicalResult
isAsyncWithNoDependency(ConversionPatternRewriter &rewriter,
                        gpu::AsyncOpInterface op) {
  if (op.getAsyncDependencies().size() != 0)
    return rewriter.notifyMatchFailure(
        op, "Can only convert with exactly one async dependency.");

  if (op.getAsyncToken())
    return rewriter.notifyMatchFailure(op, "Can convert only async version.");

  return success();
}

int64_t xla_type_id(mlir::Type T) {
  return xla::ConvertMlirTypeToPrimitiveType(T);
}
/// A rewrite pattern to convert gpu.alloc operations into a GPU runtime
/// call. Currently it supports CUDA, CPU, and XLA.
template <bool cStyle>
class ConvertAllocOpToGpuRuntimeCallPattern
    : public ConvertOpToGpuRuntimeCallPattern<gpu::AllocOp> {
public:
  /// The attribute name to use instead of `gpu.kernel`.
  StringRef backend;

  ConvertAllocOpToGpuRuntimeCallPattern(LLVMTypeConverter &typeConverter,
                                        StringRef backend)
      : ConvertOpToGpuRuntimeCallPattern<gpu::AllocOp>(typeConverter),
        backend(backend) {}

private:
  LogicalResult
  matchAndRewrite(gpu::AllocOp allocOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    MemRefType memRefType = allocOp.getType();

    if (failed(areAllLLVMTypes(allocOp, adaptor.getOperands(), rewriter)) ||
        !isConvertibleAndHasIdentityMaps(memRefType))
      return failure();

    auto loc = allocOp.getLoc();

    bool isShared = allocOp.getHostShared();

    if (isShared && allocOp.getAsyncToken())
      return rewriter.notifyMatchFailure(
          allocOp, "Host Shared allocation cannot be done async");
    if (!isShared && failed(isAsyncWithNoDependency(rewriter, allocOp)))
      return failure();

    // Get shape of the memref as values: static sizes are constant
    // values and dynamic sizes are passed to 'alloc' as operands.
    SmallVector<Value, 4> shape;
    SmallVector<Value, 4> strides;
    Value sizeBytes;
    getMemRefDescriptorSizes(loc, memRefType, adaptor.getDynamicSizes(),
                             rewriter, shape, strides, sizeBytes);

    // Allocate the underlying buffer and store a pointer to it in the MemRef
    // descriptor.
    auto nullPtr = rewriter.create<mlir::LLVM::ZeroOp>(loc, llvmPointerType);
    Value stream = adaptor.getAsyncDependencies().empty()
                       ? nullPtr
                       : adaptor.getAsyncDependencies().front();

    Value allocatedPtr;
    if (allocOp.getAsyncDependencies().size() == 0 &&
        !allocOp.getAsyncToken()) {
      auto i64 = rewriter.getIntegerType(64);
      auto i32 = rewriter.getIntegerType(32);
      auto moduleOp = allocOp->getParentOfType<ModuleOp>();

      auto ptrty = LLVM::LLVMPointerType::get(rewriter.getContext());
      auto ptr1ty = LLVM::LLVMPointerType::get(rewriter.getContext(), 1);

      if (backend == "cuda") {
        auto one = rewriter.create<LLVM::ConstantOp>(
            loc, i64, rewriter.getI64IntegerAttr(1));

        auto ptr = rewriter.create<LLVM::AllocaOp>(loc, ptrty, ptr1ty, one);
        Type tys[] = {ptrty, i64};
        auto cudaMallocFn =
            LLVM::lookupOrCreateFn(rewriter, moduleOp, "cudaMalloc", tys, i32);
        if (failed(cudaMallocFn)) {
          llvm::errs() << " cudamalloc already exists with different types\n";
          return failure();
        }

        Value args[] = {
            ptr,
            sizeBytes,
        };
        rewriter.create<LLVM::CallOp>(loc, cudaMallocFn.value(), args);
        allocatedPtr = rewriter.create<LLVM::LoadOp>(loc, ptr1ty, ptr);
      } else if (backend.starts_with("cpu")) {
        Type convertedIndex =
            typeConverter->convertType(rewriter.getIndexType());

        FailureOr<LLVM::LLVMFuncOp> mallocFunc =
            LLVM::lookupOrCreateMallocFn(rewriter, moduleOp, convertedIndex);

        if (failed(mallocFunc)) {
          llvm::errs() << " cudamalloc already exists with different types\n";
          return failure();
        }

        Value args[] = {
            sizeBytes,
        };
        allocatedPtr =
            rewriter.create<LLVM::CallOp>(loc, mallocFunc.value(), args)
                ->getResult(0);
      } else if (backend.starts_with("xla")) {

        auto zero = rewriter.create<LLVM::ConstantOp>(
            loc, i64, rewriter.getI64IntegerAttr(0));

        auto one = rewriter.create<LLVM::ConstantOp>(
            loc, i64, rewriter.getI64IntegerAttr(1));

        auto tyid = rewriter.create<LLVM::ConstantOp>(
            loc, i64,
            rewriter.getI64IntegerAttr(
                xla_type_id(memRefType.getElementType())));

        Type convertedIndex =
            typeConverter->convertType(rewriter.getIndexType());

        auto shapeDim = rewriter.create<LLVM::ConstantOp>(
            loc, i64, rewriter.getI64IntegerAttr(memRefType.getShape().size()));

        auto AT = LLVM::LLVMArrayType::get(i64, memRefType.getShape().size());

        auto shapePtr = rewriter.create<LLVM::AllocaOp>(loc, ptrty, AT, one);

        int dynIdx = 0;
        for (int i = 0; i < memRefType.getShape().size(); i++) {
          auto idx = rewriter.create<LLVM::ConstantOp>(
              loc, i64, rewriter.getI64IntegerAttr(i));
          Value idxs[] = {zero, idx};

          auto gep =
              rewriter.create<LLVM::GEPOp>(loc, ptrty, AT, shapePtr, idxs);

          Value val;

          if (memRefType.getShape()[i] == ShapedType::kDynamic) {
            val = adaptor.getDynamicSizes()[dynIdx];
            dynIdx++;
          } else {
            val = rewriter.create<LLVM::ConstantOp>(
                loc, i64, rewriter.getI64IntegerAttr(memRefType.getShape()[i]));
          }

          rewriter.create<LLVM::StoreOp>(loc, val, gep);
        }

        // handle, type id, shape len, shape ptr
        Type tys[] = {ptrty, i64, i64, ptrty};

        auto xlaMallocFn = LLVM::lookupOrCreateFn(
            rewriter, moduleOp, "reactantXLAMalloc", tys, ptrty);
        if (failed(xlaMallocFn)) {
          llvm::errs() << " xlaMalloc already exists with different types\n";
          return failure();
        }

        auto xdata = insertXLAInitDeinit(moduleOp, backend, rewriter);
        Value args[] = {xdata, tyid, shapeDim, shapePtr};
        allocatedPtr =
            rewriter.create<LLVM::CallOp>(loc, xlaMallocFn.value(), args)
                ->getResult(0);

        allocatedPtr =
            rewriter.create<LLVM::AddrSpaceCastOp>(loc, ptr1ty, allocatedPtr);

      } else {
        llvm_unreachable("unknown backend");
      }
    } else {

      auto isHostShared = rewriter.create<mlir::LLVM::ConstantOp>(
          loc, llvmInt8Type, rewriter.getI8IntegerAttr(isShared));
      allocatedPtr =
          allocCallBuilder
              .create(loc, rewriter, {sizeBytes, stream, isHostShared})
              .getResult();
    }

    // No alignment.
    Value alignedPtr = allocatedPtr;

    // Create the MemRef descriptor.
    Value memRefDescriptor = alignedPtr;
    if constexpr (!cStyle)
      this->createMemRefDescriptor(loc, memRefType, allocatedPtr, alignedPtr,
                                   shape, strides, rewriter);

    if (allocOp.getAsyncToken()) {
      // Async alloc: make dependent ops use the same stream.
      rewriter.replaceOp(allocOp, {memRefDescriptor, stream});
    } else {
      rewriter.replaceOp(allocOp, {memRefDescriptor});
    }

    return success();
  }
};

/// A rewrite pattern to convert gpu.alloc operations into a GPU runtime
/// call. Currently it supports CUDA, CPU, and XLA.
template <bool cStyle>
class ConvertDeallocOpToGpuRuntimeCallPattern
    : public ConvertOpToGpuRuntimeCallPattern<gpu::DeallocOp> {
public:
  /// The attribute name to use instead of `gpu.kernel`.
  StringRef backend;

  ConvertDeallocOpToGpuRuntimeCallPattern(LLVMTypeConverter &typeConverter,
                                          StringRef backend)
      : ConvertOpToGpuRuntimeCallPattern<gpu::DeallocOp>(typeConverter),
        backend(backend) {}

private:
  LogicalResult
  matchAndRewrite(gpu::DeallocOp deallocOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    MemRefType memRefType = deallocOp.getMemref().getType();

    if (failed(areAllLLVMTypes(deallocOp, adaptor.getOperands(), rewriter)) ||
        !isConvertibleAndHasIdentityMaps(memRefType))
      return failure();

    auto loc = deallocOp.getLoc();

    if (deallocOp.getAsyncToken())
      return rewriter.notifyMatchFailure(deallocOp, "Async free not supported");

    auto ptr = adaptor.getMemref();

    if (failed(isAsyncWithNoDependency(rewriter, deallocOp)))
      return failure();

    auto i64 = rewriter.getIntegerType(64);
    auto i32 = rewriter.getIntegerType(32);
    auto moduleOp = deallocOp->getParentOfType<ModuleOp>();

    auto ptr1ty = LLVM::LLVMPointerType::get(rewriter.getContext(), 1);

    if (backend == "cuda") {
      auto one = rewriter.create<LLVM::ConstantOp>(
          loc, i64, rewriter.getI64IntegerAttr(1));

      Type tys[] = {ptr1ty};
      auto cudaFreeFn =
          LLVM::lookupOrCreateFn(rewriter, moduleOp, "cudaFree", tys, i32);
      if (failed(cudaFreeFn)) {
        llvm::errs() << " cudafree already exists with different types\n";
        return failure();
      }

      Value args[] = {
          ptr,
      };
      rewriter.create<LLVM::CallOp>(loc, cudaFreeFn.value(), args);
    } else if (backend.starts_with("cpu")) {

      FailureOr<LLVM::LLVMFuncOp> freeFunc =
          LLVM::lookupOrCreateFreeFn(rewriter, moduleOp);

      if (failed(freeFunc)) {
        llvm::errs() << " free already exists with different types\n";
        return failure();
      }

      Value args[] = {
          ptr,
      };
      rewriter.create<LLVM::CallOp>(loc, freeFunc.value(), args)->getResult(0);
    } else if (backend.starts_with("xla")) {
      auto ptrty = LLVM::LLVMPointerType::get(rewriter.getContext());

      // handle, ptr
      Type tys[] = {ptrty, ptrty};

      auto xlaFreeFn = LLVM::lookupOrCreateFn(
          rewriter, moduleOp, "reactantXLAFree", tys,
          LLVM::LLVMVoidType::get(moduleOp->getContext()));
      if (failed(xlaFreeFn)) {
        llvm::errs() << " xlaMalloc already exists with different types\n";
        return failure();
      }

      auto xdata = insertXLAInitDeinit(moduleOp, backend, rewriter);

      Value args[] = {xdata, ptr};

      rewriter.create<LLVM::CallOp>(loc, xlaFreeFn.value(), args)->getResult(0);
    } else {
      llvm::errs() << " unknown backend: " << backend << "\n";
      return failure();
    }

    rewriter.eraseOp(deallocOp);

    return success();
  }
};

template <bool cStyle>
class ConvertXLAWrapperPattern
    : public ConvertOpToGpuRuntimeCallPattern<enzymexla::XLAWrapperOp> {
public:
  /// The attribute name to use instead of `gpu.kernel`.
  StringRef backend;

  ConvertXLAWrapperPattern(LLVMTypeConverter &typeConverter, StringRef backend)
      : ConvertOpToGpuRuntimeCallPattern<enzymexla::XLAWrapperOp>(
            typeConverter),
        backend(backend) {}

private:
  LogicalResult
  matchAndRewrite(enzymexla::XLAWrapperOp wrap, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    if (failed(areAllLLVMTypes(wrap, adaptor.getOperands(), rewriter)))
      return failure();

    auto loc = wrap.getLoc();

    std::string str;
    llvm::raw_string_ostream stream(str);

    auto i64 = rewriter.getIntegerType(64);

    auto fn = cast<FunctionOpInterface>(
        SymbolTable::lookupNearestSymbolFrom(wrap, wrap.getFn()));
    stream << fn << "\n" << '\0';

    auto stringval = mlir::LLVM::createGlobalString(
        loc, rewriter, "xlamod", str, LLVM::Linkage::Internal);

    auto ptrty = LLVM::LLVMPointerType::get(rewriter.getContext());

    auto zero = rewriter.create<LLVM::ConstantOp>(
        loc, i64, rewriter.getI64IntegerAttr(0));

    auto one = rewriter.create<LLVM::ConstantOp>(loc, i64,
                                                 rewriter.getI64IntegerAttr(1));

    auto nargs = rewriter.create<LLVM::ConstantOp>(
        loc, i64, rewriter.getI64IntegerAttr(adaptor.getInputs().size()));

    auto AT = LLVM::LLVMArrayType::get(i64, adaptor.getInputs().size());

    auto argsPtr = rewriter.create<LLVM::AllocaOp>(loc, ptrty, AT, one);

    for (int i = 0; i < adaptor.getInputs().size(); i++) {
      auto idx = rewriter.create<LLVM::ConstantOp>(
          loc, i64, rewriter.getI64IntegerAttr(i));
      Value idxs[] = {zero, idx};

      auto gep = rewriter.create<LLVM::GEPOp>(loc, ptrty, AT, argsPtr, idxs);

      rewriter.create<LLVM::StoreOp>(loc, adaptor.getInputs()[i], gep);
    }

    // handle, module, nargs, argptr
    Type tys[] = {ptrty, ptrty, i64, ptrty};

    auto moduleOp = wrap->getParentOfType<ModuleOp>();
    auto xlaExecFn = LLVM::lookupOrCreateFn(
        rewriter, moduleOp, "reactantXLAExec", tys,
        LLVM::LLVMVoidType::get(moduleOp->getContext()), true);
    if (failed(xlaExecFn)) {
      llvm::errs() << " reactantXLAExec already exists with different types\n";
      return failure();
    }

    auto xdata = insertXLAInitDeinit(moduleOp, backend, rewriter);
    Value args[4] = {xdata, stringval, nargs, argsPtr};

    rewriter.create<LLVM::CallOp>(loc, xlaExecFn.value(), args);

    wrap.setFnAttr(
        FlatSymbolRefAttr::get(rewriter.getStringAttr("<undefined>")));

    bool baduser = false;
    for (auto use :
         *SymbolTable::getSymbolUses(fn.getOperation(), fn->getParentOp())) {
      if (use.getUser() == wrap)
        continue;
      baduser = true;
    }
    if (!baduser)
      rewriter.eraseOp(fn.getOperation());

    rewriter.eraseOp(wrap);

    return success();
  }
};

struct ReplaceErrOpWithSuccess : public OpRewritePattern<GPUErrorOp> {
  using OpRewritePattern<GPUErrorOp>::OpRewritePattern;
  const char *PATTERN = "lower-gpu-alternatives";

  LogicalResult matchAndRewrite(GPUErrorOp errOp,
                                PatternRewriter &rewriter) const override {
    rewriter.setInsertionPoint(errOp);
    if (errOp->getRegions()[0].hasOneBlock()) {
      rewriter.eraseOp(errOp.getBody()->getTerminator());
      rewriter.inlineBlockBefore(errOp.getBody(), errOp);
      rewriter.setInsertionPoint(errOp);
    } else {
      auto *condBlock = rewriter.getInsertionBlock();
      auto opPosition = rewriter.getInsertionPoint();
      auto *remainingOpsBlock = rewriter.splitBlock(condBlock, opPosition);

      auto &region = errOp.getRegion();
      rewriter.setInsertionPointToEnd(condBlock);
      rewriter.create<cf::BranchOp>(errOp.getLoc(), &region.front());

      for (Block &block : errOp->getRegions()[0]) {
        if (auto terminator =
                dyn_cast<enzymexla::PolygeistYieldOp>(block.getTerminator())) {
          ValueRange terminatorOperands = terminator->getOperands();
          rewriter.setInsertionPointToEnd(&block);
          rewriter.create<cf::BranchOp>(errOp.getLoc(), remainingOpsBlock,
                                        terminatorOperands);
          rewriter.eraseOp(terminator);
        }
      }
      rewriter.inlineRegionBefore(region, remainingOpsBlock);
    }
    auto zero = rewriter.create<arith::ConstantIndexOp>(errOp->getLoc(), 0);
    rewriter.replaceOp(errOp, zero->getResults());
    return success();
  }
};

/// Pattern for gpu function declarations and definitions.
struct GPUFuncOpLowering : public ConvertOpToLLVMPattern<gpu::GPUFuncOp> {
private:
  /// The address spcae to use for `alloca`s in private memory.
  unsigned allocaAddrSpace;

  /// The attribute name to use instead of `gpu.kernel`.
  StringAttr kernelAttributeName;

public:
  using ConvertOpToLLVMPattern<gpu::GPUFuncOp>::ConvertOpToLLVMPattern;

  GPUFuncOpLowering(LLVMTypeConverter &converter, unsigned allocaAddrSpace,
                    StringAttr kernelAttributeName)
      : ConvertOpToLLVMPattern<gpu::GPUFuncOp>(converter),
        allocaAddrSpace(allocaAddrSpace),
        kernelAttributeName(kernelAttributeName) {}

  LogicalResult
  matchAndRewrite(gpu::GPUFuncOp gpuFuncOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = gpuFuncOp.getLoc();

    SmallVector<LLVM::GlobalOp, 3> workgroupBuffers;
    workgroupBuffers.reserve(gpuFuncOp.getNumWorkgroupAttributions());
    for (const auto &en :
         llvm::enumerate(gpuFuncOp.getWorkgroupAttributions())) {
      Value attribution = en.value();

      auto type = dyn_cast<MemRefType>(attribution.getType());
      assert(type && type.hasStaticShape() && "unexpected type in attribution");

      uint64_t numElements = type.getNumElements();

      auto elementType = typeConverter->convertType(type.getElementType());
      auto arrayType = LLVM::LLVMArrayType::get(elementType, numElements);
      std::string name = std::string(
          llvm::formatv("__wg_{0}_{1}", gpuFuncOp.getName(), en.index()));
      auto globalOp = rewriter.create<LLVM::GlobalOp>(
          gpuFuncOp.getLoc(), arrayType, /*isConstant=*/false,
          LLVM::Linkage::Internal, name, /*value=*/Attribute(),
          /*alignment=*/0,
          static_cast<unsigned>(gpu::GPUDialect::getWorkgroupAddressSpace()));
      workgroupBuffers.push_back(globalOp);
    }

    auto typePair = convertFunctionType(gpuFuncOp, *typeConverter);
    if (!typePair)
      return rewriter.notifyMatchFailure(gpuFuncOp->getLoc(),
                                         "failed to convert signature");

    auto [funcType, signatureConversion] = *typePair;

    // Create the new function operation. Only copy those attributes that are
    // not specific to function modeling.
    SmallVector<NamedAttribute> attributes;
    for (const auto &attr : gpuFuncOp->getAttrs()) {
      if (attr.getName() == SymbolTable::getSymbolAttrName() ||
          attr.getName() == gpuFuncOp.getFunctionTypeAttrName() ||
          attr.getName() ==
              gpu::GPUFuncOp::getNumWorkgroupAttributionsAttrName())
        continue;
      attributes.push_back(attr);
    }
    // Add a dialect specific kernel attribute in addition to GPU kernel
    // attribute. The former is necessary for further translation while the
    // latter is expected by gpu.launch_func.
    if (gpuFuncOp.isKernel())
      attributes.emplace_back(kernelAttributeName, rewriter.getUnitAttr());
    auto llvmFuncOp = rewriter.create<LLVM::LLVMFuncOp>(
        gpuFuncOp.getLoc(), gpuFuncOp.getName(), funcType,
        LLVM::Linkage::External, /*dsoLocal*/ false, /*cconv*/ LLVM::CConv::C,
        /*comdat=*/nullptr, attributes);

    {
      // Insert operations that correspond to converted workgroup and private
      // memory attributions to the body of the function. This must operate on
      // the original function, before the body region is inlined in the new
      // function to maintain the relation between block arguments and the
      // parent operation that assigns their semantics.
      OpBuilder::InsertionGuard guard(rewriter);

      // Rewrite workgroup memory attributions to addresses of global buffers.
      rewriter.setInsertionPointToStart(&gpuFuncOp.front());
      unsigned numProperArguments = gpuFuncOp.getNumArguments();

      for (const auto &en : llvm::enumerate(workgroupBuffers)) {
        LLVM::GlobalOp global = en.value();
        Value memory = rewriter.create<LLVM::AddressOfOp>(loc, global);

        // Build a memref descriptor pointing to the buffer to plug with the
        // existing memref infrastructure. This may use more registers than
        // otherwise necessary given that memref sizes are fixed, but we can try
        // and canonicalize that away later.
        Value attribution = gpuFuncOp.getWorkgroupAttributions()[en.index()];
        auto type = cast<MemRefType>(attribution.getType());
        Value descr = MemRefDescriptor::fromStaticShape(
            rewriter, loc, *getTypeConverter(), type, memory);
        signatureConversion.remapInput(numProperArguments + en.index(), descr);
      }

      // Rewrite private memory attributions to alloca'ed buffers.
      unsigned numWorkgroupAttributions =
          gpuFuncOp.getNumWorkgroupAttributions();
      auto int64Ty = IntegerType::get(rewriter.getContext(), 64);
      for (const auto &en :
           llvm::enumerate(gpuFuncOp.getPrivateAttributions())) {
        Value attribution = en.value();
        auto type = cast<MemRefType>(attribution.getType());
        assert(type && type.hasStaticShape() &&
               "unexpected type in attribution");

        // Explicitly drop memory space when lowering private memory
        // attributions since NVVM models it as `alloca`s in the default
        // memory space and does not support `alloca`s with addrspace(5).
        auto ptrType =
            LLVM::LLVMPointerType::get(type.getContext(), allocaAddrSpace);
        Value numElements = rewriter.create<LLVM::ConstantOp>(
            gpuFuncOp.getLoc(), int64Ty, type.getNumElements());
        Value allocated = rewriter.create<LLVM::AllocaOp>(
            gpuFuncOp.getLoc(), ptrType, type.getElementType(), numElements,
            /*alignment=*/0);
        Value descr = MemRefDescriptor::fromStaticShape(
            rewriter, loc, *getTypeConverter(), type, allocated);
        signatureConversion.remapInput(
            numProperArguments + numWorkgroupAttributions + en.index(), descr);
      }
    }

    // Move the region to the new function, update the entry block signature.
    rewriter.inlineRegionBefore(gpuFuncOp.getBody(), llvmFuncOp.getBody(),
                                llvmFuncOp.end());
    if (failed(rewriter.convertRegionTypes(
            &llvmFuncOp.getBody(), *typeConverter, &signatureConversion)))
      return rewriter.notifyMatchFailure(
          gpuFuncOp->getLoc(), "failed to apply signature conversion");
    rewriter.eraseOp(gpuFuncOp);
    return success();
  }
};

static LogicalResult
isAsyncWithZeroDependencies(ConversionPatternRewriter &rewriter,
                            gpu::AsyncOpInterface op) {
  if (op.getAsyncDependencies().size() != 0)
    return rewriter.notifyMatchFailure(
        op, "Can only convert with exactly one async dependency.");

  if (!op.getAsyncToken())
    return rewriter.notifyMatchFailure(op, "Can convert only async version.");

  return success();
}

/// Pattern for function declarations and definitions.
struct FuncOpLowering : public ConvertOpToLLVMPattern<func::FuncOp> {
public:
  using ConvertOpToLLVMPattern<func::FuncOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(func::FuncOp funcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    bool anyStablehlo =
        funcOp
            ->walk([](Operation *op) -> WalkResult {
              if (op->getDialect()->getNamespace() == "stablehlo")
                return WalkResult::interrupt();
              return WalkResult::advance();
            })
            .wasInterrupted();

    if (anyStablehlo) {
      if (SymbolTable::symbolKnownUseEmpty(funcOp, funcOp->getParentOp())) {
        rewriter.eraseOp(funcOp);
        return success();
      }
      return failure();
    }

    auto typePair = convertFunctionType(funcOp, *typeConverter);
    if (!typePair)
      return rewriter.notifyMatchFailure(funcOp->getLoc(),
                                         "failed to convert signature");

    auto [convertedType, conversionSignature] = *typePair;
    SmallVector<NamedAttribute> attributes = convertFuncAttributes(
        funcOp, *typeConverter, conversionSignature, rewriter);

    LLVM::Linkage linkage = LLVM::Linkage::External;
    if (funcOp->hasAttr(kLLVMLinkageAttrName)) {
      auto attr =
          cast<mlir::LLVM::LinkageAttr>(funcOp->getAttr(kLLVMLinkageAttrName));
      linkage = attr.getLinkage();
    }
    auto newFuncOp = rewriter.create<LLVM::LLVMFuncOp>(
        funcOp.getLoc(), funcOp.getName(), convertedType, linkage,
        /*dsoLocal=*/false, /*cconv=*/LLVM::CConv::C, /*comdat=*/nullptr,
        attributes);
    rewriter.inlineRegionBefore(funcOp.getBody(), newFuncOp.getBody(),
                                newFuncOp.end());
    if (failed(rewriter.convertRegionTypes(&newFuncOp.getBody(), *typeConverter,
                                           &conversionSignature))) {
      return rewriter.notifyMatchFailure(
          funcOp->getLoc(), "failed to apply signature conversion");
    }

    rewriter.eraseOp(funcOp);
    return success();
  }
};

/// Pattern for function calls, unpacks the results from the struct.
struct CallOpLowering : public ConvertOpToLLVMPattern<func::CallOp> {
public:
  using ConvertOpToLLVMPattern<func::CallOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(func::CallOp callOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    unsigned numResults = callOp.getNumResults();
    SmallVector<Type, 1> callResultTypes;
    if (!callOp.getResults().empty()) {
      callResultTypes.push_back(convertAndPackFunctionResultType(
          callOp.getCalleeType(), *typeConverter));
      if (!callResultTypes.back()) {
        return rewriter.notifyMatchFailure(
            callOp.getLoc(), "failed to convert callee signature");
      }
    }

    auto newCallOp = rewriter.create<LLVM::CallOp>(
        callOp->getLoc(), callResultTypes, callOp.getCallee(),
        adaptor.getOperands());
    newCallOp->setAttrs(callOp->getAttrs());

    if (numResults <= 1) {
      rewriter.replaceOp(callOp, newCallOp->getResults());
      return success();
    }

    SmallVector<Value> results;
    results.reserve(numResults);
    for (auto index : llvm::seq<unsigned>(0, numResults)) {
      results.push_back(rewriter.create<LLVM::ExtractValueOp>(
          callOp->getLoc(), newCallOp->getResult(0), index));
    }
    rewriter.replaceOp(callOp, results);
    return success();
  }
};

/// Pattern for returning from a function, packs the results into a struct.
struct ReturnOpLowering : public ConvertOpToLLVMPattern<func::ReturnOp> {
public:
  using ConvertOpToLLVMPattern<func::ReturnOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(func::ReturnOp returnOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (returnOp->getNumOperands() <= 1) {
      rewriter.replaceOpWithNewOp<LLVM::ReturnOp>(returnOp,
                                                  adaptor.getOperands());
      return success();
    }

    auto returnedType = LLVM::LLVMStructType::getLiteral(
        returnOp->getContext(),
        llvm::to_vector(adaptor.getOperands().getTypes()));
    Value packed =
        rewriter.create<LLVM::UndefOp>(returnOp->getLoc(), returnedType);
    for (const auto &[index, value] : llvm::enumerate(adaptor.getOperands())) {
      packed = rewriter.create<LLVM::InsertValueOp>(returnOp->getLoc(), packed,
                                                    value, index);
    }
    rewriter.replaceOpWithNewOp<LLVM::ReturnOp>(returnOp, packed);
    return success();
  }
};

/// Pattern for returning from a function, packs the results into a struct.
struct GPUReturnOpLowering : public ConvertOpToLLVMPattern<gpu::ReturnOp> {
public:
  using ConvertOpToLLVMPattern<gpu::ReturnOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(gpu::ReturnOp returnOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (returnOp->getNumOperands() <= 1) {
      rewriter.replaceOpWithNewOp<LLVM::ReturnOp>(returnOp,
                                                  adaptor.getOperands());
      return success();
    }

    auto returnedType = LLVM::LLVMStructType::getLiteral(
        returnOp->getContext(),
        llvm::to_vector(adaptor.getOperands().getTypes()));
    Value packed =
        rewriter.create<LLVM::UndefOp>(returnOp->getLoc(), returnedType);
    for (const auto &[index, value] : llvm::enumerate(adaptor.getOperands())) {
      packed = rewriter.create<LLVM::InsertValueOp>(returnOp->getLoc(), packed,
                                                    value, index);
    }
    rewriter.replaceOpWithNewOp<LLVM::ReturnOp>(returnOp, packed);
    return success();
  }
};

/// TODO: Temporary until we migrate everything to opaque pointers
struct ReconcileUnrealizedPointerCasts
    : public OpRewritePattern<UnrealizedConversionCastOp> {
  using OpRewritePattern<UnrealizedConversionCastOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(UnrealizedConversionCastOp ucc,
                                PatternRewriter &rewriter) const override {
    auto inputs = ucc.getInputs();
    auto results = ucc.getResults();
    if (!(inputs.size() == 1 && results.size() == 1))
      return failure();
    auto inputTy = inputs[0].getType();
    auto outputTy = results[0].getType();
    if (!(isa<LLVM::LLVMPointerType>(inputTy) &&
          isa<LLVM::LLVMPointerType>(outputTy)))
      return failure();
    rewriter.replaceOpWithNewOp<LLVM::AddrSpaceCastOp>(ucc, outputTy,
                                                       inputs[0]);
    return success();
  }
};

struct AllocaScopeOpLowering
    : public ConvertOpToLLVMPattern<memref::AllocaScopeOp> {
  using ConvertOpToLLVMPattern<memref::AllocaScopeOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(memref::AllocaScopeOp allocaScopeOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    OpBuilder::InsertionGuard guard(rewriter);
    Location loc = allocaScopeOp.getLoc();

    // Split the current block before the AllocaScopeOp to create the inlining
    // point.
    auto *currentBlock = rewriter.getInsertionBlock();
    auto *remainingOpsBlock =
        rewriter.splitBlock(currentBlock, rewriter.getInsertionPoint());
    Block *continueBlock;
    if (allocaScopeOp.getNumResults() == 0) {
      continueBlock = remainingOpsBlock;
    } else {
      continueBlock = rewriter.createBlock(
          remainingOpsBlock, allocaScopeOp.getResultTypes(),
          SmallVector<Location>(allocaScopeOp->getNumResults(),
                                allocaScopeOp.getLoc()));
      rewriter.create<LLVM::BrOp>(loc, ValueRange(), remainingOpsBlock);
    }

    // Inline body region.
    Block *beforeBody = &allocaScopeOp.getBodyRegion().front();
    Block *afterBody = &allocaScopeOp.getBodyRegion().back();
    rewriter.inlineRegionBefore(allocaScopeOp.getBodyRegion(), continueBlock);

    // Save stack and then branch into the body of the region.
    rewriter.setInsertionPointToEnd(currentBlock);
    auto stackSaveOp =
        rewriter.create<LLVM::StackSaveOp>(loc, getVoidPtrType());
    rewriter.create<LLVM::BrOp>(loc, ValueRange(), beforeBody);

    // Replace the alloca_scope return with a branch that jumps out of the body.
    // Stack restore before leaving the body region.
    rewriter.setInsertionPointToEnd(afterBody);
    auto returnOp =
        cast<memref::AllocaScopeReturnOp>(afterBody->getTerminator());
    auto branchOp = rewriter.replaceOpWithNewOp<LLVM::BrOp>(
        returnOp, returnOp.getResults(), continueBlock);

    // Insert stack restore before jumping out the body of the region.
    rewriter.setInsertionPoint(branchOp);
    rewriter.create<LLVM::StackRestoreOp>(loc, stackSaveOp);

    // Replace the op with values return from the body region.
    rewriter.replaceOp(allocaScopeOp, continueBlock->getArguments());

    return success();
  }
};

/// Appends the patterns lowering operations from the Memref dialect to the LLVM
/// dialect using the C-style type conversion, i.e. converting memrefs to
/// pointer to arrays of arrays.
static void
populateCStyleMemRefLoweringPatterns(RewritePatternSet &patterns,
                                     LLVMTypeConverter &typeConverter,
                                     StringRef backend) {
  patterns.add<CAllocaOpLowering, CAllocOpLowering, CDeallocOpLowering,
               GetGlobalOpLowering, GlobalOpLowering, CLoadOpLowering,
               CStoreOpLowering, AllocaScopeOpLowering, CAtomicRMWOpLowering>(
      typeConverter);
  patterns.add<CMemcpyOpLowering>(typeConverter, backend);
}

struct GPUShuffleOpLowering : public ConvertOpToLLVMPattern<gpu::ShuffleOp> {
  using ConvertOpToLLVMPattern<gpu::ShuffleOp>::ConvertOpToLLVMPattern;

  /// Convert gpu dialect shfl mode enum to the equivalent nvvm one.
  static NVVM::ShflKind convertShflKind(gpu::ShuffleMode mode) {
    switch (mode) {
    case gpu::ShuffleMode::XOR:
      return NVVM::ShflKind::bfly;
    case gpu::ShuffleMode::UP:
      return NVVM::ShflKind::up;
    case gpu::ShuffleMode::DOWN:
      return NVVM::ShflKind::down;
    case gpu::ShuffleMode::IDX:
      return NVVM::ShflKind::idx;
    }
    llvm_unreachable("unknown shuffle mode");
  }

  /// Lowers a shuffle to the corresponding NVVM op.
  ///
  /// Convert the `width` argument into an activeMask (a bitmask which specifies
  /// which threads participate in the shuffle) and a maskAndClamp (specifying
  /// the highest lane which participates in the shuffle).
  ///
  ///     %one = llvm.constant(1 : i32) : i32
  ///     %minus_one = llvm.constant(-1 : i32) : i32
  ///     %thirty_two = llvm.constant(32 : i32) : i32
  ///     %num_lanes = llvm.sub %thirty_two, %width : i32
  ///     %active_mask = llvm.lshr %minus_one, %num_lanes : i32
  ///     %mask_and_clamp = llvm.sub %width, %one : i32
  ///     %shfl = nvvm.shfl.sync.bfly %active_mask, %value, %offset,
  ///         %mask_and_clamp : !llvm<"{ float, i1 }">
  ///     %shfl_value = llvm.extractvalue %shfl[0] :
  ///         !llvm<"{ float, i1 }">
  ///     %shfl_pred = llvm.extractvalue %shfl[1] :
  ///         !llvm<"{ float, i1 }">
  LogicalResult
  matchAndRewrite(gpu::ShuffleOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();

    auto valueTy = adaptor.getValue().getType();
    auto int32Type = IntegerType::get(rewriter.getContext(), 32);
    auto predTy = IntegerType::get(rewriter.getContext(), 1);

    Value one = rewriter.create<LLVM::ConstantOp>(loc, int32Type, 1);
    Value minusOne = rewriter.create<LLVM::ConstantOp>(loc, int32Type, -1);
    Value thirtyTwo = rewriter.create<LLVM::ConstantOp>(loc, int32Type, 32);
    Value numLeadInactiveLane = rewriter.create<LLVM::SubOp>(
        loc, int32Type, thirtyTwo, adaptor.getWidth());
    // Bit mask of active lanes: `(-1) >> (32 - activeWidth)`.
    Value activeMask = rewriter.create<LLVM::LShrOp>(loc, int32Type, minusOne,
                                                     numLeadInactiveLane);
    Value maskAndClamp;
    if (op.getMode() == gpu::ShuffleMode::UP) {
      // Clamp lane: `32 - activeWidth`
      maskAndClamp = numLeadInactiveLane;
    } else {
      // Clamp lane: `activeWidth - 1`
      maskAndClamp =
          rewriter.create<LLVM::SubOp>(loc, int32Type, adaptor.getWidth(), one);
    }

    bool predIsUsed = !op->getResult(1).use_empty();
    UnitAttr returnValueAndIsValidAttr = nullptr;
    Type resultTy = valueTy;
    if (predIsUsed) {
      returnValueAndIsValidAttr = rewriter.getUnitAttr();
      resultTy = LLVM::LLVMStructType::getLiteral(rewriter.getContext(),
                                                  {valueTy, predTy});
    }
    Value shfl = rewriter.create<NVVM::ShflOp>(
        loc, resultTy, activeMask, adaptor.getValue(), adaptor.getOffset(),
        maskAndClamp, convertShflKind(op.getMode()), returnValueAndIsValidAttr);
    if (predIsUsed) {
      Value shflValue = rewriter.create<LLVM::ExtractValueOp>(loc, shfl, 0);
      Value isActiveSrcLane =
          rewriter.create<LLVM::ExtractValueOp>(loc, shfl, 1);
      rewriter.replaceOp(op, {shflValue, isActiveSrcLane});
    } else {
      rewriter.replaceOp(op, {shfl, nullptr});
    }
    return success();
  }
};

struct GPULaneIdOpToNVVM : ConvertOpToLLVMPattern<gpu::LaneIdOp> {
  using ConvertOpToLLVMPattern<gpu::LaneIdOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(gpu::LaneIdOp op, gpu::LaneIdOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    MLIRContext *context = rewriter.getContext();
    LLVM::ConstantRangeAttr bounds = nullptr;
    if (std::optional<APInt> upperBound = op.getUpperBound())
      bounds = rewriter.getAttr<LLVM::ConstantRangeAttr>(
          /*bitWidth=*/32, /*lower=*/0, upperBound->getZExtValue());
    else
      bounds = rewriter.getAttr<LLVM::ConstantRangeAttr>(
          /*bitWidth=*/32, /*lower=*/0, /*upper=*/kWarpSize);
    Value newOp =
        rewriter.create<NVVM::LaneIdOp>(loc, rewriter.getI32Type(), bounds);
    // Truncate or extend the result depending on the index bitwidth specified
    // by the LLVMTypeConverter options.
    const unsigned indexBitwidth = getTypeConverter()->getIndexTypeBitwidth();
    if (indexBitwidth > 32) {
      newOp = rewriter.create<LLVM::SExtOp>(
          loc, IntegerType::get(context, indexBitwidth), newOp);
    } else if (indexBitwidth < 32) {
      newOp = rewriter.create<LLVM::TruncOp>(
          loc, IntegerType::get(context, indexBitwidth), newOp);
    }
    rewriter.replaceOp(op, {newOp});
    return success();
  }
};

struct GPUBarrierToNVVM : ConvertOpToLLVMPattern<gpu::BarrierOp> {
  using ConvertOpToLLVMPattern<gpu::BarrierOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(gpu::BarrierOp op, gpu::BarrierOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<NVVM::Barrier0Op>(op);
    return success();
  }
};

namespace mlir {
namespace gpu {
namespace index_lowering {
enum class IndexKind : uint32_t { Other = 0, Block = 1, Grid = 2 };
enum class IntrType : uint32_t {
  None = 0,
  Id = 1,
  Dim = 2,
};

// Rewriting that replaces Op with XOp, YOp, or ZOp depending on the dimension
// that Op operates on.  Op is assumed to return an `index` value and
// XOp, YOp and ZOp are assumed to return an `llvm.i32` value.  Depending on
// `indexBitwidth`, sign-extend or truncate the resulting value to match the
// bitwidth expected by the consumers of the value.
template <typename Op, typename XOp, typename YOp, typename ZOp>
struct OpLowering : public OpConversionPattern<Op> {
private:
  unsigned indexBitwidth;
  IndexKind indexKind;
  IntrType intrType;

public:
  explicit OpLowering(const LLVMTypeConverter &typeConverter,
                      PatternBenefit benefit = 1)
      : OpConversionPattern<Op>(typeConverter, &typeConverter.getContext(),
                                benefit),
        indexBitwidth(typeConverter.getIndexTypeBitwidth()),
        indexKind(IndexKind::Other), intrType(IntrType::None) {}

  explicit OpLowering(const LLVMTypeConverter &typeConverter,
                      IndexKind indexKind, IntrType intrType,
                      PatternBenefit benefit = 1)
      : OpConversionPattern<Op>(typeConverter, &typeConverter.getContext(),
                                benefit),
        indexBitwidth(typeConverter.getIndexTypeBitwidth()),
        indexKind(indexKind), intrType(intrType) {}

  // Convert the kernel arguments to an LLVM type, preserve the rest.
  LogicalResult
  matchAndRewrite(Op op, typename OpConversionPattern<Op>::OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    MLIRContext *context = rewriter.getContext();
    Operation *newOp;
    switch (op.getDimension()) {
    case gpu::Dimension::x:
      newOp = rewriter.create<XOp>(loc, IntegerType::get(context, 32));
      break;
    case gpu::Dimension::y:
      newOp = rewriter.create<YOp>(loc, IntegerType::get(context, 32));
      break;
    case gpu::Dimension::z:
      newOp = rewriter.create<ZOp>(loc, IntegerType::get(context, 32));
      break;
    }

    // Order of priority for bounds:
    // 1. The upper_bound attribute
    // 2. Inherent attributes on a surrounding gpu.func
    // 3. Discardable attributes on a surrounding function of any kind
    // The below code handles these in reverse order so that more important
    // sources overwrite less important ones.
    DenseI32ArrayAttr funcBounds = nullptr;
    if (auto funcOp = op->template getParentOfType<FunctionOpInterface>()) {
      switch (indexKind) {
      case IndexKind::Block: {
        auto blockHelper =
            gpu::GPUDialect::KnownBlockSizeAttrHelper(op.getContext());
        if (blockHelper.isAttrPresent(funcOp))
          funcBounds = blockHelper.getAttr(funcOp);
        break;
      }
      case IndexKind::Grid: {
        auto gridHelper =
            gpu::GPUDialect::KnownGridSizeAttrHelper(op.getContext());
        if (gridHelper.isAttrPresent(funcOp))
          funcBounds = gridHelper.getAttr(funcOp);
        break;
      }
      case IndexKind::Other:
        break;
      }
    }
    if (auto gpuFunc = op->template getParentOfType<gpu::GPUFuncOp>()) {
      switch (indexKind) {
      case IndexKind::Block:
        funcBounds = gpuFunc.getKnownBlockSizeAttr();
        break;
      case IndexKind::Grid:
        funcBounds = gpuFunc.getKnownGridSizeAttr();
        break;
      case IndexKind::Other:
        break;
      }
    }
    std::optional<int32_t> upperBound;
    if (funcBounds)
      upperBound =
          funcBounds.asArrayRef()[static_cast<uint32_t>(op.getDimension())];
    if (auto opBound = op.getUpperBound())
      upperBound = opBound->getZExtValue();

    if (upperBound && intrType != IntrType::None) {
      int32_t min = (intrType == IntrType::Dim ? 1 : 0);
      int32_t max = *upperBound == std::numeric_limits<int32_t>::max()
                        ? *upperBound
                        : *upperBound + (intrType == IntrType::Id ? 0 : 1);
      newOp->setAttr("range", LLVM::ConstantRangeAttr::get(
                                  rewriter.getContext(), 32, min, max));
    }
    if (indexBitwidth > 32) {
      newOp = rewriter.create<LLVM::SExtOp>(
          loc, IntegerType::get(context, indexBitwidth), newOp->getResult(0));
    } else if (indexBitwidth < 32) {
      newOp = rewriter.create<LLVM::TruncOp>(
          loc, IntegerType::get(context, indexBitwidth), newOp->getResult(0));
    }

    rewriter.replaceOpWithNewOp<UnrealizedConversionCastOp>(
        op, op->getResultTypes(), newOp->getResults());
    return success();
  }
};
} // namespace index_lowering
} // namespace gpu
} // namespace mlir

/// Appends the patterns lowering operations from the Func dialect to the LLVM
/// dialect using the C-style type conversion, i.e. converting memrefs to
/// pointer to arrays of arrays.
static void
populateCStyleGPUFuncLoweringPatterns(RewritePatternSet &patterns,
                                      LLVMTypeConverter &typeConverter,
                                      std::string gpuTarget, bool func) {
  if (func) {
    patterns.add<GPUReturnOpLowering>(typeConverter);
    patterns.add<GPUFuncOpLowering>(
        typeConverter,
        /*allocaAddrSpace=*/0,
        StringAttr::get(&typeConverter.getContext(),
                        gpuTarget == "cuda"
                            ? NVVM::NVVMDialect::getKernelFuncAttrName()
                            : ROCDL::ROCDLDialect::getKernelFuncAttrName()));
  } else {
    if (gpuTarget == "cuda") {
      using namespace mlir::gpu::index_lowering;
      PatternBenefit benefit(1);
      patterns.add<gpu::index_lowering::OpLowering<
          gpu::ThreadIdOp, NVVM::ThreadIdXOp, NVVM::ThreadIdYOp,
          NVVM::ThreadIdZOp>>(typeConverter, IndexKind::Block, IntrType::Id,
                              benefit);
      patterns.add<gpu::index_lowering::OpLowering<
          gpu::BlockDimOp, NVVM::BlockDimXOp, NVVM::BlockDimYOp,
          NVVM::BlockDimZOp>>(typeConverter, IndexKind::Block, IntrType::Dim,
                              benefit);
      patterns.add<gpu::index_lowering::OpLowering<
          gpu::ClusterIdOp, NVVM::ClusterIdXOp, NVVM::ClusterIdYOp,
          NVVM::ClusterIdZOp>>(typeConverter, IndexKind::Other, IntrType::Id,
                               benefit);
      patterns.add<gpu::index_lowering::OpLowering<
          gpu::ClusterDimOp, NVVM::ClusterDimXOp, NVVM::ClusterDimYOp,
          NVVM::ClusterDimZOp>>(typeConverter, IndexKind::Other, IntrType::Dim,
                                benefit);
      patterns.add<gpu::index_lowering::OpLowering<
          gpu::ClusterBlockIdOp, NVVM::BlockInClusterIdXOp,
          NVVM::BlockInClusterIdYOp, NVVM::BlockInClusterIdZOp>>(
          typeConverter, IndexKind::Other, IntrType::Id, benefit);
      patterns.add<gpu::index_lowering::OpLowering<
          gpu::ClusterDimBlocksOp, NVVM::ClusterDimBlocksXOp,
          NVVM::ClusterDimBlocksYOp, NVVM::ClusterDimBlocksZOp>>(
          typeConverter, IndexKind::Other, IntrType::Dim, benefit);
      patterns.add<
          gpu::index_lowering::OpLowering<gpu::BlockIdOp, NVVM::BlockIdXOp,
                                          NVVM::BlockIdYOp, NVVM::BlockIdZOp>>(
          typeConverter, IndexKind::Grid, IntrType::Id, benefit);
      patterns.add<
          gpu::index_lowering::OpLowering<gpu::GridDimOp, NVVM::GridDimXOp,
                                          NVVM::GridDimYOp, NVVM::GridDimZOp>>(
          typeConverter, IndexKind::Grid, IntrType::Dim, benefit);
      patterns.add<GPULaneIdOpToNVVM, GPUShuffleOpLowering>(typeConverter,
                                                            benefit);

      populateLibDeviceConversionPatterns(typeConverter, patterns, benefit);
      patterns.add<GPUBarrierToNVVM>(typeConverter, benefit);
    }
  }
}

/// Appends the patterns lowering operations from the Func dialect to the LLVM
/// dialect using the C-style type conversion, i.e. converting memrefs to
/// pointer to arrays of arrays.
static void
populateCStyleFuncLoweringPatterns(RewritePatternSet &patterns,
                                   LLVMTypeConverter &typeConverter) {
  patterns.add<CallOpLowering, FuncOpLowering, ReturnOpLowering>(typeConverter);
}

static void removeUnsupportedLifeTimes(mlir::Operation *root) {
  llvm::SmallVector<mlir::Operation *> toErase;
  root->walk([&](mlir::Operation *op) {
    if (auto lifetimeStart = llvm::dyn_cast<mlir::LLVM::LifetimeStartOp>(op)) {
      if (!llvm::isa_and_nonnull<mlir::LLVM::AllocaOp, mlir::LLVM::PoisonOp>(
              lifetimeStart.getPtr().getDefiningOp()))
        toErase.push_back(op);
    } else if (auto lifetimeEnd =
                   llvm::dyn_cast<mlir::LLVM::LifetimeEndOp>(op)) {
      if (!llvm::isa_and_nonnull<mlir::LLVM::AllocaOp, mlir::LLVM::PoisonOp>(
              lifetimeEnd.getPtr().getDefiningOp()))
        toErase.push_back(op);
    }
  });
  for (mlir::Operation *op : toErase)
    op->erase();
}

template <typename T>
static void addOpaquePointerConversion(LLVMTypeConverter &converter) {
  converter.addConversion([&converter](T) -> Type {
    return LLVM::LLVMPointerType::get(&converter.getContext());
  });
}

//===-----------------------------------------------------------------------===/

namespace {

struct ConvertPolygeistToLLVMPass
    : public mlir::enzyme::impl::ConvertPolygeistToLLVMBase<
          ConvertPolygeistToLLVMPass> {
  using ConvertPolygeistToLLVMBase::ConvertPolygeistToLLVMBase;

  void convertModule(ModuleOp m, bool gpuModule) {
    const auto &dataLayoutAnalysis = getAnalysis<DataLayoutAnalysis>();

    if (useCStyleMemRef && useBarePtrCallConv) {
      emitError(m.getLoc()) << "C-style memref lowering is not compatible with "
                               "bare-pointer calling convention";
      signalPassFailure();
      return;
    }
    if (gpuModule) {
      // Request C wrapper emission.
      for (auto func : m.getOps<func::FuncOp>()) {
        func->setAttr(LLVM::LLVMDialect::getEmitCWrapperAttrName(),
                      UnitAttr::get(&getContext()));
      }
    }

    LowerToLLVMOptions options(&getContext(),
                               dataLayoutAnalysis.getAtOrAbove(m));
    // TODO need to tweak options.indexBitwidth in some cases? consult
    // LowerGpuOpsToNVVMOpsPass
    options.useBarePtrCallConv = useBarePtrCallConv;
    if (indexBitwidth != kDeriveIndexBitwidthFromDataLayout)
      options.overrideIndexBitwidth(indexBitwidth);

    options.dataLayout = llvm::DataLayout(this->dataLayout);

    // Define the type converter. Override the default behavior for memrefs if
    // requested.
    LLVMTypeConverter converter(&getContext(), options, &dataLayoutAnalysis);
    if (useCStyleMemRef) {
      converter.addConversion([&](MemRefType type) -> std::optional<Type> {
        auto elTy = convertMemrefElementTypeForLLVMPointer(type, converter);
        if (!elTy)
          return Type();
        return LLVM::LLVMPointerType::get(type.getContext(),
                                          type.getMemorySpaceAsInt());
      });
    }
    addOpaquePointerConversion<gpu::AsyncTokenType>(converter);

    SmallVector<Operation *> gmods;
    m->walk([&](gpu::GPUModuleOp mod) { gmods.push_back(mod); });

    if (backend == "cuda" && gmods.size()) {
      OpBuilder rewriter(m);
      auto i32 = rewriter.getIntegerType(32);
      auto i64 = rewriter.getIntegerType(64);
      auto ptrty = LLVM::LLVMPointerType::get(rewriter.getContext());
      Type tys[] = {ptrty, i64, i32, i64, i32, ptrty, i64, ptrty};
      LLVM::lookupOrCreateFn(rewriter, m, "cudaLaunchKernel", tys, i32);
    }

    for (auto mod : gmods) {
      RewritePatternSet patterns(&getContext());

      // Insert our custom version of GPUFuncLowering
      ConversionTarget target(*mod->getContext());
      if (useCStyleMemRef) {
        populateCStyleGPUFuncLoweringPatterns(patterns, converter, backend,
                                              false);
        if (backend == "cuda") {
          target.addIllegalDialect<gpu::GPUDialect>();
          target.addLegalOp<gpu::GPUModuleOp, gpu::GPUFuncOp, gpu::ReturnOp>();
          target.addLegalDialect<NVVM::NVVMDialect, LLVM::LLVMDialect>();
          target.addLegalOp<UnrealizedConversionCastOp>();
        }
      }

      ConversionConfig conversionConfig;
      if (failed(applyPartialConversion(mod, target, std::move(patterns),
                                        conversionConfig))) {
        mod->emitError() << "failed to apply conversion patterns";
        return signalPassFailure();
      }
      if (failed(applyPatternsAndFoldGreedily(mod, {}))) {
        mod->emitError() << "failed to apply folding";
        return signalPassFailure();
      }
    };

    LLVMConversionTarget target(getContext());
    RewritePatternSet patterns(&getContext());

    std::string gpuTarget = backend;

    populatePolygeistToLLVMConversionPatterns(converter, patterns);
    populateSCFToControlFlowConversionPatterns(patterns);
    // populateForBreakToWhilePatterns(patterns);
    cf::populateControlFlowToLLVMConversionPatterns(converter, patterns);
    if (useCStyleMemRef) {
      populateCStyleMemRefLoweringPatterns(patterns, converter, backend);
      populateCStyleFuncLoweringPatterns(patterns, converter);
    } else {
      populateFinalizeMemRefToLLVMConversionPatterns(converter, patterns);
      populateFuncToLLVMConversionPatterns(converter, patterns);
    }

    ub::populateUBToLLVMConversionPatterns(converter, patterns);

    // TODO use lower priority for libm pending
    // https://github.com/llvm/llvm-project/pull/127291
    populateMathToLLVMConversionPatterns(converter, patterns);
    populateMathToLibmConversionPatterns(patterns);

    populateOpenMPToLLVMConversionPatterns(converter, patterns);
    arith::populateArithToLLVMConversionPatterns(converter, patterns);

    // Insert our custom version of GPUFuncLowering
    if (useCStyleMemRef) {
      populateCStyleGPUFuncLoweringPatterns(patterns, converter, backend, true);
    }

    // Our custom versions of the gpu patterns
    if (useCStyleMemRef) {
      patterns.add<ConvertLaunchFuncOpToGpuRuntimeCallPattern>(
          converter, "gpu.binary", gpuTarget);

      patterns.add<ConvertGPUModuleOp>(converter, "gpu.binary", gpuTarget);
      // patterns.add<LegalizeLaunchFuncOpPattern>(
      //     converter, /*kernelBarePtrCallConv*/ true,
      //     /*kernelIntersperseSizeCallConv*/ false);
      patterns.add<ConvertAllocOpToGpuRuntimeCallPattern<true>>(converter,
                                                                gpuTarget);
      patterns.add<ConvertDeallocOpToGpuRuntimeCallPattern<true>>(converter,
                                                                  gpuTarget);
      patterns.add<ConvertXLAWrapperPattern<true>>(converter, gpuTarget);
    } else {
      patterns.add<ConvertAllocOpToGpuRuntimeCallPattern<false>>(converter,
                                                                 gpuTarget);
      patterns.add<ConvertDeallocOpToGpuRuntimeCallPattern<false>>(converter,
                                                                   gpuTarget);
      patterns.add<ConvertXLAWrapperPattern<false>>(converter, gpuTarget);
    }

    patterns
        .add<LLVMOpLowering, GlobalOpTypeConversion, ReturnOpTypeConversion>(
            converter);
    patterns.add<URLLVMOpLowering>(converter);

    // Legality callback for operations that checks whether their operand and
    // results types are converted.
    auto areAllTypesConverted = [&](Operation *op) -> std::optional<bool> {
      // Check if TyepAttrs got converted
      for (auto &attr : op->getAttrs())
        if (auto tyAttr = dyn_cast<TypeAttr>(attr.getValue()))
          if (converter.convertType(tyAttr.getValue()) != tyAttr.getValue())
            return std::nullopt;

      SmallVector<Type> convertedResultTypes;
      if (failed(converter.convertTypes(op->getResultTypes(),
                                        convertedResultTypes)))
        return std::nullopt;
      SmallVector<Type> convertedOperandTypes;
      if (failed(converter.convertTypes(op->getOperandTypes(),
                                        convertedOperandTypes)))
        return std::nullopt;

      return convertedResultTypes == op->getResultTypes() &&
             convertedOperandTypes == op->getOperandTypes();
    };

    configureOpenMPToLLVMConversionLegality(target, converter);
    target.addIllegalOp<scf::ForOp, scf::IfOp, scf::ParallelOp, scf::WhileOp,
                        scf::ExecuteRegionOp, func::FuncOp>();
    target.addDynamicallyLegalDialect<LLVM::LLVMDialect>(areAllTypesConverted);
    target.addDynamicallyLegalOp<LLVM::GlobalOp>(
        [&](LLVM::GlobalOp op) -> std::optional<bool> {
          if (converter.convertType(op.getGlobalType()) == op.getGlobalType())
            return true;
          return std::nullopt;
        });
    target.addDynamicallyLegalOp<gpu::GPUModuleOp>(
        [&](gpu::GPUModuleOp op) -> std::optional<bool> {
          if (op->hasAttr("polygeist_stubs"))
            return true;
          return std::nullopt;
        });
    target.addDynamicallyLegalOp<LLVM::ReturnOp>(
        [&](LLVM::ReturnOp op) -> std::optional<bool> {
          // Outside global ops, defer to the normal type-based check. Note
          // that the infrastructure will not do it automatically because
          // per-op checks override dialect-level checks unconditionally.
          if (!isa<LLVM::GlobalOp>(op->getParentOp()))
            return areAllTypesConverted(op);

          SmallVector<Type> convertedOperandTypes;
          if (failed(converter.convertTypes(op->getOperandTypes(),
                                            convertedOperandTypes)))
            return std::nullopt;
          return convertedOperandTypes == op->getOperandTypes();
        });
    /*
    target.addDynamicallyLegalOp<UnrealizedConversionCastOp>(
        [&](Operation *op) { return op->getOperand(0).getType() !=
    op->getResult(0).getType(); });
        */

    // target.addIllegalOp<UnrealizedConversionCastOp>();
    if (failed(applyPartialConversion(m, target, std::move(patterns))))
      signalPassFailure();
    {
      RewritePatternSet patterns(&getContext());
      patterns.insert<ReconcileUnrealizedPointerCasts>(&getContext());
      if (failed(applyPatternsAndFoldGreedily(m, std::move(patterns)))) {
        llvm::errs() << " failed to reconcile unrealized pointer casts\n";
        llvm::errs() << *m << "\n";
        signalPassFailure();
      }
    }

    if (StringRef(gpuTarget).starts_with("xla")) {
      m->walk([](LLVM::CallOp call) {
        if (auto callee = call.getCallee()) {
          if (callee == "cudaDeviceSynchronize") {
            call->erase();
          }
        }
      });
      m->walk([](LLVM::LLVMFuncOp call) {
        if (call.getName() == "cudaDeviceSynchronize") {
          call->erase();
        }
      });
    }

    removeUnsupportedLifeTimes(m);
  }

  void runOnOperation() override {
    ModuleOp m = getOperation();
    convertModule(m, /* gpuModule */ false);
  }
};
} // namespace
