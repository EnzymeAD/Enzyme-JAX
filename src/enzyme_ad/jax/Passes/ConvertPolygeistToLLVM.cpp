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
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/Passes.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/FunctionCallUtils.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Target/LLVMIR/Import.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"

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

struct Memref2PointerOpLowering
    : public ConvertOpToLLVMPattern<Memref2PointerOp> {
  using ConvertOpToLLVMPattern<Memref2PointerOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(Memref2PointerOp op, OpAdaptor transformed,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    auto LPT = op.getType().cast<LLVM::LLVMPointerType>();
    auto space0 = op.getSource().getType().getMemorySpaceAsInt();
    if (transformed.getSource().getType().isa<LLVM::LLVMPointerType>()) {
      mlir::Value ptr = rewriter.create<LLVM::BitcastOp>(
          loc, LLVM::LLVMPointerType::get(op.getContext(), space0),
          transformed.getSource());
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
    ptr = rewriter.create<LLVM::BitcastOp>(
        loc, LLVM::LLVMPointerType::get(op.getContext(), space0), ptr);
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
    if (auto PT = dyn_cast<LLVM::LLVMPointerType>(convertedType)) {
      rewriter.replaceOpWithNewOp<LLVM::BitcastOp>(op, PT, adaptor.getSource());
      return success();
    }

    auto descr = MemRefDescriptor::poison(rewriter, loc, convertedType);
    auto ptr = rewriter.create<LLVM::BitcastOp>(
        op.getLoc(), descr.getElementPtrType(), adaptor.getSource());

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
              ? LLVM::lookupOrCreateGenericAllocFn(module, getIndexType())
              : LLVM::lookupOrCreateMallocFn(module, getIndexType());
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
              ? LLVM::lookupOrCreateGenericFreeFn(module)
              : LLVM::lookupOrCreateFreeFn(module);
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
      auto elementsAttr = globalOp.getInitialValue()->cast<ElementsAttr>();
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
      auto attrsDict = argAttrDicts[i].cast<DictionaryAttr>();
      convertedAttrs.reserve(attrsDict.size());
      for (const NamedAttribute &attr : attrsDict) {
        const auto convert = [&](const NamedAttribute &attr) {
          return TypeAttr::get(typeConverter.convertType(
              attr.getValue().cast<TypeAttr>().getValue()));
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

/// Pattern for function declarations and definitions.
struct FuncOpLowering : public ConvertOpToLLVMPattern<func::FuncOp> {
public:
  using ConvertOpToLLVMPattern<func::FuncOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(func::FuncOp funcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
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
          funcOp->getAttr(kLLVMLinkageAttrName).cast<mlir::LLVM::LinkageAttr>();
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
        callOp->getLoc(), callResultTypes, adaptor.getOperands(),
        callOp->getAttrs());

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
    if (!(inputTy.isa<LLVM::LLVMPointerType>() &&
          outputTy.isa<LLVM::LLVMPointerType>()))
      return failure();
    rewriter.replaceOpWithNewOp<LLVM::BitcastOp>(ucc, outputTy, inputs[0]);
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
                                     LLVMTypeConverter &typeConverter) {
  patterns.add<CAllocaOpLowering, CAllocOpLowering, CDeallocOpLowering,
               GetGlobalOpLowering, GlobalOpLowering, CLoadOpLowering,
               CStoreOpLowering, AllocaScopeOpLowering, CAtomicRMWOpLowering>(
      typeConverter);
}

/// Appends the patterns lowering operations from the Func dialect to the LLVM
/// dialect using the C-style type conversion, i.e. converting memrefs to
/// pointer to arrays of arrays.
static void
populateCStyleFuncLoweringPatterns(RewritePatternSet &patterns,
                                   LLVMTypeConverter &typeConverter) {
  patterns.add<CallOpLowering, FuncOpLowering, ReturnOpLowering>(typeConverter);
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

    RewritePatternSet patterns(&getContext());

    populatePolygeistToLLVMConversionPatterns(converter, patterns);
    populateSCFToControlFlowConversionPatterns(patterns);
    // populateForBreakToWhilePatterns(patterns);
    cf::populateControlFlowToLLVMConversionPatterns(converter, patterns);
    if (useCStyleMemRef) {
      populateCStyleMemRefLoweringPatterns(patterns, converter);
      populateCStyleFuncLoweringPatterns(patterns, converter);
    } else {
      populateFinalizeMemRefToLLVMConversionPatterns(converter, patterns);
      populateFuncToLLVMConversionPatterns(converter, patterns);
    }

    // TODO use lower priority for libm pending
    // https://github.com/llvm/llvm-project/pull/127291
    populateMathToLLVMConversionPatterns(converter, patterns);
    populateMathToLibmConversionPatterns(patterns);

    populateOpenMPToLLVMConversionPatterns(converter, patterns);
    arith::populateArithToLLVMConversionPatterns(converter, patterns);

    // Our custom versions of the gpu patterns
    if (useCStyleMemRef) {
      // patterns.add<ConvertLaunchFuncOpToGpuRuntimeCallPattern>(
      //     converter, gpu::getDefaultGpuBinaryAnnotation(), gpuTarget);
      // patterns.add<ConvertAllocOpToGpuRuntimeCallPattern>(converter);
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

    LLVMConversionTarget target(getContext());
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
      (void)applyPatternsAndFoldGreedily(m, std::move(patterns));
    }
  }

  void runOnOperation() override {
    ModuleOp m = getOperation();
    convertModule(m, /* gpuModule */ false);
  }
};
} // namespace
