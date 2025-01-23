#include "mlir/Analysis/DataLayoutAnalysis.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/Affine/Analysis/LoopAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineMemoryOpInterfaces.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/IR/AffineValueMap.h"
#include "mlir/Dialect/Affine/Transforms/Transforms.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Utils/MemRefUtils.h"
#include "mlir/Dialect/NVGPU/IR/NVGPUDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/LogicalResult.h"
#include "src/enzyme_ad/jax/Dialect/Ops.h"

#include "GPULowering.h"
#include "Utils.h"

using namespace mlir;

namespace {

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

struct ReinterpretCastOpLowering
    : public ConvertOpToLLVMPattern<memref::ReinterpretCastOp> {
public:
  using ConvertOpToLLVMPattern<
      memref::ReinterpretCastOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(memref::ReinterpretCastOp castOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto sourceType = castOp.getSource().getType();
    sourceType.getElementType();

    Value newAddr = adaptor.getSource();
    Type newAddrType = getTypeConverter()->convertType(castOp.getType());
    Type newElType =
        getTypeConverter()->convertType(castOp.getType().getElementType());
    MemRefType targetMemRefType =
        cast<MemRefType>(castOp.getResult().getType());

    Value offset;
    // Set offset.
    if (castOp.isDynamicOffset(0))
      offset = adaptor.getOffsets()[0];
    else
      offset = rewriter.create<LLVM::ConstantOp>(
          castOp->getLoc(), getTypeConverter()->getIndexType(),
          castOp.getStaticOffset(0));

    // // Set sizes and strides.
    // unsigned dynSizeId = 0;
    // unsigned dynStrideId = 0;
    // Value curStride = nullptr;
    // for (int e = -1, i = targetMemRefType.getRank(); i > e; --i) {
    //   Value size, stride;
    //   if (castOp.isDynamicSize(i)) {
    //     size = adaptor.getSizes()[dynSizeId++];
    //   } else {
    //     size = rewriter.create<arith::ConstantIndexOp>(castOp->getLoc(),
    //     castOp.getStaticSize(i));
    //   }
    //   if (castOp.isDynamicStride(i)) {
    //     stride = adaptor.getStrides()[dynStrideId++];
    //   } else {
    //     stride = rewriter.create<arith::ConstantIndexOp>(castOp->getLoc(),
    //     castOp.getStaticStride(i));
    //   }
    //   if (!curStride)
    //     curStride = stride;
    // }

    rewriter.replaceOpWithNewOp<LLVM::GEPOp>(castOp, newAddrType, newElType,
                                             newAddr, ValueRange{offset});

    return success();
  }
};

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

    IntegerAttr alignment = globalOp.getAlignmentAttr();
    bool dso_local = globalOp->getAttr("polygeist.cuda_device") ||
                     globalOp->getAttr("polygeist.cuda_constant");
    bool thread_local_ = false;
    LLVM::UnnamedAddrAttr unnamed_addr = nullptr;
    StringAttr section = nullptr;
    bool externallyInitialized = false;
    auto newGlobal = rewriter.replaceOpWithNewOp<LLVM::GlobalOp>(
        globalOp, convertedType, globalOp.getConstant(), globalOp.getSymName(),
        linkage, dso_local, thread_local_, externallyInitialized, initialValue,
        alignment, originalType.getMemorySpaceAsInt(), unnamed_addr, section,
        /*comdat=*/nullptr, /*dbg_expr=*/nullptr);
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

/// Pattern for lowering automatic stack allocations.
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

    // TODO index size
    Type indexType = rewriter.getI64Type();
    return rewriter.create<LLVM::ConstantOp>(
        original.getLoc(), indexType,
        rewriter.getIntegerAttr(indexType,
                                original.getType().getRank() == 0
                                    ? 1
                                    : original.getType().getDimSize(0)));
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
    if (!originalType.getLayout().isIdentity())
      return rewriter.notifyMatchFailure(allocaOp,
                                         "Memref layout is not identity");

    if (!originalType.hasStaticShape())
      return rewriter.notifyMatchFailure(allocaOp, "Alloca with dynamic sizes");
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

struct AtAddrLower : public ConvertOpToLLVMPattern<enzymexla::AtAddrOp> {
  using ConvertOpToLLVMPattern<enzymexla::AtAddrOp>::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(enzymexla::AtAddrOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOp(op, ValueRange({adaptor.getAddr()}));
    return success();
  }
};

static bool isAtAddrMemref(Value v) {
  return v.getDefiningOp<enzymexla::AtAddrOp>();
}

static llvm::FailureOr<Value>
getAccessPointer(RewriterBase &rewriter, const LLVMTypeConverter &typeConverter,
                 Operation *op, TypedValue<MemRefType> baseMemref,
                 Value basePtr, ValueRange indices) {
  auto baseMemrefTy = baseMemref.getType();
  auto isContiguious =
      mlir::trailingNDimsContiguous(baseMemrefTy, baseMemrefTy.getRank());
  if (!isContiguious && !isAtAddrMemref(baseMemref))
    return rewriter.notifyMatchFailure(op, "Memref layout is not contiguous");

  if (indices.size() != 1)
    return rewriter.notifyMatchFailure(op, "Only 1-d memrefs for now");

  Type tyForOffset = typeConverter.convertType(baseMemrefTy.getElementType());
  if (!tyForOffset)
    return rewriter.notifyMatchFailure(op, "Could not convert el type");
  basePtr =
      rewriter.create<LLVM::GEPOp>(op->getLoc(), basePtr.getType(), tyForOffset,
                                   basePtr, ValueRange{indices[0]});
  return basePtr;
}

struct DeviceAsyncCopyOpLowerig
    : public ConvertOpToLLVMPattern<nvgpu::DeviceAsyncCopyOp> {
  using ConvertOpToLLVMPattern<
      nvgpu::DeviceAsyncCopyOp>::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(nvgpu::DeviceAsyncCopyOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    Location loc = op.getLoc();
    auto dstMemrefType = cast<MemRefType>(op.getDst().getType());
    auto dstPtrRes =
        getAccessPointer(rewriter, *getTypeConverter(), op, op.getDst(),
                         adaptor.getDst(), adaptor.getDstIndices());
    if (failed(dstPtrRes))
      return failure();
    Value dstPtr = *dstPtrRes;
    FailureOr<unsigned> dstAddressSpace =
        getTypeConverter()->getMemRefAddressSpace(dstMemrefType);
    if (failed(dstAddressSpace))
      return rewriter.notifyMatchFailure(
          loc, "destination memref address space not convertible to integer");

    auto srcMemrefType = cast<MemRefType>(op.getSrc().getType());
    FailureOr<unsigned> srcAddressSpace =
        getTypeConverter()->getMemRefAddressSpace(srcMemrefType);
    if (failed(srcAddressSpace))
      return rewriter.notifyMatchFailure(
          loc, "source memref address space not convertible to integer");

    auto scrPtrRes =
        getAccessPointer(rewriter, *getTypeConverter(), op, op.getSrc(),
                         adaptor.getSrc(), adaptor.getSrcIndices());
    if (failed(scrPtrRes))
      return failure();
    Value scrPtr = *scrPtrRes;
    // Intrinsics takes a global pointer so we need an address space cast.
    auto srcPointerGlobalType = LLVM::LLVMPointerType::get(
        op->getContext(), NVVM::NVVMMemorySpace::kGlobalMemorySpace);
    scrPtr = b.create<LLVM::AddrSpaceCastOp>(srcPointerGlobalType, scrPtr);
    int64_t dstElements = adaptor.getDstElements().getZExtValue();
    int64_t sizeInBytes =
        (dstMemrefType.getElementTypeBitWidth() * dstElements) / 8;
    // When the optional SrcElements argument is *not* present, the regular
    // CpAsyncOp is generated. CopyAsyncOp reads bytes from source (global
    // memory) to fill DstElements number of elements in the destination
    // (shared memory).
    Value srcBytes = adaptor.getSrcElements();
    if (srcBytes) {
      // When the optional SrcElements argument is present, the source (global
      // memory) of CpAsyncOp is read only for SrcElements number of elements.
      // The rest of the DstElements in the destination (shared memory) are
      // filled with zeros.
      Value c3I32 =
          b.create<LLVM::ConstantOp>(b.getI32Type(), b.getI32IntegerAttr(3));
      Value bitwidth = b.create<LLVM::ConstantOp>(
          b.getI32Type(),
          b.getI32IntegerAttr(srcMemrefType.getElementTypeBitWidth()));
      Value srcElementsI32 = b.create<LLVM::TruncOp>(b.getI32Type(), srcBytes);
      srcBytes = b.create<LLVM::LShrOp>(
          b.create<LLVM::MulOp>(bitwidth, srcElementsI32), c3I32);
    }
    // Cache global (.cg) for 16 dst bytes, Cache all (.ca) for sizes other than
    // 16 dst bytes.
    NVVM::LoadCacheModifierKind cacheModifier =
        (op.getBypassL1().value_or(false) && sizeInBytes == 16)
            ? NVVM::LoadCacheModifierKind::CG
            : NVVM::LoadCacheModifierKind::CA;

    b.create<NVVM::CpAsyncOp>(
        dstPtr, scrPtr, rewriter.getI32IntegerAttr(sizeInBytes),
        NVVM::LoadCacheModifierKindAttr::get(op->getContext(), cacheModifier),
        srcBytes);

    // Drop the result token.
    Value zero = b.create<LLVM::ConstantOp>(
        IntegerType::get(op.getContext(), 32), rewriter.getI32IntegerAttr(0));
    rewriter.replaceOp(op, zero);
    return success();
  }
};

struct VectorLoadLower : public ConvertOpToLLVMPattern<vector::LoadOp> {
  using ConvertOpToLLVMPattern<vector::LoadOp>::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(vector::LoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto tyAttr = cast_or_null<TypeAttr>(op->getAttr("polymer.access.type"));
    if (!tyAttr)
      return rewriter.notifyMatchFailure(op, "Access type attribute missing");
    Type tyToAccess = tyAttr.getValue();

    auto baseMemref = op.getBase();
    auto basePtr = adaptor.getBase();
    auto ptr = getAccessPointer(rewriter, *getTypeConverter(), op, baseMemref,
                                basePtr, adaptor.getIndices());
    if (failed(ptr))
      return failure();

    Value newVal = bitcastToVec(
        rewriter, getTypeConverter()->getDataLayoutAnalysis()->getAbove(op),

        rewriter.create<LLVM::LoadOp>(op.getLoc(), tyToAccess, *ptr));

    rewriter.replaceOp(op, newVal);

    return success();
  }
};

struct VectorStoreLower : public ConvertOpToLLVMPattern<vector::StoreOp> {
  using ConvertOpToLLVMPattern<vector::StoreOp>::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(vector::StoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto tyAttr = cast_or_null<TypeAttr>(op->getAttr("polymer.access.type"));
    if (!tyAttr)
      return rewriter.notifyMatchFailure(op, "Access type attribute missing");
    Type tyToAccess = tyAttr.getValue();

    auto baseMemref = op.getBase();
    auto basePtr = adaptor.getBase();
    auto ptr = getAccessPointer(rewriter, *getTypeConverter(), op, baseMemref,
                                basePtr, adaptor.getIndices());
    if (failed(ptr))
      return failure();

    rewriter.replaceOpWithNewOp<LLVM::StoreOp>(
        op,
        bitcastFromVec(
            rewriter, getTypeConverter()->getDataLayoutAnalysis()->getAbove(op),
            tyToAccess, adaptor.getValueToStore()),
        *ptr);

    return success();
  }
};

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
    if (args.empty())
      return adaptor.getMemref();
    return rewriter.create<LLVM::GEPOp>(
        loc,
        LLVM::LLVMPointerType::get(op.getContext(),
                                   originalType.getMemorySpaceAsInt()),
        elTy, adaptor.getMemref(), args);
  }
};

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

void mlir::populateGPULoweringPatterns(RewritePatternSet &patterns,
                                       LLVMTypeConverter &typeConverter) {
  patterns.add<ReinterpretCastOpLowering, AtAddrLower, CLoadOpLowering,
               CStoreOpLowering, VectorStoreLower, VectorLoadLower,
               CAllocaOpLowering, GlobalOpLowering, GetGlobalOpLowering,
               DeviceAsyncCopyOpLowerig>(typeConverter);
}
