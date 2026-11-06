//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang_compile.h"

#include "absl/status/status.h"
#include "flang/Frontend/CompilerInstance.h"
#include "flang/Frontend/CompilerInvocation.h"
#include "flang/Frontend/FrontendActions.h"
#include "flang/Frontend/FrontendOptions.h"
#include "flang/Frontend/PreprocessorOptions.h"
#include "flang/Frontend/TargetOptions.h"
#include "flang/Frontend/TextDiagnosticPrinter.h"
#include "flang/ISO_Fortran_binding.h"
#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/Support/InternalNames.h"
#include "flang/Parser/message.h"
#include "flang/Parser/parsing.h"
#include "flang/Runtime/magic-numbers.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Support/LLVM.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/DiagnosticIDs.h"
#include "clang/Basic/DiagnosticOptions.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/TargetParser/Host.h"
#include "llvm/TargetParser/Triple.h"

#include <cstdlib>
#include <mutex>
#include <optional>
#include <utility>

namespace enzyme_jax {
namespace {

#define ENZYME_STRINGIFY_IMPL(X) #X
#define ENZYME_STRINGIFY(X) ENZYME_STRINGIFY_IMPL(X)

const char *getBuiltinModuleSource() {
  return R"FORTRAN(
module __fortran_builtins
  implicit none
  private

  intrinsic :: __builtin_c_loc
  public :: __builtin_c_loc

  intrinsic :: __builtin_c_devloc
  public :: __builtin_c_devloc

  intrinsic :: __builtin_c_f_pointer
  public :: __builtin_c_f_pointer

  intrinsic :: __builtin_f_c_string
  public :: __builtin_f_c_string

  intrinsic :: __builtin_c_f_strpointer
  public :: __builtin_c_f_strpointer

  intrinsic :: __builtin_show_descriptor
  public :: __builtin_show_descriptor

  intrinsic :: sizeof
  public :: sizeof

  intrinsic :: selected_int_kind
  integer, parameter :: int64 = selected_int_kind(18)

  type, bind(c), public :: __builtin_c_ptr
    integer(kind=int64), private :: __address
  end type

  type, bind(c), public :: __builtin_c_funptr
    integer(kind=int64), private :: __address
  end type

  type, public :: __builtin_event_type
    integer(kind=int64), private :: &
       __m1 = 0, __m2 = 0, __m3 = 0, __m4 = 0, &
       __m5 = 0, __m6 = 0, __m7 = 0, __m8 = 0
  end type

  type, public :: __builtin_notify_type
    integer(kind=int64), private :: __count = -1
  end type

  type, public :: __builtin_lock_type
    integer(kind=int64), private :: __count = -1
  end type

  type, public :: __builtin_ieee_flag_type
    integer(kind=1), private :: flag = 0
  end type

  type(__builtin_ieee_flag_type), parameter, public :: &
    __builtin_ieee_invalid = &
      __builtin_ieee_flag_type()FORTRAN" ENZYME_STRINGIFY(_FORTRAN_RUNTIME_IEEE_INVALID) R"FORTRAN(), &
    __builtin_ieee_overflow = &
      __builtin_ieee_flag_type()FORTRAN" ENZYME_STRINGIFY(_FORTRAN_RUNTIME_IEEE_OVERFLOW) R"FORTRAN(), &
    __builtin_ieee_divide_by_zero = &
      __builtin_ieee_flag_type()FORTRAN" ENZYME_STRINGIFY(
      _FORTRAN_RUNTIME_IEEE_DIVIDE_BY_ZERO) R"FORTRAN(), &
    __builtin_ieee_underflow = &
      __builtin_ieee_flag_type()FORTRAN" ENZYME_STRINGIFY(_FORTRAN_RUNTIME_IEEE_UNDERFLOW) R"FORTRAN(), &
    __builtin_ieee_inexact = &
      __builtin_ieee_flag_type()FORTRAN" ENZYME_STRINGIFY(_FORTRAN_RUNTIME_IEEE_INEXACT) R"FORTRAN(), &
    __builtin_ieee_denorm = &
      __builtin_ieee_flag_type()FORTRAN" ENZYME_STRINGIFY(_FORTRAN_RUNTIME_IEEE_DENORM) R"FORTRAN()

  type, public :: __builtin_ieee_round_type
    integer(kind=1), private :: mode = 0
  end type

  type(__builtin_ieee_round_type), parameter, public :: &
    __builtin_ieee_to_zero = &
      __builtin_ieee_round_type()FORTRAN" ENZYME_STRINGIFY(_FORTRAN_RUNTIME_IEEE_TO_ZERO) R"FORTRAN(), &
    __builtin_ieee_nearest = &
      __builtin_ieee_round_type()FORTRAN" ENZYME_STRINGIFY(_FORTRAN_RUNTIME_IEEE_NEAREST) R"FORTRAN(), &
    __builtin_ieee_up = &
      __builtin_ieee_round_type()FORTRAN" ENZYME_STRINGIFY(_FORTRAN_RUNTIME_IEEE_UP) R"FORTRAN(), &
    __builtin_ieee_down = &
      __builtin_ieee_round_type()FORTRAN" ENZYME_STRINGIFY(_FORTRAN_RUNTIME_IEEE_DOWN) R"FORTRAN(), &
    __builtin_ieee_away = &
      __builtin_ieee_round_type()FORTRAN" ENZYME_STRINGIFY(_FORTRAN_RUNTIME_IEEE_AWAY) R"FORTRAN(), &
    __builtin_ieee_other = &
      __builtin_ieee_round_type()FORTRAN" ENZYME_STRINGIFY(_FORTRAN_RUNTIME_IEEE_OTHER) R"FORTRAN()

  type, private :: __builtin_dummy_team_descriptor_type
    integer(kind=int64), private :: __placeholder = -1
  end type

  type, public :: __builtin_team_type
    type(__builtin_dummy_team_descriptor_type), pointer, private :: info => null()
  end type

  type, public, bind(C) :: __builtin_prif_coarray_handle_type
    type(__builtin_c_ptr), private :: info
  end type

  integer, parameter, public :: __builtin_atomic_int_kind = selected_int_kind(18)
  integer, parameter, public :: &
    __builtin_atomic_logical_kind = __builtin_atomic_int_kind

  type, public :: __builtin_dim3
    integer :: x=1, y=1, z=1
  end type
  type(__builtin_dim3), public :: &
    __builtin_threadIdx, __builtin_blockDim, __builtin_blockIdx, &
    __builtin_gridDim
  integer, parameter, public :: __builtin_warpsize = 32

  type, public, bind(c) :: __builtin_c_devptr
    type(__builtin_c_ptr) :: cptr
  end type

  intrinsic :: __builtin_fma
  intrinsic :: __builtin_ieee_int
  intrinsic :: __builtin_ieee_is_nan, __builtin_ieee_is_negative, &
    __builtin_ieee_is_normal
  intrinsic :: __builtin_ieee_next_after, __builtin_ieee_next_down, &
    __builtin_ieee_next_up
  intrinsic :: scale
  intrinsic :: __builtin_ieee_real
  intrinsic :: __builtin_ieee_selected_real_kind
  intrinsic :: __builtin_ieee_support_datatype, &
    __builtin_ieee_support_denormal, __builtin_ieee_support_divide, &
    __builtin_ieee_support_flag, __builtin_ieee_support_halting, &
    __builtin_ieee_support_inf, __builtin_ieee_support_io, &
    __builtin_ieee_support_nan, __builtin_ieee_support_rounding, &
    __builtin_ieee_support_sqrt, &
    __builtin_ieee_support_standard, __builtin_ieee_support_subnormal, &
    __builtin_ieee_support_underflow_control
  public :: __builtin_fma
  public :: __builtin_ieee_int
  public :: __builtin_ieee_is_nan, __builtin_ieee_is_negative, &
    __builtin_ieee_is_normal
  public :: __builtin_ieee_next_after, __builtin_ieee_next_down, &
    __builtin_ieee_next_up
  public :: __builtin_ieee_real
  public :: scale
  public :: __builtin_ieee_selected_real_kind
  public :: __builtin_ieee_support_datatype, &
    __builtin_ieee_support_denormal, __builtin_ieee_support_divide, &
    __builtin_ieee_support_flag, __builtin_ieee_support_halting, &
    __builtin_ieee_support_inf, __builtin_ieee_support_io, &
    __builtin_ieee_support_nan, __builtin_ieee_support_rounding, &
    __builtin_ieee_support_sqrt, &
    __builtin_ieee_support_standard, __builtin_ieee_support_subnormal, &
    __builtin_ieee_support_underflow_control

  type :: __force_derived_type_instantiations
    type(__builtin_c_ptr) :: c_ptr
    type(__builtin_c_devptr) :: c_devptr
    type(__builtin_c_funptr) :: c_funptr
    type(__builtin_event_type) :: event_type
    type(__builtin_lock_type) :: lock_type
    type(__builtin_team_type) :: team_type
  end type

  intrinsic :: __builtin_compiler_options, __builtin_compiler_version
  public :: __builtin_compiler_options, __builtin_compiler_version

  interface operator(==)
    module procedure __builtin_c_ptr_eq
    module procedure __builtin_c_devptr_eq
  end interface
  public :: operator(==)

  interface operator(/=)
    module procedure __builtin_c_ptr_ne
    module procedure __builtin_c_devptr_ne
  end interface
  public :: operator(/=)

  interface __builtin_c_associated
    module procedure c_associated_c_ptr
    module procedure c_associated_c_funptr
  end interface
  public :: __builtin_c_associated

  type(__builtin_c_ptr), parameter, public :: __builtin_c_null_ptr = __builtin_c_ptr(0)
  type(__builtin_c_devptr), parameter, public :: __builtin_c_null_devptr = __builtin_c_devptr(__builtin_c_null_ptr)
  type(__builtin_c_funptr), parameter, public :: &
    __builtin_c_null_funptr = __builtin_c_funptr(0)

  public :: __builtin_c_ptr_eq
  public :: __builtin_c_ptr_ne
  public :: __builtin_c_devptr_eq
  public :: __builtin_c_devptr_ne
  public :: __builtin_c_funloc

  contains

  elemental logical function __builtin_c_ptr_eq(x, y)
    type(__builtin_c_ptr), intent(in) :: x, y
    __builtin_c_ptr_eq = x%__address == y%__address
  end function

  elemental logical function __builtin_c_ptr_ne(x, y)
    type(__builtin_c_ptr), intent(in) :: x, y
    __builtin_c_ptr_ne = x%__address /= y%__address
  end function

  elemental logical function __builtin_c_devptr_eq(x, y)
    type(__builtin_c_devptr), intent(in) :: x, y
    __builtin_c_devptr_eq = x%cptr == y%cptr
  end function

  elemental logical function __builtin_c_devptr_ne(x, y)
    type(__builtin_c_devptr), intent(in) :: x, y
    __builtin_c_devptr_ne = x%cptr /= y%cptr
  end function

  pure function __builtin_c_funloc(x)
    type(__builtin_c_funptr) :: __builtin_c_funloc
    external :: x
    __builtin_c_funloc = __builtin_c_funptr(loc(x))
  end function

  pure logical function c_associated_c_ptr(c_ptr_1, c_ptr_2)
    type(__builtin_c_ptr), intent(in) :: c_ptr_1
    type(__builtin_c_ptr), intent(in), optional :: c_ptr_2
    if (c_ptr_1%__address == __builtin_c_null_ptr%__address) then
      c_associated_c_ptr = .false.
    else if (present(c_ptr_2)) then
      c_associated_c_ptr = c_ptr_1%__address == c_ptr_2%__address
    else
      c_associated_c_ptr = .true.
    end if
  end function c_associated_c_ptr

  pure logical function c_associated_c_funptr(c_ptr_1, c_ptr_2)
    type(__builtin_c_funptr), intent(in) :: c_ptr_1
    type(__builtin_c_funptr), intent(in), optional :: c_ptr_2
    if (c_ptr_1%__address == __builtin_c_null_ptr%__address) then
      c_associated_c_funptr = .false.
    else if (present(c_ptr_2)) then
      c_associated_c_funptr = c_ptr_1%__address == c_ptr_2%__address
    else
      c_associated_c_funptr = .true.
    end if
  end function c_associated_c_funptr

end module
)FORTRAN";
}

const char *getTypeInfoModuleSource() {
  return R"FORTRAN(
module __fortran_type_info

  use __fortran_builtins, &
    only: __builtin_c_ptr, __builtin_c_devptr, __builtin_c_funptr
  implicit none
  private

  integer, parameter :: int64 = selected_int_kind(18)

  type :: DerivedType
    type(Binding), pointer, contiguous :: binding(:)
    character(len=:), pointer :: name
    integer(kind=int64) :: sizeInBytes
    type(DerivedType), pointer :: uninstantiated
    integer(kind=int64), pointer, contiguous :: kindParameter(:)
    integer(1), pointer, contiguous :: lenParameterKind(:)
    type(Component), pointer, contiguous :: component(:)
    type(ProcPtrComponent), pointer, contiguous :: procptr(:)
    type(SpecialBinding), pointer, contiguous :: special(:)
    integer(4) :: specialBitSet
    integer(1) :: hasParent
    integer(1) :: noInitializationNeeded
    integer(1) :: noDestructionNeeded
    integer(1) :: noFinalizationNeeded
    integer(1) :: noDefinedAssignment
    integer(1) :: __padding0(3)
  end type

  type :: Binding
    type(__builtin_c_funptr) :: proc
    character(len=:), pointer :: name
  end type

  enum, bind(c)
    enumerator :: Deferred = 1, Explicit = 2, LenParameter = 3
  end enum

  type, bind(c) :: Value
    integer(1) :: genre
    integer(1) :: __padding0(7)
    integer(kind=int64) :: value
  end type

  enum, bind(c)
    enumerator :: Data = 1, Pointer = 2, Allocatable = 3, Automatic = 4
  end enum

  enum, bind(c)
    enumerator :: Host = 0, Device = 1, Managed = 2, Unified = 3
  end enum

  enum, bind(c)
    enumerator :: CategoryInteger = 0, CategoryReal = 1, &
      CategoryComplex = 2, CategoryCharacter = 3, &
      CategoryLogical = 4, CategoryDerived = 5
  end enum

  type :: Component
    character(len=:), pointer :: name
    integer(1) :: genre
    integer(1) :: category
    integer(1) :: kind
    integer(1) :: rank
    integer(1) :: memorySpace
    integer(1) :: __padding0(3)
    integer(kind=int64) :: offset
    type(Value) :: characterLen
    type(DerivedType), pointer :: derived
    type(Value), pointer, contiguous :: lenValue(:)
    type(Value), pointer, contiguous :: bounds(:, :)
    type(__builtin_c_ptr) :: initialization
  end type

  type :: ProcPtrComponent
    character(len=:), pointer :: name
    integer(kind=int64) :: offset
    type(__builtin_c_funptr) :: initialization
  end type

  enum, bind(c)
    enumerator :: ScalarAssignment = 1, ElementalAssignment = 2
    enumerator :: ReadFormatted = 3, ReadUnformatted = 4
    enumerator :: WriteFormatted = 5, WriteUnformatted = 6
    enumerator :: ElementalFinal = 7, AssumedRankFinal = 8
    enumerator :: ScalarFinal = 9
  end enum

  type, bind(c) :: SpecialBinding
    integer(1) :: which
    integer(1) :: isArgDescriptorSet
    integer(1) :: isTypeBound
    integer(1) :: specialCaseFlag
    integer(1) :: __padding0(4)
    type(__builtin_c_funptr) :: proc
  end type

end module
)FORTRAN";
}

const char *getIsoCBindingModuleSource() {
  return R"FORTRAN(
module iso_c_binding

  use __fortran_builtins, only: &
    c_associated => __builtin_c_associated, &
    c_funloc => __builtin_c_funloc, &
    c_funptr => __builtin_c_funptr, &
    c_f_pointer => __builtin_c_f_pointer, &
    c_f_strpointer => __builtin_c_f_strpointer, &
    c_loc => __builtin_c_loc, &
    c_null_funptr => __builtin_c_null_funptr, &
    c_null_ptr => __builtin_c_null_ptr, &
    c_ptr => __builtin_c_ptr, &
    c_sizeof => sizeof, &
    f_c_string => __builtin_f_c_string, &
    operator(==), operator(/=)

  implicit none
  private

  public :: c_associated, c_funloc, c_funptr, c_f_pointer, c_f_strpointer, &
    c_loc, c_null_funptr, c_null_ptr, c_ptr, c_sizeof, f_c_string, &
    operator(==), operator(/=)

  integer, parameter, public :: &
    c_int8_t = 1, &
    c_int16_t = 2, &
    c_int32_t = 4, &
    c_int64_t = 8, &
    c_int128_t = 16
  integer, parameter, public :: &
    c_int = c_int32_t, &
    c_short = c_int16_t, &
#if defined(_WIN32)
    c_long = c_int32_t, &
#else
    c_long = c_int64_t, &
#endif
    c_long_long = c_int64_t, &
    c_signed_char = c_int8_t, &
    c_size_t = kind(c_sizeof(1)), &
    c_intmax_t = c_int64_t, &
    c_intptr_t = c_size_t, &
    c_ptrdiff_t = c_size_t
  integer, parameter, public :: &
    c_int_least8_t = c_int8_t, &
    c_int_fast8_t = c_int8_t, &
    c_int_least16_t = c_int16_t, &
#if defined(__linux__) && defined(__powerpc__)
    c_int_fast16_t = c_long, &
#else
    c_int_fast16_t = c_int16_t, &
#endif
    c_int_least32_t = c_int32_t, &
#if defined(__linux__) && defined(__powerpc__)
    c_int_fast32_t = c_long, &
#else
    c_int_fast32_t = c_int32_t, &
#endif
    c_int_least64_t = c_int64_t, &
    c_int_fast64_t = c_int64_t, &
    c_int_least128_t = c_int128_t, &
    c_int_fast128_t = c_int128_t

  integer, parameter, public :: &
    c_float = 4, &
    c_double = 8, &
#if defined(__x86_64__)
    c_long_double = 10
#elif defined(_WIN32) && defined(__aarch64__)
    c_long_double = 8
#else
    c_long_double = 16
#endif

  integer, parameter, public :: &
    c_float_complex = c_float, &
    c_double_complex = c_double, &
    c_long_double_complex = c_long_double

  integer, parameter, public :: c_bool = 1
  integer, parameter, public :: c_char = 1

  character(kind=c_char, len=1), parameter, public :: c_null_char = achar(0)
  character(kind=c_char, len=1), parameter, public :: c_alert = achar(7)
  character(kind=c_char, len=1), parameter, public :: c_backspace = achar(8)
  character(kind=c_char, len=1), parameter, public :: c_form_feed = achar(12)
  character(kind=c_char, len=1), parameter, public :: c_new_line = achar(10)
  character(kind=c_char, len=1), parameter, public :: c_carriage_return = achar(13)
  character(kind=c_char, len=1), parameter, public :: c_horizontal_tab = achar(9)
  character(kind=c_char, len=1), parameter, public :: c_vertical_tab =  achar(11)

  interface c_f_procpointer
    module procedure c_f_procpointer
  end interface
  public :: c_f_procpointer

  integer, parameter, public :: &
    c_float16 = 2, &
    c_float16_complex = c_float16
  integer, parameter, public :: &
    c_float128 = 16, &
    c_float128_complex = c_float128
  integer, parameter, public :: &
    c_uint8_t = 1, &
    c_uint16_t = 2, &
    c_uint32_t = 4, &
    c_uint64_t = 8, &
    c_uint128_t = 16
  integer, parameter, public :: &
    c_unsigned_char = c_uint8_t, &
    c_unsigned_short = c_uint16_t, &
    c_unsigned = c_uint32_t, &
#if defined(_WIN32)
    c_unsigned_long = c_uint32_t, &
#else
    c_unsigned_long = c_uint64_t, &
#endif
    c_unsigned_long_long = c_uint64_t, &
#if defined(__powerpc__) || defined(_WIN32)
    c_uintmax_t = c_uint64_t
#else
    c_uintmax_t = c_uint128_t
#endif
  integer, parameter, public :: &
    c_uint_fast8_t = c_uint8_t, &
    c_uint_fast16_t = c_uint16_t, &
    c_uint_fast32_t = c_uint32_t, &
    c_uint_fast64_t = c_uint64_t, &
    c_uint_fast128_t = c_uint128_t
  integer, parameter, public :: &
    c_uint_least8_t = c_uint8_t, &
    c_uint_least16_t = c_uint16_t, &
    c_uint_least32_t = c_uint32_t, &
    c_uint_least64_t = c_uint64_t, &
    c_uint_least128_t = c_uint128_t

 contains

  subroutine c_f_procpointer(cptr, fptr)
    type(c_funptr), intent(in) :: cptr
    procedure(), pointer, intent(out) :: fptr
  end subroutine c_f_procpointer

end module iso_c_binding
)FORTRAN";
}

const char *getIsoFortranEnvImplModuleSource() {
  return R"FORTRAN(
module iso_fortran_env_impl
  implicit none
  private

  integer, parameter, public :: &
    selectedInt8 = selected_int_kind(2), &
    selectedInt16 = selected_int_kind(4), &
    selectedInt32 = selected_int_kind(9), &
    selectedInt64 = selected_int_kind(18),&
    selectedInt128 = selected_int_kind(38), &
    safeInt8 = merge(selectedInt8, selected_int_kind(0), &
                     selectedInt8 >= 0), &
    safeInt16 = merge(selectedInt16, selected_int_kind(0), &
                      selectedInt16 >= 0), &
    safeInt32 = merge(selectedInt32, selected_int_kind(0), &
                      selectedInt32 >= 0), &
    safeInt64 = merge(selectedInt64, selected_int_kind(0), &
                      selectedInt64 >= 0), &
    safeInt128 = merge(selectedInt128, selected_int_kind(0), &
                       selectedInt128 >= 0)

  integer, parameter, public :: &
    int8 = merge(selectedInt8, merge(-2, -1, selectedInt8 >= 0), &
                 digits(int(0,kind=safeInt8)) == 7), &
    int16 = merge(selectedInt16, merge(-2, -1, selectedInt16 >= 0), &
                  digits(int(0,kind=safeInt16)) == 15), &
    int32 = merge(selectedInt32, merge(-2, -1, selectedInt32 >= 0), &
                  digits(int(0,kind=safeInt32)) == 31), &
    int64 = merge(selectedInt64, merge(-2, -1, selectedInt64 >= 0), &
                  digits(int(0,kind=safeInt64)) == 63), &
    int128 = merge(selectedInt128, merge(-2, -1, selectedInt128 >= 0), &
                   digits(int(0,kind=safeInt128)) == 127)

  integer, parameter, public :: &
    selectedUInt8 = selected_unsigned_kind(2), &
    selectedUInt16 = selected_unsigned_kind(4), &
    selectedUInt32 = selected_unsigned_kind(9), &
    selectedUInt64 = selected_unsigned_kind(18),&
    selectedUInt128 = selected_unsigned_kind(38), &
    safeUInt8 = merge(selectedUInt8, selected_unsigned_kind(0), &
                     selectedUInt8 >= 0), &
    safeUInt16 = merge(selectedUInt16, selected_unsigned_kind(0), &
                      selectedUInt16 >= 0), &
    safeUInt32 = merge(selectedUInt32, selected_unsigned_kind(0), &
                      selectedUInt32 >= 0), &
    safeUInt64 = merge(selectedUInt64, selected_unsigned_kind(0), &
                      selectedUInt64 >= 0), &
    safeUInt128 = merge(selectedUInt128, selected_unsigned_kind(0), &
                       selectedUInt128 >= 0)

  integer, parameter, public :: &
    uint8 = merge(selectedUInt8, merge(-2, -1, selectedUInt8 >= 0), &
                 digits(uint(0,kind=safeUInt8)) == 8), &
    uint16 = merge(selectedUInt16, merge(-2, -1, selectedUInt16 >= 0), &
                  digits(uint(0,kind=safeUInt16)) == 16), &
    uint32 = merge(selectedUInt32, merge(-2, -1, selectedUInt32 >= 0), &
                  digits(uint(0,kind=safeUInt32)) == 32), &
    uint64 = merge(selectedUInt64, merge(-2, -1, selectedUInt64 >= 0), &
                  digits(uint(0,kind=safeUInt64)) == 64), &
    uint128 = merge(selectedUInt128, merge(-2, -1, selectedUInt128 >= 0), &
                   digits(uint(0,kind=safeUInt128)) == 128)

  integer, parameter, dimension(*), public :: __builtin_integer_kinds = [ &
      selected_int_kind(0), &
      [(pack([selected_int_kind(k)], &
             selected_int_kind(k) >= 0 .and. &
               selected_int_kind(k) /= selected_int_kind(k-1)), &
        integer :: k=1, 39)]]

  integer, parameter, public :: &
    logical8 = int8, logical16 = int16, logical32 = int32, logical64 = int64

  integer, parameter, dimension(*), public :: __builtin_logical_kinds = [ &
      pack([logical8],  logical8 >= 0), &
      pack([logical16], logical16 >= 0), &
      pack([logical32], logical32 >= 0), &
      pack([logical64], logical64 >= 0) &
    ]

  integer, parameter, public :: &
    selectedReal16 = selected_real_kind(3, 4), &
    selectedBfloat16 = selected_real_kind(2, 37), &
    selectedReal32 = selected_real_kind(6, 37), &
    selectedReal64 = selected_real_kind(15, 307), &
    selectedReal80 = selected_real_kind(18, 4931), &
    selectedReal64x2 = selected_real_kind(31, 307), &
    selectedReal128 = selected_real_kind(33, 4931), &
    safeReal16 = merge(selectedReal16, selected_real_kind(0,0), &
                       selectedReal16 >= 0), &
    safeBfloat16 = merge(selectedBfloat16, selected_real_kind(0,0), &
                         selectedBfloat16 >= 0), &
    safeReal32 = merge(selectedReal32, selected_real_kind(0,0), &
                       selectedReal32 >= 0), &
    safeReal64 = merge(selectedReal64, selected_real_kind(0,0), &
                       selectedReal64 >= 0), &
    safeReal80 = merge(selectedReal80, selected_real_kind(0,0), &
                       selectedReal80 >= 0), &
    safeReal64x2 = merge(selectedReal64x2, selected_real_kind(0,0), &
                         selectedReal64x2 >= 0), &
    safeReal128 = merge(selectedReal128, selected_real_kind(0,0), &
                        selectedReal128 >= 0)

  integer, parameter, public :: &
    real16 = merge(selectedReal16, merge(-2, -1, selectedReal16 >= 0), &
                   digits(real(0,kind=safeReal16)) == 11), &
    bfloat16 = merge(selectedBfloat16, merge(-2, -1, selectedBfloat16 >= 0), &
                     digits(real(0,kind=safeBfloat16)) == 8), &
    real32 = merge(selectedReal32, merge(-2, -1, selectedReal32 >= 0), &
                   digits(real(0,kind=safeReal32)) == 24), &
    real64 = merge(selectedReal64, merge(-2, -1, selectedReal64 >= 0), &
                   digits(real(0,kind=safeReal64)) == 53), &
    real80 = merge(selectedReal80, merge(-2, -1, selectedReal80 >= 0), &
                   digits(real(0,kind=safeReal80)) == 64), &
    real64x2 = merge(selectedReal64x2, merge(-2, -1, selectedReal64x2 >= 0), &
                     digits(real(0,kind=safeReal64x2)) == 106), &
    real128 = merge(selectedReal128, merge(-2, -1, selectedReal128 >= 0), &
                    digits(real(0,kind=safeReal128)) == 113)

  integer, parameter, dimension(*), public :: __builtin_real_kinds = [ &
      pack([real16], real16 >= 0), &
      pack([bfloat16], bfloat16 >= 0), &
      pack([real32], real32 >= 0), &
      pack([real64], real64 >= 0), &
      pack([real80], real80 >= 0), &
      pack([real64x2], real64x2 >= 0), &
      pack([real128], real128 >= 0) &
    ]
end module iso_fortran_env_impl
)FORTRAN";
}

const char *getIsoFortranEnvModuleSource() {
  return R"FORTRAN(
module iso_fortran_env

  use __fortran_builtins, only: &
    event_type => __builtin_event_type, &
    notify_type => __builtin_notify_type, &
    lock_type => __builtin_lock_type, &
    team_type => __builtin_team_type, &
    atomic_int_kind => __builtin_atomic_int_kind, &
    atomic_logical_kind => __builtin_atomic_logical_kind, &
    compiler_options => __builtin_compiler_options, &
    compiler_version => __builtin_compiler_version

  use iso_fortran_env_impl, only: &
    selectedInt8, selectedInt16, selectedInt32, selectedInt64, selectedInt128, &
    safeInt8, safeInt16, safeInt32, safeInt64, safeInt128, &
    int8, int16, int32, int64, int128, &
    selectedUInt8, selectedUInt16, selectedUInt32, selectedUInt64, selectedUInt128, &
    safeUInt8, safeUInt16, safeUInt32, safeUInt64, safeUInt128, &
    uint8, uint16, uint32, uint64, uint128, &
    logical8, logical16, logical32, logical64, &
    selectedReal16, selectedBfloat16, selectedReal32, &
    selectedReal64, selectedReal80, selectedReal64x2, &
    selectedReal128, &
    safeReal16, safeBfloat16, safeReal32, &
    safeReal64, safeReal80, safeReal64x2, &
    safeReal128, &
    real16, bfloat16, real32, real64, &
    real80, real64x2, real128, &
    integer_kinds => __builtin_integer_kinds, &
    real_kinds => __builtin_real_kinds, &
    logical_kinds => __builtin_logical_kinds

  implicit none
  private

  public :: event_type, notify_type, lock_type, team_type, &
    atomic_int_kind, atomic_logical_kind, compiler_options, &
    compiler_version

  integer, parameter :: &
    selectedASCII = selected_char_kind('ASCII'), &
    selectedUCS_2 = selected_char_kind('UCS-2'), &
    selectedUnicode = selected_char_kind('ISO_10646')
  integer, parameter, public :: character_kinds(*) = [ &
    pack([selectedASCII], selectedASCII >= 0), &
    pack([selectedUCS_2], selectedUCS_2 >= 0), &
    pack([selectedUnicode], selectedUnicode >= 0)]

  public :: selectedInt8, selectedInt16, selectedInt32, selectedInt64, selectedInt128, &
    safeInt8, safeInt16, safeInt32, safeInt64, safeInt128, &
    int8, int16, int32, int64, int128

  public :: selectedUInt8, selectedUInt16, selectedUInt32, selectedUInt64, selectedUInt128, &
    safeUInt8, safeUInt16, safeUInt32, safeUInt64, safeUInt128, &
    uint8, uint16, uint32, uint64, uint128

  public :: logical8, logical16, logical32, logical64

  public :: selectedReal16, selectedBfloat16, selectedReal32, &
    selectedReal64, selectedReal80, selectedReal64x2, &
    selectedReal128, &
    safeReal16, safeBfloat16, safeReal32, &
    safeReal64, safeReal80, safeReal64x2, &
    safeReal128, &
    real16, bfloat16, real32, real64, &
    real80, real64x2, real128

  public :: integer_kinds, real_kinds, logical_kinds

  integer, parameter, public :: current_team = -1, &
    initial_team = -2, &
    parent_team = -3

  integer, parameter, public :: character_storage_size = 8
  integer, parameter, public :: file_storage_size = 8

  intrinsic :: __builtin_numeric_storage_size
  integer, parameter, public :: numeric_storage_size = &
    __builtin_numeric_storage_size()

  integer, parameter, public :: &
    output_unit = )FORTRAN" ENZYME_STRINGIFY(FORTRAN_DEFAULT_OUTPUT_UNIT) R"FORTRAN(, &
    input_unit = )FORTRAN" ENZYME_STRINGIFY(FORTRAN_DEFAULT_INPUT_UNIT) R"FORTRAN(, &
    error_unit = )FORTRAN" ENZYME_STRINGIFY(FORTRAN_ERROR_UNIT) R"FORTRAN(, &
    iostat_end = )FORTRAN" ENZYME_STRINGIFY(FORTRAN_RUNTIME_IOSTAT_END) R"FORTRAN(, &
    iostat_eor = )FORTRAN" ENZYME_STRINGIFY(FORTRAN_RUNTIME_IOSTAT_EOR) R"FORTRAN(, &
    iostat_inquire_internal_unit = )FORTRAN" ENZYME_STRINGIFY(
      FORTRAN_RUNTIME_IOSTAT_INQUIRE_INTERNAL_UNIT) R"FORTRAN(, &
    stat_failed_image = )FORTRAN" ENZYME_STRINGIFY(FORTRAN_RUNTIME_STAT_FAILED_IMAGE) R"FORTRAN(, &
    stat_locked = )FORTRAN" ENZYME_STRINGIFY(FORTRAN_RUNTIME_STAT_LOCKED) R"FORTRAN(, &
    stat_locked_other_image = )FORTRAN" ENZYME_STRINGIFY(FORTRAN_RUNTIME_STAT_LOCKED_OTHER_IMAGE) R"FORTRAN(, &
    stat_stopped_image = )FORTRAN" ENZYME_STRINGIFY(FORTRAN_RUNTIME_STAT_STOPPED_IMAGE) R"FORTRAN(, &
    stat_unlocked = )FORTRAN" ENZYME_STRINGIFY(FORTRAN_RUNTIME_STAT_UNLOCKED) R"FORTRAN(, &
    stat_unlocked_failed_image = )FORTRAN" ENZYME_STRINGIFY(FORTRAN_RUNTIME_STAT_UNLOCKED_FAILED_IMAGE) R"FORTRAN(

end module iso_fortran_env
)FORTRAN";
}

#undef ENZYME_STRINGIFY
#undef ENZYME_STRINGIFY_IMPL

absl::Status compileModuleFile(llvm::StringRef SourcePath,
                               llvm::StringRef ModuleOutputDirectory) {
  auto Invocation = std::make_shared<Fortran::frontend::CompilerInvocation>();
  llvm::SmallString<256> OutputDirStorage(ModuleOutputDirectory);
  std::string OutputDirString(OutputDirStorage.str());
  std::string SourcePathString(SourcePath);

  llvm::SmallVector<const char *, 8> Args = {"-cpp", "-fsyntax-only", "-J",
                                             OutputDirString.c_str(),
                                             SourcePathString.c_str()};

  clang::DiagnosticOptions DiagnosticOptions;
  std::string DiagnosticBuffer;
  llvm::raw_string_ostream DiagnosticStream(DiagnosticBuffer);
  Fortran::frontend::TextDiagnosticPrinter DiagnosticPrinter(DiagnosticStream,
                                                             DiagnosticOptions);
  auto Diagnostics = Fortran::frontend::CompilerInstance::createDiagnostics(
      DiagnosticOptions, &DiagnosticPrinter, /*ShouldOwnClient=*/false);
  if (!Diagnostics) {
    return absl::InternalError(
        "Failed to create diagnostics engine for intrinsic module");
  }

  if (!Fortran::frontend::CompilerInvocation::createFromArgs(
          *Invocation, Args, *Diagnostics, "flang")) {
    return absl::InternalError("Failed to parse intrinsic module arguments: " +
                               DiagnosticBuffer);
  }

  Invocation->getPreprocessorOpts().searchDirectoriesFromIntrModPath.push_back(
      OutputDirString);

  Fortran::frontend::CompilerInstance Instance(Invocation);
  Instance.createDiagnostics(&DiagnosticPrinter, /*ShouldOwnClient=*/false);
  Instance.setSemaOutputStream(DiagnosticStream);

  Fortran::frontend::ParseSyntaxOnlyAction Action;
  if (!Instance.executeAction(Action)) {
    if (!Instance.getParsing().messages().empty()) {
      Instance.getParsing().messages().Emit(DiagnosticStream,
                                            Instance.getAllCookedSources(),
                                            /*echoSourceLines=*/true);
    }
    return absl::InternalError("Failed to compile intrinsic module " +
                               SourcePath.str() + ": " + DiagnosticBuffer);
  }
  return absl::OkStatus();
}

std::string *gIntrinsicModulesDirectory = nullptr;

void cleanupIntrinsicModulesDirectory() {
  if (gIntrinsicModulesDirectory && !gIntrinsicModulesDirectory->empty()) {
    llvm::sys::fs::remove_directories(*gIntrinsicModulesDirectory);
    delete gIntrinsicModulesDirectory;
    gIntrinsicModulesDirectory = nullptr;
  }
}

/// Generates precompiled `.mod` files for Flang's standard intrinsic modules
/// (`__fortran_builtins`, `__fortran_type_info`, `iso_c_binding`,
/// `iso_fortran_env_impl`, `iso_fortran_env`) once per process.
absl::StatusOr<llvm::StringRef> ensureIntrinsicModulesDirectory() {
  static std::once_flag OnceFlag;
  static absl::Status InitStatus;
  static std::string DirectoryPath;

  std::call_once(OnceFlag, [&]() {
    llvm::SmallString<128> TempDirectory;
    if (std::error_code ErrorCode = llvm::sys::fs::createUniqueDirectory(
            "enzyme_jax_flang_intrinsics", TempDirectory)) {
      InitStatus = absl::InternalError(
          "Failed to create temporary directory for Flang intrinsic modules: " +
          ErrorCode.message());
      return;
    }

    DirectoryPath = std::string(TempDirectory.str());
    gIntrinsicModulesDirectory = new std::string(DirectoryPath);
    std::atexit(cleanupIntrinsicModulesDirectory);

    struct ModuleSpec {
      const char *FileName;
      const char *Source;
    };
    const ModuleSpec Modules[] = {
        {"__fortran_builtins.f90", getBuiltinModuleSource()},
        {"__fortran_type_info.f90", getTypeInfoModuleSource()},
        {"iso_c_binding.f90", getIsoCBindingModuleSource()},
        {"iso_fortran_env_impl.f90", getIsoFortranEnvImplModuleSource()},
        {"iso_fortran_env.f90", getIsoFortranEnvModuleSource()},
    };

    for (const ModuleSpec &Spec : Modules) {
      llvm::SmallString<256> FilePath(TempDirectory);
      llvm::sys::path::append(FilePath, Spec.FileName);
      std::error_code ErrorCode;
      llvm::raw_fd_ostream FileStream(FilePath, ErrorCode,
                                      llvm::sys::fs::OF_Text);
      if (ErrorCode) {
        InitStatus = absl::InternalError("Failed to write intrinsic module " +
                                         std::string(Spec.FileName) + ": " +
                                         ErrorCode.message());
        return;
      }
      FileStream << Spec.Source;
      FileStream.close();

      InitStatus = compileModuleFile(FilePath, DirectoryPath);
      if (!InitStatus.ok()) {
        return;
      }
    }
  });

  if (!InitStatus.ok()) {
    return InitStatus;
  }
  return llvm::StringRef(DirectoryPath);
}

struct ExpectedCTypeInfo {
  enum class Category {
    Integer,
    Real,
    Complex,
    Logical,
  };
  Category TypeCategory;
  size_t ElementSizeInBytes;
  int CfiTypeCode;
};

absl::StatusOr<ExpectedCTypeInfo>
mapCTypeNameToExpectedInfo(llvm::StringRef CTypeName) {
  if (CTypeName == "float")
    return ExpectedCTypeInfo{ExpectedCTypeInfo::Category::Real, 4,
                             CFI_type_float};
  if (CTypeName == "double")
    return ExpectedCTypeInfo{ExpectedCTypeInfo::Category::Real, 8,
                             CFI_type_double};
  if (CTypeName == "half")
    return ExpectedCTypeInfo{ExpectedCTypeInfo::Category::Real, 2,
                             CFI_type_half_float};
  if (CTypeName == "bfloat16")
    return ExpectedCTypeInfo{ExpectedCTypeInfo::Category::Real, 2,
                             CFI_type_bfloat};
  if (CTypeName == "int8_t" || CTypeName == "char")
    return ExpectedCTypeInfo{ExpectedCTypeInfo::Category::Integer, 1,
                             CFI_type_int8_t};
  if (CTypeName == "int16_t")
    return ExpectedCTypeInfo{ExpectedCTypeInfo::Category::Integer, 2,
                             CFI_type_int16_t};
  if (CTypeName == "int32_t")
    return ExpectedCTypeInfo{ExpectedCTypeInfo::Category::Integer, 4,
                             CFI_type_int32_t};
  if (CTypeName == "int64_t")
    return ExpectedCTypeInfo{ExpectedCTypeInfo::Category::Integer, 8,
                             CFI_type_int64_t};
  if (CTypeName == "uint8_t")
    return ExpectedCTypeInfo{ExpectedCTypeInfo::Category::Integer, 1,
                             CFI_type_uint8_t};
  if (CTypeName == "uint16_t")
    return ExpectedCTypeInfo{ExpectedCTypeInfo::Category::Integer, 2,
                             CFI_type_uint16_t};
  if (CTypeName == "uint32_t")
    return ExpectedCTypeInfo{ExpectedCTypeInfo::Category::Integer, 4,
                             CFI_type_uint32_t};
  if (CTypeName == "uint64_t")
    return ExpectedCTypeInfo{ExpectedCTypeInfo::Category::Integer, 8,
                             CFI_type_uint64_t};
  if (CTypeName == "bool")
    return ExpectedCTypeInfo{ExpectedCTypeInfo::Category::Logical, 1,
                             CFI_type_Bool};
  if (CTypeName == "std::complex<float>" || CTypeName == "_Complex float")
    return ExpectedCTypeInfo{ExpectedCTypeInfo::Category::Complex, 8,
                             CFI_type_float_Complex};
  if (CTypeName == "std::complex<double>" || CTypeName == "_Complex double")
    return ExpectedCTypeInfo{ExpectedCTypeInfo::Category::Complex, 16,
                             CFI_type_double_Complex};
  return absl::InvalidArgumentError(
      "Unsupported JAX element type for Fortran: " + CTypeName.str());
}

absl::Status validateScalarFirType(mlir::Type FirElementType,
                                   llvm::StringRef ExpectedCTypeName,
                                   size_t ParameterIndex) {
  auto ExpectedOrError = mapCTypeNameToExpectedInfo(ExpectedCTypeName);
  if (!ExpectedOrError.ok()) {
    return ExpectedOrError.status();
  }
  ExpectedCTypeInfo Expected = *ExpectedOrError;

  if (auto FloatType = mlir::dyn_cast<mlir::FloatType>(FirElementType)) {
    size_t BitWidth = FloatType.getWidth();
    if (Expected.TypeCategory != ExpectedCTypeInfo::Category::Real ||
        Expected.ElementSizeInBytes * 8 != BitWidth) {
      return absl::InvalidArgumentError(
          "Fortran parameter " + std::to_string(ParameterIndex) + " has real(" +
          std::to_string(BitWidth / 8) + "), which does not match JAX type " +
          ExpectedCTypeName.str());
    }
    if (ExpectedCTypeName == "bfloat16" && !FloatType.isBF16()) {
      return absl::InvalidArgumentError("Fortran parameter " +
                                        std::to_string(ParameterIndex) +
                                        " is not bfloat16");
    }
    if (ExpectedCTypeName == "half" && !FloatType.isF16()) {
      return absl::InvalidArgumentError("Fortran parameter " +
                                        std::to_string(ParameterIndex) +
                                        " is not float16");
    }
    return absl::OkStatus();
  }

  if (auto IntType = mlir::dyn_cast<mlir::IntegerType>(FirElementType)) {
    size_t BitWidth = IntType.getWidth();
    if (Expected.TypeCategory != ExpectedCTypeInfo::Category::Integer ||
        Expected.ElementSizeInBytes * 8 != BitWidth) {
      return absl::InvalidArgumentError(
          "Fortran parameter " + std::to_string(ParameterIndex) +
          " has integer(" + std::to_string(BitWidth / 8) +
          "), which does not match JAX type " + ExpectedCTypeName.str());
    }
    return absl::OkStatus();
  }

  if (auto FirIntType = mlir::dyn_cast<fir::IntegerType>(FirElementType)) {
    size_t BitWidth = static_cast<size_t>(FirIntType.getFKind()) * 8;
    if (Expected.TypeCategory != ExpectedCTypeInfo::Category::Integer ||
        Expected.ElementSizeInBytes * 8 != BitWidth) {
      return absl::InvalidArgumentError(
          "Fortran parameter " + std::to_string(ParameterIndex) +
          " has integer(" + std::to_string(BitWidth / 8) +
          "), which does not match JAX type " + ExpectedCTypeName.str());
    }
    return absl::OkStatus();
  }

  if (auto ComplexType = mlir::dyn_cast<mlir::ComplexType>(FirElementType)) {
    auto ElementFloat =
        mlir::dyn_cast<mlir::FloatType>(ComplexType.getElementType());
    size_t TotalBytes = ElementFloat ? (ElementFloat.getWidth() / 8) * 2 : 0;
    if (Expected.TypeCategory != ExpectedCTypeInfo::Category::Complex ||
        Expected.ElementSizeInBytes != TotalBytes) {
      return absl::InvalidArgumentError(
          "Fortran parameter " + std::to_string(ParameterIndex) +
          " has complex type that does not match JAX type " +
          ExpectedCTypeName.str());
    }
    return absl::OkStatus();
  }

  if (auto LogicalType = mlir::dyn_cast<fir::LogicalType>(FirElementType)) {
    if (Expected.TypeCategory != ExpectedCTypeInfo::Category::Logical ||
        LogicalType.getFKind() != 1) {
      return absl::InvalidArgumentError(
          "Fortran parameter " + std::to_string(ParameterIndex) +
          " has logical(kind=" + std::to_string(LogicalType.getFKind()) +
          "); JAX bool requires logical(1) or logical(c_bool)");
    }
    return absl::OkStatus();
  }

  return absl::InvalidArgumentError(
      "Unsupported Fortran parameter type at index " +
      std::to_string(ParameterIndex));
}

absl::StatusOr<FortranParameter> classifyDummyParameter(
    mlir::Type FirType, llvm::ArrayRef<int64_t> ExpectedShape,
    llvm::StringRef ExpectedCTypeName, size_t ParameterIndex) {
  auto ExpectedOrError = mapCTypeNameToExpectedInfo(ExpectedCTypeName);
  if (!ExpectedOrError.ok()) {
    return ExpectedOrError.status();
  }
  ExpectedCTypeInfo Expected = *ExpectedOrError;

  FortranParameter Result;
  Result.CfiTypeCode = Expected.CfiTypeCode;
  Result.ElementSizeInBytes = Expected.ElementSizeInBytes;

  if (mlir::isa<fir::BoxCharType>(FirType)) {
    return absl::InvalidArgumentError(
        "CHARACTER dummy arguments with hidden length parameters are not "
        "supported at parameter " +
        std::to_string(ParameterIndex));
  }

  auto classifyBox = [&](fir::BaseBoxType BoxType) -> absl::Status {
    Result.PassingKind = ParameterPassingKind::BoxDescriptor;
    if (BoxType.isPointer()) {
      Result.CfiAttributeCode = CFI_attribute_pointer;
    } else if (BoxType.isPointerOrAllocatable()) {
      Result.CfiAttributeCode = CFI_attribute_allocatable;
    } else {
      Result.CfiAttributeCode = CFI_attribute_other;
    }

    if (BoxType.isAssumedRank()) {
      Result.Rank = -1;
    } else if (auto SequenceType = fir::unwrapUntilSeqType(BoxType)) {
      Result.Rank = static_cast<int>(SequenceType.getDimension());
      if (Result.Rank != static_cast<int>(ExpectedShape.size())) {
        return absl::InvalidArgumentError(
            "Fortran descriptor parameter " + std::to_string(ParameterIndex) +
            " has rank " + std::to_string(Result.Rank) +
            ", but JAX argument has rank " +
            std::to_string(ExpectedShape.size()));
      }
    } else {
      Result.Rank = 0;
      if (!ExpectedShape.empty()) {
        return absl::InvalidArgumentError(
            "Fortran scalar descriptor parameter " +
            std::to_string(ParameterIndex) +
            " cannot receive JAX array of rank " +
            std::to_string(ExpectedShape.size()));
      }
    }

    mlir::Type ElementType = fir::getFortranElementType(BoxType);
    return validateScalarFirType(ElementType, ExpectedCTypeName,
                                 ParameterIndex);
  };

  if (auto BoxType = mlir::dyn_cast<fir::BaseBoxType>(FirType)) {
    if (absl::Status Status = classifyBox(BoxType); !Status.ok()) {
      return Status;
    }
    return Result;
  }

  if (fir::isa_ref_type(FirType)) {
    mlir::Type PointeeType = fir::dyn_cast_ptrEleTy(FirType);
    if (auto InnerBoxType =
            mlir::dyn_cast_or_null<fir::BaseBoxType>(PointeeType)) {
      if (absl::Status Status = classifyBox(InnerBoxType); !Status.ok()) {
        return Status;
      }
      return Result;
    }

    Result.PassingKind = ParameterPassingKind::RawPointer;
    Result.Rank = static_cast<int>(ExpectedShape.size());

    if (auto SequenceType =
            mlir::dyn_cast_or_null<fir::SequenceType>(PointeeType)) {
      int FirRank = static_cast<int>(SequenceType.getDimension());
      auto SequenceShape = SequenceType.getShape();
      int64_t ExpectedTotalElements = 1;
      for (int64_t Extent : ExpectedShape) {
        ExpectedTotalElements *= Extent;
      }

      if (FirRank == static_cast<int>(ExpectedShape.size())) {
        // Fortran column-major dimensions are reversed relative to JAX
        // row-major.
        for (int Dim = 0; Dim < FirRank; ++Dim) {
          int64_t FirExtent = SequenceShape[Dim];
          int64_t JaxExtent = ExpectedShape[FirRank - 1 - Dim];
          if (FirExtent != fir::SequenceType::getUnknownExtent() &&
              FirExtent != JaxExtent) {
            return absl::InvalidArgumentError(
                "Fortran parameter " + std::to_string(ParameterIndex) +
                " dimension " + std::to_string(Dim + 1) + " has extent " +
                std::to_string(FirExtent) + ", expected " +
                std::to_string(JaxExtent) +
                " (reversed from JAX row-major shape)");
          }
        }
      } else if (!SequenceType.hasDynamicExtents() &&
                 !SequenceType.hasUnknownShape()) {
        int64_t FirTotalElements = 1;
        for (int64_t FirExtent : SequenceShape) {
          FirTotalElements *= FirExtent;
        }
        if (FirTotalElements != ExpectedTotalElements) {
          return absl::InvalidArgumentError(
              "Fortran parameter " + std::to_string(ParameterIndex) + " has " +
              std::to_string(FirTotalElements) +
              " elements, but JAX argument has " +
              std::to_string(ExpectedTotalElements) + " elements");
        }
      }
    } else if (!ExpectedShape.empty()) {
      return absl::InvalidArgumentError("Fortran scalar parameter " +
                                        std::to_string(ParameterIndex) +
                                        " cannot receive JAX array of rank " +
                                        std::to_string(ExpectedShape.size()));
    }

    mlir::Type ElementType = fir::unwrapAllRefAndSeqType(PointeeType);
    if (absl::Status Status = validateScalarFirType(
            ElementType, ExpectedCTypeName, ParameterIndex);
        !Status.ok()) {
      return Status;
    }
    return Result;
  }

  if (fir::isa_trivial(FirType)) {
    if (!ExpectedShape.empty()) {
      return absl::InvalidArgumentError("Fortran VALUE scalar parameter " +
                                        std::to_string(ParameterIndex) +
                                        " cannot receive JAX array of rank " +
                                        std::to_string(ExpectedShape.size()));
    }
    Result.PassingKind = ParameterPassingKind::ByValue;
    Result.Rank = 0;
    if (absl::Status Status =
            validateScalarFirType(FirType, ExpectedCTypeName, ParameterIndex);
        !Status.ok()) {
      return Status;
    }
    return Result;
  }

  return absl::InvalidArgumentError(
      "Unsupported Fortran parameter passing convention at index " +
      std::to_string(ParameterIndex));
}

absl::StatusOr<mlir::func::FuncOp>
resolveProcedureInModule(mlir::ModuleOp ModuleOperation,
                         llvm::StringRef ProcedureName) {
  if (auto ExactMatch =
          ModuleOperation.lookupSymbol<mlir::func::FuncOp>(ProcedureName)) {
    if (!ExactMatch.isDeclaration()) {
      return ExactMatch;
    }
  }

  std::string LowerName = ProcedureName.lower();
  size_t ScopeSeparator = LowerName.find("::");
  if (ScopeSeparator != std::string::npos) {
    llvm::StringRef ModulePart =
        llvm::StringRef(LowerName).take_front(ScopeSeparator);
    llvm::StringRef ProcPart =
        llvm::StringRef(LowerName).drop_front(ScopeSeparator + 2);
    std::string MangledName =
        fir::NameUniquer::doProcedure({ModulePart}, {}, ProcPart);
    if (auto ModuleMatch =
            ModuleOperation.lookupSymbol<mlir::func::FuncOp>(MangledName)) {
      if (!ModuleMatch.isDeclaration()) {
        return ModuleMatch;
      }
    }
    return absl::InvalidArgumentError(
        "Could not find Fortran module procedure '" + ProcedureName.str() +
        "' (mangled '" + MangledName + "')");
  }

  std::string TopLevelMangled =
      fir::NameUniquer::doProcedure({}, {}, LowerName);
  if (auto TopLevelMatch =
          ModuleOperation.lookupSymbol<mlir::func::FuncOp>(TopLevelMangled)) {
    if (!TopLevelMatch.isDeclaration()) {
      return TopLevelMatch;
    }
  }

  if (auto LowerBindCMatch =
          ModuleOperation.lookupSymbol<mlir::func::FuncOp>(LowerName)) {
    if (!LowerBindCMatch.isDeclaration()) {
      return LowerBindCMatch;
    }
  }

  llvm::SmallVector<mlir::func::FuncOp, 4> ModuleCandidates;
  llvm::SmallVector<std::string, 8> DefinedNames;
  for (mlir::func::FuncOp FunctionOp :
       ModuleOperation.getOps<mlir::func::FuncOp>()) {
    if (FunctionOp.isDeclaration()) {
      continue;
    }
    llvm::StringRef SymbolName = FunctionOp.getSymName();
    DefinedNames.push_back(SymbolName.str());
    auto [Kind, Deconstructed] = fir::NameUniquer::deconstruct(SymbolName);
    if (Kind == fir::NameUniquer::NameKind::PROCEDURE &&
        Deconstructed.name == LowerName) {
      ModuleCandidates.push_back(FunctionOp);
    }
  }

  if (ModuleCandidates.size() == 1) {
    return ModuleCandidates.front();
  }
  if (ModuleCandidates.size() > 1) {
    return absl::InvalidArgumentError(
        "Ambiguous Fortran procedure name '" + ProcedureName.str() +
        "'; qualify with 'module_name::procedure_name'");
  }

  std::string Available = llvm::join(DefinedNames, ", ");
  return absl::InvalidArgumentError(
      "Could not find Fortran procedure '" + ProcedureName.str() +
      "' in compiled module (defined symbols: [" + Available + "])");
}

/// Custom Flang `CodeGenAction` that inspects the lowered FIR `FuncOp`
/// signature before translating to LLVM IR and skips Flang's LLVM middle-end
/// pass pipeline so Enzyme differentiates unvectorized scalar loops.
class ExtractLLVMFromFortranAction : public Fortran::frontend::CodeGenAction {
public:
  ExtractLLVMFromFortranAction(
      llvm::StringRef ProcedureName,
      llvm::ArrayRef<llvm::SmallVector<int64_t>> OutputShapes,
      llvm::ArrayRef<std::string> OutputTypeNames,
      llvm::ArrayRef<llvm::SmallVector<int64_t>> InputShapes,
      llvm::ArrayRef<std::string> InputTypeNames)
      : CodeGenAction(Fortran::frontend::BackendActionTy::Backend_EmitLL),
        ProcedureName(ProcedureName), OutputShapes(OutputShapes),
        OutputTypeNames(OutputTypeNames), InputShapes(InputShapes),
        InputTypeNames(InputTypeNames) {}

  void executeAction() override {
    if (!mlirModule) {
      ExtractionStatus =
          absl::InternalError("Flang did not produce an MLIR module");
      return;
    }

    auto FunctionOrError = resolveProcedureInModule(*mlirModule, ProcedureName);
    if (!FunctionOrError.ok()) {
      ExtractionStatus = FunctionOrError.status();
      return;
    }

    mlir::func::FuncOp FunctionOp = *FunctionOrError;
    ResolvedSymbolName = FunctionOp.getSymName().str();
    mlir::FunctionType FunctionType = FunctionOp.getFunctionType();

    if (FunctionType.getNumResults() == 1) {
      if (OutputShapes.size() != 1 || !OutputShapes[0].empty()) {
        ExtractionStatus = absl::InvalidArgumentError(
            "Fortran function '" + ProcedureName.str() +
            "' returns a scalar value, so out_shapes must contain a single "
            "scalar ShapedArray((), dtype)");
        return;
      }
      if (FunctionType.getNumInputs() != InputShapes.size()) {
        ExtractionStatus = absl::InvalidArgumentError(
            "Fortran function '" + ProcedureName.str() + "' expects " +
            std::to_string(FunctionType.getNumInputs()) +
            " input arguments, but " + std::to_string(InputShapes.size()) +
            " were provided");
        return;
      }
      if (absl::Status Status = validateScalarFirType(FunctionType.getResult(0),
                                                      OutputTypeNames[0], 0);
          !Status.ok()) {
        ExtractionStatus = Status;
        return;
      }
      ReturnsScalarByValue = true;
      for (size_t Index = 0; Index < InputShapes.size(); ++Index) {
        auto ParamOrError = classifyDummyParameter(
            FunctionType.getInput(Index), InputShapes[Index],
            InputTypeNames[Index], Index);
        if (!ParamOrError.ok()) {
          ExtractionStatus = ParamOrError.status();
          return;
        }
        Parameters.push_back(*ParamOrError);
      }
    } else if (FunctionType.getNumResults() == 0) {
      size_t ExpectedArgCount = OutputShapes.size() + InputShapes.size();
      if (FunctionType.getNumInputs() != ExpectedArgCount) {
        ExtractionStatus = absl::InvalidArgumentError(
            "Fortran subroutine '" + ProcedureName.str() + "' expects " +
            std::to_string(FunctionType.getNumInputs()) +
            " arguments (outputs followed by inputs), but " +
            std::to_string(ExpectedArgCount) + " were provided");
        return;
      }
      for (size_t Index = 0; Index < OutputShapes.size(); ++Index) {
        auto ParamOrError = classifyDummyParameter(
            FunctionType.getInput(Index), OutputShapes[Index],
            OutputTypeNames[Index], Index);
        if (!ParamOrError.ok()) {
          ExtractionStatus = ParamOrError.status();
          return;
        }
        Parameters.push_back(*ParamOrError);
      }
      for (size_t Index = 0; Index < InputShapes.size(); ++Index) {
        size_t ArgIndex = OutputShapes.size() + Index;
        auto ParamOrError = classifyDummyParameter(
            FunctionType.getInput(ArgIndex), InputShapes[Index],
            InputTypeNames[Index], ArgIndex);
        if (!ParamOrError.ok()) {
          ExtractionStatus = ParamOrError.status();
          return;
        }
        Parameters.push_back(*ParamOrError);
      }
    } else {
      ExtractionStatus = absl::InvalidArgumentError(
          "Fortran procedure '" + ProcedureName.str() + "' has " +
          std::to_string(FunctionType.getNumResults()) +
          " MLIR results; use a subroutine with intent(out) arguments or a "
          "scalar-returning function");
      return;
    }

    llvmCtx = std::make_unique<llvm::LLVMContext>();
    generateLLVMIR();
    if (!llvmModule) {
      ExtractionStatus =
          absl::InternalError("Flang failed to lower FIR module to LLVM IR");
      return;
    }

    llvm::TargetMachine &TargetMachine = getInstance().getTargetMachine();
    const llvm::Triple &TargetTriple = TargetMachine.getTargetTriple();
    llvmModule->setTargetTriple(TargetTriple);
    llvmModule->setDataLayout(TargetTriple.computeDataLayout(
        getInstance().getInvocation().getTargetOpts().abi));

    // Resolve final LLVM function symbol name if ExternalNameConversion renamed
    // it.
    if (!llvmModule->getFunction(ResolvedSymbolName)) {
      if (llvm::StringRef(ResolvedSymbolName).starts_with("_QP")) {
        std::string BaseName =
            llvm::StringRef(ResolvedSymbolName).drop_front(3).str();
        if (llvmModule->getFunction(BaseName + "_")) {
          ResolvedSymbolName = BaseName + "_";
        } else if (llvmModule->getFunction(BaseName)) {
          ResolvedSymbolName = BaseName;
        }
      }
    }
    if (!llvmModule->getFunction(ResolvedSymbolName)) {
      std::string LowerProc = ProcedureName.lower();
      size_t Sep = LowerProc.rfind("::");
      if (Sep != std::string::npos) {
        LowerProc = LowerProc.substr(Sep + 2);
      }
      for (const llvm::Function &Func : *llvmModule) {
        if (Func.isDeclaration())
          continue;
        llvm::StringRef FName = Func.getName();
        if (FName.starts_with("llvm.") || FName.starts_with("fir.") ||
            FName.starts_with("_Fortran"))
          continue;
        if (FName == LowerProc || FName == (LowerProc + "_") ||
            FName.equals_insensitive(LowerProc)) {
          ResolvedSymbolName = FName.str();
          break;
        }
      }
    }

    // Release MLIR objects while `mlirCtx` is still alive.
    mlirModule = nullptr;
    mlirCtx.reset();
  }

  absl::StatusOr<CompiledFortranModule> takeCompiledModule() {
    if (!ExtractionStatus.ok()) {
      return ExtractionStatus;
    }
    if (!llvmModule || !llvmCtx) {
      return absl::InternalError("Compiled LLVM module is null");
    }
    CompiledFortranModule Result;
    Result.Context = std::move(llvmCtx);
    Result.Module = std::move(llvmModule);
    Result.ResolvedSymbolName = std::move(ResolvedSymbolName);
    Result.Parameters = std::move(Parameters);
    Result.ReturnsScalarByValue = ReturnsScalarByValue;
    return Result;
  }

private:
  llvm::StringRef ProcedureName;
  llvm::ArrayRef<llvm::SmallVector<int64_t>> OutputShapes;
  llvm::ArrayRef<std::string> OutputTypeNames;
  llvm::ArrayRef<llvm::SmallVector<int64_t>> InputShapes;
  llvm::ArrayRef<std::string> InputTypeNames;

  absl::Status ExtractionStatus = absl::OkStatus();
  std::string ResolvedSymbolName;
  std::vector<FortranParameter> Parameters;
  bool ReturnsScalarByValue = false;
};

bool isClangOnlyFlag(llvm::StringRef Flag, bool &ConsumesNextArg) {
  ConsumesNextArg = false;
  if (Flag == "-resource-dir" || Flag == "-isysroot" || Flag == "-isystem" ||
      Flag == "-internal-isystem" || Flag == "-internal-externc-isystem" ||
      Flag == "-x" || Flag == "-std" || Flag == "-stdlib") {
    ConsumesNextArg = true;
    return true;
  }
  if (Flag.starts_with("-resource-dir=") || Flag.starts_with("-isysroot=") ||
      Flag.starts_with("-isystem=") || Flag.starts_with("-std=") ||
      Flag.starts_with("-stdlib=") || Flag.starts_with("-fgnuc-version=")) {
    return true;
  }
  return false;
}

} // namespace

absl::StatusOr<CompiledFortranModule> compileFortranSourceToLLVM(
    llvm::StringRef ProcedureName, llvm::StringRef FortranSource,
    llvm::ArrayRef<llvm::SmallVector<int64_t>> OutputShapes,
    llvm::ArrayRef<std::string> OutputTypeNames,
    llvm::ArrayRef<llvm::SmallVector<int64_t>> InputShapes,
    llvm::ArrayRef<std::string> InputTypeNames,
    llvm::ArrayRef<std::string> CommandLineArgs) {
  auto IntrinsicDirOrError = ensureIntrinsicModulesDirectory();
  if (!IntrinsicDirOrError.ok()) {
    return IntrinsicDirOrError.status();
  }
  std::string IntrinsicModulesDir = IntrinsicDirOrError->str();

  llvm::SmallString<128> TempDirectory;
  if (std::error_code ErrorCode = llvm::sys::fs::createUniqueDirectory(
          "enzyme_jax_fortran", TempDirectory)) {
    return absl::InternalError(
        "Failed to create temporary directory for Fortran compilation: " +
        ErrorCode.message());
  }
  llvm::scope_exit CleanupDirectory(
      [&]() { llvm::sys::fs::remove_directories(TempDirectory); });

  llvm::SmallString<256> SourceFilePath(TempDirectory);
  llvm::sys::path::append(SourceFilePath, "kernel.f90");
  {
    std::error_code ErrorCode;
    llvm::raw_fd_ostream SourceStream(SourceFilePath, ErrorCode,
                                      llvm::sys::fs::OF_Text);
    if (ErrorCode) {
      return absl::InternalError("Failed to write temporary Fortran source: " +
                                 ErrorCode.message());
    }
    SourceStream << FortranSource;
  }

  std::string TempDirString = std::string(TempDirectory.str());
  std::string SourcePathString = std::string(SourceFilePath.str());

  // Default to -O2 and -fstack-arrays before user flags so Flang's HLFIR/FIR
  // passes inline array expressions and intrinsics (matmul, sum, dot_product,
  // transpose) into scalar loops before Enzyme differentiates them.
  llvm::SmallVector<const char *, 16> FlangArgs = {"-O2", "-fstack-arrays"};
  bool UserSpecifiedModuleDir = false;
  for (size_t Index = 0; Index < CommandLineArgs.size(); ++Index) {
    llvm::StringRef Arg = CommandLineArgs[Index];
    bool ConsumesNext = false;
    if (isClangOnlyFlag(Arg, ConsumesNext)) {
      if (ConsumesNext && Index + 1 < CommandLineArgs.size()) {
        ++Index;
      }
      continue;
    }
    if (Arg == "-J" || Arg == "-module-dir" || Arg.starts_with("-J") ||
        Arg.starts_with("-module-dir=")) {
      UserSpecifiedModuleDir = true;
    }
    FlangArgs.push_back(CommandLineArgs[Index].c_str());
  }
  FlangArgs.push_back(SourcePathString.c_str());

  auto Invocation = std::make_shared<Fortran::frontend::CompilerInvocation>();
  clang::DiagnosticOptions DiagnosticOptions;
  std::string DiagnosticBuffer;
  llvm::raw_string_ostream DiagnosticStream(DiagnosticBuffer);
  Fortran::frontend::TextDiagnosticPrinter DiagnosticPrinter(DiagnosticStream,
                                                             DiagnosticOptions);
  auto Diagnostics = Fortran::frontend::CompilerInstance::createDiagnostics(
      DiagnosticOptions, &DiagnosticPrinter, /*ShouldOwnClient=*/false);
  if (!Diagnostics) {
    return absl::InternalError("Failed to create Flang diagnostics engine");
  }

  if (!Fortran::frontend::CompilerInvocation::createFromArgs(
          *Invocation, FlangArgs, *Diagnostics, "flang")) {
    return absl::InvalidArgumentError("Invalid Fortran compiler arguments: " +
                                      DiagnosticBuffer);
  }

  if (Invocation->getFrontendOpts().inputs.empty()) {
    Invocation->getFrontendOpts().inputs.emplace_back(
        SourcePathString,
        Fortran::frontend::InputKind(Fortran::frontend::Language::Fortran));
  }
  if (Invocation->getTargetOpts().triple.empty()) {
    Invocation->getTargetOpts().triple =
        llvm::Triple::normalize(llvm::sys::getDefaultTargetTriple());
  }
  if (!UserSpecifiedModuleDir) {
    Invocation->setModuleDir(TempDirString);
  }
  Invocation->getPreprocessorOpts().searchDirectoriesFromIntrModPath.push_back(
      IntrinsicModulesDir);

  Fortran::frontend::CompilerInstance Instance(Invocation);
  Instance.createDiagnostics(&DiagnosticPrinter, /*ShouldOwnClient=*/false);
  Instance.setSemaOutputStream(DiagnosticStream);

  ExtractLLVMFromFortranAction Action(ProcedureName, OutputShapes,
                                      OutputTypeNames, InputShapes,
                                      InputTypeNames);
  if (!Instance.executeAction(Action)) {
    if (!Instance.getParsing().messages().empty()) {
      Instance.getParsing().messages().Emit(DiagnosticStream,
                                            Instance.getAllCookedSources(),
                                            /*echoSourceLines=*/true);
    }
    if (DiagnosticBuffer.empty()) {
      DiagnosticBuffer = "Fortran compilation failed";
    }
    return absl::InvalidArgumentError(DiagnosticBuffer);
  }

  auto CompiledOrError = Action.takeCompiledModule();
  if (!CompiledOrError.ok()) {
    if (!DiagnosticBuffer.empty()) {
      return absl::InvalidArgumentError(
          CompiledOrError.status().message().data() + std::string("\n") +
          DiagnosticBuffer);
    }
    return CompiledOrError.status();
  }
  return CompiledOrError;
}

} // namespace enzyme_jax
