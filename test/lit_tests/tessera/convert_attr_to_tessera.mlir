// RUN: enzymexlamlir-opt %s -func-attr-to-tessera-attr | FileCheck %s

module attributes {dlti.dl_spec = #dlti.dl_spec<!llvm.ptr<270> = dense<32> : vector<4xi64>, !llvm.ptr<271> = dense<32> : vector<4xi64>, !llvm.ptr<272> = dense<64> : vector<4xi64>, i64 = dense<64> : vector<2xi64>, i128 = dense<128> : vector<2xi64>, f80 = dense<128> : vector<2xi64>, !llvm.ptr = dense<64> : vector<4xi64>, i1 = dense<8> : vector<2xi64>, i8 = dense<8> : vector<2xi64>, i16 = dense<16> : vector<2xi64>, i32 = dense<32> : vector<2xi64>, f16 = dense<16> : vector<2xi64>, f64 = dense<64> : vector<2xi64>, f128 = dense<128> : vector<2xi64>, "dlti.endianness" = "little", "dlti.mangling_mode" = "e", "dlti.legal_int_widths" = array<i32: 8, 16, 32, 64>, "dlti.stack_alignment" = 128 : i64>, llvm.ident = "clang version 22.0.0git (https://github.com/llvm/llvm-project 4749bf56a65e38ee7b05ac7f9fe261aab6cb5bc6)", llvm.module_asm = [], llvm.target_triple = "x86_64-unknown-linux-gnu"} {
  llvm.mlir.global private unnamed_addr constant @".str"("tessera_op=reciprocal\00") {addr_space = 0 : i32, dso_local, section = "llvm.metadata"}
  llvm.mlir.global private unnamed_addr constant @".str.1"("/home/jessicacotturone21/Reactant/enzyme/test/tessera_test.c\00") {addr_space = 0 : i32, dso_local, section = "llvm.metadata"}
  llvm.module_flags [#llvm.mlir.module_flag<error, "wchar_size", 4 : i32>, #llvm.mlir.module_flag<min, "PIC Level", 2 : i32>, #llvm.mlir.module_flag<max, "PIE Level", 2 : i32>, #llvm.mlir.module_flag<max, "uwtable", 2 : i32>]
  llvm.func local_unnamed_addr @g(%arg0: i32 {llvm.noundef}) -> i32 attributes {dso_local, no_unwind, passthrough = [["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>, tune_cpu = "generic", uwtable_kind = #llvm.uwtableKind<async>} {
    %0 = llvm.call @reciprocal(%arg0) {no_unwind} : (i32 {llvm.noundef}) -> i32
    %1 = llvm.call @reciprocal(%0) {no_unwind} : (i32 {llvm.noundef}) -> i32
    llvm.return %1 : i32
  }
  // CHECK: llvm.func @reciprocal
  // CHECK-SAME: tessera.convert = #tessera<convert reciprocal>
  llvm.func @reciprocal(i32 {llvm.noundef}) -> i32 attributes {passthrough = [["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>, tessera_op = "reciprocal", tune_cpu = "generic"}
}
