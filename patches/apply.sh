#!/bin/sh

cd llvm-project
patch -p1 < ../patches/llvm_build.patch
# patch -p1 < ../patches/clang_macos.patch
cd -

cd jax
patch -p1 < ../patches/jax_workspace.patch
touch llvm_dummy.BUILD
cd -

