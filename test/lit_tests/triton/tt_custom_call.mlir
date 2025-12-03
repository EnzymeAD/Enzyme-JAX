// RUN: enzymexlamlir-opt %s --raise-triton-custom-call --canonicalize | FileCheck %s

module @jit_add attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<8xi32>, %arg1: tensor<8xi32>) -> (tensor<8xi32> {jax.result_info = "result"}) {
    %0 = stablehlo.custom_call @triton_kernel_call(%arg0, %arg1) {api_version = 2 : i32, backend_config = "x\9C\A5\99\DDr\DB6\16\80\C7;\BD\11\AFr\B9\97\D8\ECz\C7\EA\D24\01R\14\A58\99\B8M\9Av\C6iv\D2l\BB3\9D\0C\87\12a\855E\AA$\94J\F5\F8-\FB\00\FB\08{\BF7{\00\F0\9F\A2\05\C9\1E\8Fe\11\E7|\E7\078\07\80\A4\FD\EF\8D\F6\9F7\9A\E6\07\81wK\D3\98FO\BEx\FA_tq\A1]\\\A074\A6\A9\CFh\80f[t}\FD\E3[\F4\FD\8F\FF\FC\F0o\F4\95?\BF=\7F\1D\07 \A3i\C6g\9Afa\12#\D7\18k\06\F3\D3\05e([z\98\98\BEf\007\A5Y\E6e\E1\EF\149\B6\A6\0D\80k,\A2d\16\0D*\9B\A8\F1\03\12\E7\E7\E8+\BA\08ct\B3\8E\E7\8C\D3+a\0D)\FE\00\E7eM\CD\F8\1Cf\E1,\A2\C8\A01K\B75\E2\9960V~\EA/\91\B1vld\ACX*}\F4#d\F8Q\B8\88\11\AE\89{B\D63\F5\A3\D4\F0qj\E485\EB85[\1BjFJ\7F\8DY\18 L\\\ED\0E0)]\80~\0A\CBap\BA\BA\1C\BDxV<\9BY\04\1E\A5\97\D8\AC=\03c\F0,\B8t\C5\B3(\99\0F0\9A \F3\E1\F9b\EC\17\7Fc\AC\B6\D3\C9\D4\D4\FEv\EDy|\FE\BD\19_\0A\E6\F4H\0C_\C9\A7\B3\99\C1\01Q \93Q\BAg\EB\E8\E7\EE\CC~|\B6Ct\B4K\14\83(\F7\93-W5\07\B1\83\08Qr\10;SB\B4\C12\F9l\ACe\16\C1\A3\D39\F3\C3\C0\D8T\89\C3ce\E0X\00\B3OQ1-\E0\B7\A0Z\BB\82rv\05E>\D6,\BB\C8\9A\A8Yv\A7\D6\A4\11\0A\C0OY\1E\88\1F\07\85Cc\EE\10\8C\8D\1BV\88\A3j\858\DA I\0B\9A\CBi\22\C6:o\C2\7F\95x\13\F8\85|Q\B62\22fd\02\BA\C2\9C\07d\B7B\12Su\0A\88)\E7t\1D\19\BF\85\01\CD\99i0\CE\A16OG\10\18Y>\07\C2\18_\89\\\A6a\10\DB\AA\06\B1-:\AB\A8\14\14\C6Q\18S\E4g\CBj:NS0cnL\E0\BF\84\F8\10\AC\04\D9\05D\1E\EF\F88\BA\87\C5\C0\9D\C0\E8\1FPa|\11\00\92\C6A\03Xx\87\95\D3\81E:\EA\11\13\11\EB\A8\131V\8E\18\ABDL\F6DL\CA\88\89J\C4\04\E1\91\9Awd\8AGy\C4r\EE-]\BA\03Y\AE\85k!\A2\18\AE5%v3\85\82\C8\CB\B7\95B\0BY\96*\D3\B2\1Au9\C9\EB\12v\EEgyE\D0_\F3\C1\95X\9E a\E6\B5\\\EC\03\DC\0F1\B8\C2\CFz&\04\B2o\A1\8C\D5\B3/\92n\C9\A4\EBb.,t\FFp\F6!Y\BDqu\92\A5\0DR\CA\CA\CE\8C\A7\D5^\02xh\D4\FD\A4\0E\17\8E\22p\D4)\0F\22\DA=\B8t\13F\14|zz\B1\8C\D9\C5\CA\8F\17I\EA\FFvQx\F0\14$2*\A4\07F@g\EB\85\E7\CFf)\FD\AC\0D\EE4c\E6\C2\86\ABd\F9J(\85\BE8\FF|\9D\04T*\8F\95\94_\FD\E4}\B8z\E3\CD\93\E5\0A\9C\F5\D6q\C8\84\FAC;gS\FD\EBo\BF\BB~\F5\FE\F5\F7^\9C\08MK\D9\F0\D5\07o\95&\C1zNS\A1\E9*\DB\FC\E6\DD\FB\B7^\C6\D20^\C8`\FB\9Bx\D7f\04\13\B1\F6\172M\FD\95\BA\D3f\E03\9F\C80\95\15\C1d\EC/\E9cC\EC\DF\F7\BA\F62\B6d^\14fr.\154\DB1\DAB\91\1C2\95|\09yA\F8\D8\A9T\\x\AF\DF\BD=\C3\C3C5\C8\C1\1A\D6P\1B\DCw\0B5\8Co\12Y\A6\D0\A5\EC\FE\EE\\\A3]\D3x\C1>\A1\E4\06\FD\AB(\B2\07:U]\F3\D5OW\EF\BFA\C5\FD)^/gy\C1\98\D2~\A3y\F4R\DE\DD\DC@\AFF\DF\C5,\C9[\86\81~\90Q\1D2eW\F2\A6\86~\E07\B53h\E1\B3-\A3\D9\F0\F0\86\85~\C6\1Fa\DF\9DM\CD\0D\19\F5\F6!\AC\B0\EEww\13\8C\E5\22\C6\E6\A8 \E5\AF8\7F5\CB$\1E0\15\9D\0Eb\1E\EEg\D9\0EJ\9FL\F9:\19\CB\B7DB\ED\C2c\92?\C6;\A6]l\81\FB,6\1B\82}t]cs\D2H^\11\80=nxZ\C4QH\99V+\F5vSl\92\C7\B5\AB\D4\96\FE\9CW\DB\E0\0E\06\BF\FC\E3\EF\7F\85}\1E=G\F0\F7\ACo_\E5\D7\B9\A1\C6\05\E1\98\98\8Bn<\B8\D1>=\E3\0F\87\F9\D8\A8\18\DBv\C7\9Cb,Y\B3\D5\9A5\04\96\B0\C8\22\8A\EE\E0t\C0\98\C17|\B4Z\CF\A2p^\FF\18\E1\ECTX\9C\A2\BF\80\0C\FCs\19Z\E4\05\BA\837A(>`\08\A3\90m\C1\0A\98\9A\22\18\BC\DF\E1'\1C\97\B6GQ\B6-J\15\C5\C1\A8n\02\86\C8g\D0\AFgk\A8{t\17'\F9)\EC9\BA\F1\A3\8C\DE\8B\C4 8\E8\85\01<\03\FE\822^\99\0B~e\84g\1B\89\17t\91\EB\F1P*\CC\E0\CD-\ACT?e\A0\E8\A7!\FBd\CC\93\18\1E\C4\0C\BASK\CB\EDjy\E6\91z\B8\D4\A3\1B\96\85\D2w\A9\07\CD2\84\A3\F4\83\EA\A4\AD\DE\F4I\1Dd\95 \B8\14\B68\FC\F6\D74:\DD\0F\B4\BB\09!\D8\1E\DB\AE\E5@\0FP\00\8C\BA\80\F3\92\E0\AA\10\9C\8A\00\AD\1De\11m\05b\E9m\9F\15\A8\E3\16u\B1\8F:R\A1\BA%\15\AE.\AD\FC;z\DB\03\00\E2\87y\93\D6|\C2\B2\D2w\AF\8E\0E&\11\BBu&+h\E9\DFR/\85>\07]\87\DF{\9E\17\CBZGE\B9\98E\D1N\11\A3q\96\A4\97\EEFTxI\9E4\C9\1E6%<[E>k\FB-\BD:\7F\D1\03#f\1B\D6)\A1\9A\99\B6KP\0A\D5\03\C7~\08\DB)\AD\22/\8F`VU\06\ED\BA\EE)/\B1\9A\E5\A92\F1\902\EBj\1FTc]u\A7\AB\1E\80\DF\F4\B2r\E1\C5\01\B1\B4+K\D4k\95;\BDaY\1D\EB\F6yY\0B\F5\107'\BB\1A\C0n7]u,1[\1D\A0J\8B\DE0^'\E2\87\80\B8o\B1\99z\FFb\EE\E0\96~v\BB\7Fo#\B8&\EE\11\D2\97\F2V\A2\1B\16\9B\08\AB\BD\18\98^\8FM\AFL\ED\05n\1A\DDF\1CsP\EB0Ro7\F5\81\1A\8D\144\8F\D8\12\08i\E5\A8\D3M\C3\B3\BA?\0D\94\DE\E7g\8D<\92\E4(\F1\03iI\AF\F2\D1\C7\AD@V\0E\DA6\02\DE\1E\19\B0]\D0<\E24\03\DE>2\E0\1Ay\\\0F\98[:(\E0Q\B1\E6\C5I\B1\DD\B0E>\D5\BB\B5\D3\80y\C4m\F3\84\BB\C7\F3&\EDz\14\CF\F5\BA\C1\DEn\D1bY;\8E\9B\FDm\BF\AD\8C\0F\EA\FAm\ED\DE\DAVj\FAmZ\A7\CCi-!\13\BDnV\99\B9cO<\A4\E1\B7q\A3\9D\FD~\87\8B\B6:\D3i7\FB\22\1Dz\DDn_\ABo\D3\C6\AD\95\C5\17\BE\DE\B3^;\90\E6Y\AC\BAs\1D\DE1p\D1pq\B3]\98\8Fk\17\05\96;\C9\92\94\02\BF\96\A5\F1!\0D\03\93\12\95R\B6N\E3jD\F4\CE\FB\F2\FDP\AB\FF/$\F6]\FD\C5\17\A5R\96\EC\95\1D\97\B2\D6^Y\FE\AD\A5\94\B5\15d\F9\C4r\D9\D1^Y\FE\BD\A2\94u\F6\C9\8A/\0C\A5\ECXA\16\DBR\D6\DD+\8BK\EEDA\B6\E0\8A\0B\C4\C3\C2\FC\0B\AE\\x\EF\CC\89/\AFr\E1\BDS'\BE\95\CA\85\F7\CE\1D'\E7\B2e\DE\E0*&?\CF\C0\C5G>e\9Aj7!)B\0A\912;y!\C9a+\1F&\E6\CEa\BB\18.3\C0+E\8E\8D\8A\B12\E0\8D\1Cp\8A\01\AB50.\06\CAU\B8\95\03n10j\0DL\8A\81\D6gZy\F8&\0C\9B\1B\F7\E4\E5\C9\B7'ON\FE|\82N\BE\FCB\FB\D3\93'\F5\BF\FF\07\0E4\B1L", operand_layouts = [dense<0> : tensor<1xindex>, dense<0> : tensor<1xindex>], result_layouts = [dense<0> : tensor<1xindex>]} : (tensor<8xi32>, tensor<8xi32>) -> tensor<8xi32>
    return %0 : tensor<8xi32>
  }
}

// CHECK:  func.func public @main(%arg0: tensor<8xi32>, %arg1: tensor<8xi32>) -> (tensor<8xi32> {jax.result_info = "result"}) {
// CHECK-NEXT:    %c = stablehlo.constant dense<1> : tensor<i64>
// CHECK-NEXT:    %0 = enzymexla_tt_ext.call @triton_module::@triton_module_inner::@add_kernel clusters in(%c, %c, %c) blocks in(%c, %c, %c) (%arg0, %arg1) {arg_attrs = [], operand_layouts = [dense<0> : tensor<1xindex>, dense<0> : tensor<1xindex>], res_attrs = [], result_layouts = [dense<0> : tensor<1xindex>]} : (tensor<8xi32>, tensor<8xi32>) -> tensor<8xi32>
// CHECK-NEXT:    return %0 : tensor<8xi32>
// CHECK-NEXT:  }

// CHECK:  enzymexla_tt_ext.module @triton_module {
// CHECK-NEXT:    builtin.module @triton_module_inner {
// CHECK-NEXT:      tt.func public @add_kernel(%arg0: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<i32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
// CHECK-NEXT:        %cst = arith.constant dense<8> : tensor<8xi32>
// CHECK-NEXT:        %c8_i32 = arith.constant 8 : i32
// CHECK-NEXT:        %0 = tt.get_program_id x : i32
// CHECK-NEXT:        %1 = arith.muli %0, %c8_i32 : i32
// CHECK-NEXT:        %2 = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
// CHECK-NEXT:        %3 = tt.splat %1 : i32 -> tensor<8xi32>
// CHECK-NEXT:        %4 = arith.addi %3, %2 : tensor<8xi32>
// CHECK-NEXT:        %5 = arith.cmpi slt, %4, %cst : tensor<8xi32>
// CHECK-NEXT:        %6 = tt.splat %arg0 : !tt.ptr<i32> -> tensor<8x!tt.ptr<i32>>
// CHECK-NEXT:        %7 = tt.addptr %6, %4 : tensor<8x!tt.ptr<i32>>, tensor<8xi32>
// CHECK-NEXT:        %8 = tt.load %7, %5 : tensor<8x!tt.ptr<i32>>
// CHECK-NEXT:        %9 = tt.splat %arg1 : !tt.ptr<i32> -> tensor<8x!tt.ptr<i32>>
// CHECK-NEXT:        %10 = tt.addptr %9, %4 : tensor<8x!tt.ptr<i32>>, tensor<8xi32>
// CHECK-NEXT:        %11 = tt.load %10, %5 : tensor<8x!tt.ptr<i32>>
// CHECK-NEXT:        %12 = arith.addi %8, %11 : tensor<8xi32>
// CHECK-NEXT:        %13 = tt.splat %arg2 : !tt.ptr<i32> -> tensor<8x!tt.ptr<i32>>
// CHECK-NEXT:        %14 = tt.addptr %13, %4 : tensor<8x!tt.ptr<i32>>, tensor<8xi32>
// CHECK-NEXT:        tt.store %14, %12, %5 : tensor<8x!tt.ptr<i32>>
// CHECK-NEXT:        tt.return
// CHECK-NEXT:      }
// CHECK-NEXT:    }
// CHECK-NEXT:  }
