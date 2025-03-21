module @jit__unnamed_wrapped_function_ attributes {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<1x3x224x224xf32>) -> (tensor<1x2048x7x7xf32> {jax.result_info = ".last_hidden_state"}, tensor<1x2048x1x1xf32> {jax.result_info = ".pooler_output"}) {
    %c = stablehlo.constant dense<49> : tensor<i32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %cst_0 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %cst_2 = stablehlo.constant dense_resource<__elided__> : tensor<7x7x3x64xf32>
    %cst_3 = stablehlo.constant dense_resource<__elided__> : tensor<64xf32>
    %cst_4 = stablehlo.constant dense_resource<__elided__> : tensor<64xf32>
    %cst_5 = stablehlo.constant dense_resource<__elided__> : tensor<64xf32>
    %cst_6 = stablehlo.constant dense_resource<__elided__> : tensor<64xf32>
    %cst_7 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x64x256xf32>
    %cst_8 = stablehlo.constant dense_resource<__elided__> : tensor<256xf32>
    %cst_9 = stablehlo.constant dense_resource<__elided__> : tensor<256xf32>
    %cst_10 = stablehlo.constant dense_resource<__elided__> : tensor<256xf32>
    %cst_11 = stablehlo.constant dense_resource<__elided__> : tensor<256xf32>
    %cst_12 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x64x64xf32>
    %cst_13 = stablehlo.constant dense_resource<__elided__> : tensor<64xf32>
    %cst_14 = stablehlo.constant dense_resource<__elided__> : tensor<64xf32>
    %cst_15 = stablehlo.constant dense_resource<__elided__> : tensor<64xf32>
    %cst_16 = stablehlo.constant dense_resource<__elided__> : tensor<64xf32>
    %cst_17 = stablehlo.constant dense_resource<__elided__> : tensor<3x3x64x64xf32>
    %cst_18 = stablehlo.constant dense_resource<__elided__> : tensor<64xf32>
    %cst_19 = stablehlo.constant dense_resource<__elided__> : tensor<64xf32>
    %cst_20 = stablehlo.constant dense_resource<__elided__> : tensor<64xf32>
    %cst_21 = stablehlo.constant dense_resource<__elided__> : tensor<64xf32>
    %cst_22 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x64x256xf32>
    %cst_23 = stablehlo.constant dense_resource<__elided__> : tensor<256xf32>
    %cst_24 = stablehlo.constant dense_resource<__elided__> : tensor<256xf32>
    %cst_25 = stablehlo.constant dense_resource<__elided__> : tensor<256xf32>
    %cst_26 = stablehlo.constant dense_resource<__elided__> : tensor<256xf32>
    %cst_27 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x256x64xf32>
    %cst_28 = stablehlo.constant dense_resource<__elided__> : tensor<64xf32>
    %cst_29 = stablehlo.constant dense_resource<__elided__> : tensor<64xf32>
    %cst_30 = stablehlo.constant dense_resource<__elided__> : tensor<64xf32>
    %cst_31 = stablehlo.constant dense_resource<__elided__> : tensor<64xf32>
    %cst_32 = stablehlo.constant dense_resource<__elided__> : tensor<3x3x64x64xf32>
    %cst_33 = stablehlo.constant dense_resource<__elided__> : tensor<64xf32>
    %cst_34 = stablehlo.constant dense_resource<__elided__> : tensor<64xf32>
    %cst_35 = stablehlo.constant dense_resource<__elided__> : tensor<64xf32>
    %cst_36 = stablehlo.constant dense_resource<__elided__> : tensor<64xf32>
    %cst_37 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x64x256xf32>
    %cst_38 = stablehlo.constant dense_resource<__elided__> : tensor<256xf32>
    %cst_39 = stablehlo.constant dense_resource<__elided__> : tensor<256xf32>
    %cst_40 = stablehlo.constant dense_resource<__elided__> : tensor<256xf32>
    %cst_41 = stablehlo.constant dense_resource<__elided__> : tensor<256xf32>
    %cst_42 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x256x64xf32>
    %cst_43 = stablehlo.constant dense_resource<__elided__> : tensor<64xf32>
    %cst_44 = stablehlo.constant dense_resource<__elided__> : tensor<64xf32>
    %cst_45 = stablehlo.constant dense_resource<__elided__> : tensor<64xf32>
    %cst_46 = stablehlo.constant dense_resource<__elided__> : tensor<64xf32>
    %cst_47 = stablehlo.constant dense_resource<__elided__> : tensor<3x3x64x64xf32>
    %cst_48 = stablehlo.constant dense_resource<__elided__> : tensor<64xf32>
    %cst_49 = stablehlo.constant dense_resource<__elided__> : tensor<64xf32>
    %cst_50 = stablehlo.constant dense_resource<__elided__> : tensor<64xf32>
    %cst_51 = stablehlo.constant dense_resource<__elided__> : tensor<64xf32>
    %cst_52 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x64x256xf32>
    %cst_53 = stablehlo.constant dense_resource<__elided__> : tensor<256xf32>
    %cst_54 = stablehlo.constant dense_resource<__elided__> : tensor<256xf32>
    %cst_55 = stablehlo.constant dense_resource<__elided__> : tensor<256xf32>
    %cst_56 = stablehlo.constant dense_resource<__elided__> : tensor<256xf32>
    %cst_57 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x256x512xf32>
    %cst_58 = stablehlo.constant dense_resource<__elided__> : tensor<512xf32>
    %cst_59 = stablehlo.constant dense_resource<__elided__> : tensor<512xf32>
    %cst_60 = stablehlo.constant dense_resource<__elided__> : tensor<512xf32>
    %cst_61 = stablehlo.constant dense_resource<__elided__> : tensor<512xf32>
    %cst_62 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x256x128xf32>
    %cst_63 = stablehlo.constant dense_resource<__elided__> : tensor<128xf32>
    %cst_64 = stablehlo.constant dense_resource<__elided__> : tensor<128xf32>
    %cst_65 = stablehlo.constant dense_resource<__elided__> : tensor<128xf32>
    %cst_66 = stablehlo.constant dense_resource<__elided__> : tensor<128xf32>
    %cst_67 = stablehlo.constant dense_resource<__elided__> : tensor<3x3x128x128xf32>
    %cst_68 = stablehlo.constant dense_resource<__elided__> : tensor<128xf32>
    %cst_69 = stablehlo.constant dense_resource<__elided__> : tensor<128xf32>
    %cst_70 = stablehlo.constant dense_resource<__elided__> : tensor<128xf32>
    %cst_71 = stablehlo.constant dense_resource<__elided__> : tensor<128xf32>
    %cst_72 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x128x512xf32>
    %cst_73 = stablehlo.constant dense_resource<__elided__> : tensor<512xf32>
    %cst_74 = stablehlo.constant dense_resource<__elided__> : tensor<512xf32>
    %cst_75 = stablehlo.constant dense_resource<__elided__> : tensor<512xf32>
    %cst_76 = stablehlo.constant dense_resource<__elided__> : tensor<512xf32>
    %cst_77 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x512x128xf32>
    %cst_78 = stablehlo.constant dense_resource<__elided__> : tensor<128xf32>
    %cst_79 = stablehlo.constant dense_resource<__elided__> : tensor<128xf32>
    %cst_80 = stablehlo.constant dense_resource<__elided__> : tensor<128xf32>
    %cst_81 = stablehlo.constant dense_resource<__elided__> : tensor<128xf32>
    %cst_82 = stablehlo.constant dense_resource<__elided__> : tensor<3x3x128x128xf32>
    %cst_83 = stablehlo.constant dense_resource<__elided__> : tensor<128xf32>
    %cst_84 = stablehlo.constant dense_resource<__elided__> : tensor<128xf32>
    %cst_85 = stablehlo.constant dense_resource<__elided__> : tensor<128xf32>
    %cst_86 = stablehlo.constant dense_resource<__elided__> : tensor<128xf32>
    %cst_87 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x128x512xf32>
    %cst_88 = stablehlo.constant dense_resource<__elided__> : tensor<512xf32>
    %cst_89 = stablehlo.constant dense_resource<__elided__> : tensor<512xf32>
    %cst_90 = stablehlo.constant dense_resource<__elided__> : tensor<512xf32>
    %cst_91 = stablehlo.constant dense_resource<__elided__> : tensor<512xf32>
    %cst_92 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x512x128xf32>
    %cst_93 = stablehlo.constant dense_resource<__elided__> : tensor<128xf32>
    %cst_94 = stablehlo.constant dense_resource<__elided__> : tensor<128xf32>
    %cst_95 = stablehlo.constant dense_resource<__elided__> : tensor<128xf32>
    %cst_96 = stablehlo.constant dense_resource<__elided__> : tensor<128xf32>
    %cst_97 = stablehlo.constant dense_resource<__elided__> : tensor<3x3x128x128xf32>
    %cst_98 = stablehlo.constant dense_resource<__elided__> : tensor<128xf32>
    %cst_99 = stablehlo.constant dense_resource<__elided__> : tensor<128xf32>
    %cst_100 = stablehlo.constant dense_resource<__elided__> : tensor<128xf32>
    %cst_101 = stablehlo.constant dense_resource<__elided__> : tensor<128xf32>
    %cst_102 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x128x512xf32>
    %cst_103 = stablehlo.constant dense_resource<__elided__> : tensor<512xf32>
    %cst_104 = stablehlo.constant dense_resource<__elided__> : tensor<512xf32>
    %cst_105 = stablehlo.constant dense_resource<__elided__> : tensor<512xf32>
    %cst_106 = stablehlo.constant dense_resource<__elided__> : tensor<512xf32>
    %cst_107 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x512x128xf32>
    %cst_108 = stablehlo.constant dense_resource<__elided__> : tensor<128xf32>
    %cst_109 = stablehlo.constant dense_resource<__elided__> : tensor<128xf32>
    %cst_110 = stablehlo.constant dense_resource<__elided__> : tensor<128xf32>
    %cst_111 = stablehlo.constant dense_resource<__elided__> : tensor<128xf32>
    %cst_112 = stablehlo.constant dense_resource<__elided__> : tensor<3x3x128x128xf32>
    %cst_113 = stablehlo.constant dense_resource<__elided__> : tensor<128xf32>
    %cst_114 = stablehlo.constant dense_resource<__elided__> : tensor<128xf32>
    %cst_115 = stablehlo.constant dense_resource<__elided__> : tensor<128xf32>
    %cst_116 = stablehlo.constant dense_resource<__elided__> : tensor<128xf32>
    %cst_117 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x128x512xf32>
    %cst_118 = stablehlo.constant dense_resource<__elided__> : tensor<512xf32>
    %cst_119 = stablehlo.constant dense_resource<__elided__> : tensor<512xf32>
    %cst_120 = stablehlo.constant dense_resource<__elided__> : tensor<512xf32>
    %cst_121 = stablehlo.constant dense_resource<__elided__> : tensor<512xf32>
    %cst_122 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x512x1024xf32>
    %cst_123 = stablehlo.constant dense_resource<__elided__> : tensor<1024xf32>
    %cst_124 = stablehlo.constant dense_resource<__elided__> : tensor<1024xf32>
    %cst_125 = stablehlo.constant dense_resource<__elided__> : tensor<1024xf32>
    %cst_126 = stablehlo.constant dense_resource<__elided__> : tensor<1024xf32>
    %cst_127 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x512x256xf32>
    %cst_128 = stablehlo.constant dense_resource<__elided__> : tensor<256xf32>
    %cst_129 = stablehlo.constant dense_resource<__elided__> : tensor<256xf32>
    %cst_130 = stablehlo.constant dense_resource<__elided__> : tensor<256xf32>
    %cst_131 = stablehlo.constant dense_resource<__elided__> : tensor<256xf32>
    %cst_132 = stablehlo.constant dense_resource<__elided__> : tensor<3x3x256x256xf32>
    %cst_133 = stablehlo.constant dense_resource<__elided__> : tensor<256xf32>
    %cst_134 = stablehlo.constant dense_resource<__elided__> : tensor<256xf32>
    %cst_135 = stablehlo.constant dense_resource<__elided__> : tensor<256xf32>
    %cst_136 = stablehlo.constant dense_resource<__elided__> : tensor<256xf32>
    %cst_137 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x256x1024xf32>
    %cst_138 = stablehlo.constant dense_resource<__elided__> : tensor<1024xf32>
    %cst_139 = stablehlo.constant dense_resource<__elided__> : tensor<1024xf32>
    %cst_140 = stablehlo.constant dense_resource<__elided__> : tensor<1024xf32>
    %cst_141 = stablehlo.constant dense_resource<__elided__> : tensor<1024xf32>
    %cst_142 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x1024x256xf32>
    %cst_143 = stablehlo.constant dense_resource<__elided__> : tensor<256xf32>
    %cst_144 = stablehlo.constant dense_resource<__elided__> : tensor<256xf32>
    %cst_145 = stablehlo.constant dense_resource<__elided__> : tensor<256xf32>
    %cst_146 = stablehlo.constant dense_resource<__elided__> : tensor<256xf32>
    %cst_147 = stablehlo.constant dense_resource<__elided__> : tensor<3x3x256x256xf32>
    %cst_148 = stablehlo.constant dense_resource<__elided__> : tensor<256xf32>
    %cst_149 = stablehlo.constant dense_resource<__elided__> : tensor<256xf32>
    %cst_150 = stablehlo.constant dense_resource<__elided__> : tensor<256xf32>
    %cst_151 = stablehlo.constant dense_resource<__elided__> : tensor<256xf32>
    %cst_152 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x256x1024xf32>
    %cst_153 = stablehlo.constant dense_resource<__elided__> : tensor<1024xf32>
    %cst_154 = stablehlo.constant dense_resource<__elided__> : tensor<1024xf32>
    %cst_155 = stablehlo.constant dense_resource<__elided__> : tensor<1024xf32>
    %cst_156 = stablehlo.constant dense_resource<__elided__> : tensor<1024xf32>
    %cst_157 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x1024x256xf32>
    %cst_158 = stablehlo.constant dense_resource<__elided__> : tensor<256xf32>
    %cst_159 = stablehlo.constant dense_resource<__elided__> : tensor<256xf32>
    %cst_160 = stablehlo.constant dense_resource<__elided__> : tensor<256xf32>
    %cst_161 = stablehlo.constant dense_resource<__elided__> : tensor<256xf32>
    %cst_162 = stablehlo.constant dense_resource<__elided__> : tensor<3x3x256x256xf32>
    %cst_163 = stablehlo.constant dense_resource<__elided__> : tensor<256xf32>
    %cst_164 = stablehlo.constant dense_resource<__elided__> : tensor<256xf32>
    %cst_165 = stablehlo.constant dense_resource<__elided__> : tensor<256xf32>
    %cst_166 = stablehlo.constant dense_resource<__elided__> : tensor<256xf32>
    %cst_167 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x256x1024xf32>
    %cst_168 = stablehlo.constant dense_resource<__elided__> : tensor<1024xf32>
    %cst_169 = stablehlo.constant dense_resource<__elided__> : tensor<1024xf32>
    %cst_170 = stablehlo.constant dense_resource<__elided__> : tensor<1024xf32>
    %cst_171 = stablehlo.constant dense_resource<__elided__> : tensor<1024xf32>
    %cst_172 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x1024x256xf32>
    %cst_173 = stablehlo.constant dense_resource<__elided__> : tensor<256xf32>
    %cst_174 = stablehlo.constant dense_resource<__elided__> : tensor<256xf32>
    %cst_175 = stablehlo.constant dense_resource<__elided__> : tensor<256xf32>
    %cst_176 = stablehlo.constant dense_resource<__elided__> : tensor<256xf32>
    %cst_177 = stablehlo.constant dense_resource<__elided__> : tensor<3x3x256x256xf32>
    %cst_178 = stablehlo.constant dense_resource<__elided__> : tensor<256xf32>
    %cst_179 = stablehlo.constant dense_resource<__elided__> : tensor<256xf32>
    %cst_180 = stablehlo.constant dense_resource<__elided__> : tensor<256xf32>
    %cst_181 = stablehlo.constant dense_resource<__elided__> : tensor<256xf32>
    %cst_182 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x256x1024xf32>
    %cst_183 = stablehlo.constant dense_resource<__elided__> : tensor<1024xf32>
    %cst_184 = stablehlo.constant dense_resource<__elided__> : tensor<1024xf32>
    %cst_185 = stablehlo.constant dense_resource<__elided__> : tensor<1024xf32>
    %cst_186 = stablehlo.constant dense_resource<__elided__> : tensor<1024xf32>
    %cst_187 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x1024x256xf32>
    %cst_188 = stablehlo.constant dense_resource<__elided__> : tensor<256xf32>
    %cst_189 = stablehlo.constant dense_resource<__elided__> : tensor<256xf32>
    %cst_190 = stablehlo.constant dense_resource<__elided__> : tensor<256xf32>
    %cst_191 = stablehlo.constant dense_resource<__elided__> : tensor<256xf32>
    %cst_192 = stablehlo.constant dense_resource<__elided__> : tensor<3x3x256x256xf32>
    %cst_193 = stablehlo.constant dense_resource<__elided__> : tensor<256xf32>
    %cst_194 = stablehlo.constant dense_resource<__elided__> : tensor<256xf32>
    %cst_195 = stablehlo.constant dense_resource<__elided__> : tensor<256xf32>
    %cst_196 = stablehlo.constant dense_resource<__elided__> : tensor<256xf32>
    %cst_197 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x256x1024xf32>
    %cst_198 = stablehlo.constant dense_resource<__elided__> : tensor<1024xf32>
    %cst_199 = stablehlo.constant dense_resource<__elided__> : tensor<1024xf32>
    %cst_200 = stablehlo.constant dense_resource<__elided__> : tensor<1024xf32>
    %cst_201 = stablehlo.constant dense_resource<__elided__> : tensor<1024xf32>
    %cst_202 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x1024x256xf32>
    %cst_203 = stablehlo.constant dense_resource<__elided__> : tensor<256xf32>
    %cst_204 = stablehlo.constant dense_resource<__elided__> : tensor<256xf32>
    %cst_205 = stablehlo.constant dense_resource<__elided__> : tensor<256xf32>
    %cst_206 = stablehlo.constant dense_resource<__elided__> : tensor<256xf32>
    %cst_207 = stablehlo.constant dense_resource<__elided__> : tensor<3x3x256x256xf32>
    %cst_208 = stablehlo.constant dense_resource<__elided__> : tensor<256xf32>
    %cst_209 = stablehlo.constant dense_resource<__elided__> : tensor<256xf32>
    %cst_210 = stablehlo.constant dense_resource<__elided__> : tensor<256xf32>
    %cst_211 = stablehlo.constant dense_resource<__elided__> : tensor<256xf32>
    %cst_212 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x256x1024xf32>
    %cst_213 = stablehlo.constant dense_resource<__elided__> : tensor<1024xf32>
    %cst_214 = stablehlo.constant dense_resource<__elided__> : tensor<1024xf32>
    %cst_215 = stablehlo.constant dense_resource<__elided__> : tensor<1024xf32>
    %cst_216 = stablehlo.constant dense_resource<__elided__> : tensor<1024xf32>
    %cst_217 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x1024x2048xf32>
    %cst_218 = stablehlo.constant dense_resource<__elided__> : tensor<2048xf32>
    %cst_219 = stablehlo.constant dense_resource<__elided__> : tensor<2048xf32>
    %cst_220 = stablehlo.constant dense_resource<__elided__> : tensor<2048xf32>
    %cst_221 = stablehlo.constant dense_resource<__elided__> : tensor<2048xf32>
    %cst_222 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x1024x512xf32>
    %cst_223 = stablehlo.constant dense_resource<__elided__> : tensor<512xf32>
    %cst_224 = stablehlo.constant dense_resource<__elided__> : tensor<512xf32>
    %cst_225 = stablehlo.constant dense_resource<__elided__> : tensor<512xf32>
    %cst_226 = stablehlo.constant dense_resource<__elided__> : tensor<512xf32>
    %cst_227 = stablehlo.constant dense_resource<__elided__> : tensor<3x3x512x512xf32>
    %cst_228 = stablehlo.constant dense_resource<__elided__> : tensor<512xf32>
    %cst_229 = stablehlo.constant dense_resource<__elided__> : tensor<512xf32>
    %cst_230 = stablehlo.constant dense_resource<__elided__> : tensor<512xf32>
    %cst_231 = stablehlo.constant dense_resource<__elided__> : tensor<512xf32>
    %cst_232 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x512x2048xf32>
    %cst_233 = stablehlo.constant dense_resource<__elided__> : tensor<2048xf32>
    %cst_234 = stablehlo.constant dense_resource<__elided__> : tensor<2048xf32>
    %cst_235 = stablehlo.constant dense_resource<__elided__> : tensor<2048xf32>
    %cst_236 = stablehlo.constant dense_resource<__elided__> : tensor<2048xf32>
    %cst_237 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x2048x512xf32>
    %cst_238 = stablehlo.constant dense_resource<__elided__> : tensor<512xf32>
    %cst_239 = stablehlo.constant dense_resource<__elided__> : tensor<512xf32>
    %cst_240 = stablehlo.constant dense_resource<__elided__> : tensor<512xf32>
    %cst_241 = stablehlo.constant dense_resource<__elided__> : tensor<512xf32>
    %cst_242 = stablehlo.constant dense_resource<__elided__> : tensor<3x3x512x512xf32>
    %cst_243 = stablehlo.constant dense_resource<__elided__> : tensor<512xf32>
    %cst_244 = stablehlo.constant dense_resource<__elided__> : tensor<512xf32>
    %cst_245 = stablehlo.constant dense_resource<__elided__> : tensor<512xf32>
    %cst_246 = stablehlo.constant dense_resource<__elided__> : tensor<512xf32>
    %cst_247 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x512x2048xf32>
    %cst_248 = stablehlo.constant dense_resource<__elided__> : tensor<2048xf32>
    %cst_249 = stablehlo.constant dense_resource<__elided__> : tensor<2048xf32>
    %cst_250 = stablehlo.constant dense_resource<__elided__> : tensor<2048xf32>
    %cst_251 = stablehlo.constant dense_resource<__elided__> : tensor<2048xf32>
    %cst_252 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x2048x512xf32>
    %cst_253 = stablehlo.constant dense_resource<__elided__> : tensor<512xf32>
    %cst_254 = stablehlo.constant dense_resource<__elided__> : tensor<512xf32>
    %cst_255 = stablehlo.constant dense_resource<__elided__> : tensor<512xf32>
    %cst_256 = stablehlo.constant dense_resource<__elided__> : tensor<512xf32>
    %cst_257 = stablehlo.constant dense_resource<__elided__> : tensor<3x3x512x512xf32>
    %cst_258 = stablehlo.constant dense_resource<__elided__> : tensor<512xf32>
    %cst_259 = stablehlo.constant dense_resource<__elided__> : tensor<512xf32>
    %cst_260 = stablehlo.constant dense_resource<__elided__> : tensor<512xf32>
    %cst_261 = stablehlo.constant dense_resource<__elided__> : tensor<512xf32>
    %cst_262 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x512x2048xf32>
    %cst_263 = stablehlo.constant dense_resource<__elided__> : tensor<2048xf32>
    %cst_264 = stablehlo.constant dense_resource<__elided__> : tensor<2048xf32>
    %cst_265 = stablehlo.constant dense_resource<__elided__> : tensor<2048xf32>
    %cst_266 = stablehlo.constant dense_resource<__elided__> : tensor<2048xf32>
    %0 = stablehlo.transpose %arg0, dims = [0, 2, 3, 1] : (tensor<1x3x224x224xf32>) -> tensor<1x224x224x3xf32>
    %1 = stablehlo.convolution(%0, %cst_2) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [2, 2], pad = [[3, 3], [3, 3]]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x224x224x3xf32>, tensor<7x7x3x64xf32>) -> tensor<1x112x112x64xf32>
    %2 = stablehlo.broadcast_in_dim %cst_3, dims = [3] : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %3 = stablehlo.broadcast_in_dim %cst_4, dims = [3] : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %4 = stablehlo.broadcast_in_dim %2, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<1x112x112x64xf32>
    %5 = stablehlo.subtract %1, %4 : tensor<1x112x112x64xf32>
    %6 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x1x1x64xf32>
    %7 = stablehlo.add %3, %6 : tensor<1x1x1x64xf32>
    %8 = stablehlo.rsqrt %7 : tensor<1x1x1x64xf32>
    %9 = stablehlo.reshape %cst_5 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %10 = stablehlo.multiply %8, %9 : tensor<1x1x1x64xf32>
    %11 = stablehlo.broadcast_in_dim %10, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<1x112x112x64xf32>
    %12 = stablehlo.multiply %5, %11 : tensor<1x112x112x64xf32>
    %13 = stablehlo.reshape %cst_6 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %14 = stablehlo.broadcast_in_dim %13, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<1x112x112x64xf32>
    %15 = stablehlo.add %12, %14 : tensor<1x112x112x64xf32>
    %16 = call @relu(%15) : (tensor<1x112x112x64xf32>) -> tensor<1x112x112x64xf32>
    %17 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<f32>
    %18 = "stablehlo.reduce_window"(%16, %17) <{padding = dense<[[0, 0], [1, 1], [1, 1], [0, 0]]> : tensor<4x2xi64>, window_dimensions = array<i64: 1, 3, 3, 1>, window_strides = array<i64: 1, 2, 2, 1>}> ({
    ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
      %870 = stablehlo.maximum %arg1, %arg2 : tensor<f32>
      stablehlo.return %870 : tensor<f32>
    }) : (tensor<1x112x112x64xf32>, tensor<f32>) -> tensor<1x56x56x64xf32>
    %19 = stablehlo.convolution(%18, %cst_7) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x56x56x64xf32>, tensor<1x1x64x256xf32>) -> tensor<1x56x56x256xf32>
    %20 = stablehlo.broadcast_in_dim %cst_8, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %21 = stablehlo.broadcast_in_dim %cst_9, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %22 = stablehlo.broadcast_in_dim %20, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<1x56x56x256xf32>
    %23 = stablehlo.subtract %19, %22 : tensor<1x56x56x256xf32>
    %24 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
    %25 = stablehlo.add %21, %24 : tensor<1x1x1x256xf32>
    %26 = stablehlo.rsqrt %25 : tensor<1x1x1x256xf32>
    %27 = stablehlo.reshape %cst_10 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %28 = stablehlo.multiply %26, %27 : tensor<1x1x1x256xf32>
    %29 = stablehlo.broadcast_in_dim %28, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<1x56x56x256xf32>
    %30 = stablehlo.multiply %23, %29 : tensor<1x56x56x256xf32>
    %31 = stablehlo.reshape %cst_11 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %32 = stablehlo.broadcast_in_dim %31, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<1x56x56x256xf32>
    %33 = stablehlo.add %30, %32 : tensor<1x56x56x256xf32>
    %34 = stablehlo.convolution(%18, %cst_12) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x56x56x64xf32>, tensor<1x1x64x64xf32>) -> tensor<1x56x56x64xf32>
    %35 = stablehlo.broadcast_in_dim %cst_13, dims = [3] : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %36 = stablehlo.broadcast_in_dim %cst_14, dims = [3] : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %37 = stablehlo.broadcast_in_dim %35, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<1x56x56x64xf32>
    %38 = stablehlo.subtract %34, %37 : tensor<1x56x56x64xf32>
    %39 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x1x1x64xf32>
    %40 = stablehlo.add %36, %39 : tensor<1x1x1x64xf32>
    %41 = stablehlo.rsqrt %40 : tensor<1x1x1x64xf32>
    %42 = stablehlo.reshape %cst_15 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %43 = stablehlo.multiply %41, %42 : tensor<1x1x1x64xf32>
    %44 = stablehlo.broadcast_in_dim %43, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<1x56x56x64xf32>
    %45 = stablehlo.multiply %38, %44 : tensor<1x56x56x64xf32>
    %46 = stablehlo.reshape %cst_16 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %47 = stablehlo.broadcast_in_dim %46, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<1x56x56x64xf32>
    %48 = stablehlo.add %45, %47 : tensor<1x56x56x64xf32>
    %49 = call @relu_0(%48) : (tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32>
    %50 = stablehlo.convolution(%49, %cst_17) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {pad = [[1, 1], [1, 1]]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x56x56x64xf32>, tensor<3x3x64x64xf32>) -> tensor<1x56x56x64xf32>
    %51 = stablehlo.broadcast_in_dim %cst_18, dims = [3] : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %52 = stablehlo.broadcast_in_dim %cst_19, dims = [3] : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %53 = stablehlo.broadcast_in_dim %51, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<1x56x56x64xf32>
    %54 = stablehlo.subtract %50, %53 : tensor<1x56x56x64xf32>
    %55 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x1x1x64xf32>
    %56 = stablehlo.add %52, %55 : tensor<1x1x1x64xf32>
    %57 = stablehlo.rsqrt %56 : tensor<1x1x1x64xf32>
    %58 = stablehlo.reshape %cst_20 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %59 = stablehlo.multiply %57, %58 : tensor<1x1x1x64xf32>
    %60 = stablehlo.broadcast_in_dim %59, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<1x56x56x64xf32>
    %61 = stablehlo.multiply %54, %60 : tensor<1x56x56x64xf32>
    %62 = stablehlo.reshape %cst_21 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %63 = stablehlo.broadcast_in_dim %62, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<1x56x56x64xf32>
    %64 = stablehlo.add %61, %63 : tensor<1x56x56x64xf32>
    %65 = call @relu_0(%64) : (tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32>
    %66 = stablehlo.convolution(%65, %cst_22) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x56x56x64xf32>, tensor<1x1x64x256xf32>) -> tensor<1x56x56x256xf32>
    %67 = stablehlo.broadcast_in_dim %cst_23, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %68 = stablehlo.broadcast_in_dim %cst_24, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %69 = stablehlo.broadcast_in_dim %67, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<1x56x56x256xf32>
    %70 = stablehlo.subtract %66, %69 : tensor<1x56x56x256xf32>
    %71 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
    %72 = stablehlo.add %68, %71 : tensor<1x1x1x256xf32>
    %73 = stablehlo.rsqrt %72 : tensor<1x1x1x256xf32>
    %74 = stablehlo.reshape %cst_25 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %75 = stablehlo.multiply %73, %74 : tensor<1x1x1x256xf32>
    %76 = stablehlo.broadcast_in_dim %75, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<1x56x56x256xf32>
    %77 = stablehlo.multiply %70, %76 : tensor<1x56x56x256xf32>
    %78 = stablehlo.reshape %cst_26 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %79 = stablehlo.broadcast_in_dim %78, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<1x56x56x256xf32>
    %80 = stablehlo.add %77, %79 : tensor<1x56x56x256xf32>
    %81 = stablehlo.add %80, %33 : tensor<1x56x56x256xf32>
    %82 = call @relu_1(%81) : (tensor<1x56x56x256xf32>) -> tensor<1x56x56x256xf32>
    %83 = stablehlo.convolution(%82, %cst_27) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x56x56x256xf32>, tensor<1x1x256x64xf32>) -> tensor<1x56x56x64xf32>
    %84 = stablehlo.broadcast_in_dim %cst_28, dims = [3] : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %85 = stablehlo.broadcast_in_dim %cst_29, dims = [3] : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %86 = stablehlo.broadcast_in_dim %84, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<1x56x56x64xf32>
    %87 = stablehlo.subtract %83, %86 : tensor<1x56x56x64xf32>
    %88 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x1x1x64xf32>
    %89 = stablehlo.add %85, %88 : tensor<1x1x1x64xf32>
    %90 = stablehlo.rsqrt %89 : tensor<1x1x1x64xf32>
    %91 = stablehlo.reshape %cst_30 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %92 = stablehlo.multiply %90, %91 : tensor<1x1x1x64xf32>
    %93 = stablehlo.broadcast_in_dim %92, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<1x56x56x64xf32>
    %94 = stablehlo.multiply %87, %93 : tensor<1x56x56x64xf32>
    %95 = stablehlo.reshape %cst_31 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %96 = stablehlo.broadcast_in_dim %95, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<1x56x56x64xf32>
    %97 = stablehlo.add %94, %96 : tensor<1x56x56x64xf32>
    %98 = call @relu_0(%97) : (tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32>
    %99 = stablehlo.convolution(%98, %cst_32) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {pad = [[1, 1], [1, 1]]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x56x56x64xf32>, tensor<3x3x64x64xf32>) -> tensor<1x56x56x64xf32>
    %100 = stablehlo.broadcast_in_dim %cst_33, dims = [3] : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %101 = stablehlo.broadcast_in_dim %cst_34, dims = [3] : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %102 = stablehlo.broadcast_in_dim %100, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<1x56x56x64xf32>
    %103 = stablehlo.subtract %99, %102 : tensor<1x56x56x64xf32>
    %104 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x1x1x64xf32>
    %105 = stablehlo.add %101, %104 : tensor<1x1x1x64xf32>
    %106 = stablehlo.rsqrt %105 : tensor<1x1x1x64xf32>
    %107 = stablehlo.reshape %cst_35 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %108 = stablehlo.multiply %106, %107 : tensor<1x1x1x64xf32>
    %109 = stablehlo.broadcast_in_dim %108, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<1x56x56x64xf32>
    %110 = stablehlo.multiply %103, %109 : tensor<1x56x56x64xf32>
    %111 = stablehlo.reshape %cst_36 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %112 = stablehlo.broadcast_in_dim %111, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<1x56x56x64xf32>
    %113 = stablehlo.add %110, %112 : tensor<1x56x56x64xf32>
    %114 = call @relu_0(%113) : (tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32>
    %115 = stablehlo.convolution(%114, %cst_37) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x56x56x64xf32>, tensor<1x1x64x256xf32>) -> tensor<1x56x56x256xf32>
    %116 = stablehlo.broadcast_in_dim %cst_38, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %117 = stablehlo.broadcast_in_dim %cst_39, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %118 = stablehlo.broadcast_in_dim %116, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<1x56x56x256xf32>
    %119 = stablehlo.subtract %115, %118 : tensor<1x56x56x256xf32>
    %120 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
    %121 = stablehlo.add %117, %120 : tensor<1x1x1x256xf32>
    %122 = stablehlo.rsqrt %121 : tensor<1x1x1x256xf32>
    %123 = stablehlo.reshape %cst_40 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %124 = stablehlo.multiply %122, %123 : tensor<1x1x1x256xf32>
    %125 = stablehlo.broadcast_in_dim %124, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<1x56x56x256xf32>
    %126 = stablehlo.multiply %119, %125 : tensor<1x56x56x256xf32>
    %127 = stablehlo.reshape %cst_41 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %128 = stablehlo.broadcast_in_dim %127, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<1x56x56x256xf32>
    %129 = stablehlo.add %126, %128 : tensor<1x56x56x256xf32>
    %130 = stablehlo.add %129, %82 : tensor<1x56x56x256xf32>
    %131 = call @relu_1(%130) : (tensor<1x56x56x256xf32>) -> tensor<1x56x56x256xf32>
    %132 = stablehlo.convolution(%131, %cst_42) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x56x56x256xf32>, tensor<1x1x256x64xf32>) -> tensor<1x56x56x64xf32>
    %133 = stablehlo.broadcast_in_dim %cst_43, dims = [3] : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %134 = stablehlo.broadcast_in_dim %cst_44, dims = [3] : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %135 = stablehlo.broadcast_in_dim %133, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<1x56x56x64xf32>
    %136 = stablehlo.subtract %132, %135 : tensor<1x56x56x64xf32>
    %137 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x1x1x64xf32>
    %138 = stablehlo.add %134, %137 : tensor<1x1x1x64xf32>
    %139 = stablehlo.rsqrt %138 : tensor<1x1x1x64xf32>
    %140 = stablehlo.reshape %cst_45 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %141 = stablehlo.multiply %139, %140 : tensor<1x1x1x64xf32>
    %142 = stablehlo.broadcast_in_dim %141, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<1x56x56x64xf32>
    %143 = stablehlo.multiply %136, %142 : tensor<1x56x56x64xf32>
    %144 = stablehlo.reshape %cst_46 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %145 = stablehlo.broadcast_in_dim %144, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<1x56x56x64xf32>
    %146 = stablehlo.add %143, %145 : tensor<1x56x56x64xf32>
    %147 = call @relu_0(%146) : (tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32>
    %148 = stablehlo.convolution(%147, %cst_47) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {pad = [[1, 1], [1, 1]]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x56x56x64xf32>, tensor<3x3x64x64xf32>) -> tensor<1x56x56x64xf32>
    %149 = stablehlo.broadcast_in_dim %cst_48, dims = [3] : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %150 = stablehlo.broadcast_in_dim %cst_49, dims = [3] : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %151 = stablehlo.broadcast_in_dim %149, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<1x56x56x64xf32>
    %152 = stablehlo.subtract %148, %151 : tensor<1x56x56x64xf32>
    %153 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x1x1x64xf32>
    %154 = stablehlo.add %150, %153 : tensor<1x1x1x64xf32>
    %155 = stablehlo.rsqrt %154 : tensor<1x1x1x64xf32>
    %156 = stablehlo.reshape %cst_50 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %157 = stablehlo.multiply %155, %156 : tensor<1x1x1x64xf32>
    %158 = stablehlo.broadcast_in_dim %157, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<1x56x56x64xf32>
    %159 = stablehlo.multiply %152, %158 : tensor<1x56x56x64xf32>
    %160 = stablehlo.reshape %cst_51 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %161 = stablehlo.broadcast_in_dim %160, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<1x56x56x64xf32>
    %162 = stablehlo.add %159, %161 : tensor<1x56x56x64xf32>
    %163 = call @relu_0(%162) : (tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32>
    %164 = stablehlo.convolution(%163, %cst_52) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x56x56x64xf32>, tensor<1x1x64x256xf32>) -> tensor<1x56x56x256xf32>
    %165 = stablehlo.broadcast_in_dim %cst_53, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %166 = stablehlo.broadcast_in_dim %cst_54, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %167 = stablehlo.broadcast_in_dim %165, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<1x56x56x256xf32>
    %168 = stablehlo.subtract %164, %167 : tensor<1x56x56x256xf32>
    %169 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
    %170 = stablehlo.add %166, %169 : tensor<1x1x1x256xf32>
    %171 = stablehlo.rsqrt %170 : tensor<1x1x1x256xf32>
    %172 = stablehlo.reshape %cst_55 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %173 = stablehlo.multiply %171, %172 : tensor<1x1x1x256xf32>
    %174 = stablehlo.broadcast_in_dim %173, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<1x56x56x256xf32>
    %175 = stablehlo.multiply %168, %174 : tensor<1x56x56x256xf32>
    %176 = stablehlo.reshape %cst_56 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %177 = stablehlo.broadcast_in_dim %176, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<1x56x56x256xf32>
    %178 = stablehlo.add %175, %177 : tensor<1x56x56x256xf32>
    %179 = stablehlo.add %178, %131 : tensor<1x56x56x256xf32>
    %180 = call @relu_1(%179) : (tensor<1x56x56x256xf32>) -> tensor<1x56x56x256xf32>
    %181 = stablehlo.convolution(%180, %cst_57) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [2, 2]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x56x56x256xf32>, tensor<1x1x256x512xf32>) -> tensor<1x28x28x512xf32>
    %182 = stablehlo.broadcast_in_dim %cst_58, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %183 = stablehlo.broadcast_in_dim %cst_59, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %184 = stablehlo.broadcast_in_dim %182, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<1x28x28x512xf32>
    %185 = stablehlo.subtract %181, %184 : tensor<1x28x28x512xf32>
    %186 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x1x1x512xf32>
    %187 = stablehlo.add %183, %186 : tensor<1x1x1x512xf32>
    %188 = stablehlo.rsqrt %187 : tensor<1x1x1x512xf32>
    %189 = stablehlo.reshape %cst_60 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %190 = stablehlo.multiply %188, %189 : tensor<1x1x1x512xf32>
    %191 = stablehlo.broadcast_in_dim %190, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<1x28x28x512xf32>
    %192 = stablehlo.multiply %185, %191 : tensor<1x28x28x512xf32>
    %193 = stablehlo.reshape %cst_61 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %194 = stablehlo.broadcast_in_dim %193, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<1x28x28x512xf32>
    %195 = stablehlo.add %192, %194 : tensor<1x28x28x512xf32>
    %196 = stablehlo.convolution(%180, %cst_62) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x56x56x256xf32>, tensor<1x1x256x128xf32>) -> tensor<1x56x56x128xf32>
    %197 = stablehlo.broadcast_in_dim %cst_63, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %198 = stablehlo.broadcast_in_dim %cst_64, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %199 = stablehlo.broadcast_in_dim %197, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<1x56x56x128xf32>
    %200 = stablehlo.subtract %196, %199 : tensor<1x56x56x128xf32>
    %201 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x1x1x128xf32>
    %202 = stablehlo.add %198, %201 : tensor<1x1x1x128xf32>
    %203 = stablehlo.rsqrt %202 : tensor<1x1x1x128xf32>
    %204 = stablehlo.reshape %cst_65 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %205 = stablehlo.multiply %203, %204 : tensor<1x1x1x128xf32>
    %206 = stablehlo.broadcast_in_dim %205, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<1x56x56x128xf32>
    %207 = stablehlo.multiply %200, %206 : tensor<1x56x56x128xf32>
    %208 = stablehlo.reshape %cst_66 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %209 = stablehlo.broadcast_in_dim %208, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<1x56x56x128xf32>
    %210 = stablehlo.add %207, %209 : tensor<1x56x56x128xf32>
    %211 = call @relu_2(%210) : (tensor<1x56x56x128xf32>) -> tensor<1x56x56x128xf32>
    %212 = stablehlo.convolution(%211, %cst_67) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [2, 2], pad = [[1, 1], [1, 1]]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x56x56x128xf32>, tensor<3x3x128x128xf32>) -> tensor<1x28x28x128xf32>
    %213 = stablehlo.broadcast_in_dim %cst_68, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %214 = stablehlo.broadcast_in_dim %cst_69, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %215 = stablehlo.broadcast_in_dim %213, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<1x28x28x128xf32>
    %216 = stablehlo.subtract %212, %215 : tensor<1x28x28x128xf32>
    %217 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x1x1x128xf32>
    %218 = stablehlo.add %214, %217 : tensor<1x1x1x128xf32>
    %219 = stablehlo.rsqrt %218 : tensor<1x1x1x128xf32>
    %220 = stablehlo.reshape %cst_70 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %221 = stablehlo.multiply %219, %220 : tensor<1x1x1x128xf32>
    %222 = stablehlo.broadcast_in_dim %221, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<1x28x28x128xf32>
    %223 = stablehlo.multiply %216, %222 : tensor<1x28x28x128xf32>
    %224 = stablehlo.reshape %cst_71 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %225 = stablehlo.broadcast_in_dim %224, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<1x28x28x128xf32>
    %226 = stablehlo.add %223, %225 : tensor<1x28x28x128xf32>
    %227 = call @relu_3(%226) : (tensor<1x28x28x128xf32>) -> tensor<1x28x28x128xf32>
    %228 = stablehlo.convolution(%227, %cst_72) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x28x28x128xf32>, tensor<1x1x128x512xf32>) -> tensor<1x28x28x512xf32>
    %229 = stablehlo.broadcast_in_dim %cst_73, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %230 = stablehlo.broadcast_in_dim %cst_74, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %231 = stablehlo.broadcast_in_dim %229, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<1x28x28x512xf32>
    %232 = stablehlo.subtract %228, %231 : tensor<1x28x28x512xf32>
    %233 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x1x1x512xf32>
    %234 = stablehlo.add %230, %233 : tensor<1x1x1x512xf32>
    %235 = stablehlo.rsqrt %234 : tensor<1x1x1x512xf32>
    %236 = stablehlo.reshape %cst_75 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %237 = stablehlo.multiply %235, %236 : tensor<1x1x1x512xf32>
    %238 = stablehlo.broadcast_in_dim %237, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<1x28x28x512xf32>
    %239 = stablehlo.multiply %232, %238 : tensor<1x28x28x512xf32>
    %240 = stablehlo.reshape %cst_76 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %241 = stablehlo.broadcast_in_dim %240, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<1x28x28x512xf32>
    %242 = stablehlo.add %239, %241 : tensor<1x28x28x512xf32>
    %243 = stablehlo.add %242, %195 : tensor<1x28x28x512xf32>
    %244 = call @relu_4(%243) : (tensor<1x28x28x512xf32>) -> tensor<1x28x28x512xf32>
    %245 = stablehlo.convolution(%244, %cst_77) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x28x28x512xf32>, tensor<1x1x512x128xf32>) -> tensor<1x28x28x128xf32>
    %246 = stablehlo.broadcast_in_dim %cst_78, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %247 = stablehlo.broadcast_in_dim %cst_79, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %248 = stablehlo.broadcast_in_dim %246, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<1x28x28x128xf32>
    %249 = stablehlo.subtract %245, %248 : tensor<1x28x28x128xf32>
    %250 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x1x1x128xf32>
    %251 = stablehlo.add %247, %250 : tensor<1x1x1x128xf32>
    %252 = stablehlo.rsqrt %251 : tensor<1x1x1x128xf32>
    %253 = stablehlo.reshape %cst_80 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %254 = stablehlo.multiply %252, %253 : tensor<1x1x1x128xf32>
    %255 = stablehlo.broadcast_in_dim %254, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<1x28x28x128xf32>
    %256 = stablehlo.multiply %249, %255 : tensor<1x28x28x128xf32>
    %257 = stablehlo.reshape %cst_81 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %258 = stablehlo.broadcast_in_dim %257, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<1x28x28x128xf32>
    %259 = stablehlo.add %256, %258 : tensor<1x28x28x128xf32>
    %260 = call @relu_3(%259) : (tensor<1x28x28x128xf32>) -> tensor<1x28x28x128xf32>
    %261 = stablehlo.convolution(%260, %cst_82) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {pad = [[1, 1], [1, 1]]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x28x28x128xf32>, tensor<3x3x128x128xf32>) -> tensor<1x28x28x128xf32>
    %262 = stablehlo.broadcast_in_dim %cst_83, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %263 = stablehlo.broadcast_in_dim %cst_84, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %264 = stablehlo.broadcast_in_dim %262, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<1x28x28x128xf32>
    %265 = stablehlo.subtract %261, %264 : tensor<1x28x28x128xf32>
    %266 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x1x1x128xf32>
    %267 = stablehlo.add %263, %266 : tensor<1x1x1x128xf32>
    %268 = stablehlo.rsqrt %267 : tensor<1x1x1x128xf32>
    %269 = stablehlo.reshape %cst_85 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %270 = stablehlo.multiply %268, %269 : tensor<1x1x1x128xf32>
    %271 = stablehlo.broadcast_in_dim %270, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<1x28x28x128xf32>
    %272 = stablehlo.multiply %265, %271 : tensor<1x28x28x128xf32>
    %273 = stablehlo.reshape %cst_86 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %274 = stablehlo.broadcast_in_dim %273, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<1x28x28x128xf32>
    %275 = stablehlo.add %272, %274 : tensor<1x28x28x128xf32>
    %276 = call @relu_3(%275) : (tensor<1x28x28x128xf32>) -> tensor<1x28x28x128xf32>
    %277 = stablehlo.convolution(%276, %cst_87) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x28x28x128xf32>, tensor<1x1x128x512xf32>) -> tensor<1x28x28x512xf32>
    %278 = stablehlo.broadcast_in_dim %cst_88, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %279 = stablehlo.broadcast_in_dim %cst_89, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %280 = stablehlo.broadcast_in_dim %278, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<1x28x28x512xf32>
    %281 = stablehlo.subtract %277, %280 : tensor<1x28x28x512xf32>
    %282 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x1x1x512xf32>
    %283 = stablehlo.add %279, %282 : tensor<1x1x1x512xf32>
    %284 = stablehlo.rsqrt %283 : tensor<1x1x1x512xf32>
    %285 = stablehlo.reshape %cst_90 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %286 = stablehlo.multiply %284, %285 : tensor<1x1x1x512xf32>
    %287 = stablehlo.broadcast_in_dim %286, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<1x28x28x512xf32>
    %288 = stablehlo.multiply %281, %287 : tensor<1x28x28x512xf32>
    %289 = stablehlo.reshape %cst_91 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %290 = stablehlo.broadcast_in_dim %289, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<1x28x28x512xf32>
    %291 = stablehlo.add %288, %290 : tensor<1x28x28x512xf32>
    %292 = stablehlo.add %291, %244 : tensor<1x28x28x512xf32>
    %293 = call @relu_4(%292) : (tensor<1x28x28x512xf32>) -> tensor<1x28x28x512xf32>
    %294 = stablehlo.convolution(%293, %cst_92) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x28x28x512xf32>, tensor<1x1x512x128xf32>) -> tensor<1x28x28x128xf32>
    %295 = stablehlo.broadcast_in_dim %cst_93, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %296 = stablehlo.broadcast_in_dim %cst_94, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %297 = stablehlo.broadcast_in_dim %295, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<1x28x28x128xf32>
    %298 = stablehlo.subtract %294, %297 : tensor<1x28x28x128xf32>
    %299 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x1x1x128xf32>
    %300 = stablehlo.add %296, %299 : tensor<1x1x1x128xf32>
    %301 = stablehlo.rsqrt %300 : tensor<1x1x1x128xf32>
    %302 = stablehlo.reshape %cst_95 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %303 = stablehlo.multiply %301, %302 : tensor<1x1x1x128xf32>
    %304 = stablehlo.broadcast_in_dim %303, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<1x28x28x128xf32>
    %305 = stablehlo.multiply %298, %304 : tensor<1x28x28x128xf32>
    %306 = stablehlo.reshape %cst_96 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %307 = stablehlo.broadcast_in_dim %306, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<1x28x28x128xf32>
    %308 = stablehlo.add %305, %307 : tensor<1x28x28x128xf32>
    %309 = call @relu_3(%308) : (tensor<1x28x28x128xf32>) -> tensor<1x28x28x128xf32>
    %310 = stablehlo.convolution(%309, %cst_97) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {pad = [[1, 1], [1, 1]]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x28x28x128xf32>, tensor<3x3x128x128xf32>) -> tensor<1x28x28x128xf32>
    %311 = stablehlo.broadcast_in_dim %cst_98, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %312 = stablehlo.broadcast_in_dim %cst_99, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %313 = stablehlo.broadcast_in_dim %311, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<1x28x28x128xf32>
    %314 = stablehlo.subtract %310, %313 : tensor<1x28x28x128xf32>
    %315 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x1x1x128xf32>
    %316 = stablehlo.add %312, %315 : tensor<1x1x1x128xf32>
    %317 = stablehlo.rsqrt %316 : tensor<1x1x1x128xf32>
    %318 = stablehlo.reshape %cst_100 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %319 = stablehlo.multiply %317, %318 : tensor<1x1x1x128xf32>
    %320 = stablehlo.broadcast_in_dim %319, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<1x28x28x128xf32>
    %321 = stablehlo.multiply %314, %320 : tensor<1x28x28x128xf32>
    %322 = stablehlo.reshape %cst_101 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %323 = stablehlo.broadcast_in_dim %322, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<1x28x28x128xf32>
    %324 = stablehlo.add %321, %323 : tensor<1x28x28x128xf32>
    %325 = call @relu_3(%324) : (tensor<1x28x28x128xf32>) -> tensor<1x28x28x128xf32>
    %326 = stablehlo.convolution(%325, %cst_102) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x28x28x128xf32>, tensor<1x1x128x512xf32>) -> tensor<1x28x28x512xf32>
    %327 = stablehlo.broadcast_in_dim %cst_103, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %328 = stablehlo.broadcast_in_dim %cst_104, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %329 = stablehlo.broadcast_in_dim %327, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<1x28x28x512xf32>
    %330 = stablehlo.subtract %326, %329 : tensor<1x28x28x512xf32>
    %331 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x1x1x512xf32>
    %332 = stablehlo.add %328, %331 : tensor<1x1x1x512xf32>
    %333 = stablehlo.rsqrt %332 : tensor<1x1x1x512xf32>
    %334 = stablehlo.reshape %cst_105 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %335 = stablehlo.multiply %333, %334 : tensor<1x1x1x512xf32>
    %336 = stablehlo.broadcast_in_dim %335, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<1x28x28x512xf32>
    %337 = stablehlo.multiply %330, %336 : tensor<1x28x28x512xf32>
    %338 = stablehlo.reshape %cst_106 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %339 = stablehlo.broadcast_in_dim %338, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<1x28x28x512xf32>
    %340 = stablehlo.add %337, %339 : tensor<1x28x28x512xf32>
    %341 = stablehlo.add %340, %293 : tensor<1x28x28x512xf32>
    %342 = call @relu_4(%341) : (tensor<1x28x28x512xf32>) -> tensor<1x28x28x512xf32>
    %343 = stablehlo.convolution(%342, %cst_107) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x28x28x512xf32>, tensor<1x1x512x128xf32>) -> tensor<1x28x28x128xf32>
    %344 = stablehlo.broadcast_in_dim %cst_108, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %345 = stablehlo.broadcast_in_dim %cst_109, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %346 = stablehlo.broadcast_in_dim %344, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<1x28x28x128xf32>
    %347 = stablehlo.subtract %343, %346 : tensor<1x28x28x128xf32>
    %348 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x1x1x128xf32>
    %349 = stablehlo.add %345, %348 : tensor<1x1x1x128xf32>
    %350 = stablehlo.rsqrt %349 : tensor<1x1x1x128xf32>
    %351 = stablehlo.reshape %cst_110 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %352 = stablehlo.multiply %350, %351 : tensor<1x1x1x128xf32>
    %353 = stablehlo.broadcast_in_dim %352, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<1x28x28x128xf32>
    %354 = stablehlo.multiply %347, %353 : tensor<1x28x28x128xf32>
    %355 = stablehlo.reshape %cst_111 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %356 = stablehlo.broadcast_in_dim %355, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<1x28x28x128xf32>
    %357 = stablehlo.add %354, %356 : tensor<1x28x28x128xf32>
    %358 = call @relu_3(%357) : (tensor<1x28x28x128xf32>) -> tensor<1x28x28x128xf32>
    %359 = stablehlo.convolution(%358, %cst_112) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {pad = [[1, 1], [1, 1]]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x28x28x128xf32>, tensor<3x3x128x128xf32>) -> tensor<1x28x28x128xf32>
    %360 = stablehlo.broadcast_in_dim %cst_113, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %361 = stablehlo.broadcast_in_dim %cst_114, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %362 = stablehlo.broadcast_in_dim %360, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<1x28x28x128xf32>
    %363 = stablehlo.subtract %359, %362 : tensor<1x28x28x128xf32>
    %364 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x1x1x128xf32>
    %365 = stablehlo.add %361, %364 : tensor<1x1x1x128xf32>
    %366 = stablehlo.rsqrt %365 : tensor<1x1x1x128xf32>
    %367 = stablehlo.reshape %cst_115 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %368 = stablehlo.multiply %366, %367 : tensor<1x1x1x128xf32>
    %369 = stablehlo.broadcast_in_dim %368, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<1x28x28x128xf32>
    %370 = stablehlo.multiply %363, %369 : tensor<1x28x28x128xf32>
    %371 = stablehlo.reshape %cst_116 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %372 = stablehlo.broadcast_in_dim %371, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<1x28x28x128xf32>
    %373 = stablehlo.add %370, %372 : tensor<1x28x28x128xf32>
    %374 = call @relu_3(%373) : (tensor<1x28x28x128xf32>) -> tensor<1x28x28x128xf32>
    %375 = stablehlo.convolution(%374, %cst_117) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x28x28x128xf32>, tensor<1x1x128x512xf32>) -> tensor<1x28x28x512xf32>
    %376 = stablehlo.broadcast_in_dim %cst_118, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %377 = stablehlo.broadcast_in_dim %cst_119, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %378 = stablehlo.broadcast_in_dim %376, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<1x28x28x512xf32>
    %379 = stablehlo.subtract %375, %378 : tensor<1x28x28x512xf32>
    %380 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x1x1x512xf32>
    %381 = stablehlo.add %377, %380 : tensor<1x1x1x512xf32>
    %382 = stablehlo.rsqrt %381 : tensor<1x1x1x512xf32>
    %383 = stablehlo.reshape %cst_120 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %384 = stablehlo.multiply %382, %383 : tensor<1x1x1x512xf32>
    %385 = stablehlo.broadcast_in_dim %384, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<1x28x28x512xf32>
    %386 = stablehlo.multiply %379, %385 : tensor<1x28x28x512xf32>
    %387 = stablehlo.reshape %cst_121 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %388 = stablehlo.broadcast_in_dim %387, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<1x28x28x512xf32>
    %389 = stablehlo.add %386, %388 : tensor<1x28x28x512xf32>
    %390 = stablehlo.add %389, %342 : tensor<1x28x28x512xf32>
    %391 = call @relu_4(%390) : (tensor<1x28x28x512xf32>) -> tensor<1x28x28x512xf32>
    %392 = stablehlo.convolution(%391, %cst_122) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [2, 2]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x28x28x512xf32>, tensor<1x1x512x1024xf32>) -> tensor<1x14x14x1024xf32>
    %393 = stablehlo.broadcast_in_dim %cst_123, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %394 = stablehlo.broadcast_in_dim %cst_124, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %395 = stablehlo.broadcast_in_dim %393, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<1x14x14x1024xf32>
    %396 = stablehlo.subtract %392, %395 : tensor<1x14x14x1024xf32>
    %397 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x1x1x1024xf32>
    %398 = stablehlo.add %394, %397 : tensor<1x1x1x1024xf32>
    %399 = stablehlo.rsqrt %398 : tensor<1x1x1x1024xf32>
    %400 = stablehlo.reshape %cst_125 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %401 = stablehlo.multiply %399, %400 : tensor<1x1x1x1024xf32>
    %402 = stablehlo.broadcast_in_dim %401, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<1x14x14x1024xf32>
    %403 = stablehlo.multiply %396, %402 : tensor<1x14x14x1024xf32>
    %404 = stablehlo.reshape %cst_126 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %405 = stablehlo.broadcast_in_dim %404, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<1x14x14x1024xf32>
    %406 = stablehlo.add %403, %405 : tensor<1x14x14x1024xf32>
    %407 = stablehlo.convolution(%391, %cst_127) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x28x28x512xf32>, tensor<1x1x512x256xf32>) -> tensor<1x28x28x256xf32>
    %408 = stablehlo.broadcast_in_dim %cst_128, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %409 = stablehlo.broadcast_in_dim %cst_129, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %410 = stablehlo.broadcast_in_dim %408, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<1x28x28x256xf32>
    %411 = stablehlo.subtract %407, %410 : tensor<1x28x28x256xf32>
    %412 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
    %413 = stablehlo.add %409, %412 : tensor<1x1x1x256xf32>
    %414 = stablehlo.rsqrt %413 : tensor<1x1x1x256xf32>
    %415 = stablehlo.reshape %cst_130 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %416 = stablehlo.multiply %414, %415 : tensor<1x1x1x256xf32>
    %417 = stablehlo.broadcast_in_dim %416, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<1x28x28x256xf32>
    %418 = stablehlo.multiply %411, %417 : tensor<1x28x28x256xf32>
    %419 = stablehlo.reshape %cst_131 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %420 = stablehlo.broadcast_in_dim %419, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<1x28x28x256xf32>
    %421 = stablehlo.add %418, %420 : tensor<1x28x28x256xf32>
    %422 = call @relu_5(%421) : (tensor<1x28x28x256xf32>) -> tensor<1x28x28x256xf32>
    %423 = stablehlo.convolution(%422, %cst_132) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [2, 2], pad = [[1, 1], [1, 1]]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x28x28x256xf32>, tensor<3x3x256x256xf32>) -> tensor<1x14x14x256xf32>
    %424 = stablehlo.broadcast_in_dim %cst_133, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %425 = stablehlo.broadcast_in_dim %cst_134, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %426 = stablehlo.broadcast_in_dim %424, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<1x14x14x256xf32>
    %427 = stablehlo.subtract %423, %426 : tensor<1x14x14x256xf32>
    %428 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
    %429 = stablehlo.add %425, %428 : tensor<1x1x1x256xf32>
    %430 = stablehlo.rsqrt %429 : tensor<1x1x1x256xf32>
    %431 = stablehlo.reshape %cst_135 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %432 = stablehlo.multiply %430, %431 : tensor<1x1x1x256xf32>
    %433 = stablehlo.broadcast_in_dim %432, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<1x14x14x256xf32>
    %434 = stablehlo.multiply %427, %433 : tensor<1x14x14x256xf32>
    %435 = stablehlo.reshape %cst_136 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %436 = stablehlo.broadcast_in_dim %435, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<1x14x14x256xf32>
    %437 = stablehlo.add %434, %436 : tensor<1x14x14x256xf32>
    %438 = call @relu_6(%437) : (tensor<1x14x14x256xf32>) -> tensor<1x14x14x256xf32>
    %439 = stablehlo.convolution(%438, %cst_137) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x14x14x256xf32>, tensor<1x1x256x1024xf32>) -> tensor<1x14x14x1024xf32>
    %440 = stablehlo.broadcast_in_dim %cst_138, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %441 = stablehlo.broadcast_in_dim %cst_139, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %442 = stablehlo.broadcast_in_dim %440, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<1x14x14x1024xf32>
    %443 = stablehlo.subtract %439, %442 : tensor<1x14x14x1024xf32>
    %444 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x1x1x1024xf32>
    %445 = stablehlo.add %441, %444 : tensor<1x1x1x1024xf32>
    %446 = stablehlo.rsqrt %445 : tensor<1x1x1x1024xf32>
    %447 = stablehlo.reshape %cst_140 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %448 = stablehlo.multiply %446, %447 : tensor<1x1x1x1024xf32>
    %449 = stablehlo.broadcast_in_dim %448, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<1x14x14x1024xf32>
    %450 = stablehlo.multiply %443, %449 : tensor<1x14x14x1024xf32>
    %451 = stablehlo.reshape %cst_141 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %452 = stablehlo.broadcast_in_dim %451, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<1x14x14x1024xf32>
    %453 = stablehlo.add %450, %452 : tensor<1x14x14x1024xf32>
    %454 = stablehlo.add %453, %406 : tensor<1x14x14x1024xf32>
    %455 = call @relu_7(%454) : (tensor<1x14x14x1024xf32>) -> tensor<1x14x14x1024xf32>
    %456 = stablehlo.convolution(%455, %cst_142) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x14x14x1024xf32>, tensor<1x1x1024x256xf32>) -> tensor<1x14x14x256xf32>
    %457 = stablehlo.broadcast_in_dim %cst_143, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %458 = stablehlo.broadcast_in_dim %cst_144, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %459 = stablehlo.broadcast_in_dim %457, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<1x14x14x256xf32>
    %460 = stablehlo.subtract %456, %459 : tensor<1x14x14x256xf32>
    %461 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
    %462 = stablehlo.add %458, %461 : tensor<1x1x1x256xf32>
    %463 = stablehlo.rsqrt %462 : tensor<1x1x1x256xf32>
    %464 = stablehlo.reshape %cst_145 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %465 = stablehlo.multiply %463, %464 : tensor<1x1x1x256xf32>
    %466 = stablehlo.broadcast_in_dim %465, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<1x14x14x256xf32>
    %467 = stablehlo.multiply %460, %466 : tensor<1x14x14x256xf32>
    %468 = stablehlo.reshape %cst_146 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %469 = stablehlo.broadcast_in_dim %468, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<1x14x14x256xf32>
    %470 = stablehlo.add %467, %469 : tensor<1x14x14x256xf32>
    %471 = call @relu_6(%470) : (tensor<1x14x14x256xf32>) -> tensor<1x14x14x256xf32>
    %472 = stablehlo.convolution(%471, %cst_147) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {pad = [[1, 1], [1, 1]]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x14x14x256xf32>, tensor<3x3x256x256xf32>) -> tensor<1x14x14x256xf32>
    %473 = stablehlo.broadcast_in_dim %cst_148, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %474 = stablehlo.broadcast_in_dim %cst_149, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %475 = stablehlo.broadcast_in_dim %473, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<1x14x14x256xf32>
    %476 = stablehlo.subtract %472, %475 : tensor<1x14x14x256xf32>
    %477 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
    %478 = stablehlo.add %474, %477 : tensor<1x1x1x256xf32>
    %479 = stablehlo.rsqrt %478 : tensor<1x1x1x256xf32>
    %480 = stablehlo.reshape %cst_150 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %481 = stablehlo.multiply %479, %480 : tensor<1x1x1x256xf32>
    %482 = stablehlo.broadcast_in_dim %481, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<1x14x14x256xf32>
    %483 = stablehlo.multiply %476, %482 : tensor<1x14x14x256xf32>
    %484 = stablehlo.reshape %cst_151 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %485 = stablehlo.broadcast_in_dim %484, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<1x14x14x256xf32>
    %486 = stablehlo.add %483, %485 : tensor<1x14x14x256xf32>
    %487 = call @relu_6(%486) : (tensor<1x14x14x256xf32>) -> tensor<1x14x14x256xf32>
    %488 = stablehlo.convolution(%487, %cst_152) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x14x14x256xf32>, tensor<1x1x256x1024xf32>) -> tensor<1x14x14x1024xf32>
    %489 = stablehlo.broadcast_in_dim %cst_153, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %490 = stablehlo.broadcast_in_dim %cst_154, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %491 = stablehlo.broadcast_in_dim %489, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<1x14x14x1024xf32>
    %492 = stablehlo.subtract %488, %491 : tensor<1x14x14x1024xf32>
    %493 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x1x1x1024xf32>
    %494 = stablehlo.add %490, %493 : tensor<1x1x1x1024xf32>
    %495 = stablehlo.rsqrt %494 : tensor<1x1x1x1024xf32>
    %496 = stablehlo.reshape %cst_155 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %497 = stablehlo.multiply %495, %496 : tensor<1x1x1x1024xf32>
    %498 = stablehlo.broadcast_in_dim %497, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<1x14x14x1024xf32>
    %499 = stablehlo.multiply %492, %498 : tensor<1x14x14x1024xf32>
    %500 = stablehlo.reshape %cst_156 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %501 = stablehlo.broadcast_in_dim %500, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<1x14x14x1024xf32>
    %502 = stablehlo.add %499, %501 : tensor<1x14x14x1024xf32>
    %503 = stablehlo.add %502, %455 : tensor<1x14x14x1024xf32>
    %504 = call @relu_7(%503) : (tensor<1x14x14x1024xf32>) -> tensor<1x14x14x1024xf32>
    %505 = stablehlo.convolution(%504, %cst_157) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x14x14x1024xf32>, tensor<1x1x1024x256xf32>) -> tensor<1x14x14x256xf32>
    %506 = stablehlo.broadcast_in_dim %cst_158, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %507 = stablehlo.broadcast_in_dim %cst_159, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %508 = stablehlo.broadcast_in_dim %506, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<1x14x14x256xf32>
    %509 = stablehlo.subtract %505, %508 : tensor<1x14x14x256xf32>
    %510 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
    %511 = stablehlo.add %507, %510 : tensor<1x1x1x256xf32>
    %512 = stablehlo.rsqrt %511 : tensor<1x1x1x256xf32>
    %513 = stablehlo.reshape %cst_160 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %514 = stablehlo.multiply %512, %513 : tensor<1x1x1x256xf32>
    %515 = stablehlo.broadcast_in_dim %514, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<1x14x14x256xf32>
    %516 = stablehlo.multiply %509, %515 : tensor<1x14x14x256xf32>
    %517 = stablehlo.reshape %cst_161 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %518 = stablehlo.broadcast_in_dim %517, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<1x14x14x256xf32>
    %519 = stablehlo.add %516, %518 : tensor<1x14x14x256xf32>
    %520 = call @relu_6(%519) : (tensor<1x14x14x256xf32>) -> tensor<1x14x14x256xf32>
    %521 = stablehlo.convolution(%520, %cst_162) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {pad = [[1, 1], [1, 1]]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x14x14x256xf32>, tensor<3x3x256x256xf32>) -> tensor<1x14x14x256xf32>
    %522 = stablehlo.broadcast_in_dim %cst_163, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %523 = stablehlo.broadcast_in_dim %cst_164, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %524 = stablehlo.broadcast_in_dim %522, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<1x14x14x256xf32>
    %525 = stablehlo.subtract %521, %524 : tensor<1x14x14x256xf32>
    %526 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
    %527 = stablehlo.add %523, %526 : tensor<1x1x1x256xf32>
    %528 = stablehlo.rsqrt %527 : tensor<1x1x1x256xf32>
    %529 = stablehlo.reshape %cst_165 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %530 = stablehlo.multiply %528, %529 : tensor<1x1x1x256xf32>
    %531 = stablehlo.broadcast_in_dim %530, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<1x14x14x256xf32>
    %532 = stablehlo.multiply %525, %531 : tensor<1x14x14x256xf32>
    %533 = stablehlo.reshape %cst_166 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %534 = stablehlo.broadcast_in_dim %533, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<1x14x14x256xf32>
    %535 = stablehlo.add %532, %534 : tensor<1x14x14x256xf32>
    %536 = call @relu_6(%535) : (tensor<1x14x14x256xf32>) -> tensor<1x14x14x256xf32>
    %537 = stablehlo.convolution(%536, %cst_167) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x14x14x256xf32>, tensor<1x1x256x1024xf32>) -> tensor<1x14x14x1024xf32>
    %538 = stablehlo.broadcast_in_dim %cst_168, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %539 = stablehlo.broadcast_in_dim %cst_169, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %540 = stablehlo.broadcast_in_dim %538, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<1x14x14x1024xf32>
    %541 = stablehlo.subtract %537, %540 : tensor<1x14x14x1024xf32>
    %542 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x1x1x1024xf32>
    %543 = stablehlo.add %539, %542 : tensor<1x1x1x1024xf32>
    %544 = stablehlo.rsqrt %543 : tensor<1x1x1x1024xf32>
    %545 = stablehlo.reshape %cst_170 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %546 = stablehlo.multiply %544, %545 : tensor<1x1x1x1024xf32>
    %547 = stablehlo.broadcast_in_dim %546, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<1x14x14x1024xf32>
    %548 = stablehlo.multiply %541, %547 : tensor<1x14x14x1024xf32>
    %549 = stablehlo.reshape %cst_171 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %550 = stablehlo.broadcast_in_dim %549, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<1x14x14x1024xf32>
    %551 = stablehlo.add %548, %550 : tensor<1x14x14x1024xf32>
    %552 = stablehlo.add %551, %504 : tensor<1x14x14x1024xf32>
    %553 = call @relu_7(%552) : (tensor<1x14x14x1024xf32>) -> tensor<1x14x14x1024xf32>
    %554 = stablehlo.convolution(%553, %cst_172) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x14x14x1024xf32>, tensor<1x1x1024x256xf32>) -> tensor<1x14x14x256xf32>
    %555 = stablehlo.broadcast_in_dim %cst_173, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %556 = stablehlo.broadcast_in_dim %cst_174, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %557 = stablehlo.broadcast_in_dim %555, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<1x14x14x256xf32>
    %558 = stablehlo.subtract %554, %557 : tensor<1x14x14x256xf32>
    %559 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
    %560 = stablehlo.add %556, %559 : tensor<1x1x1x256xf32>
    %561 = stablehlo.rsqrt %560 : tensor<1x1x1x256xf32>
    %562 = stablehlo.reshape %cst_175 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %563 = stablehlo.multiply %561, %562 : tensor<1x1x1x256xf32>
    %564 = stablehlo.broadcast_in_dim %563, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<1x14x14x256xf32>
    %565 = stablehlo.multiply %558, %564 : tensor<1x14x14x256xf32>
    %566 = stablehlo.reshape %cst_176 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %567 = stablehlo.broadcast_in_dim %566, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<1x14x14x256xf32>
    %568 = stablehlo.add %565, %567 : tensor<1x14x14x256xf32>
    %569 = call @relu_6(%568) : (tensor<1x14x14x256xf32>) -> tensor<1x14x14x256xf32>
    %570 = stablehlo.convolution(%569, %cst_177) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {pad = [[1, 1], [1, 1]]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x14x14x256xf32>, tensor<3x3x256x256xf32>) -> tensor<1x14x14x256xf32>
    %571 = stablehlo.broadcast_in_dim %cst_178, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %572 = stablehlo.broadcast_in_dim %cst_179, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %573 = stablehlo.broadcast_in_dim %571, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<1x14x14x256xf32>
    %574 = stablehlo.subtract %570, %573 : tensor<1x14x14x256xf32>
    %575 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
    %576 = stablehlo.add %572, %575 : tensor<1x1x1x256xf32>
    %577 = stablehlo.rsqrt %576 : tensor<1x1x1x256xf32>
    %578 = stablehlo.reshape %cst_180 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %579 = stablehlo.multiply %577, %578 : tensor<1x1x1x256xf32>
    %580 = stablehlo.broadcast_in_dim %579, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<1x14x14x256xf32>
    %581 = stablehlo.multiply %574, %580 : tensor<1x14x14x256xf32>
    %582 = stablehlo.reshape %cst_181 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %583 = stablehlo.broadcast_in_dim %582, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<1x14x14x256xf32>
    %584 = stablehlo.add %581, %583 : tensor<1x14x14x256xf32>
    %585 = call @relu_6(%584) : (tensor<1x14x14x256xf32>) -> tensor<1x14x14x256xf32>
    %586 = stablehlo.convolution(%585, %cst_182) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x14x14x256xf32>, tensor<1x1x256x1024xf32>) -> tensor<1x14x14x1024xf32>
    %587 = stablehlo.broadcast_in_dim %cst_183, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %588 = stablehlo.broadcast_in_dim %cst_184, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %589 = stablehlo.broadcast_in_dim %587, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<1x14x14x1024xf32>
    %590 = stablehlo.subtract %586, %589 : tensor<1x14x14x1024xf32>
    %591 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x1x1x1024xf32>
    %592 = stablehlo.add %588, %591 : tensor<1x1x1x1024xf32>
    %593 = stablehlo.rsqrt %592 : tensor<1x1x1x1024xf32>
    %594 = stablehlo.reshape %cst_185 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %595 = stablehlo.multiply %593, %594 : tensor<1x1x1x1024xf32>
    %596 = stablehlo.broadcast_in_dim %595, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<1x14x14x1024xf32>
    %597 = stablehlo.multiply %590, %596 : tensor<1x14x14x1024xf32>
    %598 = stablehlo.reshape %cst_186 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %599 = stablehlo.broadcast_in_dim %598, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<1x14x14x1024xf32>
    %600 = stablehlo.add %597, %599 : tensor<1x14x14x1024xf32>
    %601 = stablehlo.add %600, %553 : tensor<1x14x14x1024xf32>
    %602 = call @relu_7(%601) : (tensor<1x14x14x1024xf32>) -> tensor<1x14x14x1024xf32>
    %603 = stablehlo.convolution(%602, %cst_187) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x14x14x1024xf32>, tensor<1x1x1024x256xf32>) -> tensor<1x14x14x256xf32>
    %604 = stablehlo.broadcast_in_dim %cst_188, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %605 = stablehlo.broadcast_in_dim %cst_189, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %606 = stablehlo.broadcast_in_dim %604, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<1x14x14x256xf32>
    %607 = stablehlo.subtract %603, %606 : tensor<1x14x14x256xf32>
    %608 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
    %609 = stablehlo.add %605, %608 : tensor<1x1x1x256xf32>
    %610 = stablehlo.rsqrt %609 : tensor<1x1x1x256xf32>
    %611 = stablehlo.reshape %cst_190 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %612 = stablehlo.multiply %610, %611 : tensor<1x1x1x256xf32>
    %613 = stablehlo.broadcast_in_dim %612, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<1x14x14x256xf32>
    %614 = stablehlo.multiply %607, %613 : tensor<1x14x14x256xf32>
    %615 = stablehlo.reshape %cst_191 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %616 = stablehlo.broadcast_in_dim %615, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<1x14x14x256xf32>
    %617 = stablehlo.add %614, %616 : tensor<1x14x14x256xf32>
    %618 = call @relu_6(%617) : (tensor<1x14x14x256xf32>) -> tensor<1x14x14x256xf32>
    %619 = stablehlo.convolution(%618, %cst_192) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {pad = [[1, 1], [1, 1]]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x14x14x256xf32>, tensor<3x3x256x256xf32>) -> tensor<1x14x14x256xf32>
    %620 = stablehlo.broadcast_in_dim %cst_193, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %621 = stablehlo.broadcast_in_dim %cst_194, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %622 = stablehlo.broadcast_in_dim %620, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<1x14x14x256xf32>
    %623 = stablehlo.subtract %619, %622 : tensor<1x14x14x256xf32>
    %624 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
    %625 = stablehlo.add %621, %624 : tensor<1x1x1x256xf32>
    %626 = stablehlo.rsqrt %625 : tensor<1x1x1x256xf32>
    %627 = stablehlo.reshape %cst_195 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %628 = stablehlo.multiply %626, %627 : tensor<1x1x1x256xf32>
    %629 = stablehlo.broadcast_in_dim %628, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<1x14x14x256xf32>
    %630 = stablehlo.multiply %623, %629 : tensor<1x14x14x256xf32>
    %631 = stablehlo.reshape %cst_196 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %632 = stablehlo.broadcast_in_dim %631, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<1x14x14x256xf32>
    %633 = stablehlo.add %630, %632 : tensor<1x14x14x256xf32>
    %634 = call @relu_6(%633) : (tensor<1x14x14x256xf32>) -> tensor<1x14x14x256xf32>
    %635 = stablehlo.convolution(%634, %cst_197) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x14x14x256xf32>, tensor<1x1x256x1024xf32>) -> tensor<1x14x14x1024xf32>
    %636 = stablehlo.broadcast_in_dim %cst_198, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %637 = stablehlo.broadcast_in_dim %cst_199, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %638 = stablehlo.broadcast_in_dim %636, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<1x14x14x1024xf32>
    %639 = stablehlo.subtract %635, %638 : tensor<1x14x14x1024xf32>
    %640 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x1x1x1024xf32>
    %641 = stablehlo.add %637, %640 : tensor<1x1x1x1024xf32>
    %642 = stablehlo.rsqrt %641 : tensor<1x1x1x1024xf32>
    %643 = stablehlo.reshape %cst_200 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %644 = stablehlo.multiply %642, %643 : tensor<1x1x1x1024xf32>
    %645 = stablehlo.broadcast_in_dim %644, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<1x14x14x1024xf32>
    %646 = stablehlo.multiply %639, %645 : tensor<1x14x14x1024xf32>
    %647 = stablehlo.reshape %cst_201 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %648 = stablehlo.broadcast_in_dim %647, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<1x14x14x1024xf32>
    %649 = stablehlo.add %646, %648 : tensor<1x14x14x1024xf32>
    %650 = stablehlo.add %649, %602 : tensor<1x14x14x1024xf32>
    %651 = call @relu_7(%650) : (tensor<1x14x14x1024xf32>) -> tensor<1x14x14x1024xf32>
    %652 = stablehlo.convolution(%651, %cst_202) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x14x14x1024xf32>, tensor<1x1x1024x256xf32>) -> tensor<1x14x14x256xf32>
    %653 = stablehlo.broadcast_in_dim %cst_203, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %654 = stablehlo.broadcast_in_dim %cst_204, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %655 = stablehlo.broadcast_in_dim %653, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<1x14x14x256xf32>
    %656 = stablehlo.subtract %652, %655 : tensor<1x14x14x256xf32>
    %657 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
    %658 = stablehlo.add %654, %657 : tensor<1x1x1x256xf32>
    %659 = stablehlo.rsqrt %658 : tensor<1x1x1x256xf32>
    %660 = stablehlo.reshape %cst_205 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %661 = stablehlo.multiply %659, %660 : tensor<1x1x1x256xf32>
    %662 = stablehlo.broadcast_in_dim %661, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<1x14x14x256xf32>
    %663 = stablehlo.multiply %656, %662 : tensor<1x14x14x256xf32>
    %664 = stablehlo.reshape %cst_206 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %665 = stablehlo.broadcast_in_dim %664, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<1x14x14x256xf32>
    %666 = stablehlo.add %663, %665 : tensor<1x14x14x256xf32>
    %667 = call @relu_6(%666) : (tensor<1x14x14x256xf32>) -> tensor<1x14x14x256xf32>
    %668 = stablehlo.convolution(%667, %cst_207) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {pad = [[1, 1], [1, 1]]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x14x14x256xf32>, tensor<3x3x256x256xf32>) -> tensor<1x14x14x256xf32>
    %669 = stablehlo.broadcast_in_dim %cst_208, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %670 = stablehlo.broadcast_in_dim %cst_209, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %671 = stablehlo.broadcast_in_dim %669, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<1x14x14x256xf32>
    %672 = stablehlo.subtract %668, %671 : tensor<1x14x14x256xf32>
    %673 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
    %674 = stablehlo.add %670, %673 : tensor<1x1x1x256xf32>
    %675 = stablehlo.rsqrt %674 : tensor<1x1x1x256xf32>
    %676 = stablehlo.reshape %cst_210 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %677 = stablehlo.multiply %675, %676 : tensor<1x1x1x256xf32>
    %678 = stablehlo.broadcast_in_dim %677, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<1x14x14x256xf32>
    %679 = stablehlo.multiply %672, %678 : tensor<1x14x14x256xf32>
    %680 = stablehlo.reshape %cst_211 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %681 = stablehlo.broadcast_in_dim %680, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<1x14x14x256xf32>
    %682 = stablehlo.add %679, %681 : tensor<1x14x14x256xf32>
    %683 = call @relu_6(%682) : (tensor<1x14x14x256xf32>) -> tensor<1x14x14x256xf32>
    %684 = stablehlo.convolution(%683, %cst_212) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x14x14x256xf32>, tensor<1x1x256x1024xf32>) -> tensor<1x14x14x1024xf32>
    %685 = stablehlo.broadcast_in_dim %cst_213, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %686 = stablehlo.broadcast_in_dim %cst_214, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %687 = stablehlo.broadcast_in_dim %685, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<1x14x14x1024xf32>
    %688 = stablehlo.subtract %684, %687 : tensor<1x14x14x1024xf32>
    %689 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x1x1x1024xf32>
    %690 = stablehlo.add %686, %689 : tensor<1x1x1x1024xf32>
    %691 = stablehlo.rsqrt %690 : tensor<1x1x1x1024xf32>
    %692 = stablehlo.reshape %cst_215 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %693 = stablehlo.multiply %691, %692 : tensor<1x1x1x1024xf32>
    %694 = stablehlo.broadcast_in_dim %693, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<1x14x14x1024xf32>
    %695 = stablehlo.multiply %688, %694 : tensor<1x14x14x1024xf32>
    %696 = stablehlo.reshape %cst_216 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %697 = stablehlo.broadcast_in_dim %696, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<1x14x14x1024xf32>
    %698 = stablehlo.add %695, %697 : tensor<1x14x14x1024xf32>
    %699 = stablehlo.add %698, %651 : tensor<1x14x14x1024xf32>
    %700 = call @relu_7(%699) : (tensor<1x14x14x1024xf32>) -> tensor<1x14x14x1024xf32>
    %701 = stablehlo.convolution(%700, %cst_217) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [2, 2]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x14x14x1024xf32>, tensor<1x1x1024x2048xf32>) -> tensor<1x7x7x2048xf32>
    %702 = stablehlo.broadcast_in_dim %cst_218, dims = [3] : (tensor<2048xf32>) -> tensor<1x1x1x2048xf32>
    %703 = stablehlo.broadcast_in_dim %cst_219, dims = [3] : (tensor<2048xf32>) -> tensor<1x1x1x2048xf32>
    %704 = stablehlo.broadcast_in_dim %702, dims = [0, 1, 2, 3] : (tensor<1x1x1x2048xf32>) -> tensor<1x7x7x2048xf32>
    %705 = stablehlo.subtract %701, %704 : tensor<1x7x7x2048xf32>
    %706 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x1x1x2048xf32>
    %707 = stablehlo.add %703, %706 : tensor<1x1x1x2048xf32>
    %708 = stablehlo.rsqrt %707 : tensor<1x1x1x2048xf32>
    %709 = stablehlo.reshape %cst_220 : (tensor<2048xf32>) -> tensor<1x1x1x2048xf32>
    %710 = stablehlo.multiply %708, %709 : tensor<1x1x1x2048xf32>
    %711 = stablehlo.broadcast_in_dim %710, dims = [0, 1, 2, 3] : (tensor<1x1x1x2048xf32>) -> tensor<1x7x7x2048xf32>
    %712 = stablehlo.multiply %705, %711 : tensor<1x7x7x2048xf32>
    %713 = stablehlo.reshape %cst_221 : (tensor<2048xf32>) -> tensor<1x1x1x2048xf32>
    %714 = stablehlo.broadcast_in_dim %713, dims = [0, 1, 2, 3] : (tensor<1x1x1x2048xf32>) -> tensor<1x7x7x2048xf32>
    %715 = stablehlo.add %712, %714 : tensor<1x7x7x2048xf32>
    %716 = stablehlo.convolution(%700, %cst_222) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x14x14x1024xf32>, tensor<1x1x1024x512xf32>) -> tensor<1x14x14x512xf32>
    %717 = stablehlo.broadcast_in_dim %cst_223, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %718 = stablehlo.broadcast_in_dim %cst_224, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %719 = stablehlo.broadcast_in_dim %717, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<1x14x14x512xf32>
    %720 = stablehlo.subtract %716, %719 : tensor<1x14x14x512xf32>
    %721 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x1x1x512xf32>
    %722 = stablehlo.add %718, %721 : tensor<1x1x1x512xf32>
    %723 = stablehlo.rsqrt %722 : tensor<1x1x1x512xf32>
    %724 = stablehlo.reshape %cst_225 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %725 = stablehlo.multiply %723, %724 : tensor<1x1x1x512xf32>
    %726 = stablehlo.broadcast_in_dim %725, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<1x14x14x512xf32>
    %727 = stablehlo.multiply %720, %726 : tensor<1x14x14x512xf32>
    %728 = stablehlo.reshape %cst_226 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %729 = stablehlo.broadcast_in_dim %728, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<1x14x14x512xf32>
    %730 = stablehlo.add %727, %729 : tensor<1x14x14x512xf32>
    %731 = call @relu_8(%730) : (tensor<1x14x14x512xf32>) -> tensor<1x14x14x512xf32>
    %732 = stablehlo.convolution(%731, %cst_227) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [2, 2], pad = [[1, 1], [1, 1]]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x14x14x512xf32>, tensor<3x3x512x512xf32>) -> tensor<1x7x7x512xf32>
    %733 = stablehlo.broadcast_in_dim %cst_228, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %734 = stablehlo.broadcast_in_dim %cst_229, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %735 = stablehlo.broadcast_in_dim %733, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<1x7x7x512xf32>
    %736 = stablehlo.subtract %732, %735 : tensor<1x7x7x512xf32>
    %737 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x1x1x512xf32>
    %738 = stablehlo.add %734, %737 : tensor<1x1x1x512xf32>
    %739 = stablehlo.rsqrt %738 : tensor<1x1x1x512xf32>
    %740 = stablehlo.reshape %cst_230 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %741 = stablehlo.multiply %739, %740 : tensor<1x1x1x512xf32>
    %742 = stablehlo.broadcast_in_dim %741, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<1x7x7x512xf32>
    %743 = stablehlo.multiply %736, %742 : tensor<1x7x7x512xf32>
    %744 = stablehlo.reshape %cst_231 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %745 = stablehlo.broadcast_in_dim %744, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<1x7x7x512xf32>
    %746 = stablehlo.add %743, %745 : tensor<1x7x7x512xf32>
    %747 = call @relu_9(%746) : (tensor<1x7x7x512xf32>) -> tensor<1x7x7x512xf32>
    %748 = stablehlo.convolution(%747, %cst_232) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x7x7x512xf32>, tensor<1x1x512x2048xf32>) -> tensor<1x7x7x2048xf32>
    %749 = stablehlo.broadcast_in_dim %cst_233, dims = [3] : (tensor<2048xf32>) -> tensor<1x1x1x2048xf32>
    %750 = stablehlo.broadcast_in_dim %cst_234, dims = [3] : (tensor<2048xf32>) -> tensor<1x1x1x2048xf32>
    %751 = stablehlo.broadcast_in_dim %749, dims = [0, 1, 2, 3] : (tensor<1x1x1x2048xf32>) -> tensor<1x7x7x2048xf32>
    %752 = stablehlo.subtract %748, %751 : tensor<1x7x7x2048xf32>
    %753 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x1x1x2048xf32>
    %754 = stablehlo.add %750, %753 : tensor<1x1x1x2048xf32>
    %755 = stablehlo.rsqrt %754 : tensor<1x1x1x2048xf32>
    %756 = stablehlo.reshape %cst_235 : (tensor<2048xf32>) -> tensor<1x1x1x2048xf32>
    %757 = stablehlo.multiply %755, %756 : tensor<1x1x1x2048xf32>
    %758 = stablehlo.broadcast_in_dim %757, dims = [0, 1, 2, 3] : (tensor<1x1x1x2048xf32>) -> tensor<1x7x7x2048xf32>
    %759 = stablehlo.multiply %752, %758 : tensor<1x7x7x2048xf32>
    %760 = stablehlo.reshape %cst_236 : (tensor<2048xf32>) -> tensor<1x1x1x2048xf32>
    %761 = stablehlo.broadcast_in_dim %760, dims = [0, 1, 2, 3] : (tensor<1x1x1x2048xf32>) -> tensor<1x7x7x2048xf32>
    %762 = stablehlo.add %759, %761 : tensor<1x7x7x2048xf32>
    %763 = stablehlo.add %762, %715 : tensor<1x7x7x2048xf32>
    %764 = call @relu_10(%763) : (tensor<1x7x7x2048xf32>) -> tensor<1x7x7x2048xf32>
    %765 = stablehlo.convolution(%764, %cst_237) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x7x7x2048xf32>, tensor<1x1x2048x512xf32>) -> tensor<1x7x7x512xf32>
    %766 = stablehlo.broadcast_in_dim %cst_238, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %767 = stablehlo.broadcast_in_dim %cst_239, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %768 = stablehlo.broadcast_in_dim %766, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<1x7x7x512xf32>
    %769 = stablehlo.subtract %765, %768 : tensor<1x7x7x512xf32>
    %770 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x1x1x512xf32>
    %771 = stablehlo.add %767, %770 : tensor<1x1x1x512xf32>
    %772 = stablehlo.rsqrt %771 : tensor<1x1x1x512xf32>
    %773 = stablehlo.reshape %cst_240 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %774 = stablehlo.multiply %772, %773 : tensor<1x1x1x512xf32>
    %775 = stablehlo.broadcast_in_dim %774, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<1x7x7x512xf32>
    %776 = stablehlo.multiply %769, %775 : tensor<1x7x7x512xf32>
    %777 = stablehlo.reshape %cst_241 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %778 = stablehlo.broadcast_in_dim %777, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<1x7x7x512xf32>
    %779 = stablehlo.add %776, %778 : tensor<1x7x7x512xf32>
    %780 = call @relu_9(%779) : (tensor<1x7x7x512xf32>) -> tensor<1x7x7x512xf32>
    %781 = stablehlo.convolution(%780, %cst_242) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {pad = [[1, 1], [1, 1]]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x7x7x512xf32>, tensor<3x3x512x512xf32>) -> tensor<1x7x7x512xf32>
    %782 = stablehlo.broadcast_in_dim %cst_243, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %783 = stablehlo.broadcast_in_dim %cst_244, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %784 = stablehlo.broadcast_in_dim %782, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<1x7x7x512xf32>
    %785 = stablehlo.subtract %781, %784 : tensor<1x7x7x512xf32>
    %786 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x1x1x512xf32>
    %787 = stablehlo.add %783, %786 : tensor<1x1x1x512xf32>
    %788 = stablehlo.rsqrt %787 : tensor<1x1x1x512xf32>
    %789 = stablehlo.reshape %cst_245 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %790 = stablehlo.multiply %788, %789 : tensor<1x1x1x512xf32>
    %791 = stablehlo.broadcast_in_dim %790, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<1x7x7x512xf32>
    %792 = stablehlo.multiply %785, %791 : tensor<1x7x7x512xf32>
    %793 = stablehlo.reshape %cst_246 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %794 = stablehlo.broadcast_in_dim %793, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<1x7x7x512xf32>
    %795 = stablehlo.add %792, %794 : tensor<1x7x7x512xf32>
    %796 = call @relu_9(%795) : (tensor<1x7x7x512xf32>) -> tensor<1x7x7x512xf32>
    %797 = stablehlo.convolution(%796, %cst_247) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x7x7x512xf32>, tensor<1x1x512x2048xf32>) -> tensor<1x7x7x2048xf32>
    %798 = stablehlo.broadcast_in_dim %cst_248, dims = [3] : (tensor<2048xf32>) -> tensor<1x1x1x2048xf32>
    %799 = stablehlo.broadcast_in_dim %cst_249, dims = [3] : (tensor<2048xf32>) -> tensor<1x1x1x2048xf32>
    %800 = stablehlo.broadcast_in_dim %798, dims = [0, 1, 2, 3] : (tensor<1x1x1x2048xf32>) -> tensor<1x7x7x2048xf32>
    %801 = stablehlo.subtract %797, %800 : tensor<1x7x7x2048xf32>
    %802 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x1x1x2048xf32>
    %803 = stablehlo.add %799, %802 : tensor<1x1x1x2048xf32>
    %804 = stablehlo.rsqrt %803 : tensor<1x1x1x2048xf32>
    %805 = stablehlo.reshape %cst_250 : (tensor<2048xf32>) -> tensor<1x1x1x2048xf32>
    %806 = stablehlo.multiply %804, %805 : tensor<1x1x1x2048xf32>
    %807 = stablehlo.broadcast_in_dim %806, dims = [0, 1, 2, 3] : (tensor<1x1x1x2048xf32>) -> tensor<1x7x7x2048xf32>
    %808 = stablehlo.multiply %801, %807 : tensor<1x7x7x2048xf32>
    %809 = stablehlo.reshape %cst_251 : (tensor<2048xf32>) -> tensor<1x1x1x2048xf32>
    %810 = stablehlo.broadcast_in_dim %809, dims = [0, 1, 2, 3] : (tensor<1x1x1x2048xf32>) -> tensor<1x7x7x2048xf32>
    %811 = stablehlo.add %808, %810 : tensor<1x7x7x2048xf32>
    %812 = stablehlo.add %811, %764 : tensor<1x7x7x2048xf32>
    %813 = call @relu_10(%812) : (tensor<1x7x7x2048xf32>) -> tensor<1x7x7x2048xf32>
    %814 = stablehlo.convolution(%813, %cst_252) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x7x7x2048xf32>, tensor<1x1x2048x512xf32>) -> tensor<1x7x7x512xf32>
    %815 = stablehlo.broadcast_in_dim %cst_253, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %816 = stablehlo.broadcast_in_dim %cst_254, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %817 = stablehlo.broadcast_in_dim %815, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<1x7x7x512xf32>
    %818 = stablehlo.subtract %814, %817 : tensor<1x7x7x512xf32>
    %819 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x1x1x512xf32>
    %820 = stablehlo.add %816, %819 : tensor<1x1x1x512xf32>
    %821 = stablehlo.rsqrt %820 : tensor<1x1x1x512xf32>
    %822 = stablehlo.reshape %cst_255 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %823 = stablehlo.multiply %821, %822 : tensor<1x1x1x512xf32>
    %824 = stablehlo.broadcast_in_dim %823, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<1x7x7x512xf32>
    %825 = stablehlo.multiply %818, %824 : tensor<1x7x7x512xf32>
    %826 = stablehlo.reshape %cst_256 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %827 = stablehlo.broadcast_in_dim %826, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<1x7x7x512xf32>
    %828 = stablehlo.add %825, %827 : tensor<1x7x7x512xf32>
    %829 = call @relu_9(%828) : (tensor<1x7x7x512xf32>) -> tensor<1x7x7x512xf32>
    %830 = stablehlo.convolution(%829, %cst_257) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {pad = [[1, 1], [1, 1]]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x7x7x512xf32>, tensor<3x3x512x512xf32>) -> tensor<1x7x7x512xf32>
    %831 = stablehlo.broadcast_in_dim %cst_258, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %832 = stablehlo.broadcast_in_dim %cst_259, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %833 = stablehlo.broadcast_in_dim %831, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<1x7x7x512xf32>
    %834 = stablehlo.subtract %830, %833 : tensor<1x7x7x512xf32>
    %835 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x1x1x512xf32>
    %836 = stablehlo.add %832, %835 : tensor<1x1x1x512xf32>
    %837 = stablehlo.rsqrt %836 : tensor<1x1x1x512xf32>
    %838 = stablehlo.reshape %cst_260 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %839 = stablehlo.multiply %837, %838 : tensor<1x1x1x512xf32>
    %840 = stablehlo.broadcast_in_dim %839, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<1x7x7x512xf32>
    %841 = stablehlo.multiply %834, %840 : tensor<1x7x7x512xf32>
    %842 = stablehlo.reshape %cst_261 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %843 = stablehlo.broadcast_in_dim %842, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<1x7x7x512xf32>
    %844 = stablehlo.add %841, %843 : tensor<1x7x7x512xf32>
    %845 = call @relu_9(%844) : (tensor<1x7x7x512xf32>) -> tensor<1x7x7x512xf32>
    %846 = stablehlo.convolution(%845, %cst_262) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x7x7x512xf32>, tensor<1x1x512x2048xf32>) -> tensor<1x7x7x2048xf32>
    %847 = stablehlo.broadcast_in_dim %cst_263, dims = [3] : (tensor<2048xf32>) -> tensor<1x1x1x2048xf32>
    %848 = stablehlo.broadcast_in_dim %cst_264, dims = [3] : (tensor<2048xf32>) -> tensor<1x1x1x2048xf32>
    %849 = stablehlo.broadcast_in_dim %847, dims = [0, 1, 2, 3] : (tensor<1x1x1x2048xf32>) -> tensor<1x7x7x2048xf32>
    %850 = stablehlo.subtract %846, %849 : tensor<1x7x7x2048xf32>
    %851 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x1x1x2048xf32>
    %852 = stablehlo.add %848, %851 : tensor<1x1x1x2048xf32>
    %853 = stablehlo.rsqrt %852 : tensor<1x1x1x2048xf32>
    %854 = stablehlo.reshape %cst_265 : (tensor<2048xf32>) -> tensor<1x1x1x2048xf32>
    %855 = stablehlo.multiply %853, %854 : tensor<1x1x1x2048xf32>
    %856 = stablehlo.broadcast_in_dim %855, dims = [0, 1, 2, 3] : (tensor<1x1x1x2048xf32>) -> tensor<1x7x7x2048xf32>
    %857 = stablehlo.multiply %850, %856 : tensor<1x7x7x2048xf32>
    %858 = stablehlo.reshape %cst_266 : (tensor<2048xf32>) -> tensor<1x1x1x2048xf32>
    %859 = stablehlo.broadcast_in_dim %858, dims = [0, 1, 2, 3] : (tensor<1x1x1x2048xf32>) -> tensor<1x7x7x2048xf32>
    %860 = stablehlo.add %857, %859 : tensor<1x7x7x2048xf32>
    %861 = stablehlo.add %860, %813 : tensor<1x7x7x2048xf32>
    %862 = call @relu_10(%861) : (tensor<1x7x7x2048xf32>) -> tensor<1x7x7x2048xf32>
    %863 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<f32>
    %864 = "stablehlo.reduce_window"(%862, %863) <{window_dimensions = array<i64: 1, 7, 7, 1>, window_strides = array<i64: 1, 7, 7, 1>}> ({
    ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
      %870 = stablehlo.add %arg1, %arg2 : tensor<f32>
      stablehlo.return %870 : tensor<f32>
    }) : (tensor<1x7x7x2048xf32>, tensor<f32>) -> tensor<1x1x1x2048xf32>
    %865 = stablehlo.convert %c : (tensor<i32>) -> tensor<f32>
    %866 = stablehlo.broadcast_in_dim %865, dims = [] : (tensor<f32>) -> tensor<1x1x1x2048xf32>
    %867 = stablehlo.divide %864, %866 : tensor<1x1x1x2048xf32>
    %868 = stablehlo.transpose %867, dims = [0, 3, 1, 2] : (tensor<1x1x1x2048xf32>) -> tensor<1x2048x1x1xf32>
    %869 = stablehlo.transpose %862, dims = [0, 3, 1, 2] : (tensor<1x7x7x2048xf32>) -> tensor<1x2048x7x7xf32>
    return %869, %868 : tensor<1x2048x7x7xf32>, tensor<1x2048x1x1xf32>
  }
  func.func private @relu(%arg0: tensor<1x112x112x64xf32>) -> tensor<1x112x112x64xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x112x112x64xf32>
    %1 = stablehlo.maximum %arg0, %0 : tensor<1x112x112x64xf32>
    return %1 : tensor<1x112x112x64xf32>
  }
  func.func private @relu_0(%arg0: tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x56x56x64xf32>
    %1 = stablehlo.maximum %arg0, %0 : tensor<1x56x56x64xf32>
    return %1 : tensor<1x56x56x64xf32>
  }
  func.func private @relu_1(%arg0: tensor<1x56x56x256xf32>) -> tensor<1x56x56x256xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x56x56x256xf32>
    %1 = stablehlo.maximum %arg0, %0 : tensor<1x56x56x256xf32>
    return %1 : tensor<1x56x56x256xf32>
  }
  func.func private @relu_2(%arg0: tensor<1x56x56x128xf32>) -> tensor<1x56x56x128xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x56x56x128xf32>
    %1 = stablehlo.maximum %arg0, %0 : tensor<1x56x56x128xf32>
    return %1 : tensor<1x56x56x128xf32>
  }
  func.func private @relu_3(%arg0: tensor<1x28x28x128xf32>) -> tensor<1x28x28x128xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x28x28x128xf32>
    %1 = stablehlo.maximum %arg0, %0 : tensor<1x28x28x128xf32>
    return %1 : tensor<1x28x28x128xf32>
  }
  func.func private @relu_4(%arg0: tensor<1x28x28x512xf32>) -> tensor<1x28x28x512xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x28x28x512xf32>
    %1 = stablehlo.maximum %arg0, %0 : tensor<1x28x28x512xf32>
    return %1 : tensor<1x28x28x512xf32>
  }
  func.func private @relu_5(%arg0: tensor<1x28x28x256xf32>) -> tensor<1x28x28x256xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x28x28x256xf32>
    %1 = stablehlo.maximum %arg0, %0 : tensor<1x28x28x256xf32>
    return %1 : tensor<1x28x28x256xf32>
  }
  func.func private @relu_6(%arg0: tensor<1x14x14x256xf32>) -> tensor<1x14x14x256xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x14x14x256xf32>
    %1 = stablehlo.maximum %arg0, %0 : tensor<1x14x14x256xf32>
    return %1 : tensor<1x14x14x256xf32>
  }
  func.func private @relu_7(%arg0: tensor<1x14x14x1024xf32>) -> tensor<1x14x14x1024xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x14x14x1024xf32>
    %1 = stablehlo.maximum %arg0, %0 : tensor<1x14x14x1024xf32>
    return %1 : tensor<1x14x14x1024xf32>
  }
  func.func private @relu_8(%arg0: tensor<1x14x14x512xf32>) -> tensor<1x14x14x512xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x14x14x512xf32>
    %1 = stablehlo.maximum %arg0, %0 : tensor<1x14x14x512xf32>
    return %1 : tensor<1x14x14x512xf32>
  }
  func.func private @relu_9(%arg0: tensor<1x7x7x512xf32>) -> tensor<1x7x7x512xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x7x7x512xf32>
    %1 = stablehlo.maximum %arg0, %0 : tensor<1x7x7x512xf32>
    return %1 : tensor<1x7x7x512xf32>
  }
  func.func private @relu_10(%arg0: tensor<1x7x7x2048xf32>) -> tensor<1x7x7x2048xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x7x7x2048xf32>
    %1 = stablehlo.maximum %arg0, %0 : tensor<1x7x7x2048xf32>
    return %1 : tensor<1x7x7x2048xf32>
  }
}
