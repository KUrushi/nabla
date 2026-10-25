module {
func.func @main(%a0: tensor<3xf32>, %a1: tensor<3xf32>, %a2: tensor<3xf32>, %a3: tensor<3xf32>, %a4: tensor<3xf32>, %a5: tensor<3xf32>, %a6: tensor<3xf32>, %a7: tensor<3xf32>, %a8: tensor<3xf32>, %a9: tensor<3xf32>, %a10: tensor<3xf32>, %a11: tensor<3xf32>, %a12: tensor<3xf32>, %a13: tensor<3xf32>, %a14: tensor<3xf32>, %a15: tensor<3xf32>, %a16: tensor<3xf32>, %a17: tensor<3xf32>, %a18: tensor<3xf32>, %a19: tensor<3xf32>, %a20: tensor<3xf32>, %a21: tensor<3xf32>, %a22: tensor<3xf32>, %a23: tensor<3xf32>, %a24: tensor<3xf32>, %a25: tensor<3xf32>, %a26: tensor<3xf32>, %a27: tensor<3xf32>, %a28: tensor<3xf32>, %a29: tensor<3xf32>, %a30: tensor<3xf32>, %a31: tensor<3xf32>) -> tensor<3xf32> {
  %r1 = stablehlo.add %a0, %a1 : tensor<3xf32>
  %r2 = stablehlo.add %r1, %a2 : tensor<3xf32>
  %r3 = stablehlo.add %r2, %a3 : tensor<3xf32>
  %r4 = stablehlo.add %r3, %a4 : tensor<3xf32>
  %r5 = stablehlo.add %r4, %a5 : tensor<3xf32>
  %r6 = stablehlo.add %r5, %a6 : tensor<3xf32>
  %r7 = stablehlo.add %r6, %a7 : tensor<3xf32>
  %r8 = stablehlo.add %r7, %a8 : tensor<3xf32>
  %r9 = stablehlo.add %r8, %a9 : tensor<3xf32>
  %r10 = stablehlo.add %r9, %a10 : tensor<3xf32>
  %r11 = stablehlo.add %r10, %a11 : tensor<3xf32>
  %r12 = stablehlo.add %r11, %a12 : tensor<3xf32>
  %r13 = stablehlo.add %r12, %a13 : tensor<3xf32>
  %r14 = stablehlo.add %r13, %a14 : tensor<3xf32>
  %r15 = stablehlo.add %r14, %a15 : tensor<3xf32>
  %r16 = stablehlo.add %r15, %a16 : tensor<3xf32>
  %r17 = stablehlo.add %r16, %a17 : tensor<3xf32>
  %r18 = stablehlo.add %r17, %a18 : tensor<3xf32>
  %r19 = stablehlo.add %r18, %a19 : tensor<3xf32>
  %r20 = stablehlo.add %r19, %a20 : tensor<3xf32>
  %r21 = stablehlo.add %r20, %a21 : tensor<3xf32>
  %r22 = stablehlo.add %r21, %a22 : tensor<3xf32>
  %r23 = stablehlo.add %r22, %a23 : tensor<3xf32>
  %r24 = stablehlo.add %r23, %a24 : tensor<3xf32>
  %r25 = stablehlo.add %r24, %a25 : tensor<3xf32>
  %r26 = stablehlo.add %r25, %a26 : tensor<3xf32>
  %r27 = stablehlo.add %r26, %a27 : tensor<3xf32>
  %r28 = stablehlo.add %r27, %a28 : tensor<3xf32>
  %r29 = stablehlo.add %r28, %a29 : tensor<3xf32>
  %r30 = stablehlo.add %r29, %a30 : tensor<3xf32>
  %r31 = stablehlo.add %r30, %a31 : tensor<3xf32>
  return %r31 : tensor<3xf32>
}
}
