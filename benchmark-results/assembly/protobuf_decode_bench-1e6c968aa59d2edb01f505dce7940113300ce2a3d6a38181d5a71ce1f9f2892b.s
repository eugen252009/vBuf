	.file	"protobuf_decode_bench.18d01b59cfa3324c-cgu.0"
	.section	.text._RNvXsZ_NtNtCsgEmfK2I1SDS_4core3fmt3numjNtB7_5Debug3fmt,"ax",@progbits
	.p2align	4
	.type	_RNvXsZ_NtNtCsgEmfK2I1SDS_4core3fmt3numjNtB7_5Debug3fmt,@function
_RNvXsZ_NtNtCsgEmfK2I1SDS_4core3fmt3numjNtB7_5Debug3fmt:
	.cfi_startproc
	movl	16(%rsi), %eax
	testl	$33554432, %eax
	jne	.LBB0_3
	testl	$67108864, %eax
	jne	.LBB0_2
	jmpq	*_RNvXsi_NtNtNtCsgEmfK2I1SDS_4core3fmt3num3impjNtB9_7Display3fmt@GOTPCREL(%rip)
.LBB0_3:
	jmpq	*_RNvXs6_NtNtCsgEmfK2I1SDS_4core3fmt3numjNtB7_8LowerHex3fmt@GOTPCREL(%rip)
.LBB0_2:
	jmpq	*_RNvXs8_NtNtCsgEmfK2I1SDS_4core3fmt3numjNtB7_8UpperHex3fmt@GOTPCREL(%rip)
.Lfunc_end0:
	.size	_RNvXsZ_NtNtCsgEmfK2I1SDS_4core3fmt3numjNtB7_5Debug3fmt, .Lfunc_end0-_RNvXsZ_NtNtCsgEmfK2I1SDS_4core3fmt3numjNtB7_5Debug3fmt
	.cfi_endproc

	.section	.text._RNvXsk_NtCsjrHSEGnQ3l9_3std3envNtB5_8VarErrorNtNtCsgEmfK2I1SDS_4core3fmt5Debug3fmt,"ax",@progbits
	.p2align	4
	.type	_RNvXsk_NtCsjrHSEGnQ3l9_3std3envNtB5_8VarErrorNtNtCsgEmfK2I1SDS_4core3fmt5Debug3fmt,@function
_RNvXsk_NtCsjrHSEGnQ3l9_3std3envNtB5_8VarErrorNtNtCsgEmfK2I1SDS_4core3fmt5Debug3fmt:
	.cfi_startproc
	movq	%rsi, %rax
	xorl	%ecx, %ecx
	cmpq	(%rdi), %rcx
	jno	.LBB1_1
	leaq	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.0(%rip), %rsi
	movl	$10, %edx
	movq	%rax, %rdi
	jmpq	*_RNvMsa_NtCsgEmfK2I1SDS_4core3fmtNtB5_9Formatter9write_str@GOTPCREL(%rip)
.LBB1_1:
	pushq	%rax
	.cfi_def_cfa_offset 16
	leaq	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.2(%rip), %rsi
	leaq	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.1(%rip), %r8
	movl	$10, %edx
	movq	%rdi, (%rsp)
	movq	%rsp, %rcx
	movq	%rax, %rdi
	callq	*_RNvMsa_NtCsgEmfK2I1SDS_4core3fmtNtB5_9Formatter25debug_tuple_field1_finish@GOTPCREL(%rip)
	popq	%rcx
	.cfi_def_cfa_offset 8
	retq
.Lfunc_end1:
	.size	_RNvXsk_NtCsjrHSEGnQ3l9_3std3envNtB5_8VarErrorNtNtCsgEmfK2I1SDS_4core3fmt5Debug3fmt, .Lfunc_end1-_RNvXsk_NtCsjrHSEGnQ3l9_3std3envNtB5_8VarErrorNtNtCsgEmfK2I1SDS_4core3fmt5Debug3fmt
	.cfi_endproc

	.section	.text._RNvXsq_NtCslNYArtu3iFV_5alloc6stringNtB5_6StringNtNtCsgEmfK2I1SDS_4core3fmt7Display3fmt,"ax",@progbits
	.p2align	4
	.type	_RNvXsq_NtCslNYArtu3iFV_5alloc6stringNtB5_6StringNtNtCsgEmfK2I1SDS_4core3fmt7Display3fmt,@function
_RNvXsq_NtCslNYArtu3iFV_5alloc6stringNtB5_6StringNtNtCsgEmfK2I1SDS_4core3fmt7Display3fmt:
	.cfi_startproc
	movq	%rsi, %rdx
	movq	8(%rdi), %rax
	movq	16(%rdi), %rsi
	movq	%rax, %rdi
	jmpq	*_RNvXsi_NtCsgEmfK2I1SDS_4core3fmteNtB5_7Display3fmt@GOTPCREL(%rip)
.Lfunc_end2:
	.size	_RNvXsq_NtCslNYArtu3iFV_5alloc6stringNtB5_6StringNtNtCsgEmfK2I1SDS_4core3fmt7Display3fmt, .Lfunc_end2-_RNvXsq_NtCslNYArtu3iFV_5alloc6stringNtB5_6StringNtNtCsgEmfK2I1SDS_4core3fmt7Display3fmt
	.cfi_endproc

	.section	".text.unlikely._ZN21protobuf_decode_bench11RunMetadata8required5value28_$u7b$$u7b$closure$u7d$$u7d$17hd6282e5d0d99263aE","ax",@progbits
	.p2align	4
	.type	_ZN21protobuf_decode_bench11RunMetadata8required5value28_$u7b$$u7b$closure$u7d$$u7d$17hd6282e5d0d99263aE,@function
_ZN21protobuf_decode_bench11RunMetadata8required5value28_$u7b$$u7b$closure$u7d$$u7d$17hd6282e5d0d99263aE:
.Lfunc_begin0:
	.cfi_startproc
	.cfi_personality 155, DW.ref.rust_eh_personality
	.cfi_lsda 27, .Lexception0
	pushq	%r15
	.cfi_def_cfa_offset 16
	pushq	%r14
	.cfi_def_cfa_offset 24
	pushq	%rbx
	.cfi_def_cfa_offset 32
	subq	$16, %rsp
	.cfi_def_cfa_offset 48
	.cfi_offset %rbx, -32
	.cfi_offset %r14, -24
	.cfi_offset %r15, -16
	leaq	_ZN44_$LT$$RF$T$u20$as$u20$core..fmt..Display$GT$3fmt17h63df3ed385e2537eE(%rip), %rax
	movq	%rdi, (%rsp)
	movq	%rdx, %r14
	movq	%rsi, %rbx
	movq	%rax, 8(%rsp)
.Ltmp0:
	leaq	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.11(%rip), %rdi
	leaq	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.13(%rip), %rdx
	movq	%rsp, %rsi
	callq	*_RNvNtCsgEmfK2I1SDS_4core9panicking9panic_fmt@GOTPCREL(%rip)
.Ltmp1:
	ud2
.LBB3_2:
.Ltmp2:
	movq	%rax, %r15
	testq	%rbx, %rbx
	jle	.LBB3_4
	movl	$1, %edx
	movq	%r14, %rdi
	movq	%rbx, %rsi
	callq	*_RNvCsfLfy6EI15iL_7___rustc14___rust_dealloc@GOTPCREL(%rip)
.LBB3_4:
	movq	%r15, %rdi
	callq	_Unwind_Resume@PLT
.Lfunc_end3:
	.size	_ZN21protobuf_decode_bench11RunMetadata8required5value28_$u7b$$u7b$closure$u7d$$u7d$17hd6282e5d0d99263aE, .Lfunc_end3-_ZN21protobuf_decode_bench11RunMetadata8required5value28_$u7b$$u7b$closure$u7d$$u7d$17hd6282e5d0d99263aE
	.cfi_endproc
	.section	".gcc_except_table._ZN21protobuf_decode_bench11RunMetadata8required5value28_$u7b$$u7b$closure$u7d$$u7d$17hd6282e5d0d99263aE","a",@progbits
	.p2align	2, 0x0
GCC_except_table3:
.Lexception0:
	.byte	255
	.byte	255
	.byte	1
	.uleb128 .Lcst_end0-.Lcst_begin0
.Lcst_begin0:
	.uleb128 .Ltmp0-.Lfunc_begin0
	.uleb128 .Ltmp1-.Ltmp0
	.uleb128 .Ltmp2-.Lfunc_begin0
	.byte	0
	.uleb128 .Ltmp1-.Lfunc_begin0
	.uleb128 .Lfunc_end3-.Ltmp1
	.byte	0
	.byte	0
.Lcst_end0:
	.p2align	2, 0x0

	.section	.text._ZN21protobuf_decode_bench14prefault_pages17h8bcd11700a304d31E,"ax",@progbits
	.p2align	4
	.type	_ZN21protobuf_decode_bench14prefault_pages17h8bcd11700a304d31E,@function
_ZN21protobuf_decode_bench14prefault_pages17h8bcd11700a304d31E:
	.cfi_startproc
	pushq	%r14
	.cfi_def_cfa_offset 16
	pushq	%rbx
	.cfi_def_cfa_offset 24
	pushq	%rax
	.cfi_def_cfa_offset 32
	.cfi_offset %rbx, -24
	.cfi_offset %r14, -16
	testq	%rsi, %rsi
	je	.LBB4_4
	movq	%rdi, %r14
	movl	$30, %edi
	movq	%rsi, %rbx
	callq	*sysconf@GOTPCREL(%rip)
	testq	%rax, %rax
	jle	.LBB4_5
	movq	%rax, %rcx
	movq	%rbx, %rax
	orq	%rcx, %rax
	shrq	$32, %rax
	je	.LBB4_6
	movq	%rbx, %rax
	xorl	%edx, %edx
	divq	%rcx
	jmp	.LBB4_7
.LBB4_5:
	leaq	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.15(%rip), %rax
	addq	$8, %rsp
	.cfi_def_cfa_offset 24
	popq	%rbx
	.cfi_def_cfa_offset 16
	popq	%r14
	.cfi_def_cfa_offset 8
	retq
.LBB4_6:
	.cfi_def_cfa_offset 32
	movl	%ebx, %eax
	xorl	%edx, %edx
	divl	%ecx
.LBB4_7:
	xorl	%esi, %esi
	testq	%rdx, %rdx
	setne	%sil
	addq	%rax, %rsi
	je	.LBB4_11
	xorl	%edi, %edi
	.p2align	4
.LBB4_9:
	cmpq	%rbx, %rdi
	jae	.LBB4_12
	movzbl	(%r14,%rdi), %eax
	addq	%rcx, %rdi
	decq	%rsi
	jne	.LBB4_9
.LBB4_11:
	movzbl	-1(%r14,%rbx), %eax
.LBB4_4:
	xorl	%eax, %eax
	addq	$8, %rsp
	.cfi_def_cfa_offset 24
	popq	%rbx
	.cfi_def_cfa_offset 16
	popq	%r14
	.cfi_def_cfa_offset 8
	retq
.LBB4_12:
	.cfi_def_cfa_offset 32
	leaq	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.16(%rip), %rdx
	movq	%rbx, %rsi
	callq	*_RNvNtCsgEmfK2I1SDS_4core9panicking18panic_bounds_check@GOTPCREL(%rip)
.Lfunc_end4:
	.size	_ZN21protobuf_decode_bench14prefault_pages17h8bcd11700a304d31E, .Lfunc_end4-_ZN21protobuf_decode_bench14prefault_pages17h8bcd11700a304d31E
	.cfi_endproc

	.section	.rodata.cst8,"aM",@progbits,8
	.p2align	3, 0x0
.LCPI5_0:
	.quad	0x3ff8000000000000
.LCPI5_8:
	.quad	9
.LCPI5_9:
	.quad	0x8000000000000000
.LCPI5_10:
	.quad	0x4265d3ee0b4a0000
.LCPI5_11:
	.quad	0x4272309bb4130000
.LCPI5_14:
	.quad	0x41cdcd6500000000
.LCPI5_15:
	.quad	4841369599423283200
.LCPI5_16:
	.quad	4985484787499139072
.LCPI5_17:
	.quad	0x4530000000100000
.LCPI5_18:
	.quad	0x3fe0000000000000
.LCPI5_19:
	.quad	0x3fa999999999999a
.LCPI5_20:
	.quad	0x43e0000000000000
.LCPI5_21:
	.quad	0x43efffffffffffff
.LCPI5_22:
	.quad	0x3fee666666666666
.LCPI5_23:
	.quad	0x408f400000000000
	.section	.rodata.cst16,"aM",@progbits,16
	.p2align	4, 0x0
.LCPI5_1:
	.zero	4
	.zero	4
	.long	0
	.long	4
.LCPI5_3:
	.byte	4
	.byte	3
	.byte	2
	.byte	2
	.byte	1
	.byte	1
	.byte	1
	.byte	1
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
.LCPI5_4:
	.zero	16,15
.LCPI5_12:
	.long	1127219200
	.long	1160773632
	.long	0
	.long	0
.LCPI5_13:
	.quad	0x4330000000000000
	.quad	0x4530000000000000
	.section	.rodata.cst4,"aM",@progbits,4
	.p2align	2, 0x0
.LCPI5_2:
	.long	1
.LCPI5_5:
	.long	31
.LCPI5_6:
	.short	9
	.short	0
.LCPI5_7:
	.long	73
.LCPI5_25:
	.byte	0
	.byte	0
	.byte	0
	.byte	4
	.section	.rodata,"a",@progbits
.LCPI5_24:
	.byte	15
	.section	.text._ZN21protobuf_decode_bench4main17ha1ffaea49e936ebfE,"ax",@progbits
	.hidden	_ZN21protobuf_decode_bench4main17ha1ffaea49e936ebfE
	.globl	_ZN21protobuf_decode_bench4main17ha1ffaea49e936ebfE
	.p2align	4
	.type	_ZN21protobuf_decode_bench4main17ha1ffaea49e936ebfE,@function
_ZN21protobuf_decode_bench4main17ha1ffaea49e936ebfE:
.Lfunc_begin1:
	.cfi_startproc
	.cfi_personality 155, DW.ref.rust_eh_personality
	.cfi_lsda 27, .Lexception1
	pushq	%rbp
	.cfi_def_cfa_offset 16
	pushq	%r15
	.cfi_def_cfa_offset 24
	pushq	%r14
	.cfi_def_cfa_offset 32
	pushq	%r13
	.cfi_def_cfa_offset 40
	pushq	%r12
	.cfi_def_cfa_offset 48
	pushq	%rbx
	.cfi_def_cfa_offset 56
	subq	$1080, %rsp
	.cfi_def_cfa_offset 1136
	.cfi_offset %rbx, -56
	.cfi_offset %r12, -48
	.cfi_offset %r13, -40
	.cfi_offset %r14, -32
	.cfi_offset %r15, -24
	.cfi_offset %rbp, -16
	leaq	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.3(%rip), %rsi
	leaq	112(%rsp), %rdi
	movl	$15, %edx
	movq	%rsi, 736(%rsp)
	movq	$15, 744(%rsp)
	callq	*_RNvNtCsjrHSEGnQ3l9_3std3env4__var@GOTPCREL(%rip)
	movq	120(%rsp), %rax
	movq	128(%rsp), %r14
	cmpl	$1, 112(%rsp)
	movq	%rax, 480(%rsp)
	je	.LBB5_388
	movq	136(%rsp), %r15
	leaq	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.4(%rip), %rsi
	movq	%rsi, 736(%rsp)
	movq	$12, 744(%rsp)
.Ltmp3:
	leaq	112(%rsp), %rdi
	movl	$12, %edx
	movq	%r14, 640(%rsp)
	callq	*_RNvNtCsjrHSEGnQ3l9_3std3env4__var@GOTPCREL(%rip)
.Ltmp4:
	movq	120(%rsp), %rax
	movq	128(%rsp), %r13
	cmpl	$1, 112(%rsp)
	movq	%rax, 448(%rsp)
	je	.LBB5_389
	movq	136(%rsp), %rbp
	leaq	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.5(%rip), %rsi
	movq	%rsi, 736(%rsp)
	movq	$19, 744(%rsp)
.Ltmp5:
	leaq	112(%rsp), %rdi
	movl	$19, %edx
	movq	%r13, 672(%rsp)
	callq	*_RNvNtCsjrHSEGnQ3l9_3std3env4__var@GOTPCREL(%rip)
.Ltmp6:
	movq	120(%rsp), %rax
	movq	128(%rsp), %rcx
	cmpl	$1, 112(%rsp)
	movq	%rax, 512(%rsp)
	movq	%rcx, 336(%rsp)
	je	.LBB5_390
	movq	136(%rsp), %rax
	leaq	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.6(%rip), %rsi
	movq	%rsi, 736(%rsp)
	movq	$11, 744(%rsp)
	movq	%rax, 592(%rsp)
.Ltmp7:
	leaq	112(%rsp), %rdi
	movl	$11, %edx
	callq	*_RNvNtCsjrHSEGnQ3l9_3std3env4__var@GOTPCREL(%rip)
.Ltmp8:
	movq	120(%rsp), %rax
	movq	128(%rsp), %rbx
	cmpl	$1, 112(%rsp)
	movq	%rax, 560(%rsp)
	je	.LBB5_391
	movq	136(%rsp), %rax
	leaq	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.7(%rip), %rsi
	movq	%rsi, 736(%rsp)
	movq	$14, 744(%rsp)
	movq	%rax, 552(%rsp)
.Ltmp9:
	leaq	112(%rsp), %rdi
	movl	$14, %edx
	movq	%rbx, 320(%rsp)
	callq	*_RNvNtCsjrHSEGnQ3l9_3std3env4__var@GOTPCREL(%rip)
.Ltmp10:
	movq	120(%rsp), %rax
	movq	128(%rsp), %rcx
	cmpl	$1, 112(%rsp)
	movq	%rbp, 976(%rsp)
	movq	%rax, 64(%rsp)
	movq	%rcx, 928(%rsp)
	je	.LBB5_392
	movq	136(%rsp), %r12
	leaq	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.8(%rip), %rsi
	movq	%rsi, 736(%rsp)
	movq	$21, 744(%rsp)
.Ltmp11:
	leaq	112(%rsp), %rdi
	movl	$21, %edx
	callq	*_RNvNtCsjrHSEGnQ3l9_3std3env4__var@GOTPCREL(%rip)
.Ltmp12:
	movq	120(%rsp), %rax
	movq	128(%rsp), %rcx
	cmpl	$1, 112(%rsp)
	movq	%rax, 416(%rsp)
	movq	%rcx, 32(%rsp)
	je	.LBB5_393
	movq	136(%rsp), %rbx
	leaq	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.9(%rip), %rsi
	movq	%rsi, 736(%rsp)
	movq	$15, 744(%rsp)
.Ltmp13:
	leaq	112(%rsp), %rdi
	movl	$15, %edx
	callq	*_RNvNtCsjrHSEGnQ3l9_3std3env4__var@GOTPCREL(%rip)
.Ltmp14:
	movq	120(%rsp), %r14
	movq	128(%rsp), %rbp
	cmpl	$1, 112(%rsp)
	je	.LBB5_394
	movq	136(%rsp), %r13
	leaq	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.10(%rip), %rsi
	movq	%rsi, 352(%rsp)
	movq	$27, 360(%rsp)
.Ltmp15:
	leaq	112(%rsp), %rdi
	movl	$27, %edx
	movq	%rdi, 968(%rsp)
	callq	*_RNvNtCsjrHSEGnQ3l9_3std3env4__var@GOTPCREL(%rip)
.Ltmp16:
	cmpl	$1, 112(%rsp)
	je	.LBB5_395
	leaq	120(%rsp), %rax
	movq	640(%rsp), %rdx
	movq	16(%rax), %rcx
	vmovupd	(%rax), %xmm0
	movq	480(%rsp), %rax
	movq	%rcx, 920(%rsp)
	movq	448(%rsp), %rcx
	vmovupd	%xmm0, 904(%rsp)
	movq	%rax, 736(%rsp)
	movq	%rdx, 744(%rsp)
	movq	%r15, 752(%rsp)
	movq	672(%rsp), %rdx
	movq	%rcx, 760(%rsp)
	movq	976(%rsp), %rcx
	movq	%rdx, 768(%rsp)
	movq	512(%rsp), %rdx
	movq	%rcx, 776(%rsp)
	movq	336(%rsp), %rcx
	movq	%rdx, 784(%rsp)
	movq	592(%rsp), %rdx
	movq	%rcx, 792(%rsp)
	movq	560(%rsp), %rcx
	movq	%rdx, 800(%rsp)
	movq	320(%rsp), %rdx
	movq	%rcx, 808(%rsp)
	movq	552(%rsp), %rcx
	movq	%rdx, 816(%rsp)
	movq	64(%rsp), %rdx
	movq	%rcx, 824(%rsp)
	movq	928(%rsp), %rcx
	movq	%rdx, 832(%rsp)
	movq	416(%rsp), %rdx
	movq	%rcx, 840(%rsp)
	movq	32(%rsp), %rcx
	movq	%r12, 848(%rsp)
	movq	%rdx, 856(%rsp)
	movq	%rcx, 864(%rsp)
	movq	%rbx, 872(%rsp)
	movq	%r14, 880(%rsp)
	movq	%rbp, 888(%rsp)
	movq	%r13, 896(%rsp)
	callq	*_RNvCsfLfy6EI15iL_7___rustc35___rust_no_alloc_shim_is_unstable_v2@GOTPCREL(%rip)
	movl	$16000000, %edi
	movl	$8, %esi
	callq	*_RNvCsfLfy6EI15iL_7___rustc12___rust_alloc@GOTPCREL(%rip)
	testq	%rax, %rax
	je	.LBB5_396
	leaq	600(%rsp), %rbx
	movq	$1000000, 600(%rsp)
	movq	%rax, 608(%rsp)
	movq	$0, 616(%rsp)
	xorl	%r14d, %r14d
	xorl	%r15d, %r15d
	jmp	.LBB5_18
	.p2align	4
.LBB5_17:
	vmovsd	480(%rsp), %xmm0
	vmovsd	%xmm0, (%rax,%r14)
	movl	%r15d, 8(%rax,%r14)
	incq	%r15
	addq	$16, %r14
	movq	%r15, 616(%rsp)
	cmpq	$1000000, %r15
	je	.LBB5_21
.LBB5_18:
	vcvtsi2sd	%r15, %xmm15, %xmm0
	vmulsd	.LCPI5_0(%rip), %xmm0, %xmm0
	vmovsd	%xmm0, 480(%rsp)
	cmpq	600(%rsp), %r15
	jne	.LBB5_17
.Ltmp17:
	movq	%rbx, %rdi
	callq	_ZN5alloc7raw_vec19RawVec$LT$T$C$A$GT$8grow_one17h6e5426fe02c91381E
.Ltmp18:
	movq	608(%rsp), %rax
	jmp	.LBB5_17
.LBB5_21:
	vbroadcastss	.LCPI5_5(%rip), %xmm5
	vbroadcastss	.LCPI5_6(%rip), %xmm4
	vpbroadcastd	.LCPI5_2(%rip), %xmm0
	vpbroadcastb	.LCPI5_24(%rip), %xmm13
	movq	608(%rsp), %r15
	movl	$200, %eax
	movq	$0, 40(%rsp)
	movq	$1, 48(%rsp)
	movq	$0, 56(%rsp)
	vpxor	%xmm2, %xmm2, %xmm2
	vpxor	%xmm15, %xmm15, %xmm15
	vpxor	%xmm14, %xmm14, %xmm14
	vpxor	%xmm3, %xmm3, %xmm3
	vpxor	%xmm1, %xmm1, %xmm1
	vmovaps	%xmm5, 64(%rsp)
	vmovaps	%xmm4, 320(%rsp)
	vpbroadcastd	.LCPI5_7(%rip), %xmm5
	vpbroadcastq	.LCPI5_8(%rip), %ymm4
	vmovdqa	%xmm0, 336(%rsp)
	vmovdqa	%xmm5, 416(%rsp)
	vmovdqu	%ymm4, 928(%rsp)
	.p2align	4
.LBB5_22:
	vmovdqu	%ymm1, 480(%rsp)
	vmovups	-168(%r15,%rax), %ymm0
	vmovups	-200(%r15,%rax), %ymm1
	vmovdqu	%ymm3, 448(%rsp)
	vmovdqa	336(%rsp), %xmm10
	vmovq	.LCPI5_3(%rip), %xmm12
	vperm2f128	$49, %ymm0, %ymm1, %ymm3
	vperm2f128	$32, %ymm0, %ymm1, %ymm0
	vmovups	-136(%r15,%rax), %ymm1
	vunpcklpd	%ymm3, %ymm0, %ymm0
	vmovups	%ymm0, 672(%rsp)
	vmovups	-104(%r15,%rax), %ymm0
	vperm2f128	$49, %ymm0, %ymm1, %ymm3
	vperm2f128	$32, %ymm0, %ymm1, %ymm0
	vmovups	-72(%r15,%rax), %ymm1
	vunpcklpd	%ymm3, %ymm0, %ymm0
	vmovups	%ymm0, 512(%rsp)
	vmovups	-40(%r15,%rax), %ymm0
	vperm2f128	$49, %ymm0, %ymm1, %ymm3
	vperm2f128	$32, %ymm0, %ymm1, %ymm0
	vmovups	-8(%r15,%rax), %ymm1
	vunpcklpd	%ymm3, %ymm0, %ymm0
	vmovups	%ymm0, 560(%rsp)
	vmovups	24(%r15,%rax), %ymm0
	vperm2f128	$49, %ymm0, %ymm1, %ymm3
	vperm2f128	$32, %ymm0, %ymm1, %ymm0
	vpmovsxbd	.LCPI5_25(%rip), %xmm1
	vunpcklpd	%ymm3, %ymm0, %ymm0
	vmovups	%ymm0, 640(%rsp)
	vmovdqu	-192(%r15,%rax), %xmm0
	vpunpckldq	-176(%r15,%rax), %xmm0, %xmm0
	vmovdqa	%ymm1, %ymm5
	vpermd	-160(%r15,%rax), %ymm1, %ymm1
	vpermd	-96(%r15,%rax), %ymm5, %ymm3
	vpermd	-32(%r15,%rax), %ymm5, %ymm4
	vpermd	32(%r15,%rax), %ymm5, %ymm5
	vpblendd	$12, %xmm1, %xmm0, %xmm1
	vmovdqu	-128(%r15,%rax), %xmm0
	vpunpckldq	-112(%r15,%rax), %xmm0, %xmm0
	vpsrlw	$4, %xmm1, %xmm6
	vpand	%xmm6, %xmm13, %xmm6
	vpcmpeqb	%xmm2, %xmm6, %xmm9
	vpshufb	%xmm6, %xmm12, %xmm6
	vpblendd	$12, %xmm3, %xmm0, %xmm3
	vmovdqu	-64(%r15,%rax), %xmm0
	vpunpckldq	-48(%r15,%rax), %xmm0, %xmm0
	vpor	%xmm3, %xmm10, %xmm7
	vpblendd	$12, %xmm4, %xmm0, %xmm0
	vmovdqu	(%r15,%rax), %xmm4
	vpunpckldq	16(%r15,%rax), %xmm4, %xmm4
	addq	$256, %rax
	vpblendd	$12, %xmm5, %xmm4, %xmm11
	vpor	%xmm1, %xmm10, %xmm4
	vpcmpeqd	%xmm2, %xmm1, %xmm1
	vpshufb	%xmm4, %xmm12, %xmm5
	vpand	%xmm5, %xmm9, %xmm5
	vpaddb	%xmm6, %xmm5, %xmm5
	vpcmpeqb	%xmm2, %xmm4, %xmm6
	vpcmpeqw	%xmm2, %xmm4, %xmm4
	vpsrlw	$8, %xmm6, %xmm6
	vpsrld	$16, %xmm4, %xmm4
	vpand	%xmm6, %xmm5, %xmm6
	vpsrlw	$8, %xmm5, %xmm5
	vpaddw	%xmm6, %xmm5, %xmm5
	vpand	%xmm4, %xmm5, %xmm4
	vpsrld	$16, %xmm5, %xmm5
	vpaddd	%xmm4, %xmm5, %xmm9
	vpsrlw	$4, %xmm3, %xmm5
	vpshufb	%xmm7, %xmm12, %xmm4
	vpcmpeqd	%xmm2, %xmm3, %xmm3
	vpand	%xmm5, %xmm13, %xmm5
	vpcmpeqb	%xmm2, %xmm5, %xmm6
	vpshufb	%xmm5, %xmm12, %xmm5
	vpand	%xmm6, %xmm4, %xmm4
	vpor	%xmm0, %xmm10, %xmm6
	vpaddb	%xmm5, %xmm4, %xmm4
	vpcmpeqb	%xmm2, %xmm7, %xmm5
	vpsrlw	$8, %xmm5, %xmm5
	vpand	%xmm5, %xmm4, %xmm5
	vpsrlw	$8, %xmm4, %xmm4
	vpaddw	%xmm5, %xmm4, %xmm4
	vpcmpeqw	%xmm2, %xmm7, %xmm5
	vpshufb	%xmm6, %xmm12, %xmm7
	vpsrld	$16, %xmm5, %xmm5
	vpand	%xmm5, %xmm4, %xmm5
	vpsrld	$16, %xmm4, %xmm4
	vpaddd	%xmm5, %xmm4, %xmm5
	vpsrlw	$4, %xmm0, %xmm4
	vpcmpeqd	%xmm2, %xmm0, %xmm0
	vpand	%xmm4, %xmm13, %xmm4
	vpcmpeqb	%xmm2, %xmm4, %xmm8
	vpshufb	%xmm4, %xmm12, %xmm4
	vpand	%xmm7, %xmm8, %xmm7
	vpaddb	%xmm4, %xmm7, %xmm4
	vpcmpeqb	%xmm2, %xmm6, %xmm7
	vpcmpeqw	%xmm2, %xmm6, %xmm6
	vpsrlw	$8, %xmm7, %xmm7
	vpsrld	$16, %xmm6, %xmm6
	vpand	%xmm7, %xmm4, %xmm7
	vpsrlw	$8, %xmm4, %xmm4
	vpaddw	%xmm7, %xmm4, %xmm4
	vpor	%xmm10, %xmm11, %xmm7
	vpand	%xmm6, %xmm4, %xmm6
	vpsrld	$16, %xmm4, %xmm4
	vpshufb	%xmm7, %xmm12, %xmm8
	vpaddd	%xmm6, %xmm4, %xmm4
	vpsrlw	$4, %xmm11, %xmm6
	vpand	%xmm6, %xmm13, %xmm6
	vpcmpeqb	%xmm2, %xmm6, %xmm10
	vpshufb	%xmm6, %xmm12, %xmm6
	vpcmpeqd	%xmm12, %xmm12, %xmm12
	vpand	%xmm10, %xmm8, %xmm8
	vmovdqa	416(%rsp), %xmm10
	vpaddb	%xmm6, %xmm8, %xmm6
	vpcmpeqb	%xmm2, %xmm7, %xmm8
	vpcmpeqw	%xmm2, %xmm7, %xmm7
	vpsrlw	$8, %xmm8, %xmm8
	vpsrld	$16, %xmm7, %xmm7
	vpand	%xmm6, %xmm8, %xmm8
	vpsrlw	$8, %xmm6, %xmm6
	vpaddw	%xmm6, %xmm8, %xmm6
	vmovdqa	64(%rsp), %xmm8
	vpand	%xmm7, %xmm6, %xmm7
	vpsrld	$16, %xmm6, %xmm6
	vpaddd	%xmm7, %xmm6, %xmm6
	vpxor	%xmm8, %xmm9, %xmm7
	vmovdqa	320(%rsp), %xmm9
	vpxor	%xmm4, %xmm8, %xmm4
	vpxor	%xmm5, %xmm8, %xmm5
	vpmaddwd	%xmm4, %xmm9, %xmm4
	vpmaddwd	%xmm5, %xmm9, %xmm5
	vpmaddwd	%xmm7, %xmm9, %xmm7
	vpaddd	%xmm4, %xmm10, %xmm4
	vpaddd	%xmm5, %xmm10, %xmm5
	vpaddd	%xmm7, %xmm10, %xmm7
	vpsrld	$6, %xmm4, %xmm4
	vpsrld	$6, %xmm5, %xmm5
	vpsrld	$6, %xmm7, %xmm7
	vpsubd	%xmm12, %xmm4, %xmm4
	vpsubd	%xmm12, %xmm5, %xmm5
	vpsubd	%xmm12, %xmm7, %xmm7
	vpandn	%xmm4, %xmm0, %xmm0
	vpxor	%xmm6, %xmm8, %xmm4
	vpandn	%xmm5, %xmm3, %xmm3
	vpcmpeqd	%xmm2, %xmm11, %xmm5
	vpandn	%xmm7, %xmm1, %xmm1
	vpxor	%xmm6, %xmm6, %xmm6
	vmovupd	928(%rsp), %ymm7
	vpmaddwd	%xmm4, %xmm9, %xmm4
	vpmovzxdq	%xmm1, %ymm1
	vpmovzxdq	%xmm3, %ymm3
	vpmovzxdq	%xmm0, %ymm0
	vpaddd	%xmm4, %xmm10, %xmm4
	vpsrld	$6, %xmm4, %xmm4
	vpsubd	%xmm12, %xmm4, %xmm4
	vpandn	%xmm4, %xmm5, %xmm4
	vcmpneqpd	672(%rsp), %ymm6, %ymm5
	vpmovzxdq	%xmm4, %ymm4
	vandpd	%ymm7, %ymm5, %ymm5
	vpaddq	%ymm5, %ymm15, %ymm5
	vpaddq	%ymm1, %ymm5, %ymm1
	vcmpneqpd	512(%rsp), %ymm6, %ymm5
	vandpd	%ymm7, %ymm5, %ymm5
	vpaddq	%ymm5, %ymm14, %ymm5
	vpaddq	%ymm3, %ymm5, %ymm3
	vcmpneqpd	560(%rsp), %ymm6, %ymm5
	vandpd	%ymm7, %ymm5, %ymm5
	vpaddq	448(%rsp), %ymm5, %ymm5
	vpaddq	%ymm0, %ymm5, %ymm0
	vcmpneqpd	640(%rsp), %ymm6, %ymm5
	vandpd	%ymm7, %ymm5, %ymm5
	vpaddq	480(%rsp), %ymm5, %ymm5
	vpaddq	%ymm4, %ymm5, %ymm4
	vpcmpeqd	%ymm5, %ymm5, %ymm5
	vpsubq	%ymm5, %ymm1, %ymm15
	vpsubq	%ymm5, %ymm3, %ymm14
	vpsubq	%ymm5, %ymm0, %ymm3
	vpsubq	%ymm5, %ymm4, %ymm1
	cmpq	$15999944, %rax
	jne	.LBB5_22
	vpaddq	%ymm15, %ymm14, %ymm0
	vpaddq	%ymm3, %ymm1, %ymm1
	xorl	%eax, %eax
	vpaddq	%ymm0, %ymm1, %ymm0
	vextracti128	$1, %ymm0, %xmm1
	vpaddq	%xmm1, %xmm0, %xmm0
	vpshufd	$238, %xmm0, %xmm1
	vpaddq	%xmm1, %xmm0, %xmm0
	vmovq	%xmm0, %rcx
	vpxor	%xmm0, %xmm0, %xmm0
	jmp	.LBB5_26
	.p2align	4
.LBB5_32:
	xorl	%edi, %edi
	movl	15999800(%r15,%rax), %r8d
	testl	%r8d, %r8d
	je	.LBB5_33
.LBB5_24:
	orl	$1, %r8d
	lzcntl	%r8d, %r8d
	xorl	$31, %r8d
	leal	73(%r8,%r8,8), %r8d
	shrl	$6, %r8d
	incl	%r8d
.LBB5_25:
	vcmpneqsd	15999744(%r15,%rax), %xmm0, %xmm1
	vcmpneqsd	15999760(%r15,%rax), %xmm0, %xmm2
	vcmpneqsd	15999776(%r15,%rax), %xmm0, %xmm3
	vmovq	%xmm1, %r9
	andl	$1, %r9d
	leaq	(%r9,%r9,8), %r9
	addq	%r9, %rcx
	leaq	1(%rdx,%rcx), %rcx
	vmovq	%xmm2, %rdx
	vcmpneqsd	15999792(%r15,%rax), %xmm0, %xmm2
	addq	$64, %rax
	andl	$1, %edx
	leaq	(%rdx,%rdx,8), %rdx
	addq	%rcx, %rdx
	leaq	1(%rsi,%rdx), %rcx
	vmovq	%xmm3, %rdx
	andl	$1, %edx
	leaq	(%rdx,%rdx,8), %rdx
	addq	%rcx, %rdx
	leaq	1(%rdi,%rdx), %rcx
	vmovq	%xmm2, %rdx
	andl	$1, %edx
	leaq	(%rdx,%rdx,8), %rdx
	addq	%rcx, %rdx
	leaq	1(%r8,%rdx), %rcx
	cmpq	$256, %rax
	je	.LBB5_34
.LBB5_26:
	movl	15999752(%r15,%rax), %edx
	testl	%edx, %edx
	je	.LBB5_30
	orl	$1, %edx
	lzcntl	%edx, %edx
	xorl	$31, %edx
	leal	73(%rdx,%rdx,8), %edx
	shrl	$6, %edx
	incl	%edx
	movl	15999768(%r15,%rax), %esi
	testl	%esi, %esi
	je	.LBB5_31
.LBB5_28:
	orl	$1, %esi
	lzcntl	%esi, %esi
	xorl	$31, %esi
	leal	73(%rsi,%rsi,8), %esi
	shrl	$6, %esi
	incl	%esi
	movl	15999784(%r15,%rax), %edi
	testl	%edi, %edi
	jne	.LBB5_29
	jmp	.LBB5_32
	.p2align	4
.LBB5_30:
	xorl	%edx, %edx
	movl	15999768(%r15,%rax), %esi
	testl	%esi, %esi
	jne	.LBB5_28
.LBB5_31:
	xorl	%esi, %esi
	movl	15999784(%r15,%rax), %edi
	testl	%edi, %edi
	je	.LBB5_32
.LBB5_29:
	orl	$1, %edi
	lzcntl	%edi, %edi
	xorl	$31, %edi
	leal	73(%rdi,%rdi,8), %edi
	shrl	$6, %edi
	incl	%edi
	movl	15999800(%r15,%rax), %r8d
	testl	%r8d, %r8d
	jne	.LBB5_24
.LBB5_33:
	xorl	%r8d, %r8d
	jmp	.LBB5_25
.LBB5_34:
	addq	$1000000, %rcx
	js	.LBB5_397
	movl	$8, %r12d
	leaq	40(%rsp), %r14
	jmp	.LBB5_39
.LBB5_36:
	movq	48(%rsp), %rcx
	movq	56(%rsp), %rsi
	.p2align	4
.LBB5_37:
	vmovsd	480(%rsp), %xmm0
	vmovsd	%xmm0, (%rcx,%rsi)
	addq	$8, %rsi
	movq	%rsi, 56(%rsp)
.LBB5_38:
	addq	$16, %r12
	cmpq	$16000008, %r12
	je	.LBB5_57
.LBB5_39:
	vmovsd	-8(%r15,%r12), %xmm0
	movl	(%r15,%r12), %ebx
	movq	56(%rsp), %rsi
	vmovsd	%xmm0, 480(%rsp)
	cmpq	%rsi, 40(%rsp)
	je	.LBB5_51
.LBB5_40:
	movq	48(%rsp), %rax
	movb	$10, (%rax,%rsi)
	incq	%rsi
	movq	%rsi, 56(%rsp)
	testq	%rbx, %rbx
	je	.LBB5_42
	movl	%ebx, %eax
	orl	$1, %eax
	lzcntl	%eax, %eax
	xorl	$31, %eax
	leal	73(%rax,%rax,8), %eax
	shrl	$6, %eax
	incl	%eax
	jmp	.LBB5_43
	.p2align	4
.LBB5_42:
	xorl	%eax, %eax
.LBB5_43:
	vxorpd	%xmm0, %xmm0, %xmm0
	vcmpneqsd	480(%rsp), %xmm0, %xmm0
	vmovq	%xmm0, %rcx
	andl	$1, %ecx
	leaq	(%rcx,%rcx,8), %rdi
	addq	%rax, %rdi
.Ltmp22:
	movq	%r14, %rsi
	vzeroupper
	callq	_ZN5prost8encoding6varint13encode_varint17hfef3876196e789c2E
.Ltmp23:
	testq	%rbx, %rbx
	je	.LBB5_47
	movq	56(%rsp), %rsi
	cmpq	%rsi, 40(%rsp)
	je	.LBB5_53
.LBB5_46:
	movq	48(%rsp), %rax
	movb	$8, (%rax,%rsi)
	incq	%rsi
	movq	%rsi, 56(%rsp)
.Ltmp26:
	movq	%rbx, %rdi
	movq	%r14, %rsi
	callq	_ZN5prost8encoding6varint13encode_varint17hfef3876196e789c2E
.Ltmp27:
.LBB5_47:
	vmovsd	480(%rsp), %xmm1
	vxorpd	%xmm0, %xmm0, %xmm0
	vucomisd	%xmm0, %xmm1
	jne	.LBB5_48
	jnp	.LBB5_38
.LBB5_48:
	movq	40(%rsp), %rax
	movq	56(%rsp), %rsi
	cmpq	%rsi, %rax
	je	.LBB5_55
.LBB5_49:
	movq	48(%rsp), %rcx
	movb	$17, (%rcx,%rsi)
	incq	%rsi
	subq	%rsi, %rax
	movq	%rsi, 56(%rsp)
	cmpq	$7, %rax
	ja	.LBB5_37
.Ltmp30:
	movl	$8, %edx
	movl	$1, %ecx
	movl	$1, %r8d
	movq	%r14, %rdi
	callq	_ZN5alloc7raw_vec20RawVecInner$LT$A$GT$7reserve21do_reserve_and_handle17h5eeef80eb9b0c257E
.Ltmp31:
	jmp	.LBB5_36
.LBB5_51:
.Ltmp20:
	movl	$1, %edx
	movl	$1, %ecx
	movl	$1, %r8d
	movq	%r14, %rdi
	vzeroupper
	callq	_ZN5alloc7raw_vec20RawVecInner$LT$A$GT$7reserve21do_reserve_and_handle17h5eeef80eb9b0c257E
.Ltmp21:
	movq	56(%rsp), %rsi
	jmp	.LBB5_40
.LBB5_53:
.Ltmp24:
	movl	$1, %edx
	movl	$1, %ecx
	movl	$1, %r8d
	movq	%r14, %rdi
	callq	_ZN5alloc7raw_vec20RawVecInner$LT$A$GT$7reserve21do_reserve_and_handle17h5eeef80eb9b0c257E
.Ltmp25:
	movq	56(%rsp), %rsi
	jmp	.LBB5_46
.LBB5_55:
.Ltmp28:
	movl	$1, %edx
	movl	$1, %ecx
	movl	$1, %r8d
	movq	%r14, %rdi
	callq	_ZN5alloc7raw_vec20RawVecInner$LT$A$GT$7reserve21do_reserve_and_handle17h5eeef80eb9b0c257E
.Ltmp29:
	movq	40(%rsp), %rax
	movq	56(%rsp), %rsi
	jmp	.LBB5_49
.LBB5_57:
	movq	48(%rsp), %rdi
	movq	56(%rsp), %rsi
.Ltmp33:
	callq	_ZN21protobuf_decode_bench14prefault_pages17h8bcd11700a304d31E
.Ltmp34:
	testq	%rax, %rax
	jne	.LBB5_363
	movq	56(%rsp), %rdx
	movq	48(%rsp), %rsi
	movabsq	$4784463198496096256, %rax
	movq	%rax, 728(%rsp)
	movabsq	$4787942799147466752, %rax
	movq	%rax, 984(%rsp)
.Ltmp35:
	leaq	112(%rsp), %rdi
	callq	_ZN5prost7message7Message6decode17h7e463807ec2b546fE
.Ltmp36:
	movq	112(%rsp), %rax
	movq	%rax, 552(%rsp)
	negq	%rax
	jo	.LBB5_365
	movq	120(%rsp), %rax
	movabsq	$-9223372036854775808, %r15
	movq	%rax, 32(%rsp)
	movq	128(%rsp), %rax
	testq	%rax, %rax
	je	.LBB5_398
	movl	%eax, %ecx
	andl	$7, %ecx
	cmpq	$8, %rax
	jae	.LBB5_64
	vmovsd	.LCPI5_9(%rip), %xmm0
	xorl	%edx, %edx
	jmp	.LBB5_67
.LBB5_64:
	vmovsd	.LCPI5_9(%rip), %xmm0
	movq	32(%rsp), %rdx
	movq	%rax, %rsi
	andq	$-8, %rsi
	leaq	112(%rdx), %rdi
	xorl	%edx, %edx
	.p2align	4
.LBB5_65:
	vaddsd	-112(%rdi), %xmm0, %xmm0
	addq	$8, %rdx
	vaddsd	-96(%rdi), %xmm0, %xmm0
	vaddsd	-80(%rdi), %xmm0, %xmm0
	vaddsd	-64(%rdi), %xmm0, %xmm0
	vaddsd	-48(%rdi), %xmm0, %xmm0
	vaddsd	-32(%rdi), %xmm0, %xmm0
	vaddsd	-16(%rdi), %xmm0, %xmm0
	vaddsd	(%rdi), %xmm0, %xmm0
	subq	$-128, %rdi
	cmpq	%rdx, %rsi
	jne	.LBB5_65
	testq	%rcx, %rcx
	je	.LBB5_69
.LBB5_67:
	shlq	$4, %rdx
	addq	32(%rsp), %rdx
	shll	$4, %ecx
	xorl	%esi, %esi
	.p2align	4
.LBB5_68:
	vaddsd	(%rdx,%rsi), %xmm0, %xmm0
	addq	$16, %rsi
	cmpq	%rsi, %rcx
	jne	.LBB5_68
.LBB5_69:
	vucomisd	.LCPI5_10(%rip), %xmm0
	vmovsd	%xmm0, 624(%rsp)
	jne	.LBB5_399
	jp	.LBB5_399
	movl	%eax, %ecx
	andl	$7, %ecx
	cmpq	$8, %rax
	jae	.LBB5_72
	vmovsd	.LCPI5_9(%rip), %xmm0
	xorl	%edx, %edx
	jmp	.LBB5_75
.LBB5_72:
	vmovsd	.LCPI5_9(%rip), %xmm0
	movq	32(%rsp), %rdx
	andq	$-8, %rax
	leaq	120(%rdx), %rsi
	xorl	%edx, %edx
	.p2align	4
.LBB5_73:
	movl	-112(%rsi), %edi
	movl	-96(%rsi), %r8d
	addq	$8, %rdx
	vcvtsi2sd	%rdi, %xmm15, %xmm1
	vaddsd	-120(%rsi), %xmm1, %xmm1
	vcvtsi2sd	%r8, %xmm15, %xmm2
	movl	-80(%rsi), %edi
	vaddsd	%xmm1, %xmm0, %xmm0
	vaddsd	-104(%rsi), %xmm2, %xmm1
	vaddsd	%xmm1, %xmm0, %xmm0
	vcvtsi2sd	%rdi, %xmm15, %xmm1
	vaddsd	-88(%rsi), %xmm1, %xmm1
	movl	-64(%rsi), %edi
	vaddsd	%xmm1, %xmm0, %xmm0
	vcvtsi2sd	%rdi, %xmm15, %xmm1
	vaddsd	-72(%rsi), %xmm1, %xmm1
	movl	-48(%rsi), %edi
	vaddsd	%xmm1, %xmm0, %xmm0
	vcvtsi2sd	%rdi, %xmm15, %xmm1
	vaddsd	-56(%rsi), %xmm1, %xmm1
	movl	-32(%rsi), %edi
	vaddsd	%xmm1, %xmm0, %xmm0
	vcvtsi2sd	%rdi, %xmm15, %xmm1
	vaddsd	-40(%rsi), %xmm1, %xmm1
	movl	-16(%rsi), %edi
	vaddsd	%xmm1, %xmm0, %xmm0
	vcvtsi2sd	%rdi, %xmm15, %xmm1
	movl	(%rsi), %edi
	vaddsd	-24(%rsi), %xmm1, %xmm1
	vcvtsi2sd	%rdi, %xmm15, %xmm2
	vaddsd	-8(%rsi), %xmm2, %xmm2
	subq	$-128, %rsi
	vaddsd	%xmm1, %xmm0, %xmm0
	vaddsd	%xmm2, %xmm0, %xmm0
	cmpq	%rdx, %rax
	jne	.LBB5_73
	testq	%rcx, %rcx
	je	.LBB5_77
.LBB5_75:
	movq	32(%rsp), %rax
	shlq	$4, %rdx
	shll	$4, %ecx
	leaq	8(%rdx,%rax), %rax
	xorl	%edx, %edx
	.p2align	4
.LBB5_76:
	movl	(%rax,%rdx), %esi
	vcvtsi2sd	%rsi, %xmm15, %xmm1
	vaddsd	-8(%rax,%rdx), %xmm1, %xmm1
	addq	$16, %rdx
	vaddsd	%xmm1, %xmm0, %xmm0
	cmpq	%rdx, %rcx
	jne	.LBB5_76
.LBB5_77:
	vucomisd	.LCPI5_11(%rip), %xmm0
	vmovsd	%xmm0, 992(%rsp)
	jne	.LBB5_366
	jp	.LBB5_366
	callq	*_RNvCsfLfy6EI15iL_7___rustc35___rust_no_alloc_shim_is_unstable_v2@GOTPCREL(%rip)
	movl	$96, %edi
	movl	$8, %esi
	callq	*_RNvCsfLfy6EI15iL_7___rustc12___rust_alloc@GOTPCREL(%rip)
	movq	%rax, 320(%rsp)
	testq	%rax, %rax
	je	.LBB5_368
	callq	*_RNvCsfLfy6EI15iL_7___rustc35___rust_no_alloc_shim_is_unstable_v2@GOTPCREL(%rip)
	movl	$8, %edi
	movl	$8, %esi
	callq	*_RNvCsfLfy6EI15iL_7___rustc12___rust_alloc@GOTPCREL(%rip)
	testq	%rax, %rax
	je	.LBB5_369
	movq	%rax, %rbx
	movq	%r14, (%rax)
	callq	*_RNvCsfLfy6EI15iL_7___rustc35___rust_no_alloc_shim_is_unstable_v2@GOTPCREL(%rip)
	movl	$8, %edi
	movl	$8, %esi
	callq	*_RNvCsfLfy6EI15iL_7___rustc12___rust_alloc@GOTPCREL(%rip)
	testq	%rax, %rax
	je	.LBB5_370
	movq	%r14, (%rax)
	movq	320(%rsp), %r14
	leaq	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.26(%rip), %rcx
	leaq	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.27(%rip), %rsi
	leaq	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.25(%rip), %rdx
	movq	$2, 1040(%rsp)
	movq	%rcx, (%r14)
	movq	(%rbx), %rcx
	movq	$10, 8(%r14)
	movq	%rsi, 16(%r14)
	movq	$53, 24(%r14)
	movq	%rbx, 32(%r14)
	movq	%rdx, 40(%r14)
	leaq	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.29(%rip), %rsi
	leaq	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.30(%rip), %rdx
	movq	%r14, 1048(%rsp)
	movq	$2, 1056(%rsp)
	movq	%rsi, 48(%r14)
	movq	$11, 56(%r14)
	movq	%rdx, 64(%r14)
	leaq	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.28(%rip), %rdx
	movq	$65, 72(%r14)
	movq	%rax, 80(%r14)
	movq	%rdx, 88(%r14)
	movq	16(%rcx), %rdx
	movq	8(%rcx), %rsi
.Ltmp37:
	leaq	112(%rsp), %rdi
	callq	_ZN5prost7message7Message6decode17h7e463807ec2b546fE
.Ltmp38:
	movq	112(%rsp), %rsi
	cmpq	%r15, %rsi
	je	.LBB5_371
	movq	120(%rsp), %rdi
	movq	128(%rsp), %rdx
	testq	%rdx, %rdx
	je	.LBB5_86
	movl	%edx, %eax
	andl	$7, %eax
	cmpq	$8, %rdx
	jae	.LBB5_87
	vmovsd	.LCPI5_9(%rip), %xmm0
	xorl	%ecx, %ecx
	jmp	.LBB5_90
.LBB5_86:
	vmovsd	.LCPI5_9(%rip), %xmm0
	jmp	.LBB5_92
.LBB5_87:
	vmovsd	.LCPI5_9(%rip), %xmm0
	andq	$-8, %rdx
	leaq	112(%rdi), %r8
	xorl	%ecx, %ecx
	.p2align	4
.LBB5_88:
	vaddsd	-112(%r8), %xmm0, %xmm0
	addq	$8, %rcx
	vaddsd	-96(%r8), %xmm0, %xmm0
	vaddsd	-80(%r8), %xmm0, %xmm0
	vaddsd	-64(%r8), %xmm0, %xmm0
	vaddsd	-48(%r8), %xmm0, %xmm0
	vaddsd	-32(%r8), %xmm0, %xmm0
	vaddsd	-16(%r8), %xmm0, %xmm0
	vaddsd	(%r8), %xmm0, %xmm0
	subq	$-128, %r8
	cmpq	%rcx, %rdx
	jne	.LBB5_88
	testq	%rax, %rax
	je	.LBB5_92
.LBB5_90:
	shlq	$4, %rcx
	shll	$4, %eax
	xorl	%edx, %edx
	addq	%rdi, %rcx
	.p2align	4
.LBB5_91:
	vaddsd	(%rcx,%rdx), %xmm0, %xmm0
	addq	$16, %rdx
	cmpq	%rdx, %rax
	jne	.LBB5_91
.LBB5_92:
	testq	%rsi, %rsi
	je	.LBB5_94
	shlq	$4, %rsi
	movl	$8, %edx
	vmovsd	%xmm0, 480(%rsp)
	callq	*_RNvCsfLfy6EI15iL_7___rustc14___rust_dealloc@GOTPCREL(%rip)
	vmovsd	480(%rsp), %xmm0
.LBB5_94:
	vmovsd	%xmm0, 112(%rsp)
	leaq	112(%rsp), %rax
	#APP
	#NO_APP
	movq	40(%r14), %rax
	movq	32(%r14), %rdi
.Ltmp39:
	callq	*40(%rax)
.Ltmp40:
	vmovsd	%xmm0, 112(%rsp)
	leaq	112(%rsp), %rax
	#APP
	#NO_APP
	movq	40(%r14), %rax
	movq	32(%r14), %rdi
.Ltmp41:
	callq	*40(%rax)
.Ltmp42:
	vmovsd	%xmm0, 112(%rsp)
	leaq	112(%rsp), %rax
	#APP
	#NO_APP
	movq	40(%r14), %rax
	movq	32(%r14), %rdi
.Ltmp43:
	callq	*40(%rax)
.Ltmp44:
	vmovsd	%xmm0, 112(%rsp)
	leaq	112(%rsp), %rax
	#APP
	#NO_APP
	movq	40(%r14), %rax
	movq	32(%r14), %rdi
.Ltmp45:
	callq	*40(%rax)
.Ltmp46:
	vmovsd	%xmm0, 112(%rsp)
	leaq	112(%rsp), %rax
	#APP
	#NO_APP
	movq	88(%r14), %rax
	movq	80(%r14), %rdi
.Ltmp47:
	callq	*40(%rax)
.Ltmp48:
	vmovsd	%xmm0, 112(%rsp)
	leaq	112(%rsp), %rax
	#APP
	#NO_APP
	movq	88(%r14), %rax
	movq	80(%r14), %rdi
.Ltmp49:
	callq	*40(%rax)
.Ltmp50:
	vmovsd	%xmm0, 112(%rsp)
	leaq	112(%rsp), %rax
	#APP
	#NO_APP
	movq	88(%r14), %rax
	movq	80(%r14), %rdi
.Ltmp51:
	callq	*40(%rax)
.Ltmp52:
	vmovsd	%xmm0, 112(%rsp)
	leaq	112(%rsp), %rax
	#APP
	#NO_APP
	movq	88(%r14), %rax
	movq	80(%r14), %rdi
.Ltmp53:
	callq	*40(%rax)
.Ltmp54:
	vmovsd	%xmm0, 112(%rsp)
	leaq	112(%rsp), %rax
	#APP
	#NO_APP
	movq	88(%r14), %rax
	movq	80(%r14), %rdi
.Ltmp55:
	callq	*40(%rax)
.Ltmp56:
	vmovsd	%xmm0, 112(%rsp)
	leaq	112(%rsp), %rax
	#APP
	#NO_APP
	callq	*_RNvCsfLfy6EI15iL_7___rustc35___rust_no_alloc_shim_is_unstable_v2@GOTPCREL(%rip)
	movl	$48, %edi
	movl	$8, %esi
	callq	*_RNvCsfLfy6EI15iL_7___rustc12___rust_alloc@GOTPCREL(%rip)
	testq	%rax, %rax
	je	.LBB5_372
	movq	$0, (%rax)
	movq	$8, 8(%rax)
	vxorpd	%xmm0, %xmm0, %xmm0
	vmovupd	%xmm0, 16(%rax)
	movq	$8, 32(%rax)
	movq	%rax, 64(%rsp)
	movq	$0, 40(%rax)
	callq	*_RNvCsfLfy6EI15iL_7___rustc35___rust_no_alloc_shim_is_unstable_v2@GOTPCREL(%rip)
	movl	$16, %edi
	movl	$8, %esi
	callq	*_RNvCsfLfy6EI15iL_7___rustc19___rust_alloc_zeroed@GOTPCREL(%rip)
	movq	%rax, 480(%rsp)
	testq	%rax, %rax
	je	.LBB5_373
	movq	sched_getcpu@GOTPCREL(%rip), %r13
	xorl	%eax, %eax
	jmp	.LBB5_107
	.p2align	4
.LBB5_106:
	vmovapd	448(%rsp), %xmm1
	movq	8(%r15), %rdx
	movq	416(%rsp), %rax
	leaq	(%rbx,%rbx,2), %rsi
	movl	%ebp, %edi
	notl	%edi
	movl	%r13d, %ecx
	notl	%r13d
	incq	%rbx
	shrl	$31, %edi
	shrl	$31, %r13d
	shlq	$4, %rsi
	movl	%edi, (%rdx,%rsi)
	movl	%ebp, 4(%rdx,%rsi)
	movl	%r13d, 8(%rdx,%rsi)
	movl	%ecx, 12(%rdx,%rsi)
	movq	%rax, 16(%rdx,%rsi)
	movq	$1, 24(%rdx,%rsi)
	movq	592(%rsp), %rax
	movq	sched_getcpu@GOTPCREL(%rip), %r13
	vshufpd	$1, %xmm1, %xmm1, %xmm0
	vaddsd	%xmm1, %xmm0, %xmm0
	vaddsd	512(%rsp), %xmm0, %xmm0
	vmovsd	%xmm0, 32(%rdx,%rsi)
	movq	%r14, 40(%rdx,%rsi)
	movq	320(%rsp), %r14
	movq	%rbx, 16(%r15)
	cmpq	$30, %rax
	je	.LBB5_126
.LBB5_107:
	movq	%rax, %rcx
	incq	%rax
	movq	%rcx, 416(%rsp)
	andl	$1, %ecx
	movl	$100, %ebp
	movq	%rax, 592(%rsp)
	movq	%rcx, %rax
	xorq	$1, %rax
	movq	%rcx, 336(%rsp)
	movq	%rax, 672(%rsp)
	leaq	(%rcx,%rcx,2), %rax
	movq	%rax, 928(%rsp)
	shll	$4, %eax
	addq	%r14, %rax
	movq	%rax, 448(%rsp)
	.p2align	4
.LBB5_108:
	callq	*%r13
	movl	%eax, %ebx
	#APP

	lfence

	#NO_APP
	#APP

	rdtsc

	#NO_APP
	movq	%rdx, %r12
	movq	%rax, 512(%rsp)
.Ltmp58:
	callq	*_RNvMNtCsjrHSEGnQ3l9_3std4timeNtB2_7Instant3now@GOTPCREL(%rip)
.Ltmp59:
	movq	448(%rsp), %rcx
	movq	%rax, 112(%rsp)
	movl	%edx, 120(%rsp)
	movq	40(%rcx), %rax
	movq	32(%rcx), %rdi
.Ltmp60:
	callq	*40(%rax)
.Ltmp61:
	vmovsd	%xmm0, 352(%rsp)
	leaq	352(%rsp), %rax
	#APP
	#NO_APP
.Ltmp62:
	leaq	112(%rsp), %rdi
	callq	*_RNvMNtCsjrHSEGnQ3l9_3std4timeNtB2_7Instant7elapsed@GOTPCREL(%rip)
.Ltmp63:
	movq	%rax, %r14
	movl	%edx, %r15d
	#APP

	rdtscp

	#NO_APP
	movq	%rdx, 640(%rsp)
	movq	%rax, 560(%rsp)
	#APP

	lfence

	#NO_APP
	callq	*%r13
	testl	%eax, %eax
	movl	%eax, %r13d
	setns	%al
	cmpl	%r13d, %ebx
	sete	%cl
	testb	%cl, %al
	jne	.LBB5_115
	testl	%ebx, %ebx
	js	.LBB5_115
	movq	480(%rsp), %rax
	movq	336(%rsp), %rcx
	incq	(%rax,%rcx,8)
	subq	$1, %rbp
	jb	.LBB5_130
	movq	sched_getcpu@GOTPCREL(%rip), %r13
	jmp	.LBB5_108
	.p2align	4
.LBB5_115:
	vmovq	%r14, %xmm0
	vpunpckldq	.LCPI5_12(%rip), %xmm0, %xmm0
	movq	640(%rsp), %rdx
	movq	64(%rsp), %rax
	movq	928(%rsp), %rcx
	shlq	$32, %r12
	orq	512(%rsp), %r12
	movl	%ebx, %ebp
	movl	$0, %r14d
	vsubpd	.LCPI5_13(%rip), %xmm0, %xmm0
	shlq	$32, %rdx
	orq	560(%rsp), %rdx
	movq	16(%rax,%rcx,8), %rbx
	subq	%r12, %rdx
	vmovapd	%xmm0, 448(%rsp)
	vcvtsi2sd	%r15d, %xmm15, %xmm0
	vdivsd	.LCPI5_14(%rip), %xmm0, %xmm0
	leaq	(%rax,%rcx,8), %r15
	cmovaeq	%rdx, %r14
	vmovsd	%xmm0, 512(%rsp)
	cmpq	(%rax,%rcx,8), %rbx
	jne	.LBB5_117
.Ltmp65:
	movq	%r15, %rdi
	callq	_ZN5alloc7raw_vec19RawVec$LT$T$C$A$GT$8grow_one17h53c42511abd71e68E
.Ltmp66:
.LBB5_117:
	vmovapd	448(%rsp), %xmm1
	movq	8(%r15), %rsi
	movq	416(%rsp), %rax
	leaq	(%rbx,%rbx,2), %rdx
	movl	%ebp, %edi
	notl	%edi
	movl	%r13d, %ecx
	notl	%r13d
	incq	%rbx
	shrl	$31, %edi
	shrl	$31, %r13d
	shlq	$4, %rdx
	movl	%edi, (%rsi,%rdx)
	movl	%ebp, 4(%rsi,%rdx)
	movl	%r13d, 8(%rsi,%rdx)
	movl	%ecx, 12(%rsi,%rdx)
	movq	%rax, 16(%rsi,%rdx)
	movq	672(%rsp), %rax
	movl	$100, %ebp
	movq	$0, 24(%rsi,%rdx)
	vshufpd	$1, %xmm1, %xmm1, %xmm0
	vaddsd	%xmm1, %xmm0, %xmm0
	vaddsd	512(%rsp), %xmm0, %xmm0
	leaq	(%rax,%rax,2), %rax
	movq	%rax, 336(%rsp)
	shll	$4, %eax
	addq	320(%rsp), %rax
	movq	%rax, 448(%rsp)
	vmovsd	%xmm0, 32(%rsi,%rdx)
	movq	%r14, 40(%rsi,%rdx)
	movq	%rbx, 16(%r15)
	.p2align	4
.LBB5_118:
	movq	sched_getcpu@GOTPCREL(%rip), %r13
	callq	*%r13
	movl	%eax, %ebx
	#APP

	lfence

	#NO_APP
	#APP

	rdtsc

	#NO_APP
	movq	%rdx, %r12
	movq	%rax, 512(%rsp)
.Ltmp67:
	callq	*_RNvMNtCsjrHSEGnQ3l9_3std4timeNtB2_7Instant3now@GOTPCREL(%rip)
.Ltmp68:
	movq	448(%rsp), %rcx
	movq	%rax, 112(%rsp)
	movl	%edx, 120(%rsp)
	movq	40(%rcx), %rax
	movq	32(%rcx), %rdi
.Ltmp69:
	callq	*40(%rax)
.Ltmp70:
	vmovsd	%xmm0, 352(%rsp)
	leaq	352(%rsp), %rax
	#APP
	#NO_APP
.Ltmp71:
	leaq	112(%rsp), %rdi
	callq	*_RNvMNtCsjrHSEGnQ3l9_3std4timeNtB2_7Instant7elapsed@GOTPCREL(%rip)
.Ltmp72:
	movq	%rax, %r14
	movl	%edx, %r15d
	#APP

	rdtscp

	#NO_APP
	movq	%rdx, 640(%rsp)
	movq	%rax, 560(%rsp)
	#APP

	lfence

	#NO_APP
	callq	*%r13
	testl	%eax, %eax
	movl	%eax, %r13d
	setns	%al
	cmpl	%r13d, %ebx
	sete	%cl
	testb	%cl, %al
	jne	.LBB5_124
	testl	%ebx, %ebx
	js	.LBB5_124
	movq	480(%rsp), %rax
	movq	672(%rsp), %rcx
	incq	(%rax,%rcx,8)
	subq	$1, %rbp
	jae	.LBB5_118
	jmp	.LBB5_130
	.p2align	4
.LBB5_124:
	vmovq	%r14, %xmm0
	vpunpckldq	.LCPI5_12(%rip), %xmm0, %xmm0
	movq	640(%rsp), %rdx
	movq	64(%rsp), %rax
	movq	336(%rsp), %rcx
	shlq	$32, %r12
	orq	512(%rsp), %r12
	movl	%ebx, %ebp
	movl	$0, %r14d
	vsubpd	.LCPI5_13(%rip), %xmm0, %xmm0
	shlq	$32, %rdx
	orq	560(%rsp), %rdx
	movq	16(%rax,%rcx,8), %rbx
	subq	%r12, %rdx
	vmovapd	%xmm0, 448(%rsp)
	vcvtsi2sd	%r15d, %xmm15, %xmm0
	vdivsd	.LCPI5_14(%rip), %xmm0, %xmm0
	leaq	(%rax,%rcx,8), %r15
	cmovaeq	%rdx, %r14
	vmovsd	%xmm0, 512(%rsp)
	cmpq	(%rax,%rcx,8), %rbx
	jne	.LBB5_106
.Ltmp77:
	movq	%r15, %rdi
	callq	_ZN5alloc7raw_vec19RawVec$LT$T$C$A$GT$8grow_one17h53c42511abd71e68E
.Ltmp78:
	jmp	.LBB5_106
.LBB5_126:
	movq	64(%rsp), %rax
	leaq	904(%rsp), %r15
	leaq	880(%rsp), %r12
	leaq	856(%rsp), %r13
	leaq	832(%rsp), %rbp
	movl	$35, %ecx
	movq	8(%rax), %rdx
	movq	16(%rax), %rax
	shlq	$4, %rax
	leaq	(%rax,%rax,2), %rsi
	leaq	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.44(%rip), %rax
	.p2align	4
.LBB5_127:
	testq	%rsi, %rsi
	je	.LBB5_131
	cmpl	$0, (%rdx)
	je	.LBB5_135
	addq	$-48, %rsi
	testb	$1, 8(%rdx)
	leaq	48(%rdx), %rdx
	jne	.LBB5_127
	jmp	.LBB5_135
.LBB5_130:
.Ltmp74:
	leaq	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.37(%rip), %rdi
	leaq	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.38(%rip), %rdx
	movl	$59, %esi
	callq	*_RNvNtCsgEmfK2I1SDS_4core9panicking9panic_fmt@GOTPCREL(%rip)
.Ltmp75:
	jmp	.LBB5_402
.LBB5_131:
	movq	64(%rsp), %rcx
	leaq	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.45(%rip), %rdx
	movq	32(%rcx), %rsi
	movq	40(%rcx), %rcx
	shlq	$4, %rcx
	leaq	(%rcx,%rcx,2), %rdi
	movl	$35, %ecx
	.p2align	4
.LBB5_132:
	testq	%rdi, %rdi
	je	.LBB5_136
	cmpl	$0, (%rsi)
	je	.LBB5_135
	addq	$-48, %rdi
	testb	$1, 8(%rsi)
	leaq	48(%rsi), %rsi
	jne	.LBB5_132
.LBB5_135:
	movq	%rax, %rdx
	jmp	.LBB5_137
.LBB5_136:
	movl	$9, %ecx
.LBB5_137:
	movq	%rdx, 1064(%rsp)
	movq	%rcx, 1072(%rsp)
	callq	*_RNvCsfLfy6EI15iL_7___rustc35___rust_no_alloc_shim_is_unstable_v2@GOTPCREL(%rip)
	movl	$45, %edi
	movl	$1, %esi
	movl	$45, %ebx
	callq	*_RNvCsfLfy6EI15iL_7___rustc12___rust_alloc@GOTPCREL(%rip)
	testq	%rax, %rax
	je	.LBB5_400
	vmovupd	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.46+13(%rip), %ymm0
	vmovupd	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.46(%rip), %ymm1
	leaq	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.47(%rip), %rdx
	movq	$0, 112(%rsp)
	movq	$8, 120(%rsp)
	leaq	736(%rsp), %rcx
	movq	$45, 8(%rsp)
	movq	%rax, 16(%rsp)
	movq	%rax, %r14
	movq	$45, 24(%rsp)
	movq	%rdx, 128(%rsp)
	leaq	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.48(%rip), %rdx
	movq	$9, 136(%rsp)
	movq	%rcx, 144(%rsp)
	leaq	760(%rsp), %rcx
	movq	%rdx, 152(%rsp)
	leaq	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.49(%rip), %rdx
	movq	$6, 160(%rsp)
	movq	%rcx, 168(%rsp)
	leaq	784(%rsp), %rcx
	movq	%rdx, 176(%rsp)
	leaq	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.50(%rip), %rdx
	movq	$13, 184(%rsp)
	movq	%rcx, 192(%rsp)
	leaq	808(%rsp), %rcx
	movq	%rdx, 200(%rsp)
	leaq	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.51(%rip), %rdx
	movq	$5, 208(%rsp)
	movq	%rcx, 216(%rsp)
	leaq	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.52(%rip), %rcx
	movq	%rdx, 224(%rsp)
	leaq	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.53(%rip), %rdx
	movq	$8, 232(%rsp)
	movq	%rbp, 240(%rsp)
	movq	%rcx, 248(%rsp)
	movq	$15, 256(%rsp)
	movq	%r13, 264(%rsp)
	leaq	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.54(%rip), %rcx
	movq	%rdx, 272(%rsp)
	movq	$9, 280(%rsp)
	movq	%r12, 288(%rsp)
	movl	$32, %r12d
	movq	%rcx, 296(%rsp)
	movq	$21, 304(%rsp)
	movq	%r15, 312(%rsp)
	vmovupd	%ymm0, 13(%rax)
	vmovupd	%ymm1, (%rax)
	jmp	.LBB5_140
	.p2align	4
.LBB5_139:
	addq	$24, %r12
	cmpq	$224, %r12
	je	.LBB5_149
.LBB5_140:
	movq	96(%rsp,%r12), %rax
	testq	%rax, %rax
	je	.LBB5_149
	movq	112(%rsp,%r12), %rdx
	movq	104(%rsp,%r12), %rcx
	movq	%rax, 704(%rsp)
	leaq	704(%rsp), %rax
	movq	%rax, 352(%rsp)
	leaq	_ZN44_$LT$$RF$T$u20$as$u20$core..fmt..Display$GT$3fmt17h63df3ed385e2537eE(%rip), %rax
	movq	%rax, 360(%rsp)
	leaq	440(%rsp), %rax
	movq	%rax, 368(%rsp)
	leaq	_ZN44_$LT$$RF$T$u20$as$u20$core..fmt..Display$GT$3fmt17h4aae8497b870fbd6E(%rip), %rax
	movq	%rax, 376(%rsp)
	movq	%rcx, 712(%rsp)
	movq	%rdx, 440(%rsp)
.Ltmp80:
	leaq	80(%rsp), %rdi
	leaq	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.64(%rip), %rsi
	leaq	352(%rsp), %rdx
	vzeroupper
	callq	*_RNvNvNtCslNYArtu3iFV_5alloc3fmt6format12format_inner@GOTPCREL(%rip)
.Ltmp81:
	movq	8(%rsp), %rax
	movq	80(%rsp), %r15
	movq	88(%rsp), %r13
	movq	96(%rsp), %rbp
	subq	%rbx, %rax
	cmpq	%rax, %rbp
	ja	.LBB5_147
	testq	%rbp, %rbp
	je	.LBB5_145
.LBB5_144:
	leaq	(%r14,%rbx), %rdi
	movq	%r13, %rsi
	movq	%rbp, %rdx
	callq	*memcpy@GOTPCREL(%rip)
.LBB5_145:
	addq	%rbp, %rbx
	movq	%rbx, 24(%rsp)
	testq	%r15, %r15
	je	.LBB5_139
	movl	$1, %edx
	movq	%r13, %rdi
	movq	%r15, %rsi
	callq	*_RNvCsfLfy6EI15iL_7___rustc14___rust_dealloc@GOTPCREL(%rip)
	jmp	.LBB5_139
.LBB5_147:
.Ltmp83:
	movl	$1, %ecx
	movl	$1, %r8d
	leaq	8(%rsp), %rdi
	movq	%rbx, %rsi
	movq	%rbp, %rdx
	callq	_ZN5alloc7raw_vec20RawVecInner$LT$A$GT$7reserve21do_reserve_and_handle17h5eeef80eb9b0c257E
.Ltmp84:
	movq	16(%rsp), %r14
	movq	24(%rsp), %rbx
	jmp	.LBB5_144
.LBB5_149:
	movq	_RNvXsi_NtNtNtCsgEmfK2I1SDS_4core3fmt3num3impjNtB9_7Display3fmt@GOTPCREL(%rip), %rcx
	leaq	1064(%rsp), %rax
	leaq	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.57(%rip), %r8
	movq	%rax, 112(%rsp)
	leaq	_ZN44_$LT$$RF$T$u20$as$u20$core..fmt..Display$GT$3fmt17h63df3ed385e2537eE(%rip), %rax
	movq	%rax, 120(%rsp)
	leaq	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.55(%rip), %rax
	movq	%rax, 128(%rsp)
	leaq	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.56(%rip), %rax
	movq	%rcx, 136(%rsp)
	movq	%rax, 144(%rsp)
	movq	%rcx, 152(%rsp)
	movq	%r8, 160(%rsp)
	movq	%rcx, 168(%rsp)
.Ltmp86:
	leaq	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.58(%rip), %rsi
	leaq	352(%rsp), %rdi
	leaq	112(%rsp), %rdx
	vzeroupper
	callq	*_RNvNvNtCslNYArtu3iFV_5alloc3fmt6format12format_inner@GOTPCREL(%rip)
.Ltmp87:
	movq	8(%rsp), %rax
	movq	352(%rsp), %r14
	movq	360(%rsp), %r15
	movq	368(%rsp), %r13
	subq	%rbx, %rax
	cmpq	%rax, %r13
	ja	.LBB5_376
	testq	%r13, %r13
	je	.LBB5_153
.LBB5_152:
	movq	16(%rsp), %rdi
	movq	%r15, %rsi
	movq	%r13, %rdx
	addq	%rbx, %rdi
	callq	*memcpy@GOTPCREL(%rip)
.LBB5_153:
	addq	%r13, %rbx
	movq	%rbx, 24(%rsp)
	testq	%r14, %r14
	je	.LBB5_155
	movl	$1, %edx
	movq	%r15, %rdi
	movq	%r14, %rsi
	callq	*_RNvCsfLfy6EI15iL_7___rustc14___rust_dealloc@GOTPCREL(%rip)
.LBB5_155:
	movq	8(%rsp), %rax
	subq	%rbx, %rax
	cmpq	$164, %rax
	jbe	.LBB5_378
.LBB5_156:
	vmovups	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.59+96(%rip), %ymm2
	vmovups	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.59+64(%rip), %ymm1
	movq	16(%rsp), %rax
	vmovupd	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.59+128(%rip), %ymm0
	movq	320(%rsp), %r14
	movabsq	$755030680729384033, %rcx
	xorl	%edx, %edx
	movq	$0, 448(%rsp)
	vmovups	%ymm2, 96(%rax,%rbx)
	vmovups	%ymm1, 64(%rax,%rbx)
	vmovupd	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.59+32(%rip), %ymm2
	vmovupd	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.59(%rip), %ymm1
	vmovupd	%ymm0, 128(%rax,%rbx)
	movq	%rcx, 157(%rax,%rbx)
	vmovupd	%ymm2, 32(%rax,%rbx)
	vmovupd	%ymm1, (%rax,%rbx)
	addq	$165, %rbx
	movq	%rbx, 24(%rsp)
	jmp	.LBB5_158
	.p2align	4
.LBB5_157:
	movq	416(%rsp), %rdx
	cmpq	$96, 448(%rsp)
	movq	320(%rsp), %r14
	je	.LBB5_293
.LBB5_158:
	movq	64(%rsp), %rcx
	leaq	(%rdx,%rdx,2), %rax
	movq	16(%rcx,%rax,8), %r15
	testq	%r15, %r15
	je	.LBB5_362
	movq	8(%rcx,%rax,8), %rbp
	movq	%rdx, 512(%rsp)
	leaq	(,%r15,8), %rbx
	vzeroupper
	callq	*_RNvCsfLfy6EI15iL_7___rustc35___rust_no_alloc_shim_is_unstable_v2@GOTPCREL(%rip)
	movl	$8, %esi
	movq	%rbx, %rdi
	movl	$8, %r12d
	callq	*_RNvCsfLfy6EI15iL_7___rustc12___rust_alloc@GOTPCREL(%rip)
	testq	%rax, %rax
	je	.LBB5_374
	movq	%rax, %r13
	cmpq	$5, %r15
	jae	.LBB5_162
	xorl	%eax, %eax
	jmp	.LBB5_169
	.p2align	4
.LBB5_162:
	cmpq	$17, %r15
	jae	.LBB5_164
	xorl	%eax, %eax
	jmp	.LBB5_167
	.p2align	4
.LBB5_164:
	movl	%r15d, %ecx
	andl	$15, %ecx
	movl	$16, %eax
	leaq	752(%rbp), %rdx
	cmoveq	%rax, %rcx
	movq	%r15, %rax
	xorl	%esi, %esi
	subq	%rcx, %rax
	.p2align	4
.LBB5_165:
	vmovsd	-720(%rdx), %xmm0
	vmovsd	-624(%rdx), %xmm1
	vmovsd	-528(%rdx), %xmm2
	vmovsd	-432(%rdx), %xmm3
	vmovsd	-336(%rdx), %xmm4
	vmovsd	-240(%rdx), %xmm5
	vmovsd	-144(%rdx), %xmm6
	vmovsd	-48(%rdx), %xmm7
	vmovhpd	-672(%rdx), %xmm0, %xmm0
	vmovhpd	-576(%rdx), %xmm1, %xmm1
	vmovhpd	-480(%rdx), %xmm2, %xmm2
	vmovhps	-384(%rdx), %xmm3, %xmm3
	vmovhps	-288(%rdx), %xmm4, %xmm4
	vmovhps	-192(%rdx), %xmm5, %xmm5
	vmovhpd	-96(%rdx), %xmm6, %xmm6
	vmovhpd	(%rdx), %xmm7, %xmm7
	addq	$768, %rdx
	vmovupd	%xmm1, 16(%r13,%rsi,8)
	vmovupd	%xmm0, (%r13,%rsi,8)
	vmovups	%xmm3, 48(%r13,%rsi,8)
	vmovupd	%xmm2, 32(%r13,%rsi,8)
	vmovups	%xmm5, 80(%r13,%rsi,8)
	vmovups	%xmm4, 64(%r13,%rsi,8)
	vmovupd	%xmm7, 112(%r13,%rsi,8)
	vmovupd	%xmm6, 96(%r13,%rsi,8)
	addq	$16, %rsi
	cmpq	%rsi, %rax
	jne	.LBB5_165
	cmpl	$5, %ecx
	jb	.LBB5_169
.LBB5_167:
	movl	%r15d, %edx
	movq	%rax, %rcx
	andl	$3, %edx
	movl	$4, %eax
	cmoveq	%rax, %rdx
	movq	%r15, %rax
	subq	%rdx, %rax
	leaq	(%rcx,%rcx,2), %rdx
	shlq	$4, %rdx
	leaq	176(%rbp,%rdx), %rdx
	.p2align	4
.LBB5_168:
	vmovsd	-144(%rdx), %xmm0
	vmovsd	-48(%rdx), %xmm1
	vmovhpd	-96(%rdx), %xmm0, %xmm0
	vmovhpd	(%rdx), %xmm1, %xmm1
	addq	$192, %rdx
	vmovupd	%xmm1, 16(%r13,%rcx,8)
	vmovupd	%xmm0, (%r13,%rcx,8)
	addq	$4, %rcx
	cmpq	%rcx, %rax
	jne	.LBB5_168
.LBB5_169:
	movq	448(%rsp), %rcx
	leaq	32(%rbp), %r12
	leaq	(%r14,%rcx), %rdx
	addq	$48, %rcx
	movq	%rcx, 448(%rsp)
	movq	512(%rsp), %rcx
	movq	%rdx, 640(%rsp)
	incq	%rcx
	movq	%rcx, 416(%rsp)
	leaq	(%rax,%rax,2), %rcx
	shlq	$4, %rcx
	leaq	32(%rbp,%rcx), %rcx
	.p2align	4
.LBB5_170:
	vmovsd	(%rcx), %xmm0
	addq	$48, %rcx
	vmovsd	%xmm0, (%r13,%rax,8)
	incq	%rax
	cmpq	%rax, %r15
	jne	.LBB5_170
	cmpq	$1, %r15
	jne	.LBB5_206
.LBB5_172:
	movl	%r15d, %eax
	leaq	-1(%r15), %rsi
	andl	$7, %eax
	cmpq	$7, %rsi
	jae	.LBB5_174
	vmovsd	.LCPI5_9(%rip), %xmm0
	xorl	%ecx, %ecx
	jmp	.LBB5_177
	.p2align	4
.LBB5_174:
	vmovsd	.LCPI5_9(%rip), %xmm0
	movq	%r15, %rdx
	andq	$-8, %rdx
	xorl	%ecx, %ecx
	.p2align	4
.LBB5_175:
	vaddsd	(%r13,%rcx,8), %xmm0, %xmm0
	vaddsd	8(%r13,%rcx,8), %xmm0, %xmm0
	vaddsd	16(%r13,%rcx,8), %xmm0, %xmm0
	vaddsd	24(%r13,%rcx,8), %xmm0, %xmm0
	vaddsd	32(%r13,%rcx,8), %xmm0, %xmm0
	vaddsd	40(%r13,%rcx,8), %xmm0, %xmm0
	vaddsd	48(%r13,%rcx,8), %xmm0, %xmm0
	vaddsd	56(%r13,%rcx,8), %xmm0, %xmm0
	addq	$8, %rcx
	cmpq	%rcx, %rdx
	jne	.LBB5_175
	testq	%rax, %rax
	je	.LBB5_179
.LBB5_177:
	leaq	(%r13,%rcx,8), %rcx
	xorl	%edx, %edx
	.p2align	4
.LBB5_178:
	vaddsd	(%rcx,%rdx,8), %xmm0, %xmm0
	incq	%rdx
	cmpq	%rdx, %rax
	jne	.LBB5_178
.LBB5_179:
	vcvtsi2sd	%r15, %xmm15, %xmm1
	vdivsd	%xmm1, %xmm0, %xmm2
	vmovsd	%xmm1, 560(%rsp)
	cmpq	$7, %rsi
	jae	.LBB5_181
	vmovsd	.LCPI5_9(%rip), %xmm3
	xorl	%ecx, %ecx
	jmp	.LBB5_184
	.p2align	4
.LBB5_181:
	vmovsd	.LCPI5_9(%rip), %xmm3
	movq	%r15, %rdx
	andq	$-8, %rdx
	xorl	%ecx, %ecx
	.p2align	4
.LBB5_182:
	vmovsd	(%r13,%rcx,8), %xmm0
	vmovsd	8(%r13,%rcx,8), %xmm1
	vsubsd	%xmm2, %xmm0, %xmm0
	vsubsd	%xmm2, %xmm1, %xmm1
	vmulsd	%xmm0, %xmm0, %xmm0
	vmulsd	%xmm1, %xmm1, %xmm1
	vaddsd	%xmm0, %xmm3, %xmm0
	vaddsd	%xmm1, %xmm0, %xmm0
	vmovsd	16(%r13,%rcx,8), %xmm1
	vsubsd	%xmm2, %xmm1, %xmm1
	vmulsd	%xmm1, %xmm1, %xmm1
	vaddsd	%xmm1, %xmm0, %xmm0
	vmovsd	24(%r13,%rcx,8), %xmm1
	vsubsd	%xmm2, %xmm1, %xmm1
	vmulsd	%xmm1, %xmm1, %xmm1
	vaddsd	%xmm1, %xmm0, %xmm0
	vmovsd	32(%r13,%rcx,8), %xmm1
	vsubsd	%xmm2, %xmm1, %xmm1
	vmulsd	%xmm1, %xmm1, %xmm1
	vaddsd	%xmm1, %xmm0, %xmm0
	vmovsd	40(%r13,%rcx,8), %xmm1
	vsubsd	%xmm2, %xmm1, %xmm1
	vmulsd	%xmm1, %xmm1, %xmm1
	vaddsd	%xmm1, %xmm0, %xmm0
	vmovsd	48(%r13,%rcx,8), %xmm1
	vsubsd	%xmm2, %xmm1, %xmm1
	vmulsd	%xmm1, %xmm1, %xmm1
	vaddsd	%xmm1, %xmm0, %xmm0
	vmovsd	56(%r13,%rcx,8), %xmm1
	addq	$8, %rcx
	vsubsd	%xmm2, %xmm1, %xmm1
	vmulsd	%xmm1, %xmm1, %xmm1
	vaddsd	%xmm1, %xmm0, %xmm3
	cmpq	%rcx, %rdx
	jne	.LBB5_182
	testb	$7, %r15b
	je	.LBB5_186
.LBB5_184:
	leaq	(%r13,%rcx,8), %rcx
	xorl	%edx, %edx
	.p2align	4
.LBB5_185:
	vmovsd	(%rcx,%rdx,8), %xmm0
	incq	%rdx
	vsubsd	%xmm2, %xmm0, %xmm0
	vmulsd	%xmm0, %xmm0, %xmm0
	vaddsd	%xmm0, %xmm3, %xmm3
	cmpq	%rdx, %rax
	jne	.LBB5_185
.LBB5_186:
	vmovsd	%xmm3, 336(%rsp)
	vmovsd	%xmm2, 672(%rsp)
	callq	*_RNvCsfLfy6EI15iL_7___rustc35___rust_no_alloc_shim_is_unstable_v2@GOTPCREL(%rip)
	movl	$8, %esi
	movq	%rbx, %rdi
	callq	*_RNvCsfLfy6EI15iL_7___rustc12___rust_alloc@GOTPCREL(%rip)
	testq	%rax, %rax
	je	.LBB5_386
	movq	%rax, %r14
	cmpq	$4, %r15
	ja	.LBB5_189
	xorl	%eax, %eax
	jmp	.LBB5_191
	.p2align	4
.LBB5_189:
	vbroadcastsd	.LCPI5_17(%rip), %ymm6
	vbroadcastsd	.LCPI5_14(%rip), %ymm7
	movl	%r15d, %ecx
	andl	$3, %ecx
	movl	$4, %eax
	vxorps	%xmm5, %xmm5, %xmm5
	cmoveq	%rax, %rcx
	movq	%r15, %rax
	subq	%rcx, %rax
	xorl	%ecx, %ecx
	.p2align	4
.LBB5_190:
	vmovups	128(%r12), %ymm0
	vbroadcastsd	104(%r12), %ymm4
	vmovups	(%r12), %xmm2
	vmovups	48(%r12), %xmm3
	vpbroadcastq	.LCPI5_16(%rip), %ymm8
	vbroadcastsd	.LCPI5_15(%rip), %ymm9
	vinsertf128	$1, 96(%r12), %ymm0, %ymm1
	vunpckhpd	%ymm0, %ymm4, %ymm4
	addq	$192, %r12
	vunpcklpd	%ymm0, %ymm1, %ymm0
	vmovlhps	%xmm3, %xmm2, %xmm1
	vunpckhpd	%xmm3, %xmm2, %xmm2
	vblendps	$240, %ymm4, %ymm2, %ymm2
	vblendps	$240, %ymm0, %ymm1, %ymm0
	vblendps	$170, %ymm5, %ymm2, %ymm3
	vpsrlq	$32, %ymm2, %ymm2
	vpor	%ymm2, %ymm8, %ymm2
	vorps	%ymm3, %ymm9, %ymm3
	vsubpd	%ymm6, %ymm2, %ymm2
	vaddpd	%ymm2, %ymm3, %ymm2
	vdivpd	%ymm0, %ymm2, %ymm0
	vdivpd	%ymm7, %ymm0, %ymm0
	vmovupd	%ymm0, (%r14,%rcx,8)
	addq	$4, %rcx
	cmpq	%rcx, %rax
	jne	.LBB5_190
.LBB5_191:
	vmovsd	.LCPI5_12(%rip), %xmm2
	vmovapd	.LCPI5_13(%rip), %xmm3
	vmovsd	.LCPI5_14(%rip), %xmm4
	leaq	(%rax,%rax,2), %rcx
	shlq	$4, %rcx
	leaq	40(%rbp,%rcx), %rcx
	.p2align	4
.LBB5_192:
	vmovsd	(%rcx), %xmm0
	vunpcklps	%xmm2, %xmm0, %xmm0
	vsubpd	%xmm3, %xmm0, %xmm0
	vshufpd	$1, %xmm0, %xmm0, %xmm1
	vaddsd	%xmm0, %xmm1, %xmm0
	vdivsd	-8(%rcx), %xmm0, %xmm0
	addq	$48, %rcx
	vdivsd	%xmm4, %xmm0, %xmm0
	vmovsd	%xmm0, (%r14,%rax,8)
	incq	%rax
	cmpq	%rax, %r15
	jne	.LBB5_192
	cmpq	$1, %r15
	jne	.LBB5_237
	xorl	%eax, %eax
.LBB5_195:
	vmovsd	(%r13,%rax,8), %xmm0
	xorl	%ecx, %ecx
.LBB5_196:
	vmovsd	560(%rsp), %xmm6
	vmovsd	.LCPI5_20(%rip), %xmm3
	vmovsd	.LCPI5_21(%rip), %xmm5
	vxorpd	%xmm4, %xmm4, %xmm4
	movl	$0, %r8d
	movq	$-1, %r9
	leaq	-1(%r15), %r10
	vmulsd	.LCPI5_19(%rip), %xmm6, %xmm1
	vsubsd	%xmm3, %xmm1, %xmm2
	vcvttsd2si	%xmm1, %rsi
	movq	%rsi, %rdi
	sarq	$63, %rdi
	vcvttsd2si	%xmm2, %rdx
	vmulsd	.LCPI5_22(%rip), %xmm6, %xmm2
	andq	%rdx, %rdi
	orq	%rsi, %rdi
	vucomisd	%xmm4, %xmm1
	cmovbq	%r8, %rdi
	vucomisd	%xmm5, %xmm1
	vsubsd	%xmm3, %xmm2, %xmm3
	vcvttsd2si	%xmm2, %rsi
	cmovaq	%r9, %rdi
	cmpq	%rdi, %r10
	cmovbq	%r10, %rdi
	vcvttsd2si	%xmm3, %rdx
	vmovsd	(%r13), %xmm3
	vmovsd	(%r13,%rdi,8), %xmm1
	movq	%rsi, %rdi
	sarq	$63, %rdi
	andq	%rdx, %rdi
	orq	%rsi, %rdi
	vucomisd	%xmm4, %xmm2
	vmovsd	-8(%r13,%r15,8), %xmm4
	cmovbq	%r8, %rdi
	vucomisd	%xmm5, %xmm2
	cmovaq	%r9, %rdi
	cmpq	%rdi, %r10
	cmovbq	%r10, %rdi
	vmovsd	(%r13,%rdi,8), %xmm2
	testb	%cl, %cl
	je	.LBB5_199
	leaq	-1(%rax), %rdi
	cmpq	%r15, %rdi
	jae	.LBB5_387
	vmovsd	-8(%r14,%rax,8), %xmm5
	vaddsd	(%r14,%rax,8), %xmm5, %xmm5
	vmulsd	.LCPI5_18(%rip), %xmm5, %xmm5
	jmp	.LBB5_200
	.p2align	4
.LBB5_199:
	vmovsd	(%r14,%rax,8), %xmm5
.LBB5_200:
	vmovsd	336(%rsp), %xmm6
	vmovsd	%xmm0, 352(%rsp)
	movq	%r14, %rdi
	movq	_RNvCsfLfy6EI15iL_7___rustc14___rust_dealloc@GOTPCREL(%rip), %r14
	movl	$8, %edx
	movq	%rbx, %rsi
	vdivsd	560(%rsp), %xmm6, %xmm6
	vsqrtsd	%xmm6, %xmm6, %xmm6
	vmovsd	672(%rsp), %xmm0
	vmovsd	%xmm0, 360(%rsp)
	vmovsd	%xmm6, 368(%rsp)
	vmovsd	%xmm1, 376(%rsp)
	vmovsd	%xmm2, 384(%rsp)
	vmovsd	%xmm3, 392(%rsp)
	vmovsd	%xmm4, 400(%rsp)
	vmovsd	%xmm5, 408(%rsp)
	vzeroupper
	callq	*%r14
	movl	$8, %edx
	movq	%r13, %rdi
	movq	%rbx, %rsi
	callq	*%r14
	vmovsd	.LCPI5_23(%rip), %xmm1
	movq	640(%rsp), %rax
	movq	480(%rsp), %rcx
	movq	512(%rsp), %rdx
	movq	%r15, 1000(%rsp)
	vmulsd	360(%rsp), %xmm1, %xmm2
	vmulsd	368(%rsp), %xmm1, %xmm3
	vmulsd	352(%rsp), %xmm1, %xmm0
	movq	%rax, 112(%rsp)
	leaq	(%rcx,%rdx,8), %rcx
	leaq	_ZN44_$LT$$RF$T$u20$as$u20$core..fmt..Display$GT$3fmt17h63df3ed385e2537eE(%rip), %rdx
	leaq	16(%rax), %rax
	movq	%rdx, 120(%rsp)
	movq	%rax, 128(%rsp)
	leaq	1000(%rsp), %rax
	movq	%rdx, 136(%rsp)
	movq	%rax, 144(%rsp)
	movq	_RNvXsi_NtNtNtCsgEmfK2I1SDS_4core3fmt3num3impjNtB9_7Display3fmt@GOTPCREL(%rip), %rax
	vmovsd	%xmm2, 1016(%rsp)
	vmulsd	376(%rsp), %xmm1, %xmm2
	vmovsd	%xmm3, 1024(%rsp)
	vmulsd	384(%rsp), %xmm1, %xmm3
	vmovsd	%xmm0, 1008(%rsp)
	movq	%rax, 152(%rsp)
	movq	%rcx, 160(%rsp)
	movq	_RNvXs7_NtNtCsgEmfK2I1SDS_4core3fmt5floatdNtB7_7Display3fmt@GOTPCREL(%rip), %rcx
	movq	%rax, 168(%rsp)
	leaq	1008(%rsp), %rax
	movq	%rax, 176(%rsp)
	leaq	1016(%rsp), %rax
	movq	%rcx, 184(%rsp)
	movq	%rax, 192(%rsp)
	leaq	1024(%rsp), %rax
	movq	%rcx, 200(%rsp)
	movq	%rax, 208(%rsp)
	leaq	1032(%rsp), %rax
	movq	%rcx, 216(%rsp)
	movq	%rax, 224(%rsp)
	leaq	632(%rsp), %rax
	movq	%rcx, 232(%rsp)
	movq	%rax, 240(%rsp)
	leaq	440(%rsp), %rax
	movq	%rcx, 248(%rsp)
	vmovsd	%xmm2, 1032(%rsp)
	vmulsd	392(%rsp), %xmm1, %xmm2
	vmulsd	400(%rsp), %xmm1, %xmm1
	movq	%rax, 256(%rsp)
	leaq	704(%rsp), %rax
	movq	%rcx, 264(%rsp)
	vmovsd	%xmm3, 632(%rsp)
	movq	%rax, 272(%rsp)
	leaq	408(%rsp), %rax
	movq	%rcx, 280(%rsp)
	movq	%rax, 288(%rsp)
	movq	%rcx, 296(%rsp)
	vmovsd	%xmm2, 440(%rsp)
	vmovsd	%xmm1, 704(%rsp)
.Ltmp105:
	leaq	80(%rsp), %rdi
	leaq	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.63(%rip), %rsi
	leaq	112(%rsp), %rdx
	callq	*_RNvNvNtCslNYArtu3iFV_5alloc3fmt6format12format_inner@GOTPCREL(%rip)
.Ltmp106:
	movq	8(%rsp), %rax
	movq	24(%rsp), %rbx
	movq	80(%rsp), %r14
	movq	88(%rsp), %r15
	movq	96(%rsp), %r13
	subq	%rbx, %rax
	cmpq	%rax, %r13
	ja	.LBB5_244
	testq	%r13, %r13
	je	.LBB5_204
.LBB5_203:
	movq	16(%rsp), %rdi
	movq	%r15, %rsi
	movq	%r13, %rdx
	addq	%rbx, %rdi
	callq	*memcpy@GOTPCREL(%rip)
.LBB5_204:
	addq	%r13, %rbx
	movq	%rbx, 24(%rsp)
	testq	%r14, %r14
	je	.LBB5_157
	movl	$1, %edx
	movq	%r15, %rdi
	movq	%r14, %rsi
	callq	*_RNvCsfLfy6EI15iL_7___rustc14___rust_dealloc@GOTPCREL(%rip)
	jmp	.LBB5_157
.LBB5_206:
	cmpq	$21, %r15
	jae	.LBB5_291
	leaq	-16(%rbx), %r8
	leaq	8(%r13), %rcx
	movl	%r8d, %eax
	notl	%eax
	testb	$24, %al
	jne	.LBB5_246
	movq	%r13, %rax
.LBB5_209:
	cmpq	$24, %r8
	jb	.LBB5_172
	leaq	16(%rcx), %rdx
	leaq	8(%rcx), %rsi
	leaq	24(%rcx), %rdi
	movq	%rdx, 560(%rsp)
	jmp	.LBB5_214
.LBB5_211:
	movq	%r13, %r11
.LBB5_212:
	movq	%r8, (%r11)
.LBB5_213:
	addq	$32, 560(%rsp)
	leaq	24(%rcx), %rax
	addq	$32, %rcx
	addq	$32, %rsi
	addq	$32, %rdi
	leaq	(%r13,%rbx), %rdx
	cmpq	%rdx, %rcx
	je	.LBB5_172
.LBB5_214:
	movq	(%rcx), %r9
	movq	(%rax), %rdx
	movq	%r9, %r10
	sarq	$63, %r10
	movq	%rdx, %r8
	sarq	$63, %r8
	shrq	%r10
	shrq	%r8
	xorq	%r9, %r10
	xorq	%rdx, %r8
	cmpq	%r8, %r10
	jge	.LBB5_221
	movq	%rcx, %r8
	.p2align	4
.LBB5_216:
	movq	%rdx, (%r8)
	cmpq	%r13, %rax
	je	.LBB5_219
	movq	-8(%rax), %rdx
	leaq	-8(%rax), %r14
	movq	%rax, %r8
	movq	%r14, %rax
	movq	%rdx, %r11
	sarq	$63, %r11
	shrq	%r11
	xorq	%rdx, %r11
	cmpq	%r11, %r10
	jl	.LBB5_216
	addq	$8, %r14
	jmp	.LBB5_220
.LBB5_219:
	movq	%r13, %r14
.LBB5_220:
	movq	%r9, (%r14)
.LBB5_221:
	movq	(%rcx), %r10
	movq	8(%rcx), %r8
	movq	%r8, %r9
	sarq	$63, %r9
	movq	%r10, %rax
	sarq	$63, %rax
	shrq	%r9
	shrq	%rax
	xorq	%r8, %r9
	xorq	%r10, %rax
	cmpq	%rax, %r9
	jge	.LBB5_227
	movq	%rsi, %r14
	.p2align	4
.LBB5_223:
	leaq	-8(%r14), %r11
	movq	%r10, (%r14)
	cmpq	%r13, %r11
	je	.LBB5_225
	movq	-16(%r14), %r10
	movq	%r11, %r14
	movq	%r10, %rax
	sarq	$63, %rax
	shrq	%rax
	xorq	%r10, %rax
	cmpq	%rax, %r9
	jl	.LBB5_223
	jmp	.LBB5_226
.LBB5_225:
	movq	%r13, %r11
.LBB5_226:
	movq	%r8, (%r11)
.LBB5_227:
	movq	8(%rcx), %r10
	movq	16(%rcx), %r8
	movq	%r8, %r9
	sarq	$63, %r9
	movq	%r10, %rax
	sarq	$63, %rax
	shrq	%r9
	shrq	%rax
	xorq	%r8, %r9
	xorq	%r10, %rax
	cmpq	%rax, %r9
	jge	.LBB5_233
	movq	560(%rsp), %r14
	.p2align	4
.LBB5_229:
	leaq	-8(%r14), %r11
	movq	%r10, (%r14)
	cmpq	%r13, %r11
	je	.LBB5_231
	movq	-16(%r14), %r10
	movq	%r11, %r14
	movq	%r10, %rax
	sarq	$63, %rax
	shrq	%rax
	xorq	%r10, %rax
	cmpq	%rax, %r9
	jl	.LBB5_229
	jmp	.LBB5_232
.LBB5_231:
	movq	%r13, %r11
.LBB5_232:
	movq	%r8, (%r11)
.LBB5_233:
	movq	16(%rcx), %r10
	movq	24(%rcx), %r8
	movq	%r8, %r9
	sarq	$63, %r9
	movq	%r10, %rax
	sarq	$63, %rax
	shrq	%r9
	shrq	%rax
	xorq	%r8, %r9
	xorq	%r10, %rax
	cmpq	%rax, %r9
	jge	.LBB5_213
	movq	%rdi, %r14
	.p2align	4
.LBB5_235:
	leaq	-8(%r14), %r11
	movq	%r10, (%r14)
	cmpq	%r13, %r11
	je	.LBB5_211
	movq	-16(%r14), %r10
	movq	%r11, %r14
	movq	%r10, %rax
	sarq	$63, %rax
	shrq	%rax
	xorq	%r10, %rax
	cmpq	%rax, %r9
	jl	.LBB5_235
	jmp	.LBB5_212
.LBB5_237:
	cmpq	$21, %r15
	jae	.LBB5_292
	leaq	-16(%rbx), %rcx
	leaq	8(%r14), %rax
	movl	%ecx, %edx
	notl	%edx
	testb	$24, %dl
	jne	.LBB5_255
	movq	%r14, %r8
.LBB5_240:
	cmpq	$24, %rcx
	jae	.LBB5_264
.LBB5_241:
	movq	%r15, %rax
	shrq	%rax
	testb	$1, %r15b
	jne	.LBB5_195
	leaq	-1(%rax), %rdi
	cmpq	%r15, %rdi
	jae	.LBB5_401
	vmovsd	-8(%r13,%rax,8), %xmm0
	movb	$1, %cl
	vaddsd	(%r13,%rax,8), %xmm0, %xmm0
	vmulsd	.LCPI5_18(%rip), %xmm0, %xmm0
	jmp	.LBB5_196
.LBB5_244:
.Ltmp108:
	movl	$1, %ecx
	movl	$1, %r8d
	leaq	8(%rsp), %rdi
	movq	%rbx, %rsi
	movq	%r13, %rdx
	callq	_ZN5alloc7raw_vec20RawVecInner$LT$A$GT$7reserve21do_reserve_and_handle17h5eeef80eb9b0c257E
.Ltmp109:
	movq	24(%rsp), %rbx
	jmp	.LBB5_203
.LBB5_246:
	movl	%r8d, %edx
	shrl	$3, %edx
	movl	$8, %esi
	movq	%r13, %rax
	xorl	%edi, %edi
	incl	%edx
	andl	$3, %edx
	jmp	.LBB5_250
.LBB5_247:
	movq	%r13, %r11
.LBB5_248:
	movq	%r9, (%r11)
.LBB5_249:
	leaq	8(%rax), %rcx
	incq	%rdi
	addq	$8, %rsi
	cmpq	%rdx, %rdi
	je	.LBB5_209
.LBB5_250:
	movq	(%rax), %r10
	movq	8(%rax), %r9
	movq	%rcx, %rax
	movq	%r9, %rcx
	sarq	$63, %rcx
	movq	%r10, %r11
	sarq	$63, %r11
	shrq	%rcx
	shrq	%r11
	xorq	%r9, %rcx
	xorq	%r10, %r11
	cmpq	%r11, %rcx
	jge	.LBB5_249
	movq	%rsi, %r11
	.p2align	4
.LBB5_252:
	movq	%r10, (%r13,%r11)
	cmpq	$8, %r11
	je	.LBB5_247
	movq	-16(%r13,%r11), %r10
	addq	$-8, %r11
	movq	%r10, %r14
	sarq	$63, %r14
	shrq	%r14
	xorq	%r10, %r14
	cmpq	%r14, %rcx
	jl	.LBB5_252
	addq	%r13, %r11
	jmp	.LBB5_248
.LBB5_255:
	movl	%ecx, %edx
	shrl	$3, %edx
	movl	$8, %esi
	movq	%r14, %r8
	xorl	%edi, %edi
	incl	%edx
	andl	$3, %edx
	jmp	.LBB5_259
.LBB5_256:
	movq	%r14, %r11
.LBB5_257:
	movq	%r9, (%r11)
.LBB5_258:
	leaq	8(%r8), %rax
	incq	%rdi
	addq	$8, %rsi
	cmpq	%rdx, %rdi
	je	.LBB5_240
.LBB5_259:
	movq	(%r8), %r10
	movq	8(%r8), %r9
	movq	%rax, %r8
	movq	%r9, %rax
	sarq	$63, %rax
	movq	%r10, %r11
	sarq	$63, %r11
	shrq	%rax
	shrq	%r11
	xorq	%r9, %rax
	xorq	%r10, %r11
	cmpq	%r11, %rax
	jge	.LBB5_258
	movq	%rsi, %r11
	.p2align	4
.LBB5_261:
	movq	%r10, (%r14,%r11)
	cmpq	$8, %r11
	je	.LBB5_256
	movq	-16(%r14,%r11), %r10
	addq	$-8, %r11
	movq	%r10, %r12
	sarq	$63, %r12
	shrq	%r12
	xorq	%r10, %r12
	cmpq	%r12, %rax
	jl	.LBB5_261
	addq	%r14, %r11
	jmp	.LBB5_257
.LBB5_264:
	leaq	(%r14,%rbx), %rbp
	leaq	8(%rax), %rdx
	leaq	16(%rax), %rsi
	leaq	24(%rax), %rdi
	jmp	.LBB5_268
.LBB5_265:
	movq	%r14, %r11
.LBB5_266:
	movq	%r8, (%r11)
.LBB5_267:
	leaq	24(%rax), %r8
	addq	$32, %rax
	addq	$32, %rdx
	addq	$32, %rsi
	addq	$32, %rdi
	cmpq	%rbp, %rax
	je	.LBB5_241
.LBB5_268:
	movq	(%rax), %r9
	movq	(%r8), %rcx
	movq	%r9, %r10
	sarq	$63, %r10
	movq	%rcx, %r12
	sarq	$63, %r12
	shrq	%r10
	shrq	%r12
	xorq	%r9, %r10
	xorq	%rcx, %r12
	cmpq	%r12, %r10
	jge	.LBB5_275
	movq	%rax, %rbp
	.p2align	4
.LBB5_270:
	movq	%rcx, (%rbp)
	cmpq	%r14, %r8
	je	.LBB5_273
	movq	-8(%r8), %rcx
	leaq	-8(%r8), %r12
	movq	%r8, %rbp
	movq	%r12, %r8
	movq	%rcx, %r11
	sarq	$63, %r11
	shrq	%r11
	xorq	%rcx, %r11
	cmpq	%r11, %r10
	jl	.LBB5_270
	addq	$8, %r12
	jmp	.LBB5_274
.LBB5_273:
	movq	%r14, %r12
.LBB5_274:
	leaq	(%r14,%rbx), %rbp
	movq	%r9, (%r12)
.LBB5_275:
	movq	(%rax), %r10
	movq	8(%rax), %r8
	movq	%r8, %r9
	sarq	$63, %r9
	movq	%r10, %rcx
	sarq	$63, %rcx
	shrq	%r9
	shrq	%rcx
	xorq	%r8, %r9
	xorq	%r10, %rcx
	cmpq	%rcx, %r9
	jge	.LBB5_281
	movq	%rdx, %r12
	.p2align	4
.LBB5_277:
	leaq	-8(%r12), %r11
	movq	%r10, (%r12)
	cmpq	%r14, %r11
	je	.LBB5_279
	movq	-16(%r12), %r10
	movq	%r11, %r12
	movq	%r10, %rcx
	sarq	$63, %rcx
	shrq	%rcx
	xorq	%r10, %rcx
	cmpq	%rcx, %r9
	jl	.LBB5_277
	jmp	.LBB5_280
.LBB5_279:
	movq	%r14, %r11
.LBB5_280:
	movq	%r8, (%r11)
.LBB5_281:
	movq	8(%rax), %r10
	movq	16(%rax), %r8
	movq	%r8, %r9
	sarq	$63, %r9
	movq	%r10, %rcx
	sarq	$63, %rcx
	shrq	%r9
	shrq	%rcx
	xorq	%r8, %r9
	xorq	%r10, %rcx
	cmpq	%rcx, %r9
	jge	.LBB5_287
	movq	%rsi, %r12
	.p2align	4
.LBB5_283:
	leaq	-8(%r12), %r11
	movq	%r10, (%r12)
	cmpq	%r14, %r11
	je	.LBB5_285
	movq	-16(%r12), %r10
	movq	%r11, %r12
	movq	%r10, %rcx
	sarq	$63, %rcx
	shrq	%rcx
	xorq	%r10, %rcx
	cmpq	%rcx, %r9
	jl	.LBB5_283
	jmp	.LBB5_286
.LBB5_285:
	movq	%r14, %r11
.LBB5_286:
	movq	%r8, (%r11)
.LBB5_287:
	movq	16(%rax), %r10
	movq	24(%rax), %r8
	movq	%r8, %r9
	sarq	$63, %r9
	movq	%r10, %rcx
	sarq	$63, %rcx
	shrq	%r9
	shrq	%rcx
	xorq	%r8, %r9
	xorq	%r10, %rcx
	cmpq	%rcx, %r9
	jge	.LBB5_267
	movq	%rdi, %r12
	.p2align	4
.LBB5_289:
	leaq	-8(%r12), %r11
	movq	%r10, (%r12)
	cmpq	%r14, %r11
	je	.LBB5_265
	movq	-16(%r12), %r10
	movq	%r11, %r12
	movq	%r10, %rcx
	sarq	$63, %rcx
	shrq	%rcx
	xorq	%r10, %rcx
	cmpq	%rcx, %r9
	jl	.LBB5_289
	jmp	.LBB5_266
.LBB5_291:
.Ltmp93:
	movq	%r13, %rdi
	movq	%r15, %rsi
	callq	_ZN4core5slice4sort6stable14driftsort_main17h1a7478f9f2f3f94aE
.Ltmp94:
	jmp	.LBB5_172
.LBB5_292:
.Ltmp96:
	movq	%r14, %rdi
	movq	%r15, %rsi
	vzeroupper
	callq	_ZN4core5slice4sort6stable14driftsort_main17h1a7478f9f2f3f94aE
.Ltmp97:
	jmp	.LBB5_241
.LBB5_293:
	movq	8(%rsp), %rax
	subq	%rbx, %rax
	cmpq	$114, %rax
	jbe	.LBB5_380
.LBB5_294:
	vmovupd	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.60+83(%rip), %ymm0
	vmovups	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.60+64(%rip), %ymm1
	movq	16(%rsp), %rax
	vmovupd	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.60+32(%rip), %ymm2
	movq	$0, 672(%rsp)
	vmovupd	%ymm0, 83(%rax,%rbx)
	vmovups	%ymm1, 64(%rax,%rbx)
	vmovupd	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.60(%rip), %ymm1
	vmovupd	%ymm2, 32(%rax,%rbx)
	vmovupd	%ymm1, (%rax,%rbx)
	addq	$115, %rbx
	xorl	%eax, %eax
	movq	%rbx, 24(%rsp)
	jmp	.LBB5_296
	.p2align	4
.LBB5_295:
	movq	672(%rsp), %rdx
	movq	336(%rsp), %rax
	addq	$48, %rdx
	incq	%rax
	movq	%rdx, 672(%rsp)
	cmpq	$96, %rdx
	je	.LBB5_341
.LBB5_296:
	movq	64(%rsp), %rdx
	leaq	(%rax,%rax,2), %rcx
	movq	%rax, 336(%rsp)
	movq	16(%rdx,%rcx,8), %rax
	testq	%rax, %rax
	je	.LBB5_295
	movq	64(%rsp), %rdi
	movq	320(%rsp), %rdx
	movq	672(%rsp), %rsi
	shlq	$4, %rax
	leaq	(%rax,%rax,2), %r12
	movq	8(%rdi,%rcx,8), %r15
	addq	%rdx, %rsi
	leaq	16(%rsi), %rax
	movq	%rsi, 512(%rsp)
	movq	%rax, 560(%rsp)
	addq	$40, %r15
	jmp	.LBB5_299
	.p2align	4
.LBB5_298:
	addq	$48, %r15
	addq	$-48, %r12
	je	.LBB5_295
.LBB5_299:
	vmovsd	-8(%r15), %xmm0
	vmovsd	.LCPI5_14(%rip), %xmm3
	movq	%r12, 448(%rsp)
	vmulsd	%xmm3, %xmm0, %xmm1
	vmovsd	%xmm1, 632(%rsp)
	vmovsd	(%r15), %xmm1
	vunpcklps	.LCPI5_12(%rip), %xmm1, %xmm1
	vsubpd	.LCPI5_13(%rip), %xmm1, %xmm1
	vshufpd	$1, %xmm1, %xmm1, %xmm2
	vaddsd	%xmm1, %xmm2, %xmm1
	vdivsd	%xmm0, %xmm1, %xmm0
	vdivsd	%xmm3, %xmm0, %xmm0
	vmovsd	%xmm0, 440(%rsp)
	movl	-40(%r15), %r13d
	movl	-36(%r15), %r14d
	vzeroupper
	callq	*_RNvCsfLfy6EI15iL_7___rustc35___rust_no_alloc_shim_is_unstable_v2@GOTPCREL(%rip)
	movl	$11, %edi
	movl	$1, %esi
	movl	$11, %ebx
	movl	$1, %r12d
	callq	*_RNvCsfLfy6EI15iL_7___rustc12___rust_alloc@GOTPCREL(%rip)
	testq	%rax, %rax
	je	.LBB5_374
	movabsq	$7020101786482601589, %rcx
	movq	%rax, %rbp
	movq	%rcx, (%rbp)
	movl	$1701601889, 7(%rbp)
	testl	%r13d, %r13d
	je	.LBB5_304
	callq	*_RNvCsfLfy6EI15iL_7___rustc35___rust_no_alloc_shim_is_unstable_v2@GOTPCREL(%rip)
	testl	%r14d, %r14d
	js	.LBB5_305
	movl	$10, %edi
	movl	$1, %esi
	movl	$10, %r12d
	callq	*_RNvCsfLfy6EI15iL_7___rustc12___rust_alloc@GOTPCREL(%rip)
	testq	%rax, %rax
	je	.LBB5_384
	movl	$10, %r12d
	movq	$10, 112(%rsp)
	movq	%rax, 120(%rsp)
	movq	%rax, %rbx
	xorl	%r13d, %r13d
	movq	$0, 128(%rsp)
	jmp	.LBB5_307
	.p2align	4
.LBB5_304:
	movq	$11, 80(%rsp)
	movq	%rbp, 88(%rsp)
	movq	$11, 96(%rsp)
	jmp	.LBB5_312
.LBB5_305:
	movl	$11, %edi
	movl	$1, %esi
	movl	$11, %r12d
	callq	*_RNvCsfLfy6EI15iL_7___rustc12___rust_alloc@GOTPCREL(%rip)
	testq	%rax, %rax
	je	.LBB5_384
	movq	$11, 112(%rsp)
	movq	%rax, 120(%rsp)
	movq	%rax, %rbx
	movb	$45, (%rax)
	movq	$1, 128(%rsp)
	negl	%r14d
	movl	$1, %r13d
	movl	$11, %r12d
.LBB5_307:
.Ltmp113:
	movl	$10, %edx
	leaq	352(%rsp), %rsi
	movl	%r14d, %edi
	callq	*_RNvMsa_NtNtNtCsgEmfK2I1SDS_4core3fmt3num3impm4__fmt@GOTPCREL(%rip)
	movq	%rax, 640(%rsp)
.Ltmp114:
	subq	%r13, %r12
	movq	%rdx, %r14
	cmpq	%r12, %rdx
	ja	.LBB5_337
	testq	%r14, %r14
	je	.LBB5_311
.LBB5_310:
	movq	640(%rsp), %rsi
	addq	%r13, %rbx
	movq	%r14, %rdx
	movq	%rbx, %rdi
	callq	*memcpy@GOTPCREL(%rip)
.LBB5_311:
	vmovupd	112(%rsp), %xmm0
	addq	%r14, %r13
	movl	$11, %esi
	movl	$1, %edx
	movq	%rbp, %rdi
	movq	%r13, 96(%rsp)
	vmovapd	%xmm0, 80(%rsp)
	callq	*_RNvCsfLfy6EI15iL_7___rustc14___rust_dealloc@GOTPCREL(%rip)
.LBB5_312:
	movl	-32(%r15), %r14d
	movl	-28(%r15), %ebp
	callq	*_RNvCsfLfy6EI15iL_7___rustc35___rust_no_alloc_shim_is_unstable_v2@GOTPCREL(%rip)
	movl	$11, %edi
	movl	$1, %esi
	callq	*_RNvCsfLfy6EI15iL_7___rustc12___rust_alloc@GOTPCREL(%rip)
	testq	%rax, %rax
	je	.LBB5_375
	movabsq	$7020101786482601589, %rcx
	movq	%rax, %rbx
	movq	%rcx, (%rbx)
	movl	$1701601889, 7(%rbx)
	testl	%r14d, %r14d
	je	.LBB5_317
	callq	*_RNvCsfLfy6EI15iL_7___rustc35___rust_no_alloc_shim_is_unstable_v2@GOTPCREL(%rip)
	testl	%ebp, %ebp
	js	.LBB5_318
	movl	$10, %edi
	movl	$1, %esi
	movl	$10, %r12d
	callq	*_RNvCsfLfy6EI15iL_7___rustc12___rust_alloc@GOTPCREL(%rip)
	testq	%rax, %rax
	je	.LBB5_385
	movl	$10, %r12d
	movq	$10, 112(%rsp)
	movq	%rax, 120(%rsp)
	movq	%rax, %r14
	xorl	%r13d, %r13d
	movq	$0, 128(%rsp)
	jmp	.LBB5_320
	.p2align	4
.LBB5_317:
	movq	$11, 352(%rsp)
	movq	%rbx, 360(%rsp)
	movq	$11, 368(%rsp)
	jmp	.LBB5_325
.LBB5_318:
	movl	$11, %edi
	movl	$1, %esi
	movl	$11, %r12d
	callq	*_RNvCsfLfy6EI15iL_7___rustc12___rust_alloc@GOTPCREL(%rip)
	testq	%rax, %rax
	je	.LBB5_385
	movq	$11, 112(%rsp)
	movq	%rax, 120(%rsp)
	movq	%rax, %r14
	movb	$45, (%rax)
	movq	$1, 128(%rsp)
	negl	%ebp
	movl	$1, %r13d
	movl	$11, %r12d
.LBB5_320:
.Ltmp118:
	movl	$10, %edx
	leaq	704(%rsp), %rsi
	movl	%ebp, %edi
	callq	*_RNvMsa_NtNtNtCsgEmfK2I1SDS_4core3fmt3num3impm4__fmt@GOTPCREL(%rip)
	movq	%rax, 640(%rsp)
.Ltmp119:
	subq	%r13, %r12
	movq	%rdx, %rbp
	cmpq	%r12, %rdx
	ja	.LBB5_339
	testq	%rbp, %rbp
	je	.LBB5_324
.LBB5_323:
	movq	640(%rsp), %rsi
	addq	%r13, %r14
	movq	%rbp, %rdx
	movq	%r14, %rdi
	callq	*memcpy@GOTPCREL(%rip)
.LBB5_324:
	vmovupd	112(%rsp), %xmm0
	addq	%rbp, %r13
	movl	$11, %esi
	movl	$1, %edx
	movq	%rbx, %rdi
	movq	%r13, 368(%rsp)
	vmovapd	%xmm0, 352(%rsp)
	callq	*_RNvCsfLfy6EI15iL_7___rustc14___rust_dealloc@GOTPCREL(%rip)
.LBB5_325:
	movq	512(%rsp), %rdx
	movq	560(%rsp), %rsi
	movq	_RNvXsi_NtNtNtCsgEmfK2I1SDS_4core3fmt3num3impjNtB9_7Display3fmt@GOTPCREL(%rip), %r8
	leaq	-24(%r15), %rax
	leaq	-16(%r15), %rcx
	movq	_RNvXsd_NtNtNtCsgEmfK2I1SDS_4core3fmt3num3impyNtB9_7Display3fmt@GOTPCREL(%rip), %r9
	movq	%rdx, 112(%rsp)
	leaq	_ZN44_$LT$$RF$T$u20$as$u20$core..fmt..Display$GT$3fmt17h63df3ed385e2537eE(%rip), %rdx
	movq	%rdx, 120(%rsp)
	movq	%rsi, 128(%rsp)
	movq	%rdx, 136(%rsp)
	movq	%rax, 144(%rsp)
	movq	%r8, 152(%rsp)
	movq	%rcx, 160(%rsp)
	leaq	632(%rsp), %rcx
	movq	%r8, 168(%rsp)
	leaq	440(%rsp), %r8
	movq	%rcx, 176(%rsp)
	movq	_RNvXs7_NtNtCsgEmfK2I1SDS_4core3fmt5floatdNtB7_7Display3fmt@GOTPCREL(%rip), %rcx
	movq	%rcx, 184(%rsp)
	movq	%r15, 192(%rsp)
	movq	%r9, 200(%rsp)
	movq	%r8, 208(%rsp)
	movq	%rcx, 216(%rsp)
	leaq	80(%rsp), %rcx
	leaq	352(%rsp), %r8
	movq	%rcx, 224(%rsp)
	leaq	_RNvXsq_NtCslNYArtu3iFV_5alloc6stringNtB5_6StringNtNtCsgEmfK2I1SDS_4core3fmt7Display3fmt(%rip), %rcx
	movq	%rcx, 232(%rsp)
	movq	%r8, 240(%rsp)
	movq	%rcx, 248(%rsp)
.Ltmp123:
	leaq	704(%rsp), %rdi
	leaq	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.62(%rip), %rsi
	leaq	112(%rsp), %rdx
	callq	*_RNvNvNtCslNYArtu3iFV_5alloc3fmt6format12format_inner@GOTPCREL(%rip)
.Ltmp124:
	movq	352(%rsp), %rsi
	testq	%rsi, %rsi
	je	.LBB5_328
	movq	360(%rsp), %rdi
	movl	$1, %edx
	callq	*_RNvCsfLfy6EI15iL_7___rustc14___rust_dealloc@GOTPCREL(%rip)
.LBB5_328:
	movq	80(%rsp), %rsi
	testq	%rsi, %rsi
	je	.LBB5_330
	movq	88(%rsp), %rdi
	movl	$1, %edx
	callq	*_RNvCsfLfy6EI15iL_7___rustc14___rust_dealloc@GOTPCREL(%rip)
.LBB5_330:
	movq	8(%rsp), %rax
	movq	24(%rsp), %r12
	movq	704(%rsp), %rbx
	movq	712(%rsp), %r14
	movq	720(%rsp), %r13
	subq	%r12, %rax
	cmpq	%rax, %r13
	ja	.LBB5_335
	testq	%r13, %r13
	je	.LBB5_333
.LBB5_332:
	movq	16(%rsp), %rdi
	movq	%r14, %rsi
	movq	%r13, %rdx
	addq	%r12, %rdi
	callq	*memcpy@GOTPCREL(%rip)
.LBB5_333:
	addq	%r13, %r12
	movq	%r12, 24(%rsp)
	movq	448(%rsp), %r12
	testq	%rbx, %rbx
	je	.LBB5_298
	movl	$1, %edx
	movq	%r14, %rdi
	movq	%rbx, %rsi
	callq	*_RNvCsfLfy6EI15iL_7___rustc14___rust_dealloc@GOTPCREL(%rip)
	jmp	.LBB5_298
.LBB5_335:
.Ltmp126:
	movl	$1, %ecx
	movl	$1, %r8d
	leaq	8(%rsp), %rdi
	movq	%r12, %rsi
	movq	%r13, %rdx
	callq	_ZN5alloc7raw_vec20RawVecInner$LT$A$GT$7reserve21do_reserve_and_handle17h5eeef80eb9b0c257E
.Ltmp127:
	movq	24(%rsp), %r12
	jmp	.LBB5_332
.LBB5_337:
.Ltmp115:
	movl	$1, %ecx
	movl	$1, %r8d
	leaq	112(%rsp), %rdi
	movq	%r13, %rsi
	movq	%r14, %rdx
	callq	_ZN5alloc7raw_vec20RawVecInner$LT$A$GT$7reserve21do_reserve_and_handle17h5eeef80eb9b0c257E
.Ltmp116:
	movq	120(%rsp), %rbx
	movq	128(%rsp), %r13
	jmp	.LBB5_310
.LBB5_339:
.Ltmp120:
	movl	$1, %ecx
	movl	$1, %r8d
	leaq	112(%rsp), %rdi
	movq	%r13, %rsi
	movq	%rbp, %rdx
	callq	_ZN5alloc7raw_vec20RawVecInner$LT$A$GT$7reserve21do_reserve_and_handle17h5eeef80eb9b0c257E
.Ltmp121:
	movq	120(%rsp), %r14
	movq	128(%rsp), %r13
	jmp	.LBB5_323
.LBB5_341:
	vmovupd	8(%rsp), %xmm0
	movq	24(%rsp), %rax
	movq	%rax, 96(%rsp)
	leaq	80(%rsp), %rax
	movq	%rax, 112(%rsp)
	leaq	_RNvXsq_NtCslNYArtu3iFV_5alloc6stringNtB5_6StringNtNtCsgEmfK2I1SDS_4core3fmt7Display3fmt(%rip), %rax
	movq	%rax, 120(%rsp)
	vmovapd	%xmm0, 80(%rsp)
.Ltmp138:
	leaq	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.31(%rip), %rdi
	leaq	112(%rsp), %rsi
	vzeroupper
	callq	*_RNvNtNtCsjrHSEGnQ3l9_3std2io5stdio6__print@GOTPCREL(%rip)
.Ltmp139:
.Ltmp140:
	leaq	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.32(%rip), %rsi
	leaq	112(%rsp), %rdi
	movl	$17, %edx
	callq	*_RNvNtCsjrHSEGnQ3l9_3std3env4__var@GOTPCREL(%rip)
.Ltmp141:
	cmpl	$1, 112(%rsp)
	je	.LBB5_382
	movq	120(%rsp), %rbx
	movq	128(%rsp), %r14
	movq	80(%rsp), %r15
	movq	88(%rsp), %r13
	movq	136(%rsp), %rsi
	movq	96(%rsp), %rcx
.Ltmp143:
	movq	%r14, %rdi
	movq	%r13, %rdx
	callq	*_RNvNvNtCsjrHSEGnQ3l9_3std2fs5write5inner@GOTPCREL(%rip)
.Ltmp144:
	movq	%rax, %r12
	testq	%r15, %r15
	je	.LBB5_347
	movl	$1, %edx
	movq	%r13, %rdi
	movq	%r15, %rsi
	callq	*_RNvCsfLfy6EI15iL_7___rustc14___rust_dealloc@GOTPCREL(%rip)
.LBB5_347:
	testq	%rbx, %rbx
	je	.LBB5_349
	movl	$1, %edx
	movq	%r14, %rdi
	movq	%rbx, %rsi
	callq	*_RNvCsfLfy6EI15iL_7___rustc14___rust_dealloc@GOTPCREL(%rip)
.LBB5_349:
	testq	%r12, %r12
	jne	.LBB5_383
	movq	480(%rsp), %rdi
	movl	$16, %esi
	movl	$8, %edx
	callq	*_RNvCsfLfy6EI15iL_7___rustc14___rust_dealloc@GOTPCREL(%rip)
	movq	64(%rsp), %rbx
	movq	(%rbx), %rax
	testq	%rax, %rax
	je	.LBB5_352
	movq	8(%rbx), %rdi
	shlq	$4, %rax
	movl	$8, %edx
	leaq	(%rax,%rax,2), %rsi
	callq	*_RNvCsfLfy6EI15iL_7___rustc14___rust_dealloc@GOTPCREL(%rip)
.LBB5_352:
	movq	24(%rbx), %rax
	testq	%rax, %rax
	je	.LBB5_354
	movq	32(%rbx), %rdi
	shlq	$4, %rax
	movl	$8, %edx
	leaq	(%rax,%rax,2), %rsi
	callq	*_RNvCsfLfy6EI15iL_7___rustc14___rust_dealloc@GOTPCREL(%rip)
.LBB5_354:
	movl	$48, %esi
	movl	$8, %edx
	movq	%rbx, %rdi
	callq	*_RNvCsfLfy6EI15iL_7___rustc14___rust_dealloc@GOTPCREL(%rip)
.Ltmp152:
	leaq	1040(%rsp), %rdi
	callq	_ZN4core3ptr73drop_in_place$LT$alloc..vec..Vec$LT$protobuf_decode_bench..Target$GT$$GT$17h84925d455482b3f0E
.Ltmp153:
	movq	552(%rsp), %rsi
	testq	%rsi, %rsi
	je	.LBB5_357
	movq	32(%rsp), %rdi
	shlq	$4, %rsi
	movl	$8, %edx
	callq	*_RNvCsfLfy6EI15iL_7___rustc14___rust_dealloc@GOTPCREL(%rip)
.LBB5_357:
	movq	40(%rsp), %rsi
	testq	%rsi, %rsi
	je	.LBB5_359
	movq	48(%rsp), %rdi
	movl	$1, %edx
	callq	*_RNvCsfLfy6EI15iL_7___rustc14___rust_dealloc@GOTPCREL(%rip)
.LBB5_359:
	movq	600(%rsp), %rsi
	testq	%rsi, %rsi
	je	.LBB5_361
	movq	608(%rsp), %rdi
	shlq	$4, %rsi
	movl	$8, %edx
	callq	*_RNvCsfLfy6EI15iL_7___rustc14___rust_dealloc@GOTPCREL(%rip)
.LBB5_361:
	leaq	736(%rsp), %rdi
	callq	_ZN4core3ptr55drop_in_place$LT$protobuf_decode_bench..RunMetadata$GT$17hc16e98b5f36cd3b8E
	addq	$1080, %rsp
	.cfi_def_cfa_offset 56
	popq	%rbx
	.cfi_def_cfa_offset 48
	popq	%r12
	.cfi_def_cfa_offset 40
	popq	%r13
	.cfi_def_cfa_offset 32
	popq	%r14
	.cfi_def_cfa_offset 24
	popq	%r15
	.cfi_def_cfa_offset 16
	popq	%rbp
	.cfi_def_cfa_offset 8
	retq
.LBB5_362:
	.cfi_def_cfa_offset 1136
.Ltmp162:
	leaq	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.42(%rip), %rdi
	leaq	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.43(%rip), %rdx
	movl	$37, %esi
	vzeroupper
	callq	*_RNvNtCsgEmfK2I1SDS_4core9panicking5panic@GOTPCREL(%rip)
.Ltmp163:
	jmp	.LBB5_402
.LBB5_363:
	movq	%rax, 112(%rsp)
	leaq	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.20(%rip), %r8
	leaq	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.71(%rip), %rcx
	movl	$24, %esi
	leaq	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.19(%rip), %rdi
	movl	$57, %eax
.LBB5_364:
	movq	968(%rsp), %rdx
	movq	%rax, 8(%rdx)
.Ltmp196:
	vzeroupper
	callq	*_RNvNtCsgEmfK2I1SDS_4core6result13unwrap_failed@GOTPCREL(%rip)
.Ltmp197:
	jmp	.LBB5_402
.LBB5_365:
	movq	120(%rsp), %rax
	movq	%rax, 352(%rsp)
.Ltmp193:
	leaq	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.21(%rip), %rdi
	leaq	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.73(%rip), %rcx
	leaq	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.22(%rip), %r8
	leaq	352(%rsp), %rdx
	movl	$23, %esi
	callq	*_RNvNtCsgEmfK2I1SDS_4core6result13unwrap_failed@GOTPCREL(%rip)
.Ltmp194:
	jmp	.LBB5_402
.LBB5_366:
	leaq	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.24(%rip), %rdx
	leaq	984(%rsp), %rsi
	leaq	992(%rsp), %rdi
.LBB5_367:
.Ltmp190:
	callq	_ZN4core9panicking13assert_failed17h37cf44da67905b27E
.Ltmp191:
	jmp	.LBB5_402
.LBB5_368:
.Ltmp188:
	movl	$8, %edi
	movl	$96, %esi
	callq	*_RNvNtCslNYArtu3iFV_5alloc5alloc18handle_alloc_error@GOTPCREL(%rip)
.Ltmp189:
	jmp	.LBB5_402
.LBB5_369:
.Ltmp185:
	movl	$8, %edi
	movl	$8, %esi
	callq	*_RNvNtCslNYArtu3iFV_5alloc5alloc18handle_alloc_error@GOTPCREL(%rip)
.Ltmp186:
	jmp	.LBB5_402
.LBB5_370:
.Ltmp179:
	movl	$8, %edi
	movl	$8, %esi
	callq	*_RNvNtCslNYArtu3iFV_5alloc5alloc18handle_alloc_error@GOTPCREL(%rip)
.Ltmp180:
	jmp	.LBB5_402
.LBB5_371:
	movq	120(%rsp), %rax
	movq	%rax, 352(%rsp)
.Ltmp174:
	leaq	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.39(%rip), %rdi
	leaq	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.73(%rip), %rcx
	leaq	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.40(%rip), %r8
	leaq	352(%rsp), %rdx
	movl	$22, %esi
	callq	*_RNvNtCsgEmfK2I1SDS_4core6result13unwrap_failed@GOTPCREL(%rip)
.Ltmp175:
	jmp	.LBB5_402
.LBB5_372:
.Ltmp171:
	movl	$8, %edi
	movl	$48, %esi
	callq	*_RNvNtCslNYArtu3iFV_5alloc5alloc18handle_alloc_error@GOTPCREL(%rip)
.Ltmp172:
	jmp	.LBB5_402
.LBB5_373:
.Ltmp168:
	movl	$8, %edi
	movl	$16, %esi
	callq	*_RNvNtCslNYArtu3iFV_5alloc5alloc18handle_alloc_error@GOTPCREL(%rip)
.Ltmp169:
	jmp	.LBB5_402
.LBB5_374:
.Ltmp160:
	movq	%r12, %rdi
	movq	%rbx, %rsi
	callq	*_RNvNtCslNYArtu3iFV_5alloc7raw_vec12handle_error@GOTPCREL(%rip)
.Ltmp161:
	jmp	.LBB5_402
.LBB5_375:
.Ltmp132:
	movl	$1, %edi
	movl	$11, %esi
	callq	*_RNvNtCslNYArtu3iFV_5alloc7raw_vec12handle_error@GOTPCREL(%rip)
.Ltmp133:
	jmp	.LBB5_402
.LBB5_376:
.Ltmp88:
	leaq	8(%rsp), %rdi
	movl	$1, %ecx
	movl	$1, %r8d
	movq	%rbx, %rsi
	movq	%r13, %rdx
	callq	_ZN5alloc7raw_vec20RawVecInner$LT$A$GT$7reserve21do_reserve_and_handle17h5eeef80eb9b0c257E
.Ltmp89:
	movq	24(%rsp), %rbx
	jmp	.LBB5_152
.LBB5_378:
.Ltmp91:
	leaq	8(%rsp), %rdi
	movl	$165, %edx
	movl	$1, %ecx
	movl	$1, %r8d
	movq	%rbx, %rsi
	callq	_ZN5alloc7raw_vec20RawVecInner$LT$A$GT$7reserve21do_reserve_and_handle17h5eeef80eb9b0c257E
.Ltmp92:
	movq	24(%rsp), %rbx
	jmp	.LBB5_156
.LBB5_380:
.Ltmp111:
	leaq	8(%rsp), %rdi
	movl	$115, %edx
	movl	$1, %ecx
	movl	$1, %r8d
	movq	%rbx, %rsi
	callq	_ZN5alloc7raw_vec20RawVecInner$LT$A$GT$7reserve21do_reserve_and_handle17h5eeef80eb9b0c257E
.Ltmp112:
	movq	24(%rsp), %rbx
	jmp	.LBB5_294
.LBB5_382:
	vmovupd	120(%rsp), %xmm0
	movq	136(%rsp), %rax
	movq	%rax, 368(%rsp)
	vmovapd	%xmm0, 352(%rsp)
.Ltmp154:
	leaq	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.33(%rip), %rdi
	leaq	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.72(%rip), %rcx
	leaq	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.34(%rip), %r8
	leaq	352(%rsp), %rdx
	movl	$54, %esi
	callq	*_RNvNtCsgEmfK2I1SDS_4core6result13unwrap_failed@GOTPCREL(%rip)
.Ltmp155:
	jmp	.LBB5_402
.LBB5_383:
	movq	%r12, 112(%rsp)
.Ltmp146:
	leaq	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.35(%rip), %rdi
	leaq	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.74(%rip), %rcx
	leaq	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.36(%rip), %r8
	leaq	112(%rsp), %rdx
	movl	$26, %esi
	callq	*_RNvNtCsgEmfK2I1SDS_4core6result13unwrap_failed@GOTPCREL(%rip)
.Ltmp147:
	jmp	.LBB5_402
.LBB5_384:
.Ltmp135:
	movl	$1, %edi
	movq	%r12, %rsi
	callq	*_RNvNtCslNYArtu3iFV_5alloc7raw_vec12handle_error@GOTPCREL(%rip)
.Ltmp136:
	jmp	.LBB5_402
.LBB5_385:
.Ltmp129:
	movl	$1, %edi
	movq	%r12, %rsi
	callq	*_RNvNtCslNYArtu3iFV_5alloc7raw_vec12handle_error@GOTPCREL(%rip)
.Ltmp130:
	jmp	.LBB5_402
.LBB5_386:
.Ltmp157:
	movl	$8, %edi
	movq	%rbx, %rsi
	callq	*_RNvNtCslNYArtu3iFV_5alloc7raw_vec12handle_error@GOTPCREL(%rip)
.Ltmp158:
	jmp	.LBB5_402
.LBB5_387:
.Ltmp102:
	leaq	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.14(%rip), %rdx
	movq	%r15, %rsi
	vzeroupper
	callq	*_RNvNtCsgEmfK2I1SDS_4core9panicking18panic_bounds_check@GOTPCREL(%rip)
.Ltmp103:
	jmp	.LBB5_402
.LBB5_388:
	movq	480(%rsp), %rsi
	leaq	736(%rsp), %rdi
	movq	%r14, %rdx
	callq	_ZN21protobuf_decode_bench11RunMetadata8required5value28_$u7b$$u7b$closure$u7d$$u7d$17hd6282e5d0d99263aE
.LBB5_389:
.Ltmp220:
	movq	448(%rsp), %rsi
	leaq	736(%rsp), %rdi
	movq	%r13, %rdx
	callq	_ZN21protobuf_decode_bench11RunMetadata8required5value28_$u7b$$u7b$closure$u7d$$u7d$17hd6282e5d0d99263aE
.Ltmp221:
	jmp	.LBB5_402
.LBB5_390:
.Ltmp217:
	movq	512(%rsp), %rsi
	movq	336(%rsp), %rdx
	leaq	736(%rsp), %rdi
	callq	_ZN21protobuf_decode_bench11RunMetadata8required5value28_$u7b$$u7b$closure$u7d$$u7d$17hd6282e5d0d99263aE
.Ltmp218:
	jmp	.LBB5_402
.LBB5_391:
.Ltmp214:
	movq	560(%rsp), %rsi
	leaq	736(%rsp), %rdi
	movq	%rbx, %rdx
	callq	_ZN21protobuf_decode_bench11RunMetadata8required5value28_$u7b$$u7b$closure$u7d$$u7d$17hd6282e5d0d99263aE
.Ltmp215:
	jmp	.LBB5_402
.LBB5_392:
.Ltmp211:
	movq	64(%rsp), %rsi
	movq	928(%rsp), %rdx
	leaq	736(%rsp), %rdi
	callq	_ZN21protobuf_decode_bench11RunMetadata8required5value28_$u7b$$u7b$closure$u7d$$u7d$17hd6282e5d0d99263aE
.Ltmp212:
	jmp	.LBB5_402
.LBB5_393:
.Ltmp208:
	movq	416(%rsp), %rsi
	movq	32(%rsp), %rdx
	leaq	736(%rsp), %rdi
	callq	_ZN21protobuf_decode_bench11RunMetadata8required5value28_$u7b$$u7b$closure$u7d$$u7d$17hd6282e5d0d99263aE
.Ltmp209:
	jmp	.LBB5_402
.LBB5_394:
.Ltmp205:
	leaq	736(%rsp), %rdi
	movq	%r14, %rsi
	movq	%rbp, %rdx
	callq	_ZN21protobuf_decode_bench11RunMetadata8required5value28_$u7b$$u7b$closure$u7d$$u7d$17hd6282e5d0d99263aE
.Ltmp206:
	jmp	.LBB5_402
.LBB5_395:
	movq	128(%rsp), %rdx
	movq	120(%rsp), %rsi
.Ltmp202:
	leaq	352(%rsp), %rdi
	callq	_ZN21protobuf_decode_bench11RunMetadata8required5value28_$u7b$$u7b$closure$u7d$$u7d$17hd6282e5d0d99263aE
.Ltmp203:
	jmp	.LBB5_402
.LBB5_396:
.Ltmp199:
	movl	$8, %edi
	movl	$16000000, %esi
	callq	*_RNvNtCslNYArtu3iFV_5alloc7raw_vec12handle_error@GOTPCREL(%rip)
.Ltmp200:
	jmp	.LBB5_402
.LBB5_397:
	movabsq	$-9223372036854775808, %rax
	movq	%rcx, 112(%rsp)
	leaq	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.18(%rip), %r8
	leaq	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.70(%rip), %rcx
	movl	$22, %esi
	leaq	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.17(%rip), %rdi
	decq	%rax
	jmp	.LBB5_364
.LBB5_398:
	movq	%r15, 624(%rsp)
.LBB5_399:
	leaq	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.23(%rip), %rdx
	leaq	728(%rsp), %rsi
	leaq	624(%rsp), %rdi
	jmp	.LBB5_367
.LBB5_400:
.Ltmp165:
	movl	$1, %edi
	movl	$45, %esi
	callq	*_RNvNtCslNYArtu3iFV_5alloc7raw_vec12handle_error@GOTPCREL(%rip)
.Ltmp166:
	jmp	.LBB5_402
.LBB5_401:
.Ltmp99:
	leaq	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.14(%rip), %rdx
	movq	%r15, %rsi
	vzeroupper
	callq	*_RNvNtCsgEmfK2I1SDS_4core9panicking18panic_bounds_check@GOTPCREL(%rip)
.Ltmp100:
.LBB5_402:
	ud2
.LBB5_403:
.Ltmp90:
	movq	%rax, %rbx
	testq	%r14, %r14
	je	.LBB5_404
.LBB5_413:
	movl	$1, %edx
	movq	%r15, %rdi
	movq	%r14, %rsi
	callq	*_RNvCsfLfy6EI15iL_7___rustc14___rust_dealloc@GOTPCREL(%rip)
	movq	%rbx, %r15
	jmp	.LBB5_490
.LBB5_405:
.Ltmp98:
	jmp	.LBB5_446
.LBB5_406:
.Ltmp95:
	jmp	.LBB5_450
.LBB5_407:
.Ltmp85:
	movq	%rax, %r14
	testq	%r15, %r15
	jne	.LBB5_409
	movq	%r14, %r15
	jmp	.LBB5_490
.LBB5_409:
	movl	$1, %edx
	movq	%r13, %rdi
	movq	%r15, %rsi
	callq	*_RNvCsfLfy6EI15iL_7___rustc14___rust_dealloc@GOTPCREL(%rip)
	movq	%r14, %r15
	jmp	.LBB5_490
.LBB5_410:
.Ltmp101:
	jmp	.LBB5_446
.LBB5_411:
.Ltmp110:
	movq	%rax, %rbx
	testq	%r14, %r14
	jne	.LBB5_413
.LBB5_404:
	movq	%rbx, %r15
	jmp	.LBB5_490
.LBB5_414:
.Ltmp128:
	movq	%rax, %r15
	testq	%rbx, %rbx
	je	.LBB5_490
	movl	$1, %edx
	movq	%r14, %rdi
	jmp	.LBB5_452
.LBB5_416:
.Ltmp145:
	movq	%rax, %r12
	testq	%r15, %r15
	je	.LBB5_418
	movl	$1, %edx
	movq	%r13, %rdi
	movq	%r15, %rsi
	callq	*_RNvCsfLfy6EI15iL_7___rustc14___rust_dealloc@GOTPCREL(%rip)
.LBB5_418:
	testq	%rbx, %rbx
	jne	.LBB5_420
	movq	%r12, %r15
	jmp	.LBB5_495
.LBB5_420:
	movl	$1, %edx
	movq	%r14, %rdi
	movq	%rbx, %rsi
	callq	*_RNvCsfLfy6EI15iL_7___rustc14___rust_dealloc@GOTPCREL(%rip)
	movq	%r12, %r15
	jmp	.LBB5_495
.LBB5_421:
.Ltmp167:
	jmp	.LBB5_494
.LBB5_422:
.Ltmp142:
	movq	%rax, %r15
	jmp	.LBB5_468
.LBB5_423:
.Ltmp201:
	movq	%rax, %r15
	jmp	.LBB5_507
.LBB5_424:
.Ltmp204:
	movq	%rax, %r15
	testq	%r14, %r14
	je	.LBB5_427
	movl	$1, %edx
	movq	%rbp, %rdi
	movq	%r14, %rsi
	callq	*_RNvCsfLfy6EI15iL_7___rustc14___rust_dealloc@GOTPCREL(%rip)
	jmp	.LBB5_427
.LBB5_426:
.Ltmp207:
	movq	%rax, %r15
.LBB5_427:
	cmpq	$0, 416(%rsp)
	je	.LBB5_430
	movq	32(%rsp), %rdi
	movq	416(%rsp), %rsi
	movl	$1, %edx
	callq	*_RNvCsfLfy6EI15iL_7___rustc14___rust_dealloc@GOTPCREL(%rip)
	jmp	.LBB5_430
.LBB5_429:
.Ltmp210:
	movq	%rax, %r15
.LBB5_430:
	cmpq	$0, 64(%rsp)
	je	.LBB5_433
	movq	928(%rsp), %rdi
	movq	64(%rsp), %rsi
	movl	$1, %edx
	callq	*_RNvCsfLfy6EI15iL_7___rustc14___rust_dealloc@GOTPCREL(%rip)
	jmp	.LBB5_433
.LBB5_432:
.Ltmp213:
	movq	%rax, %r15
.LBB5_433:
	cmpq	$0, 560(%rsp)
	je	.LBB5_436
	movq	320(%rsp), %rdi
	movq	560(%rsp), %rsi
	movl	$1, %edx
	callq	*_RNvCsfLfy6EI15iL_7___rustc14___rust_dealloc@GOTPCREL(%rip)
	jmp	.LBB5_436
.LBB5_435:
.Ltmp216:
	movq	%rax, %r15
.LBB5_436:
	cmpq	$0, 512(%rsp)
	je	.LBB5_439
	movq	336(%rsp), %rdi
	movq	512(%rsp), %rsi
	movl	$1, %edx
	callq	*_RNvCsfLfy6EI15iL_7___rustc14___rust_dealloc@GOTPCREL(%rip)
	jmp	.LBB5_439
.LBB5_438:
.Ltmp219:
	movq	%rax, %r15
.LBB5_439:
	cmpq	$0, 448(%rsp)
	je	.LBB5_442
	movq	672(%rsp), %rdi
	movq	448(%rsp), %rsi
	movl	$1, %edx
	callq	*_RNvCsfLfy6EI15iL_7___rustc14___rust_dealloc@GOTPCREL(%rip)
	jmp	.LBB5_442
.LBB5_441:
.Ltmp222:
	movq	%rax, %r15
.LBB5_442:
	cmpq	$0, 480(%rsp)
	je	.LBB5_508
	movq	640(%rsp), %rdi
	movq	480(%rsp), %rsi
	movl	$1, %edx
	callq	*_RNvCsfLfy6EI15iL_7___rustc14___rust_dealloc@GOTPCREL(%rip)
	movq	%r15, %rdi
	callq	_Unwind_Resume@PLT
.LBB5_444:
.Ltmp82:
	jmp	.LBB5_489
.LBB5_445:
.Ltmp104:
.LBB5_446:
	movl	$8, %edx
	movq	%rax, %r15
	movq	%r14, %rdi
	movq	%rbx, %rsi
	callq	*_RNvCsfLfy6EI15iL_7___rustc14___rust_dealloc@GOTPCREL(%rip)
	jmp	.LBB5_451
.LBB5_447:
.Ltmp57:
	movq	%rax, %r15
	jmp	.LBB5_500
.LBB5_448:
.Ltmp107:
	jmp	.LBB5_489
.LBB5_449:
.Ltmp159:
.LBB5_450:
	movq	%rax, %r15
.LBB5_451:
	movl	$8, %edx
	movq	%r13, %rdi
.LBB5_452:
	movq	%rbx, %rsi
	callq	*_RNvCsfLfy6EI15iL_7___rustc14___rust_dealloc@GOTPCREL(%rip)
	jmp	.LBB5_490
.LBB5_453:
.Ltmp19:
	movq	%rax, %r15
	jmp	.LBB5_505
.LBB5_454:
.Ltmp79:
	jmp	.LBB5_494
.LBB5_455:
.Ltmp32:
	jmp	.LBB5_484
.LBB5_456:
.Ltmp131:
	movq	%rax, %r15
	jmp	.LBB5_463
.LBB5_457:
.Ltmp137:
	movq	%rax, %r15
	jmp	.LBB5_460
.LBB5_458:
.Ltmp117:
	movq	112(%rsp), %rsi
	movq	%rax, %r15
	testq	%rsi, %rsi
	je	.LBB5_460
	movq	120(%rsp), %rdi
	movl	$1, %edx
	callq	*_RNvCsfLfy6EI15iL_7___rustc14___rust_dealloc@GOTPCREL(%rip)
.LBB5_460:
	movl	$11, %esi
	movl	$1, %edx
	movq	%rbp, %rdi
	callq	*_RNvCsfLfy6EI15iL_7___rustc14___rust_dealloc@GOTPCREL(%rip)
	jmp	.LBB5_490
.LBB5_461:
.Ltmp122:
	movq	112(%rsp), %rsi
	movq	%rax, %r15
	testq	%rsi, %rsi
	je	.LBB5_463
	movq	120(%rsp), %rdi
	movl	$1, %edx
	callq	*_RNvCsfLfy6EI15iL_7___rustc14___rust_dealloc@GOTPCREL(%rip)
.LBB5_463:
	movl	$11, %esi
	movl	$1, %edx
	movq	%rbx, %rdi
	jmp	.LBB5_473
.LBB5_464:
.Ltmp148:
	movq	%rax, %r15
.Ltmp149:
	leaq	112(%rsp), %rdi
	callq	_ZN4core3ptr42drop_in_place$LT$std..io..error..Error$GT$17hcedecc219889b047E
.Ltmp150:
	jmp	.LBB5_495
.LBB5_465:
.Ltmp151:
	callq	*_RNvNtCsgEmfK2I1SDS_4core9panicking16panic_in_cleanup@GOTPCREL(%rip)
.LBB5_466:
.Ltmp156:
	movq	352(%rsp), %rsi
	movq	%rax, %r15
	testq	%rsi, %rsi
	jle	.LBB5_468
	movq	360(%rsp), %rdi
	movl	$1, %edx
	callq	*_RNvCsfLfy6EI15iL_7___rustc14___rust_dealloc@GOTPCREL(%rip)
.LBB5_468:
	movq	80(%rsp), %rsi
	testq	%rsi, %rsi
	je	.LBB5_495
	movq	88(%rsp), %rdi
	jmp	.LBB5_492
.LBB5_470:
.Ltmp134:
	movq	%rax, %r15
	jmp	.LBB5_474
.LBB5_471:
.Ltmp125:
	movq	352(%rsp), %rsi
	movq	%rax, %r15
	testq	%rsi, %rsi
	je	.LBB5_474
	movq	360(%rsp), %rdi
	movl	$1, %edx
.LBB5_473:
	callq	*_RNvCsfLfy6EI15iL_7___rustc14___rust_dealloc@GOTPCREL(%rip)
.LBB5_474:
	movq	80(%rsp), %rsi
	testq	%rsi, %rsi
	je	.LBB5_490
	movq	88(%rsp), %rdi
	movl	$1, %edx
	callq	*_RNvCsfLfy6EI15iL_7___rustc14___rust_dealloc@GOTPCREL(%rip)
	jmp	.LBB5_490
.LBB5_476:
.Ltmp170:
	movq	64(%rsp), %rbx
	movq	%rax, %r15
	jmp	.LBB5_497
.LBB5_477:
.Ltmp173:
	movq	%rax, %r15
	jmp	.LBB5_500
.LBB5_478:
.Ltmp176:
	leaq	352(%rsp), %rdi
	movq	%rax, %r15
	callq	_ZN4core3ptr46drop_in_place$LT$prost..error..DecodeError$GT$17h6452743001005f2fE
	jmp	.LBB5_500
.LBB5_479:
.Ltmp181:
	movq	%rax, %r15
.Ltmp182:
	leaq	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.25(%rip), %rsi
	movq	%rbx, %rdi
	callq	_ZN4core3ptr50drop_in_place$LT$protobuf_decode_bench..Target$GT$17hc849af708376b6caE
.Ltmp183:
	movq	320(%rsp), %rdi
	jmp	.LBB5_481
.LBB5_480:
.Ltmp187:
	movq	320(%rsp), %rdi
	movq	%rax, %r15
.LBB5_481:
	movl	$96, %esi
	movl	$8, %edx
	callq	*_RNvCsfLfy6EI15iL_7___rustc14___rust_dealloc@GOTPCREL(%rip)
	jmp	.LBB5_501
.LBB5_482:
.Ltmp195:
	leaq	352(%rsp), %rdi
	movq	%rax, %r15
	callq	_ZN4core3ptr46drop_in_place$LT$prost..error..DecodeError$GT$17h6452743001005f2fE
	jmp	.LBB5_503
.LBB5_483:
.Ltmp198:
.LBB5_484:
	movq	%rax, %r15
	jmp	.LBB5_503
.LBB5_485:
.Ltmp192:
	movq	%rax, %r15
	jmp	.LBB5_501
.LBB5_486:
.Ltmp73:
	jmp	.LBB5_494
.LBB5_487:
.Ltmp64:
	jmp	.LBB5_494
.LBB5_488:
.Ltmp164:
.LBB5_489:
	movq	%rax, %r15
.LBB5_490:
	movq	8(%rsp), %rsi
	testq	%rsi, %rsi
	je	.LBB5_495
	movq	16(%rsp), %rdi
.LBB5_492:
	movl	$1, %edx
	callq	*_RNvCsfLfy6EI15iL_7___rustc14___rust_dealloc@GOTPCREL(%rip)
	jmp	.LBB5_495
.LBB5_493:
.Ltmp76:
.LBB5_494:
	movq	%rax, %r15
.LBB5_495:
	movq	480(%rsp), %rdi
	movl	$16, %esi
	movl	$8, %edx
	callq	*_RNvCsfLfy6EI15iL_7___rustc14___rust_dealloc@GOTPCREL(%rip)
	movq	64(%rsp), %rbx
	movq	(%rbx), %rax
	testq	%rax, %rax
	je	.LBB5_497
	movq	8(%rbx), %rdi
	shlq	$4, %rax
	movl	$8, %edx
	leaq	(%rax,%rax,2), %rsi
	callq	*_RNvCsfLfy6EI15iL_7___rustc14___rust_dealloc@GOTPCREL(%rip)
.LBB5_497:
	movq	24(%rbx), %rax
	testq	%rax, %rax
	je	.LBB5_499
	movq	32(%rbx), %rdi
	shlq	$4, %rax
	movl	$8, %edx
	leaq	(%rax,%rax,2), %rsi
	callq	*_RNvCsfLfy6EI15iL_7___rustc14___rust_dealloc@GOTPCREL(%rip)
.LBB5_499:
	movl	$48, %esi
	movl	$8, %edx
	movq	%rbx, %rdi
	callq	*_RNvCsfLfy6EI15iL_7___rustc14___rust_dealloc@GOTPCREL(%rip)
.LBB5_500:
.Ltmp177:
	leaq	1040(%rsp), %rdi
	callq	_ZN4core3ptr73drop_in_place$LT$alloc..vec..Vec$LT$protobuf_decode_bench..Target$GT$$GT$17h84925d455482b3f0E
.Ltmp178:
.LBB5_501:
	movq	552(%rsp), %rsi
	testq	%rsi, %rsi
	je	.LBB5_503
	movq	32(%rsp), %rdi
	shlq	$4, %rsi
	movl	$8, %edx
	callq	*_RNvCsfLfy6EI15iL_7___rustc14___rust_dealloc@GOTPCREL(%rip)
.LBB5_503:
	movq	40(%rsp), %rsi
	testq	%rsi, %rsi
	je	.LBB5_505
	movq	48(%rsp), %rdi
	movl	$1, %edx
	callq	*_RNvCsfLfy6EI15iL_7___rustc14___rust_dealloc@GOTPCREL(%rip)
.LBB5_505:
	movq	600(%rsp), %rsi
	testq	%rsi, %rsi
	je	.LBB5_507
	movq	608(%rsp), %rdi
	shlq	$4, %rsi
	movl	$8, %edx
	callq	*_RNvCsfLfy6EI15iL_7___rustc14___rust_dealloc@GOTPCREL(%rip)
.LBB5_507:
	leaq	736(%rsp), %rdi
	callq	_ZN4core3ptr55drop_in_place$LT$protobuf_decode_bench..RunMetadata$GT$17hc16e98b5f36cd3b8E
.LBB5_508:
	movq	%r15, %rdi
	callq	_Unwind_Resume@PLT
.LBB5_509:
.Ltmp184:
	callq	*_RNvNtCsgEmfK2I1SDS_4core9panicking16panic_in_cleanup@GOTPCREL(%rip)
.Lfunc_end5:
	.size	_ZN21protobuf_decode_bench4main17ha1ffaea49e936ebfE, .Lfunc_end5-_ZN21protobuf_decode_bench4main17ha1ffaea49e936ebfE
	.cfi_endproc
	.section	.gcc_except_table._ZN21protobuf_decode_bench4main17ha1ffaea49e936ebfE,"a",@progbits
	.p2align	2, 0x0
GCC_except_table5:
.Lexception1:
	.byte	255
	.byte	155
	.uleb128 .Lttbase0-.Lttbaseref0
.Lttbaseref0:
	.byte	1
	.uleb128 .Lcst_end1-.Lcst_begin1
.Lcst_begin1:
	.uleb128 .Lfunc_begin1-.Lfunc_begin1
	.uleb128 .Ltmp3-.Lfunc_begin1
	.byte	0
	.byte	0
	.uleb128 .Ltmp3-.Lfunc_begin1
	.uleb128 .Ltmp4-.Ltmp3
	.uleb128 .Ltmp222-.Lfunc_begin1
	.byte	0
	.uleb128 .Ltmp5-.Lfunc_begin1
	.uleb128 .Ltmp6-.Ltmp5
	.uleb128 .Ltmp219-.Lfunc_begin1
	.byte	0
	.uleb128 .Ltmp7-.Lfunc_begin1
	.uleb128 .Ltmp8-.Ltmp7
	.uleb128 .Ltmp216-.Lfunc_begin1
	.byte	0
	.uleb128 .Ltmp9-.Lfunc_begin1
	.uleb128 .Ltmp10-.Ltmp9
	.uleb128 .Ltmp213-.Lfunc_begin1
	.byte	0
	.uleb128 .Ltmp11-.Lfunc_begin1
	.uleb128 .Ltmp12-.Ltmp11
	.uleb128 .Ltmp210-.Lfunc_begin1
	.byte	0
	.uleb128 .Ltmp13-.Lfunc_begin1
	.uleb128 .Ltmp14-.Ltmp13
	.uleb128 .Ltmp207-.Lfunc_begin1
	.byte	0
	.uleb128 .Ltmp15-.Lfunc_begin1
	.uleb128 .Ltmp16-.Ltmp15
	.uleb128 .Ltmp204-.Lfunc_begin1
	.byte	0
	.uleb128 .Ltmp17-.Lfunc_begin1
	.uleb128 .Ltmp18-.Ltmp17
	.uleb128 .Ltmp19-.Lfunc_begin1
	.byte	0
	.uleb128 .Ltmp22-.Lfunc_begin1
	.uleb128 .Ltmp29-.Ltmp22
	.uleb128 .Ltmp32-.Lfunc_begin1
	.byte	0
	.uleb128 .Ltmp33-.Lfunc_begin1
	.uleb128 .Ltmp36-.Ltmp33
	.uleb128 .Ltmp198-.Lfunc_begin1
	.byte	0
	.uleb128 .Ltmp37-.Lfunc_begin1
	.uleb128 .Ltmp56-.Ltmp37
	.uleb128 .Ltmp57-.Lfunc_begin1
	.byte	0
	.uleb128 .Ltmp56-.Lfunc_begin1
	.uleb128 .Ltmp58-.Ltmp56
	.byte	0
	.byte	0
	.uleb128 .Ltmp58-.Lfunc_begin1
	.uleb128 .Ltmp63-.Ltmp58
	.uleb128 .Ltmp64-.Lfunc_begin1
	.byte	0
	.uleb128 .Ltmp63-.Lfunc_begin1
	.uleb128 .Ltmp65-.Ltmp63
	.byte	0
	.byte	0
	.uleb128 .Ltmp65-.Lfunc_begin1
	.uleb128 .Ltmp66-.Ltmp65
	.uleb128 .Ltmp79-.Lfunc_begin1
	.byte	0
	.uleb128 .Ltmp66-.Lfunc_begin1
	.uleb128 .Ltmp67-.Ltmp66
	.byte	0
	.byte	0
	.uleb128 .Ltmp67-.Lfunc_begin1
	.uleb128 .Ltmp72-.Ltmp67
	.uleb128 .Ltmp73-.Lfunc_begin1
	.byte	0
	.uleb128 .Ltmp72-.Lfunc_begin1
	.uleb128 .Ltmp77-.Ltmp72
	.byte	0
	.byte	0
	.uleb128 .Ltmp77-.Lfunc_begin1
	.uleb128 .Ltmp78-.Ltmp77
	.uleb128 .Ltmp79-.Lfunc_begin1
	.byte	0
	.uleb128 .Ltmp74-.Lfunc_begin1
	.uleb128 .Ltmp75-.Ltmp74
	.uleb128 .Ltmp76-.Lfunc_begin1
	.byte	0
	.uleb128 .Ltmp80-.Lfunc_begin1
	.uleb128 .Ltmp81-.Ltmp80
	.uleb128 .Ltmp82-.Lfunc_begin1
	.byte	0
	.uleb128 .Ltmp81-.Lfunc_begin1
	.uleb128 .Ltmp83-.Ltmp81
	.byte	0
	.byte	0
	.uleb128 .Ltmp83-.Lfunc_begin1
	.uleb128 .Ltmp84-.Ltmp83
	.uleb128 .Ltmp85-.Lfunc_begin1
	.byte	0
	.uleb128 .Ltmp86-.Lfunc_begin1
	.uleb128 .Ltmp87-.Ltmp86
	.uleb128 .Ltmp164-.Lfunc_begin1
	.byte	0
	.uleb128 .Ltmp87-.Lfunc_begin1
	.uleb128 .Ltmp105-.Ltmp87
	.byte	0
	.byte	0
	.uleb128 .Ltmp105-.Lfunc_begin1
	.uleb128 .Ltmp106-.Ltmp105
	.uleb128 .Ltmp107-.Lfunc_begin1
	.byte	0
	.uleb128 .Ltmp106-.Lfunc_begin1
	.uleb128 .Ltmp108-.Ltmp106
	.byte	0
	.byte	0
	.uleb128 .Ltmp108-.Lfunc_begin1
	.uleb128 .Ltmp109-.Ltmp108
	.uleb128 .Ltmp110-.Lfunc_begin1
	.byte	0
	.uleb128 .Ltmp93-.Lfunc_begin1
	.uleb128 .Ltmp94-.Ltmp93
	.uleb128 .Ltmp95-.Lfunc_begin1
	.byte	0
	.uleb128 .Ltmp96-.Lfunc_begin1
	.uleb128 .Ltmp97-.Ltmp96
	.uleb128 .Ltmp98-.Lfunc_begin1
	.byte	0
	.uleb128 .Ltmp113-.Lfunc_begin1
	.uleb128 .Ltmp114-.Ltmp113
	.uleb128 .Ltmp117-.Lfunc_begin1
	.byte	0
	.uleb128 .Ltmp114-.Lfunc_begin1
	.uleb128 .Ltmp118-.Ltmp114
	.byte	0
	.byte	0
	.uleb128 .Ltmp118-.Lfunc_begin1
	.uleb128 .Ltmp119-.Ltmp118
	.uleb128 .Ltmp122-.Lfunc_begin1
	.byte	0
	.uleb128 .Ltmp119-.Lfunc_begin1
	.uleb128 .Ltmp123-.Ltmp119
	.byte	0
	.byte	0
	.uleb128 .Ltmp123-.Lfunc_begin1
	.uleb128 .Ltmp124-.Ltmp123
	.uleb128 .Ltmp125-.Lfunc_begin1
	.byte	0
	.uleb128 .Ltmp124-.Lfunc_begin1
	.uleb128 .Ltmp126-.Ltmp124
	.byte	0
	.byte	0
	.uleb128 .Ltmp126-.Lfunc_begin1
	.uleb128 .Ltmp127-.Ltmp126
	.uleb128 .Ltmp128-.Lfunc_begin1
	.byte	0
	.uleb128 .Ltmp115-.Lfunc_begin1
	.uleb128 .Ltmp116-.Ltmp115
	.uleb128 .Ltmp117-.Lfunc_begin1
	.byte	0
	.uleb128 .Ltmp120-.Lfunc_begin1
	.uleb128 .Ltmp121-.Ltmp120
	.uleb128 .Ltmp122-.Lfunc_begin1
	.byte	0
	.uleb128 .Ltmp138-.Lfunc_begin1
	.uleb128 .Ltmp141-.Ltmp138
	.uleb128 .Ltmp142-.Lfunc_begin1
	.byte	0
	.uleb128 .Ltmp143-.Lfunc_begin1
	.uleb128 .Ltmp144-.Ltmp143
	.uleb128 .Ltmp145-.Lfunc_begin1
	.byte	0
	.uleb128 .Ltmp152-.Lfunc_begin1
	.uleb128 .Ltmp153-.Ltmp152
	.uleb128 .Ltmp192-.Lfunc_begin1
	.byte	0
	.uleb128 .Ltmp162-.Lfunc_begin1
	.uleb128 .Ltmp163-.Ltmp162
	.uleb128 .Ltmp164-.Lfunc_begin1
	.byte	0
	.uleb128 .Ltmp196-.Lfunc_begin1
	.uleb128 .Ltmp197-.Ltmp196
	.uleb128 .Ltmp198-.Lfunc_begin1
	.byte	0
	.uleb128 .Ltmp193-.Lfunc_begin1
	.uleb128 .Ltmp194-.Ltmp193
	.uleb128 .Ltmp195-.Lfunc_begin1
	.byte	0
	.uleb128 .Ltmp190-.Lfunc_begin1
	.uleb128 .Ltmp189-.Ltmp190
	.uleb128 .Ltmp192-.Lfunc_begin1
	.byte	0
	.uleb128 .Ltmp185-.Lfunc_begin1
	.uleb128 .Ltmp186-.Ltmp185
	.uleb128 .Ltmp187-.Lfunc_begin1
	.byte	0
	.uleb128 .Ltmp179-.Lfunc_begin1
	.uleb128 .Ltmp180-.Ltmp179
	.uleb128 .Ltmp181-.Lfunc_begin1
	.byte	0
	.uleb128 .Ltmp174-.Lfunc_begin1
	.uleb128 .Ltmp175-.Ltmp174
	.uleb128 .Ltmp176-.Lfunc_begin1
	.byte	0
	.uleb128 .Ltmp171-.Lfunc_begin1
	.uleb128 .Ltmp172-.Ltmp171
	.uleb128 .Ltmp173-.Lfunc_begin1
	.byte	0
	.uleb128 .Ltmp168-.Lfunc_begin1
	.uleb128 .Ltmp169-.Ltmp168
	.uleb128 .Ltmp170-.Lfunc_begin1
	.byte	0
	.uleb128 .Ltmp160-.Lfunc_begin1
	.uleb128 .Ltmp161-.Ltmp160
	.uleb128 .Ltmp164-.Lfunc_begin1
	.byte	0
	.uleb128 .Ltmp132-.Lfunc_begin1
	.uleb128 .Ltmp133-.Ltmp132
	.uleb128 .Ltmp134-.Lfunc_begin1
	.byte	0
	.uleb128 .Ltmp88-.Lfunc_begin1
	.uleb128 .Ltmp89-.Ltmp88
	.uleb128 .Ltmp90-.Lfunc_begin1
	.byte	0
	.uleb128 .Ltmp91-.Lfunc_begin1
	.uleb128 .Ltmp112-.Ltmp91
	.uleb128 .Ltmp164-.Lfunc_begin1
	.byte	0
	.uleb128 .Ltmp154-.Lfunc_begin1
	.uleb128 .Ltmp155-.Ltmp154
	.uleb128 .Ltmp156-.Lfunc_begin1
	.byte	0
	.uleb128 .Ltmp146-.Lfunc_begin1
	.uleb128 .Ltmp147-.Ltmp146
	.uleb128 .Ltmp148-.Lfunc_begin1
	.byte	0
	.uleb128 .Ltmp135-.Lfunc_begin1
	.uleb128 .Ltmp136-.Ltmp135
	.uleb128 .Ltmp137-.Lfunc_begin1
	.byte	0
	.uleb128 .Ltmp129-.Lfunc_begin1
	.uleb128 .Ltmp130-.Ltmp129
	.uleb128 .Ltmp131-.Lfunc_begin1
	.byte	0
	.uleb128 .Ltmp157-.Lfunc_begin1
	.uleb128 .Ltmp158-.Ltmp157
	.uleb128 .Ltmp159-.Lfunc_begin1
	.byte	0
	.uleb128 .Ltmp102-.Lfunc_begin1
	.uleb128 .Ltmp103-.Ltmp102
	.uleb128 .Ltmp104-.Lfunc_begin1
	.byte	0
	.uleb128 .Ltmp103-.Lfunc_begin1
	.uleb128 .Ltmp220-.Ltmp103
	.byte	0
	.byte	0
	.uleb128 .Ltmp220-.Lfunc_begin1
	.uleb128 .Ltmp221-.Ltmp220
	.uleb128 .Ltmp222-.Lfunc_begin1
	.byte	0
	.uleb128 .Ltmp217-.Lfunc_begin1
	.uleb128 .Ltmp218-.Ltmp217
	.uleb128 .Ltmp219-.Lfunc_begin1
	.byte	0
	.uleb128 .Ltmp214-.Lfunc_begin1
	.uleb128 .Ltmp215-.Ltmp214
	.uleb128 .Ltmp216-.Lfunc_begin1
	.byte	0
	.uleb128 .Ltmp211-.Lfunc_begin1
	.uleb128 .Ltmp212-.Ltmp211
	.uleb128 .Ltmp213-.Lfunc_begin1
	.byte	0
	.uleb128 .Ltmp208-.Lfunc_begin1
	.uleb128 .Ltmp209-.Ltmp208
	.uleb128 .Ltmp210-.Lfunc_begin1
	.byte	0
	.uleb128 .Ltmp205-.Lfunc_begin1
	.uleb128 .Ltmp206-.Ltmp205
	.uleb128 .Ltmp207-.Lfunc_begin1
	.byte	0
	.uleb128 .Ltmp202-.Lfunc_begin1
	.uleb128 .Ltmp203-.Ltmp202
	.uleb128 .Ltmp204-.Lfunc_begin1
	.byte	0
	.uleb128 .Ltmp199-.Lfunc_begin1
	.uleb128 .Ltmp200-.Ltmp199
	.uleb128 .Ltmp201-.Lfunc_begin1
	.byte	0
	.uleb128 .Ltmp165-.Lfunc_begin1
	.uleb128 .Ltmp166-.Ltmp165
	.uleb128 .Ltmp167-.Lfunc_begin1
	.byte	0
	.uleb128 .Ltmp99-.Lfunc_begin1
	.uleb128 .Ltmp100-.Ltmp99
	.uleb128 .Ltmp101-.Lfunc_begin1
	.byte	0
	.uleb128 .Ltmp100-.Lfunc_begin1
	.uleb128 .Ltmp149-.Ltmp100
	.byte	0
	.byte	0
	.uleb128 .Ltmp149-.Lfunc_begin1
	.uleb128 .Ltmp150-.Ltmp149
	.uleb128 .Ltmp151-.Lfunc_begin1
	.byte	1
	.uleb128 .Ltmp182-.Lfunc_begin1
	.uleb128 .Ltmp178-.Ltmp182
	.uleb128 .Ltmp184-.Lfunc_begin1
	.byte	1
	.uleb128 .Ltmp178-.Lfunc_begin1
	.uleb128 .Lfunc_end5-.Ltmp178
	.byte	0
	.byte	0
.Lcst_end1:
	.byte	127
	.byte	0
	.p2align	2, 0x0
.Lttbase0:
	.byte	0
	.p2align	2, 0x0

	.section	.rodata.cst8,"aM",@progbits,8
	.p2align	3, 0x0
.LCPI6_0:
	.quad	0x8000000000000000
	.section	".text._ZN21protobuf_decode_bench4main28_$u7b$$u7b$closure$u7d$$u7d$17h8393f7a28191b620E","ax",@progbits
	.p2align	4
	.type	_ZN21protobuf_decode_bench4main28_$u7b$$u7b$closure$u7d$$u7d$17h8393f7a28191b620E,@function
_ZN21protobuf_decode_bench4main28_$u7b$$u7b$closure$u7d$$u7d$17h8393f7a28191b620E:
.Lfunc_begin2:
	.cfi_startproc
	.cfi_personality 155, DW.ref.rust_eh_personality
	.cfi_lsda 27, .Lexception2
	pushq	%rbx
	.cfi_def_cfa_offset 16
	subq	$48, %rsp
	.cfi_def_cfa_offset 64
	.cfi_offset %rbx, -16
	movq	(%rdi), %rax
	leaq	24(%rsp), %rdi
	movq	8(%rax), %rsi
	movq	16(%rax), %rdx
	callq	_ZN5prost7message7Message6decode17h7e463807ec2b546fE
	movq	24(%rsp), %rsi
	movq	%rsi, %rax
	negq	%rax
	jo	.LBB6_1
	movq	32(%rsp), %rdi
	movq	40(%rsp), %rdx
	testq	%rdx, %rdx
	je	.LBB6_4
	movl	%edx, %eax
	andl	$7, %eax
	cmpq	$8, %rdx
	jae	.LBB6_7
	vmovsd	.LCPI6_0(%rip), %xmm0
	xorl	%ecx, %ecx
	jmp	.LBB6_10
.LBB6_4:
	vmovsd	.LCPI6_0(%rip), %xmm0
	jmp	.LBB6_12
.LBB6_7:
	vmovsd	.LCPI6_0(%rip), %xmm0
	andq	$-8, %rdx
	leaq	112(%rdi), %r8
	xorl	%ecx, %ecx
	.p2align	4
.LBB6_8:
	vaddsd	-112(%r8), %xmm0, %xmm0
	addq	$8, %rcx
	vaddsd	-96(%r8), %xmm0, %xmm0
	vaddsd	-80(%r8), %xmm0, %xmm0
	vaddsd	-64(%r8), %xmm0, %xmm0
	vaddsd	-48(%r8), %xmm0, %xmm0
	vaddsd	-32(%r8), %xmm0, %xmm0
	vaddsd	-16(%r8), %xmm0, %xmm0
	vaddsd	(%r8), %xmm0, %xmm0
	subq	$-128, %r8
	cmpq	%rcx, %rdx
	jne	.LBB6_8
	testq	%rax, %rax
	je	.LBB6_12
.LBB6_10:
	shlq	$4, %rcx
	shll	$4, %eax
	xorl	%edx, %edx
	addq	%rdi, %rcx
	.p2align	4
.LBB6_11:
	vaddsd	(%rcx,%rdx), %xmm0, %xmm0
	addq	$16, %rdx
	cmpq	%rdx, %rax
	jne	.LBB6_11
.LBB6_12:
	testq	%rsi, %rsi
	je	.LBB6_14
	shlq	$4, %rsi
	movl	$8, %edx
	vmovsd	%xmm0, 16(%rsp)
	callq	*_RNvCsfLfy6EI15iL_7___rustc14___rust_dealloc@GOTPCREL(%rip)
	vmovsd	16(%rsp), %xmm0
.LBB6_14:
	addq	$48, %rsp
	.cfi_def_cfa_offset 16
	popq	%rbx
	.cfi_def_cfa_offset 8
	retq
.LBB6_1:
	.cfi_def_cfa_offset 64
	movq	32(%rsp), %rax
	movq	%rax, 8(%rsp)
.Ltmp223:
	leaq	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.39(%rip), %rdi
	leaq	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.73(%rip), %rcx
	leaq	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.40(%rip), %r8
	leaq	8(%rsp), %rdx
	movl	$22, %esi
	callq	*_RNvNtCsgEmfK2I1SDS_4core6result13unwrap_failed@GOTPCREL(%rip)
.Ltmp224:
	ud2
.LBB6_15:
.Ltmp225:
	leaq	8(%rsp), %rdi
	movq	%rax, %rbx
	callq	_ZN4core3ptr46drop_in_place$LT$prost..error..DecodeError$GT$17h6452743001005f2fE
	movq	%rbx, %rdi
	callq	_Unwind_Resume@PLT
.Lfunc_end6:
	.size	_ZN21protobuf_decode_bench4main28_$u7b$$u7b$closure$u7d$$u7d$17h8393f7a28191b620E, .Lfunc_end6-_ZN21protobuf_decode_bench4main28_$u7b$$u7b$closure$u7d$$u7d$17h8393f7a28191b620E
	.cfi_endproc
	.section	".gcc_except_table._ZN21protobuf_decode_bench4main28_$u7b$$u7b$closure$u7d$$u7d$17h8393f7a28191b620E","a",@progbits
	.p2align	2, 0x0
GCC_except_table6:
.Lexception2:
	.byte	255
	.byte	255
	.byte	1
	.uleb128 .Lcst_end2-.Lcst_begin2
.Lcst_begin2:
	.uleb128 .Lfunc_begin2-.Lfunc_begin2
	.uleb128 .Ltmp223-.Lfunc_begin2
	.byte	0
	.byte	0
	.uleb128 .Ltmp223-.Lfunc_begin2
	.uleb128 .Ltmp224-.Ltmp223
	.uleb128 .Ltmp225-.Lfunc_begin2
	.byte	0
	.uleb128 .Ltmp224-.Lfunc_begin2
	.uleb128 .Lfunc_end6-.Ltmp224
	.byte	0
	.byte	0
.Lcst_end2:
	.p2align	2, 0x0

	.section	.rodata.cst8,"aM",@progbits,8
	.p2align	3, 0x0
.LCPI7_0:
	.quad	0x8000000000000000
	.section	".text._ZN21protobuf_decode_bench4main28_$u7b$$u7b$closure$u7d$$u7d$17he017dc1e830bdeb0E","ax",@progbits
	.p2align	4
	.type	_ZN21protobuf_decode_bench4main28_$u7b$$u7b$closure$u7d$$u7d$17he017dc1e830bdeb0E,@function
_ZN21protobuf_decode_bench4main28_$u7b$$u7b$closure$u7d$$u7d$17he017dc1e830bdeb0E:
.Lfunc_begin3:
	.cfi_startproc
	.cfi_personality 155, DW.ref.rust_eh_personality
	.cfi_lsda 27, .Lexception3
	pushq	%rbx
	.cfi_def_cfa_offset 16
	subq	$48, %rsp
	.cfi_def_cfa_offset 64
	.cfi_offset %rbx, -16
	movq	(%rdi), %rax
	leaq	24(%rsp), %rdi
	movq	8(%rax), %rsi
	movq	16(%rax), %rdx
	callq	_ZN5prost7message7Message6decode17h7e463807ec2b546fE
	movq	24(%rsp), %rsi
	movq	%rsi, %rax
	negq	%rax
	jo	.LBB7_1
	movq	32(%rsp), %rdi
	movq	40(%rsp), %rdx
	testq	%rdx, %rdx
	je	.LBB7_4
	movl	%edx, %eax
	andl	$7, %eax
	cmpq	$8, %rdx
	jae	.LBB7_7
	vmovsd	.LCPI7_0(%rip), %xmm0
	xorl	%ecx, %ecx
	jmp	.LBB7_10
.LBB7_4:
	vmovsd	.LCPI7_0(%rip), %xmm0
	jmp	.LBB7_12
.LBB7_7:
	vmovsd	.LCPI7_0(%rip), %xmm0
	andq	$-8, %rdx
	leaq	120(%rdi), %r8
	xorl	%ecx, %ecx
	.p2align	4
.LBB7_8:
	movl	-112(%r8), %r9d
	movl	-96(%r8), %r10d
	addq	$8, %rcx
	vcvtsi2sd	%r9, %xmm15, %xmm1
	vaddsd	-120(%r8), %xmm1, %xmm2
	vcvtsi2sd	%r10, %xmm15, %xmm1
	vaddsd	-104(%r8), %xmm1, %xmm1
	movl	-80(%r8), %r9d
	vaddsd	%xmm2, %xmm0, %xmm0
	vaddsd	%xmm1, %xmm0, %xmm0
	vcvtsi2sd	%r9, %xmm15, %xmm1
	vaddsd	-88(%r8), %xmm1, %xmm1
	movl	-64(%r8), %r9d
	vaddsd	%xmm1, %xmm0, %xmm0
	vcvtsi2sd	%r9, %xmm15, %xmm1
	vaddsd	-72(%r8), %xmm1, %xmm1
	movl	-48(%r8), %r9d
	vaddsd	%xmm1, %xmm0, %xmm0
	vcvtsi2sd	%r9, %xmm15, %xmm1
	vaddsd	-56(%r8), %xmm1, %xmm1
	movl	-32(%r8), %r9d
	vaddsd	%xmm1, %xmm0, %xmm0
	vcvtsi2sd	%r9, %xmm15, %xmm1
	vaddsd	-40(%r8), %xmm1, %xmm1
	movl	-16(%r8), %r9d
	vaddsd	%xmm1, %xmm0, %xmm0
	vcvtsi2sd	%r9, %xmm15, %xmm1
	movl	(%r8), %r9d
	vaddsd	-24(%r8), %xmm1, %xmm1
	vcvtsi2sd	%r9, %xmm15, %xmm2
	vaddsd	-8(%r8), %xmm2, %xmm2
	subq	$-128, %r8
	vaddsd	%xmm1, %xmm0, %xmm0
	vaddsd	%xmm2, %xmm0, %xmm0
	cmpq	%rcx, %rdx
	jne	.LBB7_8
	testq	%rax, %rax
	je	.LBB7_12
.LBB7_10:
	shlq	$4, %rcx
	shll	$4, %eax
	xorl	%edx, %edx
	leaq	8(%rcx,%rdi), %rcx
	.p2align	4
.LBB7_11:
	movl	(%rcx,%rdx), %r8d
	vcvtsi2sd	%r8, %xmm15, %xmm1
	vaddsd	-8(%rcx,%rdx), %xmm1, %xmm1
	addq	$16, %rdx
	vaddsd	%xmm1, %xmm0, %xmm0
	cmpq	%rdx, %rax
	jne	.LBB7_11
.LBB7_12:
	testq	%rsi, %rsi
	je	.LBB7_14
	shlq	$4, %rsi
	movl	$8, %edx
	vmovsd	%xmm0, 16(%rsp)
	callq	*_RNvCsfLfy6EI15iL_7___rustc14___rust_dealloc@GOTPCREL(%rip)
	vmovsd	16(%rsp), %xmm0
.LBB7_14:
	addq	$48, %rsp
	.cfi_def_cfa_offset 16
	popq	%rbx
	.cfi_def_cfa_offset 8
	retq
.LBB7_1:
	.cfi_def_cfa_offset 64
	movq	32(%rsp), %rax
	movq	%rax, 8(%rsp)
.Ltmp226:
	leaq	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.39(%rip), %rdi
	leaq	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.73(%rip), %rcx
	leaq	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.41(%rip), %r8
	leaq	8(%rsp), %rdx
	movl	$22, %esi
	callq	*_RNvNtCsgEmfK2I1SDS_4core6result13unwrap_failed@GOTPCREL(%rip)
.Ltmp227:
	ud2
.LBB7_15:
.Ltmp228:
	leaq	8(%rsp), %rdi
	movq	%rax, %rbx
	callq	_ZN4core3ptr46drop_in_place$LT$prost..error..DecodeError$GT$17h6452743001005f2fE
	movq	%rbx, %rdi
	callq	_Unwind_Resume@PLT
.Lfunc_end7:
	.size	_ZN21protobuf_decode_bench4main28_$u7b$$u7b$closure$u7d$$u7d$17he017dc1e830bdeb0E, .Lfunc_end7-_ZN21protobuf_decode_bench4main28_$u7b$$u7b$closure$u7d$$u7d$17he017dc1e830bdeb0E
	.cfi_endproc
	.section	".gcc_except_table._ZN21protobuf_decode_bench4main28_$u7b$$u7b$closure$u7d$$u7d$17he017dc1e830bdeb0E","a",@progbits
	.p2align	2, 0x0
GCC_except_table7:
.Lexception3:
	.byte	255
	.byte	255
	.byte	1
	.uleb128 .Lcst_end3-.Lcst_begin3
.Lcst_begin3:
	.uleb128 .Lfunc_begin3-.Lfunc_begin3
	.uleb128 .Ltmp226-.Lfunc_begin3
	.byte	0
	.byte	0
	.uleb128 .Ltmp226-.Lfunc_begin3
	.uleb128 .Ltmp227-.Ltmp226
	.uleb128 .Ltmp228-.Lfunc_begin3
	.byte	0
	.uleb128 .Ltmp227-.Lfunc_begin3
	.uleb128 .Lfunc_end7-.Ltmp227
	.byte	0
	.byte	0
.Lcst_end3:
	.p2align	2, 0x0

	.section	.text._ZN3std2rt10lang_start17hdf1d92676b1a174eE,"ax",@progbits
	.hidden	_ZN3std2rt10lang_start17hdf1d92676b1a174eE
	.globl	_ZN3std2rt10lang_start17hdf1d92676b1a174eE
	.p2align	4
	.type	_ZN3std2rt10lang_start17hdf1d92676b1a174eE,@function
_ZN3std2rt10lang_start17hdf1d92676b1a174eE:
	.cfi_startproc
	pushq	%rax
	.cfi_def_cfa_offset 16
	movl	%ecx, %r8d
	movq	%rdx, %rcx
	movq	%rsi, %rdx
	leaq	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.65(%rip), %rsi
	movq	%rdi, (%rsp)
	movq	%rsp, %rdi
	callq	*_RNvNtCsjrHSEGnQ3l9_3std2rt19lang_start_internal@GOTPCREL(%rip)
	popq	%rcx
	.cfi_def_cfa_offset 8
	retq
.Lfunc_end8:
	.size	_ZN3std2rt10lang_start17hdf1d92676b1a174eE, .Lfunc_end8-_ZN3std2rt10lang_start17hdf1d92676b1a174eE
	.cfi_endproc

	.section	".text._ZN3std2rt10lang_start28_$u7b$$u7b$closure$u7d$$u7d$17h5c9919aee196cda5E","ax",@progbits
	.p2align	4
	.type	_ZN3std2rt10lang_start28_$u7b$$u7b$closure$u7d$$u7d$17h5c9919aee196cda5E,@function
_ZN3std2rt10lang_start28_$u7b$$u7b$closure$u7d$$u7d$17h5c9919aee196cda5E:
	.cfi_startproc
	pushq	%rax
	.cfi_def_cfa_offset 16
	movq	(%rdi), %rdi
	callq	_ZN3std3sys9backtrace28__rust_begin_short_backtrace17h75516a0c8ac98aebE
	xorl	%eax, %eax
	popq	%rcx
	.cfi_def_cfa_offset 8
	retq
.Lfunc_end9:
	.size	_ZN3std2rt10lang_start28_$u7b$$u7b$closure$u7d$$u7d$17h5c9919aee196cda5E, .Lfunc_end9-_ZN3std2rt10lang_start28_$u7b$$u7b$closure$u7d$$u7d$17h5c9919aee196cda5E
	.cfi_endproc

	.section	.text._ZN3std3sys9backtrace28__rust_begin_short_backtrace17h75516a0c8ac98aebE,"ax",@progbits
	.p2align	4
	.type	_ZN3std3sys9backtrace28__rust_begin_short_backtrace17h75516a0c8ac98aebE,@function
_ZN3std3sys9backtrace28__rust_begin_short_backtrace17h75516a0c8ac98aebE:
	.cfi_startproc
	pushq	%rax
	.cfi_def_cfa_offset 16
	callq	*%rdi
	#APP
	#NO_APP
	popq	%rax
	.cfi_def_cfa_offset 8
	retq
.Lfunc_end10:
	.size	_ZN3std3sys9backtrace28__rust_begin_short_backtrace17h75516a0c8ac98aebE, .Lfunc_end10-_ZN3std3sys9backtrace28__rust_begin_short_backtrace17h75516a0c8ac98aebE
	.cfi_endproc

	.section	".text._ZN42_$LT$$RF$T$u20$as$u20$core..fmt..Debug$GT$3fmt17h0fab7b41fe56f931E","ax",@progbits
	.p2align	4
	.type	_ZN42_$LT$$RF$T$u20$as$u20$core..fmt..Debug$GT$3fmt17h0fab7b41fe56f931E,@function
_ZN42_$LT$$RF$T$u20$as$u20$core..fmt..Debug$GT$3fmt17h0fab7b41fe56f931E:
	.cfi_startproc
	movq	(%rdi), %rdi
	jmpq	*_RNvXs6_NtNtCsgEmfK2I1SDS_4core3fmt5floatdNtB7_5Debug3fmt@GOTPCREL(%rip)
.Lfunc_end11:
	.size	_ZN42_$LT$$RF$T$u20$as$u20$core..fmt..Debug$GT$3fmt17h0fab7b41fe56f931E, .Lfunc_end11-_ZN42_$LT$$RF$T$u20$as$u20$core..fmt..Debug$GT$3fmt17h0fab7b41fe56f931E
	.cfi_endproc

	.section	".text._ZN42_$LT$$RF$T$u20$as$u20$core..fmt..Debug$GT$3fmt17h2676d877de11e1abE","ax",@progbits
	.p2align	4
	.type	_ZN42_$LT$$RF$T$u20$as$u20$core..fmt..Debug$GT$3fmt17h2676d877de11e1abE,@function
_ZN42_$LT$$RF$T$u20$as$u20$core..fmt..Debug$GT$3fmt17h2676d877de11e1abE:
	.cfi_startproc
	movq	(%rdi), %rdi
	jmpq	*_RNvXs9_NtNtCsjrHSEGnQ3l9_3std3ffi6os_strNtB5_8OsStringNtNtCsgEmfK2I1SDS_4core3fmt5Debug3fmt@GOTPCREL(%rip)
.Lfunc_end12:
	.size	_ZN42_$LT$$RF$T$u20$as$u20$core..fmt..Debug$GT$3fmt17h2676d877de11e1abE, .Lfunc_end12-_ZN42_$LT$$RF$T$u20$as$u20$core..fmt..Debug$GT$3fmt17h2676d877de11e1abE
	.cfi_endproc

	.section	".text._ZN42_$LT$$RF$T$u20$as$u20$core..fmt..Debug$GT$3fmt17h9cb5016884c992d4E","ax",@progbits
	.p2align	4
	.type	_ZN42_$LT$$RF$T$u20$as$u20$core..fmt..Debug$GT$3fmt17h9cb5016884c992d4E,@function
_ZN42_$LT$$RF$T$u20$as$u20$core..fmt..Debug$GT$3fmt17h9cb5016884c992d4E:
	.cfi_startproc
	movq	%rsi, %rdx
	movq	(%rdi), %rax
	movq	8(%rdi), %rsi
	movq	%rax, %rdi
	jmpq	*_RNvXsh_NtCsgEmfK2I1SDS_4core3fmteNtB5_5Debug3fmt@GOTPCREL(%rip)
.Lfunc_end13:
	.size	_ZN42_$LT$$RF$T$u20$as$u20$core..fmt..Debug$GT$3fmt17h9cb5016884c992d4E, .Lfunc_end13-_ZN42_$LT$$RF$T$u20$as$u20$core..fmt..Debug$GT$3fmt17h9cb5016884c992d4E
	.cfi_endproc

	.section	".text._ZN42_$LT$$RF$T$u20$as$u20$core..fmt..Debug$GT$3fmt17hbd12bc66adbbe4b9E","ax",@progbits
	.p2align	4
	.type	_ZN42_$LT$$RF$T$u20$as$u20$core..fmt..Debug$GT$3fmt17hbd12bc66adbbe4b9E,@function
_ZN42_$LT$$RF$T$u20$as$u20$core..fmt..Debug$GT$3fmt17hbd12bc66adbbe4b9E:
	.cfi_startproc
	movq	(%rdi), %rdi
	movl	16(%rsi), %eax
	testl	$33554432, %eax
	jne	.LBB14_3
	testl	$67108864, %eax
	jne	.LBB14_2
	jmpq	*_RNvXsi_NtNtNtCsgEmfK2I1SDS_4core3fmt3num3impjNtB9_7Display3fmt@GOTPCREL(%rip)
.LBB14_3:
	jmpq	*_RNvXs6_NtNtCsgEmfK2I1SDS_4core3fmt3numjNtB7_8LowerHex3fmt@GOTPCREL(%rip)
.LBB14_2:
	jmpq	*_RNvXs8_NtNtCsgEmfK2I1SDS_4core3fmt3numjNtB7_8UpperHex3fmt@GOTPCREL(%rip)
.Lfunc_end14:
	.size	_ZN42_$LT$$RF$T$u20$as$u20$core..fmt..Debug$GT$3fmt17hbd12bc66adbbe4b9E, .Lfunc_end14-_ZN42_$LT$$RF$T$u20$as$u20$core..fmt..Debug$GT$3fmt17hbd12bc66adbbe4b9E
	.cfi_endproc

	.section	".text._ZN44_$LT$$RF$T$u20$as$u20$core..fmt..Display$GT$3fmt17h4aae8497b870fbd6E","ax",@progbits
	.p2align	4
	.type	_ZN44_$LT$$RF$T$u20$as$u20$core..fmt..Display$GT$3fmt17h4aae8497b870fbd6E,@function
_ZN44_$LT$$RF$T$u20$as$u20$core..fmt..Display$GT$3fmt17h4aae8497b870fbd6E:
	.cfi_startproc
	movq	(%rdi), %rax
	movq	%rsi, %rdx
	movq	8(%rax), %rdi
	movq	16(%rax), %rsi
	jmpq	*_RNvXsi_NtCsgEmfK2I1SDS_4core3fmteNtB5_7Display3fmt@GOTPCREL(%rip)
.Lfunc_end15:
	.size	_ZN44_$LT$$RF$T$u20$as$u20$core..fmt..Display$GT$3fmt17h4aae8497b870fbd6E, .Lfunc_end15-_ZN44_$LT$$RF$T$u20$as$u20$core..fmt..Display$GT$3fmt17h4aae8497b870fbd6E
	.cfi_endproc

	.section	".text._ZN44_$LT$$RF$T$u20$as$u20$core..fmt..Display$GT$3fmt17h63df3ed385e2537eE","ax",@progbits
	.p2align	4
	.type	_ZN44_$LT$$RF$T$u20$as$u20$core..fmt..Display$GT$3fmt17h63df3ed385e2537eE,@function
_ZN44_$LT$$RF$T$u20$as$u20$core..fmt..Display$GT$3fmt17h63df3ed385e2537eE:
	.cfi_startproc
	movq	%rsi, %rdx
	movq	(%rdi), %rax
	movq	8(%rdi), %rsi
	movq	%rax, %rdi
	jmpq	*_RNvXsi_NtCsgEmfK2I1SDS_4core3fmteNtB5_7Display3fmt@GOTPCREL(%rip)
.Lfunc_end16:
	.size	_ZN44_$LT$$RF$T$u20$as$u20$core..fmt..Display$GT$3fmt17h63df3ed385e2537eE, .Lfunc_end16-_ZN44_$LT$$RF$T$u20$as$u20$core..fmt..Display$GT$3fmt17h63df3ed385e2537eE
	.cfi_endproc

	.section	.rodata.cst8,"aM",@progbits,8
	.p2align	3, 0x0
.LCPI17_0:
	.quad	0x8000000000000000
	.section	".text._ZN4core3ops8function6FnOnce40call_once$u7b$$u7b$vtable.shim$u7d$$u7d$17h5021214c16a0eb18E","ax",@progbits
	.p2align	4
	.type	_ZN4core3ops8function6FnOnce40call_once$u7b$$u7b$vtable.shim$u7d$$u7d$17h5021214c16a0eb18E,@function
_ZN4core3ops8function6FnOnce40call_once$u7b$$u7b$vtable.shim$u7d$$u7d$17h5021214c16a0eb18E:
.Lfunc_begin4:
	.cfi_startproc
	.cfi_personality 155, DW.ref.rust_eh_personality
	.cfi_lsda 27, .Lexception4
	pushq	%rbx
	.cfi_def_cfa_offset 16
	subq	$48, %rsp
	.cfi_def_cfa_offset 64
	.cfi_offset %rbx, -16
	movq	(%rdi), %rax
	leaq	24(%rsp), %rdi
	movq	8(%rax), %rsi
	movq	16(%rax), %rdx
	callq	_ZN5prost7message7Message6decode17h7e463807ec2b546fE
	movq	24(%rsp), %rsi
	movq	%rsi, %rax
	negq	%rax
	jo	.LBB17_1
	movq	32(%rsp), %rdi
	movq	40(%rsp), %rdx
	testq	%rdx, %rdx
	je	.LBB17_4
	movl	%edx, %eax
	andl	$7, %eax
	cmpq	$8, %rdx
	jae	.LBB17_7
	vmovsd	.LCPI17_0(%rip), %xmm0
	xorl	%ecx, %ecx
	jmp	.LBB17_10
.LBB17_4:
	vmovsd	.LCPI17_0(%rip), %xmm0
	jmp	.LBB17_12
.LBB17_7:
	vmovsd	.LCPI17_0(%rip), %xmm0
	andq	$-8, %rdx
	leaq	112(%rdi), %r8
	xorl	%ecx, %ecx
	.p2align	4
.LBB17_8:
	vaddsd	-112(%r8), %xmm0, %xmm0
	addq	$8, %rcx
	vaddsd	-96(%r8), %xmm0, %xmm0
	vaddsd	-80(%r8), %xmm0, %xmm0
	vaddsd	-64(%r8), %xmm0, %xmm0
	vaddsd	-48(%r8), %xmm0, %xmm0
	vaddsd	-32(%r8), %xmm0, %xmm0
	vaddsd	-16(%r8), %xmm0, %xmm0
	vaddsd	(%r8), %xmm0, %xmm0
	subq	$-128, %r8
	cmpq	%rcx, %rdx
	jne	.LBB17_8
	testq	%rax, %rax
	je	.LBB17_12
.LBB17_10:
	shlq	$4, %rcx
	shll	$4, %eax
	xorl	%edx, %edx
	addq	%rdi, %rcx
	.p2align	4
.LBB17_11:
	vaddsd	(%rcx,%rdx), %xmm0, %xmm0
	addq	$16, %rdx
	cmpq	%rdx, %rax
	jne	.LBB17_11
.LBB17_12:
	testq	%rsi, %rsi
	je	.LBB17_14
	shlq	$4, %rsi
	movl	$8, %edx
	vmovsd	%xmm0, 16(%rsp)
	callq	*_RNvCsfLfy6EI15iL_7___rustc14___rust_dealloc@GOTPCREL(%rip)
	vmovsd	16(%rsp), %xmm0
.LBB17_14:
	addq	$48, %rsp
	.cfi_def_cfa_offset 16
	popq	%rbx
	.cfi_def_cfa_offset 8
	retq
.LBB17_1:
	.cfi_def_cfa_offset 64
	movq	32(%rsp), %rax
	movq	%rax, 8(%rsp)
.Ltmp229:
	leaq	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.39(%rip), %rdi
	leaq	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.73(%rip), %rcx
	leaq	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.40(%rip), %r8
	leaq	8(%rsp), %rdx
	movl	$22, %esi
	callq	*_RNvNtCsgEmfK2I1SDS_4core6result13unwrap_failed@GOTPCREL(%rip)
.Ltmp230:
	ud2
.LBB17_15:
.Ltmp231:
	leaq	8(%rsp), %rdi
	movq	%rax, %rbx
	callq	_ZN4core3ptr46drop_in_place$LT$prost..error..DecodeError$GT$17h6452743001005f2fE
	movq	%rbx, %rdi
	callq	_Unwind_Resume@PLT
.Lfunc_end17:
	.size	_ZN4core3ops8function6FnOnce40call_once$u7b$$u7b$vtable.shim$u7d$$u7d$17h5021214c16a0eb18E, .Lfunc_end17-_ZN4core3ops8function6FnOnce40call_once$u7b$$u7b$vtable.shim$u7d$$u7d$17h5021214c16a0eb18E
	.cfi_endproc
	.section	".gcc_except_table._ZN4core3ops8function6FnOnce40call_once$u7b$$u7b$vtable.shim$u7d$$u7d$17h5021214c16a0eb18E","a",@progbits
	.p2align	2, 0x0
GCC_except_table17:
.Lexception4:
	.byte	255
	.byte	255
	.byte	1
	.uleb128 .Lcst_end4-.Lcst_begin4
.Lcst_begin4:
	.uleb128 .Lfunc_begin4-.Lfunc_begin4
	.uleb128 .Ltmp229-.Lfunc_begin4
	.byte	0
	.byte	0
	.uleb128 .Ltmp229-.Lfunc_begin4
	.uleb128 .Ltmp230-.Ltmp229
	.uleb128 .Ltmp231-.Lfunc_begin4
	.byte	0
	.uleb128 .Ltmp230-.Lfunc_begin4
	.uleb128 .Lfunc_end17-.Ltmp230
	.byte	0
	.byte	0
.Lcst_end4:
	.p2align	2, 0x0

	.section	".text._ZN4core3ops8function6FnOnce40call_once$u7b$$u7b$vtable.shim$u7d$$u7d$17hd73245c67b3b00b0E","ax",@progbits
	.p2align	4
	.type	_ZN4core3ops8function6FnOnce40call_once$u7b$$u7b$vtable.shim$u7d$$u7d$17hd73245c67b3b00b0E,@function
_ZN4core3ops8function6FnOnce40call_once$u7b$$u7b$vtable.shim$u7d$$u7d$17hd73245c67b3b00b0E:
	.cfi_startproc
	pushq	%rax
	.cfi_def_cfa_offset 16
	movq	(%rdi), %rdi
	callq	_ZN3std3sys9backtrace28__rust_begin_short_backtrace17h75516a0c8ac98aebE
	xorl	%eax, %eax
	popq	%rcx
	.cfi_def_cfa_offset 8
	retq
.Lfunc_end18:
	.size	_ZN4core3ops8function6FnOnce40call_once$u7b$$u7b$vtable.shim$u7d$$u7d$17hd73245c67b3b00b0E, .Lfunc_end18-_ZN4core3ops8function6FnOnce40call_once$u7b$$u7b$vtable.shim$u7d$$u7d$17hd73245c67b3b00b0E
	.cfi_endproc

	.section	.rodata.cst8,"aM",@progbits,8
	.p2align	3, 0x0
.LCPI19_0:
	.quad	0x8000000000000000
	.section	".text._ZN4core3ops8function6FnOnce40call_once$u7b$$u7b$vtable.shim$u7d$$u7d$17hec0797272d3fcec1E","ax",@progbits
	.p2align	4
	.type	_ZN4core3ops8function6FnOnce40call_once$u7b$$u7b$vtable.shim$u7d$$u7d$17hec0797272d3fcec1E,@function
_ZN4core3ops8function6FnOnce40call_once$u7b$$u7b$vtable.shim$u7d$$u7d$17hec0797272d3fcec1E:
.Lfunc_begin5:
	.cfi_startproc
	.cfi_personality 155, DW.ref.rust_eh_personality
	.cfi_lsda 27, .Lexception5
	pushq	%rbx
	.cfi_def_cfa_offset 16
	subq	$48, %rsp
	.cfi_def_cfa_offset 64
	.cfi_offset %rbx, -16
	movq	(%rdi), %rax
	leaq	24(%rsp), %rdi
	movq	8(%rax), %rsi
	movq	16(%rax), %rdx
	callq	_ZN5prost7message7Message6decode17h7e463807ec2b546fE
	movq	24(%rsp), %rsi
	movq	%rsi, %rax
	negq	%rax
	jo	.LBB19_1
	movq	32(%rsp), %rdi
	movq	40(%rsp), %rdx
	testq	%rdx, %rdx
	je	.LBB19_4
	movl	%edx, %eax
	andl	$7, %eax
	cmpq	$8, %rdx
	jae	.LBB19_7
	vmovsd	.LCPI19_0(%rip), %xmm0
	xorl	%ecx, %ecx
	jmp	.LBB19_10
.LBB19_4:
	vmovsd	.LCPI19_0(%rip), %xmm0
	jmp	.LBB19_12
.LBB19_7:
	vmovsd	.LCPI19_0(%rip), %xmm0
	andq	$-8, %rdx
	leaq	120(%rdi), %r8
	xorl	%ecx, %ecx
	.p2align	4
.LBB19_8:
	movl	-112(%r8), %r9d
	movl	-96(%r8), %r10d
	addq	$8, %rcx
	vcvtsi2sd	%r9, %xmm15, %xmm1
	vaddsd	-120(%r8), %xmm1, %xmm2
	vcvtsi2sd	%r10, %xmm15, %xmm1
	vaddsd	-104(%r8), %xmm1, %xmm1
	movl	-80(%r8), %r9d
	vaddsd	%xmm2, %xmm0, %xmm0
	vaddsd	%xmm1, %xmm0, %xmm0
	vcvtsi2sd	%r9, %xmm15, %xmm1
	vaddsd	-88(%r8), %xmm1, %xmm1
	movl	-64(%r8), %r9d
	vaddsd	%xmm1, %xmm0, %xmm0
	vcvtsi2sd	%r9, %xmm15, %xmm1
	vaddsd	-72(%r8), %xmm1, %xmm1
	movl	-48(%r8), %r9d
	vaddsd	%xmm1, %xmm0, %xmm0
	vcvtsi2sd	%r9, %xmm15, %xmm1
	vaddsd	-56(%r8), %xmm1, %xmm1
	movl	-32(%r8), %r9d
	vaddsd	%xmm1, %xmm0, %xmm0
	vcvtsi2sd	%r9, %xmm15, %xmm1
	vaddsd	-40(%r8), %xmm1, %xmm1
	movl	-16(%r8), %r9d
	vaddsd	%xmm1, %xmm0, %xmm0
	vcvtsi2sd	%r9, %xmm15, %xmm1
	movl	(%r8), %r9d
	vaddsd	-24(%r8), %xmm1, %xmm1
	vcvtsi2sd	%r9, %xmm15, %xmm2
	vaddsd	-8(%r8), %xmm2, %xmm2
	subq	$-128, %r8
	vaddsd	%xmm1, %xmm0, %xmm0
	vaddsd	%xmm2, %xmm0, %xmm0
	cmpq	%rcx, %rdx
	jne	.LBB19_8
	testq	%rax, %rax
	je	.LBB19_12
.LBB19_10:
	shlq	$4, %rcx
	shll	$4, %eax
	xorl	%edx, %edx
	leaq	8(%rcx,%rdi), %rcx
	.p2align	4
.LBB19_11:
	movl	(%rcx,%rdx), %r8d
	vcvtsi2sd	%r8, %xmm15, %xmm1
	vaddsd	-8(%rcx,%rdx), %xmm1, %xmm1
	addq	$16, %rdx
	vaddsd	%xmm1, %xmm0, %xmm0
	cmpq	%rdx, %rax
	jne	.LBB19_11
.LBB19_12:
	testq	%rsi, %rsi
	je	.LBB19_14
	shlq	$4, %rsi
	movl	$8, %edx
	vmovsd	%xmm0, 16(%rsp)
	callq	*_RNvCsfLfy6EI15iL_7___rustc14___rust_dealloc@GOTPCREL(%rip)
	vmovsd	16(%rsp), %xmm0
.LBB19_14:
	addq	$48, %rsp
	.cfi_def_cfa_offset 16
	popq	%rbx
	.cfi_def_cfa_offset 8
	retq
.LBB19_1:
	.cfi_def_cfa_offset 64
	movq	32(%rsp), %rax
	movq	%rax, 8(%rsp)
.Ltmp232:
	leaq	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.39(%rip), %rdi
	leaq	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.73(%rip), %rcx
	leaq	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.41(%rip), %r8
	leaq	8(%rsp), %rdx
	movl	$22, %esi
	callq	*_RNvNtCsgEmfK2I1SDS_4core6result13unwrap_failed@GOTPCREL(%rip)
.Ltmp233:
	ud2
.LBB19_15:
.Ltmp234:
	leaq	8(%rsp), %rdi
	movq	%rax, %rbx
	callq	_ZN4core3ptr46drop_in_place$LT$prost..error..DecodeError$GT$17h6452743001005f2fE
	movq	%rbx, %rdi
	callq	_Unwind_Resume@PLT
.Lfunc_end19:
	.size	_ZN4core3ops8function6FnOnce40call_once$u7b$$u7b$vtable.shim$u7d$$u7d$17hec0797272d3fcec1E, .Lfunc_end19-_ZN4core3ops8function6FnOnce40call_once$u7b$$u7b$vtable.shim$u7d$$u7d$17hec0797272d3fcec1E
	.cfi_endproc
	.section	".gcc_except_table._ZN4core3ops8function6FnOnce40call_once$u7b$$u7b$vtable.shim$u7d$$u7d$17hec0797272d3fcec1E","a",@progbits
	.p2align	2, 0x0
GCC_except_table19:
.Lexception5:
	.byte	255
	.byte	255
	.byte	1
	.uleb128 .Lcst_end5-.Lcst_begin5
.Lcst_begin5:
	.uleb128 .Lfunc_begin5-.Lfunc_begin5
	.uleb128 .Ltmp232-.Lfunc_begin5
	.byte	0
	.byte	0
	.uleb128 .Ltmp232-.Lfunc_begin5
	.uleb128 .Ltmp233-.Ltmp232
	.uleb128 .Ltmp234-.Lfunc_begin5
	.byte	0
	.uleb128 .Ltmp233-.Lfunc_begin5
	.uleb128 .Lfunc_end19-.Ltmp233
	.byte	0
	.byte	0
.Lcst_end5:
	.p2align	2, 0x0

	.section	".text._ZN4core3ptr39drop_in_place$LT$std..env..VarError$GT$17hd6eda7ca964ee990E","ax",@progbits
	.p2align	4
	.type	_ZN4core3ptr39drop_in_place$LT$std..env..VarError$GT$17hd6eda7ca964ee990E,@function
_ZN4core3ptr39drop_in_place$LT$std..env..VarError$GT$17hd6eda7ca964ee990E:
	.cfi_startproc
	movq	(%rdi), %rsi
	testq	%rsi, %rsi
	jle	.LBB20_1
	movq	8(%rdi), %rdi
	movl	$1, %edx
	jmpq	*_RNvCsfLfy6EI15iL_7___rustc14___rust_dealloc@GOTPCREL(%rip)
.LBB20_1:
	retq
.Lfunc_end20:
	.size	_ZN4core3ptr39drop_in_place$LT$std..env..VarError$GT$17hd6eda7ca964ee990E, .Lfunc_end20-_ZN4core3ptr39drop_in_place$LT$std..env..VarError$GT$17hd6eda7ca964ee990E
	.cfi_endproc

	.section	".text._ZN4core3ptr40drop_in_place$LT$prost..error..Inner$GT$17h5eac8c554bcc6811E","ax",@progbits
	.p2align	4
	.type	_ZN4core3ptr40drop_in_place$LT$prost..error..Inner$GT$17h5eac8c554bcc6811E,@function
_ZN4core3ptr40drop_in_place$LT$prost..error..Inner$GT$17h5eac8c554bcc6811E:
	.cfi_startproc
	pushq	%rbx
	.cfi_def_cfa_offset 16
	.cfi_offset %rbx, -16
	movq	24(%rdi), %rsi
	movq	%rdi, %rbx
	testq	%rsi, %rsi
	jle	.LBB21_2
	movq	32(%rbx), %rdi
	movl	$1, %edx
	callq	*_RNvCsfLfy6EI15iL_7___rustc14___rust_dealloc@GOTPCREL(%rip)
.LBB21_2:
	movq	(%rbx), %rsi
	testq	%rsi, %rsi
	je	.LBB21_3
	movq	8(%rbx), %rdi
	shlq	$5, %rsi
	movl	$8, %edx
	popq	%rbx
	.cfi_def_cfa_offset 8
	jmpq	*_RNvCsfLfy6EI15iL_7___rustc14___rust_dealloc@GOTPCREL(%rip)
.LBB21_3:
	.cfi_def_cfa_offset 16
	popq	%rbx
	.cfi_def_cfa_offset 8
	retq
.Lfunc_end21:
	.size	_ZN4core3ptr40drop_in_place$LT$prost..error..Inner$GT$17h5eac8c554bcc6811E, .Lfunc_end21-_ZN4core3ptr40drop_in_place$LT$prost..error..Inner$GT$17h5eac8c554bcc6811E
	.cfi_endproc

	.section	".text._ZN4core3ptr42drop_in_place$LT$std..io..error..Error$GT$17hcedecc219889b047E","ax",@progbits
	.p2align	4
	.type	_ZN4core3ptr42drop_in_place$LT$std..io..error..Error$GT$17hcedecc219889b047E,@function
_ZN4core3ptr42drop_in_place$LT$std..io..error..Error$GT$17hcedecc219889b047E:
.Lfunc_begin6:
	.cfi_startproc
	.cfi_personality 155, DW.ref.rust_eh_personality
	.cfi_lsda 27, .Lexception6
	pushq	%r15
	.cfi_def_cfa_offset 16
	pushq	%r14
	.cfi_def_cfa_offset 24
	pushq	%r12
	.cfi_def_cfa_offset 32
	pushq	%rbx
	.cfi_def_cfa_offset 40
	pushq	%rax
	.cfi_def_cfa_offset 48
	.cfi_offset %rbx, -40
	.cfi_offset %r12, -32
	.cfi_offset %r14, -24
	.cfi_offset %r15, -16
	movq	(%rdi), %rax
	movl	%eax, %ecx
	andl	$3, %ecx
	leal	-2(%rcx), %edx
	cmpl	$2, %edx
	jb	.LBB22_10
	testq	%rcx, %rcx
	jne	.LBB22_2
.LBB22_10:
	addq	$8, %rsp
	.cfi_def_cfa_offset 40
	popq	%rbx
	.cfi_def_cfa_offset 32
	popq	%r12
	.cfi_def_cfa_offset 24
	popq	%r14
	.cfi_def_cfa_offset 16
	popq	%r15
	.cfi_def_cfa_offset 8
	retq
.LBB22_2:
	.cfi_def_cfa_offset 48
	movq	7(%rax), %r12
	leaq	-1(%rax), %rbx
	movq	-1(%rax), %r14
	movq	(%r12), %rax
	testq	%rax, %rax
	je	.LBB22_4
.Ltmp235:
	movq	%r14, %rdi
	callq	*%rax
.Ltmp236:
.LBB22_4:
	movq	8(%r12), %rsi
	testq	%rsi, %rsi
	je	.LBB22_6
	movq	16(%r12), %rdx
	movq	%r14, %rdi
	callq	*_RNvCsfLfy6EI15iL_7___rustc14___rust_dealloc@GOTPCREL(%rip)
.LBB22_6:
	movl	$24, %esi
	movl	$8, %edx
	movq	%rbx, %rdi
	addq	$8, %rsp
	.cfi_def_cfa_offset 40
	popq	%rbx
	.cfi_def_cfa_offset 32
	popq	%r12
	.cfi_def_cfa_offset 24
	popq	%r14
	.cfi_def_cfa_offset 16
	popq	%r15
	.cfi_def_cfa_offset 8
	jmpq	*_RNvCsfLfy6EI15iL_7___rustc14___rust_dealloc@GOTPCREL(%rip)
.LBB22_7:
	.cfi_def_cfa_offset 48
.Ltmp237:
	movq	8(%r12), %rsi
	movq	%rax, %r15
	testq	%rsi, %rsi
	je	.LBB22_9
	movq	16(%r12), %rdx
	movq	%r14, %rdi
	callq	*_RNvCsfLfy6EI15iL_7___rustc14___rust_dealloc@GOTPCREL(%rip)
.LBB22_9:
	movl	$24, %esi
	movl	$8, %edx
	movq	%rbx, %rdi
	callq	*_RNvCsfLfy6EI15iL_7___rustc14___rust_dealloc@GOTPCREL(%rip)
	movq	%r15, %rdi
	callq	_Unwind_Resume@PLT
.Lfunc_end22:
	.size	_ZN4core3ptr42drop_in_place$LT$std..io..error..Error$GT$17hcedecc219889b047E, .Lfunc_end22-_ZN4core3ptr42drop_in_place$LT$std..io..error..Error$GT$17hcedecc219889b047E
	.cfi_endproc
	.section	".gcc_except_table._ZN4core3ptr42drop_in_place$LT$std..io..error..Error$GT$17hcedecc219889b047E","a",@progbits
	.p2align	2, 0x0
GCC_except_table22:
.Lexception6:
	.byte	255
	.byte	255
	.byte	1
	.uleb128 .Lcst_end6-.Lcst_begin6
.Lcst_begin6:
	.uleb128 .Ltmp235-.Lfunc_begin6
	.uleb128 .Ltmp236-.Ltmp235
	.uleb128 .Ltmp237-.Lfunc_begin6
	.byte	0
	.uleb128 .Ltmp236-.Lfunc_begin6
	.uleb128 .Lfunc_end22-.Ltmp236
	.byte	0
	.byte	0
.Lcst_end6:
	.p2align	2, 0x0

	.section	".text._ZN4core3ptr46drop_in_place$LT$prost..error..DecodeError$GT$17h6452743001005f2fE","ax",@progbits
	.p2align	4
	.type	_ZN4core3ptr46drop_in_place$LT$prost..error..DecodeError$GT$17h6452743001005f2fE,@function
_ZN4core3ptr46drop_in_place$LT$prost..error..DecodeError$GT$17h6452743001005f2fE:
	.cfi_startproc
	pushq	%rbx
	.cfi_def_cfa_offset 16
	.cfi_offset %rbx, -16
	movq	(%rdi), %rbx
	movq	24(%rbx), %rsi
	testq	%rsi, %rsi
	jle	.LBB23_2
	movq	32(%rbx), %rdi
	movl	$1, %edx
	callq	*_RNvCsfLfy6EI15iL_7___rustc14___rust_dealloc@GOTPCREL(%rip)
.LBB23_2:
	movq	(%rbx), %rsi
	testq	%rsi, %rsi
	je	.LBB23_4
	movq	8(%rbx), %rdi
	shlq	$5, %rsi
	movl	$8, %edx
	callq	*_RNvCsfLfy6EI15iL_7___rustc14___rust_dealloc@GOTPCREL(%rip)
.LBB23_4:
	movl	$48, %esi
	movl	$8, %edx
	movq	%rbx, %rdi
	popq	%rbx
	.cfi_def_cfa_offset 8
	jmpq	*_RNvCsfLfy6EI15iL_7___rustc14___rust_dealloc@GOTPCREL(%rip)
.Lfunc_end23:
	.size	_ZN4core3ptr46drop_in_place$LT$prost..error..DecodeError$GT$17h6452743001005f2fE, .Lfunc_end23-_ZN4core3ptr46drop_in_place$LT$prost..error..DecodeError$GT$17h6452743001005f2fE
	.cfi_endproc

	.section	".text._ZN4core3ptr50drop_in_place$LT$protobuf_decode_bench..Target$GT$17hc849af708376b6caE","ax",@progbits
	.p2align	4
	.type	_ZN4core3ptr50drop_in_place$LT$protobuf_decode_bench..Target$GT$17hc849af708376b6caE,@function
_ZN4core3ptr50drop_in_place$LT$protobuf_decode_bench..Target$GT$17hc849af708376b6caE:
.Lfunc_begin7:
	.cfi_startproc
	.cfi_personality 155, DW.ref.rust_eh_personality
	.cfi_lsda 27, .Lexception7
	pushq	%r15
	.cfi_def_cfa_offset 16
	pushq	%r14
	.cfi_def_cfa_offset 24
	pushq	%rbx
	.cfi_def_cfa_offset 32
	.cfi_offset %rbx, -32
	.cfi_offset %r14, -24
	.cfi_offset %r15, -16
	movq	(%rsi), %rax
	movq	%rsi, %r14
	movq	%rdi, %rbx
	testq	%rax, %rax
	je	.LBB24_2
.Ltmp238:
	movq	%rbx, %rdi
	callq	*%rax
.Ltmp239:
.LBB24_2:
	movq	8(%r14), %rsi
	testq	%rsi, %rsi
	je	.LBB24_7
	movq	16(%r14), %rdx
	movq	%rbx, %rdi
	popq	%rbx
	.cfi_def_cfa_offset 24
	popq	%r14
	.cfi_def_cfa_offset 16
	popq	%r15
	.cfi_def_cfa_offset 8
	jmpq	*_RNvCsfLfy6EI15iL_7___rustc14___rust_dealloc@GOTPCREL(%rip)
.LBB24_7:
	.cfi_def_cfa_offset 32
	popq	%rbx
	.cfi_def_cfa_offset 24
	popq	%r14
	.cfi_def_cfa_offset 16
	popq	%r15
	.cfi_def_cfa_offset 8
	retq
.LBB24_4:
	.cfi_def_cfa_offset 32
.Ltmp240:
	movq	8(%r14), %rsi
	movq	%rax, %r15
	testq	%rsi, %rsi
	je	.LBB24_6
	movq	16(%r14), %rdx
	movq	%rbx, %rdi
	callq	*_RNvCsfLfy6EI15iL_7___rustc14___rust_dealloc@GOTPCREL(%rip)
.LBB24_6:
	movq	%r15, %rdi
	callq	_Unwind_Resume@PLT
.Lfunc_end24:
	.size	_ZN4core3ptr50drop_in_place$LT$protobuf_decode_bench..Target$GT$17hc849af708376b6caE, .Lfunc_end24-_ZN4core3ptr50drop_in_place$LT$protobuf_decode_bench..Target$GT$17hc849af708376b6caE
	.cfi_endproc
	.section	".gcc_except_table._ZN4core3ptr50drop_in_place$LT$protobuf_decode_bench..Target$GT$17hc849af708376b6caE","a",@progbits
	.p2align	2, 0x0
GCC_except_table24:
.Lexception7:
	.byte	255
	.byte	255
	.byte	1
	.uleb128 .Lcst_end7-.Lcst_begin7
.Lcst_begin7:
	.uleb128 .Ltmp238-.Lfunc_begin7
	.uleb128 .Ltmp239-.Ltmp238
	.uleb128 .Ltmp240-.Lfunc_begin7
	.byte	0
	.uleb128 .Ltmp239-.Lfunc_begin7
	.uleb128 .Lfunc_end24-.Ltmp239
	.byte	0
	.byte	0
.Lcst_end7:
	.p2align	2, 0x0

	.section	".text._ZN4core3ptr55drop_in_place$LT$protobuf_decode_bench..RunMetadata$GT$17hc16e98b5f36cd3b8E","ax",@progbits
	.p2align	4
	.type	_ZN4core3ptr55drop_in_place$LT$protobuf_decode_bench..RunMetadata$GT$17hc16e98b5f36cd3b8E,@function
_ZN4core3ptr55drop_in_place$LT$protobuf_decode_bench..RunMetadata$GT$17hc16e98b5f36cd3b8E:
	.cfi_startproc
	pushq	%rbx
	.cfi_def_cfa_offset 16
	.cfi_offset %rbx, -16
	movq	(%rdi), %rsi
	movq	%rdi, %rbx
	testq	%rsi, %rsi
	je	.LBB25_2
	movq	8(%rbx), %rdi
	movl	$1, %edx
	callq	*_RNvCsfLfy6EI15iL_7___rustc14___rust_dealloc@GOTPCREL(%rip)
.LBB25_2:
	movq	24(%rbx), %rsi
	testq	%rsi, %rsi
	je	.LBB25_4
	movq	32(%rbx), %rdi
	movl	$1, %edx
	callq	*_RNvCsfLfy6EI15iL_7___rustc14___rust_dealloc@GOTPCREL(%rip)
.LBB25_4:
	movq	48(%rbx), %rsi
	testq	%rsi, %rsi
	je	.LBB25_6
	movq	56(%rbx), %rdi
	movl	$1, %edx
	callq	*_RNvCsfLfy6EI15iL_7___rustc14___rust_dealloc@GOTPCREL(%rip)
.LBB25_6:
	movq	72(%rbx), %rsi
	testq	%rsi, %rsi
	je	.LBB25_8
	movq	80(%rbx), %rdi
	movl	$1, %edx
	callq	*_RNvCsfLfy6EI15iL_7___rustc14___rust_dealloc@GOTPCREL(%rip)
.LBB25_8:
	movq	96(%rbx), %rsi
	testq	%rsi, %rsi
	je	.LBB25_10
	movq	104(%rbx), %rdi
	movl	$1, %edx
	callq	*_RNvCsfLfy6EI15iL_7___rustc14___rust_dealloc@GOTPCREL(%rip)
.LBB25_10:
	movq	120(%rbx), %rsi
	testq	%rsi, %rsi
	je	.LBB25_12
	movq	128(%rbx), %rdi
	movl	$1, %edx
	callq	*_RNvCsfLfy6EI15iL_7___rustc14___rust_dealloc@GOTPCREL(%rip)
.LBB25_12:
	movq	144(%rbx), %rsi
	testq	%rsi, %rsi
	je	.LBB25_14
	movq	152(%rbx), %rdi
	movl	$1, %edx
	callq	*_RNvCsfLfy6EI15iL_7___rustc14___rust_dealloc@GOTPCREL(%rip)
.LBB25_14:
	movq	168(%rbx), %rsi
	testq	%rsi, %rsi
	je	.LBB25_15
	movq	176(%rbx), %rdi
	movl	$1, %edx
	popq	%rbx
	.cfi_def_cfa_offset 8
	jmpq	*_RNvCsfLfy6EI15iL_7___rustc14___rust_dealloc@GOTPCREL(%rip)
.LBB25_15:
	.cfi_def_cfa_offset 16
	popq	%rbx
	.cfi_def_cfa_offset 8
	retq
.Lfunc_end25:
	.size	_ZN4core3ptr55drop_in_place$LT$protobuf_decode_bench..RunMetadata$GT$17hc16e98b5f36cd3b8E, .Lfunc_end25-_ZN4core3ptr55drop_in_place$LT$protobuf_decode_bench..RunMetadata$GT$17hc16e98b5f36cd3b8E
	.cfi_endproc

	.section	".text._ZN4core3ptr73drop_in_place$LT$alloc..vec..Vec$LT$protobuf_decode_bench..Target$GT$$GT$17h84925d455482b3f0E","ax",@progbits
	.p2align	4
	.type	_ZN4core3ptr73drop_in_place$LT$alloc..vec..Vec$LT$protobuf_decode_bench..Target$GT$$GT$17h84925d455482b3f0E,@function
_ZN4core3ptr73drop_in_place$LT$alloc..vec..Vec$LT$protobuf_decode_bench..Target$GT$$GT$17h84925d455482b3f0E:
.Lfunc_begin8:
	.cfi_startproc
	.cfi_personality 155, DW.ref.rust_eh_personality
	.cfi_lsda 27, .Lexception8
	pushq	%rbp
	.cfi_def_cfa_offset 16
	pushq	%r15
	.cfi_def_cfa_offset 24
	pushq	%r14
	.cfi_def_cfa_offset 32
	pushq	%r13
	.cfi_def_cfa_offset 40
	pushq	%r12
	.cfi_def_cfa_offset 48
	pushq	%rbx
	.cfi_def_cfa_offset 56
	pushq	%rax
	.cfi_def_cfa_offset 64
	.cfi_offset %rbx, -56
	.cfi_offset %r12, -48
	.cfi_offset %r13, -40
	.cfi_offset %r14, -32
	.cfi_offset %r15, -24
	.cfi_offset %rbp, -16
	movq	8(%rdi), %rax
	movq	16(%rdi), %r13
	movq	%rdi, %r14
	movq	%rax, (%rsp)
	testq	%r13, %r13
	je	.LBB26_7
	movq	(%rsp), %rax
	movq	_RNvCsfLfy6EI15iL_7___rustc14___rust_dealloc@GOTPCREL(%rip), %r15
	leaq	88(%rax), %rbp
	jmp	.LBB26_2
	.p2align	4
.LBB26_6:
	addq	$48, %rbp
	decq	%r13
	je	.LBB26_7
.LBB26_2:
	movq	-48(%rbp), %rbx
	movq	-56(%rbp), %r12
	movq	(%rbx), %rax
	testq	%rax, %rax
	je	.LBB26_4
.Ltmp241:
	movq	%r12, %rdi
	callq	*%rax
.Ltmp242:
.LBB26_4:
	movq	8(%rbx), %rsi
	testq	%rsi, %rsi
	je	.LBB26_6
	movq	16(%rbx), %rdx
	movq	%r12, %rdi
	callq	*%r15
	jmp	.LBB26_6
.LBB26_7:
	movq	(%r14), %rax
	testq	%rax, %rax
	je	.LBB26_17
	movq	(%rsp), %rdi
	shlq	$4, %rax
	movl	$8, %edx
	leaq	(%rax,%rax,2), %rsi
	addq	$8, %rsp
	.cfi_def_cfa_offset 56
	popq	%rbx
	.cfi_def_cfa_offset 48
	popq	%r12
	.cfi_def_cfa_offset 40
	popq	%r13
	.cfi_def_cfa_offset 32
	popq	%r14
	.cfi_def_cfa_offset 24
	popq	%r15
	.cfi_def_cfa_offset 16
	popq	%rbp
	.cfi_def_cfa_offset 8
	jmpq	*_RNvCsfLfy6EI15iL_7___rustc14___rust_dealloc@GOTPCREL(%rip)
.LBB26_17:
	.cfi_def_cfa_offset 64
	addq	$8, %rsp
	.cfi_def_cfa_offset 56
	popq	%rbx
	.cfi_def_cfa_offset 48
	popq	%r12
	.cfi_def_cfa_offset 40
	popq	%r13
	.cfi_def_cfa_offset 32
	popq	%r14
	.cfi_def_cfa_offset 24
	popq	%r15
	.cfi_def_cfa_offset 16
	popq	%rbp
	.cfi_def_cfa_offset 8
	retq
.LBB26_9:
	.cfi_def_cfa_offset 64
.Ltmp243:
	movq	8(%rbx), %rsi
	movq	%rax, %r15
	testq	%rsi, %rsi
	je	.LBB26_11
	movq	16(%rbx), %rdx
	movq	%r12, %rdi
	callq	*_RNvCsfLfy6EI15iL_7___rustc14___rust_dealloc@GOTPCREL(%rip)
	.p2align	4
.LBB26_11:
	decq	%r13
	je	.LBB26_14
	movq	-8(%rbp), %rdi
	movq	(%rbp), %rsi
	leaq	48(%rbp), %rbx
.Ltmp244:
	callq	_ZN4core3ptr50drop_in_place$LT$protobuf_decode_bench..Target$GT$17hc849af708376b6caE
.Ltmp245:
	movq	%rbx, %rbp
	jmp	.LBB26_11
.LBB26_14:
	movq	(%r14), %rax
	testq	%rax, %rax
	je	.LBB26_16
	movq	(%rsp), %rdi
	shlq	$4, %rax
	movl	$8, %edx
	leaq	(%rax,%rax,2), %rsi
	callq	*_RNvCsfLfy6EI15iL_7___rustc14___rust_dealloc@GOTPCREL(%rip)
.LBB26_16:
	movq	%r15, %rdi
	callq	_Unwind_Resume@PLT
.LBB26_13:
.Ltmp246:
	callq	*_RNvNtCsgEmfK2I1SDS_4core9panicking16panic_in_cleanup@GOTPCREL(%rip)
.Lfunc_end26:
	.size	_ZN4core3ptr73drop_in_place$LT$alloc..vec..Vec$LT$protobuf_decode_bench..Target$GT$$GT$17h84925d455482b3f0E, .Lfunc_end26-_ZN4core3ptr73drop_in_place$LT$alloc..vec..Vec$LT$protobuf_decode_bench..Target$GT$$GT$17h84925d455482b3f0E
	.cfi_endproc
	.section	".gcc_except_table._ZN4core3ptr73drop_in_place$LT$alloc..vec..Vec$LT$protobuf_decode_bench..Target$GT$$GT$17h84925d455482b3f0E","a",@progbits
	.p2align	2, 0x0
GCC_except_table26:
.Lexception8:
	.byte	255
	.byte	155
	.uleb128 .Lttbase1-.Lttbaseref1
.Lttbaseref1:
	.byte	1
	.uleb128 .Lcst_end8-.Lcst_begin8
.Lcst_begin8:
	.uleb128 .Ltmp241-.Lfunc_begin8
	.uleb128 .Ltmp242-.Ltmp241
	.uleb128 .Ltmp243-.Lfunc_begin8
	.byte	0
	.uleb128 .Ltmp242-.Lfunc_begin8
	.uleb128 .Ltmp244-.Ltmp242
	.byte	0
	.byte	0
	.uleb128 .Ltmp244-.Lfunc_begin8
	.uleb128 .Ltmp245-.Ltmp244
	.uleb128 .Ltmp246-.Lfunc_begin8
	.byte	1
	.uleb128 .Ltmp245-.Lfunc_begin8
	.uleb128 .Lfunc_end26-.Ltmp245
	.byte	0
	.byte	0
.Lcst_end8:
	.byte	127
	.byte	0
	.p2align	2, 0x0
.Lttbase1:
	.byte	0
	.p2align	2, 0x0

	.section	.text._ZN4core5slice4sort6shared5pivot11median3_rec17hde37ea35567e1fabE,"ax",@progbits
	.p2align	4
	.type	_ZN4core5slice4sort6shared5pivot11median3_rec17hde37ea35567e1fabE,@function
_ZN4core5slice4sort6shared5pivot11median3_rec17hde37ea35567e1fabE:
	.cfi_startproc
	pushq	%rbp
	.cfi_def_cfa_offset 16
	pushq	%r15
	.cfi_def_cfa_offset 24
	pushq	%r14
	.cfi_def_cfa_offset 32
	pushq	%r13
	.cfi_def_cfa_offset 40
	pushq	%r12
	.cfi_def_cfa_offset 48
	pushq	%rbx
	.cfi_def_cfa_offset 56
	pushq	%rax
	.cfi_def_cfa_offset 64
	.cfi_offset %rbx, -56
	.cfi_offset %r12, -48
	.cfi_offset %r13, -40
	.cfi_offset %r14, -32
	.cfi_offset %r15, -24
	.cfi_offset %rbp, -16
	movq	%rdx, %r14
	movq	%rsi, %rbx
	cmpq	$8, %rcx
	jb	.LBB27_2
	shrq	$3, %rcx
	imulq	$56, %rcx, %r12
	movq	%rcx, %r15
	shlq	$5, %r15
	movq	%rcx, %r13
	leaq	(%rdi,%r15), %rsi
	leaq	(%rdi,%r12), %rdx
	callq	_ZN4core5slice4sort6shared5pivot11median3_rec17hde37ea35567e1fabE
	leaq	(%rbx,%r15), %rsi
	leaq	(%rbx,%r12), %rdx
	movq	%rax, %rbp
	movq	%rbx, %rdi
	movq	%r13, %rcx
	callq	_ZN4core5slice4sort6shared5pivot11median3_rec17hde37ea35567e1fabE
	addq	%r14, %r15
	addq	%r14, %r12
	movq	%rax, %rbx
	movq	%r14, %rdi
	movq	%r13, %rcx
	movq	%r15, %rsi
	movq	%r12, %rdx
	callq	_ZN4core5slice4sort6shared5pivot11median3_rec17hde37ea35567e1fabE
	movq	%rbp, %rdi
	movq	%rax, %r14
.LBB27_2:
	movq	(%rdi), %rax
	movq	(%rbx), %rcx
	movq	(%r14), %rsi
	movq	%rax, %rdx
	sarq	$63, %rdx
	movq	%rsi, %r8
	shrq	%rdx
	xorq	%rax, %rdx
	movq	%rcx, %rax
	sarq	$63, %rax
	shrq	%rax
	xorq	%rcx, %rax
	cmpq	%rax, %rdx
	setl	%cl
	sarq	$63, %r8
	shrq	%r8
	xorq	%rsi, %r8
	cmpq	%r8, %rdx
	setl	%dl
	xorb	%cl, %dl
	cmpq	%r8, %rax
	setl	%al
	xorb	%cl, %al
	cmovneq	%r14, %rbx
	testb	%dl, %dl
	cmovneq	%rdi, %rbx
	movq	%rbx, %rax
	addq	$8, %rsp
	.cfi_def_cfa_offset 56
	popq	%rbx
	.cfi_def_cfa_offset 48
	popq	%r12
	.cfi_def_cfa_offset 40
	popq	%r13
	.cfi_def_cfa_offset 32
	popq	%r14
	.cfi_def_cfa_offset 24
	popq	%r15
	.cfi_def_cfa_offset 16
	popq	%rbp
	.cfi_def_cfa_offset 8
	retq
.Lfunc_end27:
	.size	_ZN4core5slice4sort6shared5pivot11median3_rec17hde37ea35567e1fabE, .Lfunc_end27-_ZN4core5slice4sort6shared5pivot11median3_rec17hde37ea35567e1fabE
	.cfi_endproc

	.section	.text._ZN4core5slice4sort6shared9smallsort12sort8_stable17h6ef0fc9dfa2fbbbdE,"ax",@progbits
	.p2align	4
	.type	_ZN4core5slice4sort6shared9smallsort12sort8_stable17h6ef0fc9dfa2fbbbdE,@function
_ZN4core5slice4sort6shared9smallsort12sort8_stable17h6ef0fc9dfa2fbbbdE:
	.cfi_startproc
	pushq	%rbp
	.cfi_def_cfa_offset 16
	pushq	%r15
	.cfi_def_cfa_offset 24
	pushq	%r14
	.cfi_def_cfa_offset 32
	pushq	%r13
	.cfi_def_cfa_offset 40
	pushq	%r12
	.cfi_def_cfa_offset 48
	pushq	%rbx
	.cfi_def_cfa_offset 56
	pushq	%rax
	.cfi_def_cfa_offset 64
	.cfi_offset %rbx, -56
	.cfi_offset %r12, -48
	.cfi_offset %r13, -40
	.cfi_offset %r14, -32
	.cfi_offset %r15, -24
	.cfi_offset %rbp, -16
	vmovdqu	(%rdi), %xmm1
	vmovdqu	16(%rdi), %xmm2
	vpxor	%xmm0, %xmm0, %xmm0
	movl	$2, %ecx
	movq	%rdx, %r8
	movq	%rsi, (%rsp)
	vpcmpgtq	%xmm1, %xmm0, %xmm3
	vpsrlq	$1, %xmm3, %xmm3
	vpxor	%xmm1, %xmm3, %xmm1
	vpshufd	$238, %xmm1, %xmm3
	vpcmpgtq	%xmm3, %xmm1, %xmm1
	vmovq	%xmm1, %rax
	vpcmpgtq	%xmm2, %xmm0, %xmm1
	vpsrlq	$1, %xmm1, %xmm1
	movl	%eax, %edx
	notb	%al
	andl	$1, %edx
	vpxor	%xmm2, %xmm1, %xmm1
	movzbl	%al, %r10d
	movq	(%rdi,%rdx,8), %rsi
	vpshufd	$238, %xmm1, %xmm2
	andl	$1, %r10d
	vpcmpgtq	%xmm2, %xmm1, %xmm1
	movq	(%rdi,%r10,8), %r15
	leaq	(%rdi,%r10,8), %r11
	vmovd	%xmm1, %r14d
	subl	%r14d, %ecx
	andl	$1, %r14d
	xorl	$3, %r14d
	movq	(%rdi,%rcx,8), %rbx
	movq	%rsi, %r13
	sarq	$63, %r13
	movl	%ecx, %r9d
	movq	(%rdi,%r14,8), %rax
	shrq	%r13
	xorq	%rsi, %r13
	movq	%rbx, %r12
	sarq	$63, %r12
	movq	%rax, %rbp
	sarq	$63, %rbp
	shrq	%r12
	shrq	%rbp
	xorq	%rbx, %r12
	xorq	%rax, %rbp
	movq	%r15, %rax
	sarq	$63, %rax
	shrq	%rax
	xorq	%r15, %rax
	cmpq	%r13, %r12
	leaq	(%rdi,%r14,8), %r15
	cmovll	%r10d, %r9d
	cmpq	%rax, %rbp
	movl	$2, %eax
	cmovll	%r14d, %r9d
	cmovgel	%r10d, %ecx
	cmovlq	%r11, %r15
	cmpq	%r13, %r12
	leaq	(%rdi,%r9,8), %r10
	movq	(%rdi,%r9,8), %r9
	cmovll	%edx, %ecx
	cmovlq	%rbx, %rsi
	leaq	(%rdi,%rcx,8), %rdx
	movq	(%rdi,%rcx,8), %rcx
	movq	%rsi, (%r8)
	movq	%r9, %r11
	sarq	$63, %r11
	shrq	%r11
	xorq	%r9, %r11
	movq	%rcx, %r9
	sarq	$63, %r9
	shrq	%r9
	xorq	%rcx, %r9
	movq	%rdx, %rcx
	cmpq	%r9, %r11
	cmovlq	%r10, %rcx
	cmovlq	%rdx, %r10
	movq	(%rcx), %rcx
	movq	%rcx, 8(%r8)
	movq	(%r10), %rcx
	movq	%rcx, 16(%r8)
	movq	(%r15), %r9
	movq	%r9, 24(%r8)
	vmovdqu	32(%rdi), %xmm1
	vmovdqu	48(%rdi), %xmm2
	vpcmpgtq	%xmm1, %xmm0, %xmm3
	vpcmpgtq	%xmm2, %xmm0, %xmm0
	vpsrlq	$1, %xmm0, %xmm0
	vpsrlq	$1, %xmm3, %xmm3
	vpxor	%xmm2, %xmm0, %xmm0
	vpxor	%xmm1, %xmm3, %xmm1
	vpshufd	$238, %xmm0, %xmm2
	vpshufd	$238, %xmm1, %xmm3
	vpcmpgtq	%xmm2, %xmm0, %xmm0
	vpcmpgtq	%xmm3, %xmm1, %xmm1
	vmovd	%xmm0, %r14d
	vmovq	%xmm1, %rdx
	subl	%r14d, %eax
	andl	$1, %r14d
	movl	%edx, %ecx
	notb	%dl
	andl	$1, %ecx
	xorl	$3, %r14d
	movzbl	%dl, %ebx
	movq	32(%rdi,%rax,8), %r11
	movq	32(%rdi,%rcx,8), %r10
	movl	%eax, %ebp
	movq	32(%rdi,%r14,8), %rdx
	andl	$1, %ebx
	movq	32(%rdi,%rbx,8), %r12
	movq	%r11, %r15
	sarq	$63, %r15
	movq	%rdx, %r13
	sarq	$63, %r13
	shrq	%r15
	shrq	%r13
	xorq	%r11, %r15
	xorq	%rdx, %r13
	movq	%r12, %rdx
	sarq	$63, %rdx
	shrq	%rdx
	xorq	%r12, %rdx
	movq	%r10, %r12
	sarq	$63, %r12
	shrq	%r12
	xorq	%r10, %r12
	cmpq	%r12, %r15
	cmovll	%ebx, %ebp
	cmpq	%rdx, %r13
	leaq	32(%rdi,%rbx,8), %rdx
	cmovgel	%ebx, %eax
	leaq	32(%rdi,%r14,8), %rbx
	cmovll	%r14d, %ebp
	movq	32(%rdi,%rbp,8), %r14
	cmovlq	%rdx, %rbx
	cmpq	%r12, %r15
	leaq	32(%rdi,%rbp,8), %rdx
	movq	(%rsp), %rbp
	cmovll	%ecx, %eax
	cmovlq	%r11, %r10
	leaq	32(%rdi,%rax,8), %rcx
	movq	32(%rdi,%rax,8), %rdi
	movq	%r10, 32(%r8)
	leaq	24(%r8), %rax
	movq	%r14, %r11
	sarq	$63, %r11
	shrq	%r11
	xorq	%r14, %r11
	movq	%rdi, %r14
	sarq	$63, %r14
	shrq	%r14
	xorq	%rdi, %r14
	movq	%rcx, %rdi
	cmpq	%r14, %r11
	movq	%rsi, %r14
	cmovlq	%rdx, %rdi
	cmovlq	%rcx, %rdx
	sarq	$63, %r14
	xorl	%r11d, %r11d
	movq	(%rdi), %rdi
	shrq	%r14
	xorq	%rsi, %r14
	movq	%rdi, 40(%r8)
	xorl	%edi, %edi
	movq	(%rdx), %rcx
	movq	%r10, %rdx
	sarq	$63, %rdx
	shrq	%rdx
	xorq	%r10, %rdx
	cmpq	%r14, %rdx
	movq	%r9, %r14
	setge	%dil
	setl	%r11b
	cmovgeq	%rsi, %r10
	sarq	$63, %r14
	xorl	%r15d, %r15d
	xorl	%r12d, %r12d
	movq	%rcx, 48(%r8)
	shrq	%r14
	leaq	56(%r8), %rcx
	leaq	32(%r8,%r11,8), %rsi
	movq	(%rbx), %rbx
	xorq	%r9, %r14
	movq	%rbx, %rdx
	sarq	$63, %rdx
	movq	%rbx, 56(%r8)
	movq	%r10, (%rbp)
	leaq	(%r8,%rdi,8), %r10
	shrq	%rdx
	xorq	%rbx, %rdx
	cmpq	%r14, %rdx
	cmovlq	%r9, %rbx
	setl	%r15b
	setge	%r12b
	xorl	%r9d, %r9d
	movq	%rbx, 56(%rbp)
	shll	$3, %r12d
	shll	$3, %r15d
	movq	32(%r8,%r11,8), %rdx
	movq	(%r8,%rdi,8), %rdi
	subq	%r12, %rcx
	subq	%r15, %rax
	xorl	%r11d, %r11d
	movq	%rdx, %r8
	sarq	$63, %r8
	movq	%rdi, %rbx
	sarq	$63, %rbx
	shrq	%r8
	shrq	%rbx
	xorq	%rdx, %r8
	xorq	%rdi, %rbx
	cmpq	%rbx, %r8
	cmovlq	%rdx, %rdi
	setge	%r9b
	setl	%r11b
	xorl	%r12d, %r12d
	xorl	%r13d, %r13d
	movq	%rdi, 8(%rbp)
	leaq	(%rsi,%r11,8), %rdi
	leaq	(%r10,%r9,8), %rdx
	movq	(%rcx), %r8
	movq	(%rax), %rbx
	movq	%r8, %r14
	sarq	$63, %r14
	movq	%rbx, %r15
	sarq	$63, %r15
	shrq	%r14
	shrq	%r15
	xorq	%r8, %r14
	xorq	%rbx, %r15
	cmpq	%r15, %r14
	cmovlq	%rbx, %r8
	setl	%r12b
	setge	%r13b
	movq	%r8, 48(%rbp)
	shll	$3, %r13d
	shll	$3, %r12d
	xorl	%r8d, %r8d
	movq	(%rsi,%r11,8), %rsi
	movq	(%r10,%r9,8), %r10
	subq	%r13, %rcx
	subq	%r12, %rax
	xorl	%r9d, %r9d
	movq	%rsi, %r11
	sarq	$63, %r11
	movq	%r10, %rbx
	sarq	$63, %rbx
	shrq	%r11
	shrq	%rbx
	xorq	%rsi, %r11
	xorq	%r10, %rbx
	cmpq	%rbx, %r11
	cmovlq	%rsi, %r10
	setge	%r8b
	setl	%r9b
	xorl	%r14d, %r14d
	xorl	%r15d, %r15d
	movq	%r10, 16(%rbp)
	movq	(%rcx), %rsi
	movq	(%rax), %r10
	movq	%rsi, %r11
	sarq	$63, %r11
	movq	%r10, %rbx
	sarq	$63, %rbx
	shrq	%r11
	shrq	%rbx
	xorq	%rsi, %r11
	xorq	%r10, %rbx
	cmpq	%rbx, %r11
	cmovlq	%r10, %rsi
	setl	%r14b
	setge	%r15b
	xorl	%r10d, %r10d
	xorl	%r11d, %r11d
	movq	%rsi, 40(%rbp)
	shll	$3, %r15d
	shll	$3, %r14d
	movq	(%rdi,%r9,8), %rsi
	movq	(%rdx,%r8,8), %rbx
	subq	%r15, %rcx
	subq	%r14, %rax
	leaq	(%rdi,%r9,8), %rdi
	movq	%rsi, %r14
	sarq	$63, %r14
	movq	%rbx, %r15
	sarq	$63, %r15
	shrq	%r14
	shrq	%r15
	xorq	%rsi, %r14
	xorq	%rbx, %r15
	cmpq	%r15, %r14
	cmovlq	%rsi, %rbx
	setge	%r10b
	setl	%r11b
	xorl	%r12d, %r12d
	xorl	%r13d, %r13d
	movq	%rbx, 24(%rbp)
	movq	(%rcx), %rsi
	movq	(%rax), %rbx
	movq	%rsi, %r14
	sarq	$63, %r14
	movq	%rbx, %r15
	sarq	$63, %r15
	shrq	%r14
	shrq	%r15
	xorq	%rsi, %r14
	xorq	%rbx, %r15
	cmpq	%r15, %r14
	setge	%r13b
	cmovlq	%rbx, %rsi
	setl	%r12b
	shll	$3, %r13d
	movq	%rsi, 32(%rbp)
	shll	$3, %r12d
	leaq	(%rdi,%r11,8), %rsi
	subq	%r13, %rcx
	subq	%r12, %rax
	addq	$8, %rcx
	addq	$8, %rax
	cmpq	%rcx, %rsi
	leaq	(%rdx,%r8,8), %rcx
	leaq	(%rcx,%r10,8), %rdx
	setne	%cl
	cmpq	%rax, %rdx
	jne	.LBB28_1
	testb	%cl, %cl
	jne	.LBB28_4
.LBB28_3:
	addq	$8, %rsp
	.cfi_def_cfa_offset 56
	popq	%rbx
	.cfi_def_cfa_offset 48
	popq	%r12
	.cfi_def_cfa_offset 40
	popq	%r13
	.cfi_def_cfa_offset 32
	popq	%r14
	.cfi_def_cfa_offset 24
	popq	%r15
	.cfi_def_cfa_offset 16
	popq	%rbp
	.cfi_def_cfa_offset 8
	retq
.LBB28_1:
	.cfi_def_cfa_offset 64
	movb	$1, %cl
	testb	%cl, %cl
	je	.LBB28_3
.LBB28_4:
	callq	*_RNvNtNtNtNtCsgEmfK2I1SDS_4core5slice4sort6shared9smallsort22panic_on_ord_violation@GOTPCREL(%rip)
.Lfunc_end28:
	.size	_ZN4core5slice4sort6shared9smallsort12sort8_stable17h6ef0fc9dfa2fbbbdE, .Lfunc_end28-_ZN4core5slice4sort6shared9smallsort12sort8_stable17h6ef0fc9dfa2fbbbdE
	.cfi_endproc

	.section	.text._ZN4core5slice4sort6stable14driftsort_main17h1a7478f9f2f3f94aE,"ax",@progbits
	.p2align	4
	.type	_ZN4core5slice4sort6stable14driftsort_main17h1a7478f9f2f3f94aE,@function
_ZN4core5slice4sort6stable14driftsort_main17h1a7478f9f2f3f94aE:
.Lfunc_begin9:
	.cfi_startproc
	.cfi_personality 155, DW.ref.rust_eh_personality
	.cfi_lsda 27, .Lexception9
	pushq	%rbp
	.cfi_def_cfa_offset 16
	pushq	%r15
	.cfi_def_cfa_offset 24
	pushq	%r14
	.cfi_def_cfa_offset 32
	pushq	%r13
	.cfi_def_cfa_offset 40
	pushq	%r12
	.cfi_def_cfa_offset 48
	pushq	%rbx
	.cfi_def_cfa_offset 56
	subq	$4096, %rsp
	.cfi_adjust_cfa_offset 4096
	movq	$0, (%rsp)
	pushq	%rax
	.cfi_def_cfa_offset 4160
	.cfi_offset %rbx, -56
	.cfi_offset %r12, -48
	.cfi_offset %r13, -40
	.cfi_offset %r14, -32
	.cfi_offset %r15, -24
	.cfi_offset %rbp, -16
	movq	%rsi, %rcx
	shrq	%rcx
	movq	%rsi, %rax
	movl	$1000000, %r13d
	movl	$48, %ebx
	subq	%rcx, %rax
	cmpq	$1000000, %rsi
	cmovbq	%rsi, %r13
	cmpq	%rax, %r13
	cmovbeq	%rax, %r13
	cmpq	$49, %r13
	cmovaeq	%r13, %rbx
	cmpq	$513, %r13
	jb	.LBB29_1
	leaq	(,%rbx,8), %r15
	shrq	$61, %rax
	movabsq	$9223372036854775800, %rcx
	setne	%al
	cmpq	%rcx, %r15
	seta	%cl
	orb	%al, %cl
	je	.LBB29_7
	xorl	%r12d, %r12d
.LBB29_8:
	movq	%r12, %rdi
	movq	%r15, %rsi
	callq	*_RNvNtCslNYArtu3iFV_5alloc7raw_vec12handle_error@GOTPCREL(%rip)
.LBB29_1:
	leaq	8(%rsp), %r14
	movl	$512, %ecx
	jmp	.LBB29_2
.LBB29_7:
	movq	%rsi, %r14
	movq	%rdi, %rbp
	callq	*_RNvCsfLfy6EI15iL_7___rustc35___rust_no_alloc_shim_is_unstable_v2@GOTPCREL(%rip)
	movl	$8, %esi
	movq	%r15, %rdi
	movl	$8, %r12d
	callq	*_RNvCsfLfy6EI15iL_7___rustc12___rust_alloc@GOTPCREL(%rip)
	movq	%r14, %rsi
	movq	%rbp, %rdi
	movq	%rax, %r14
	movq	%rbx, %rcx
	testq	%rax, %rax
	je	.LBB29_8
.LBB29_2:
	xorl	%r8d, %r8d
	cmpq	$65, %rsi
	setb	%r8b
.Ltmp247:
	movq	%r14, %rdx
	callq	_ZN4core5slice4sort6stable5drift4sort17hefe6c3538f6cd592E
.Ltmp248:
	cmpq	$512, %r13
	jbe	.LBB29_4
	shlq	$3, %rbx
	movl	$8, %edx
	movq	%r14, %rdi
	movq	%rbx, %rsi
	callq	*_RNvCsfLfy6EI15iL_7___rustc14___rust_dealloc@GOTPCREL(%rip)
.LBB29_4:
	addq	$4104, %rsp
	.cfi_def_cfa_offset 56
	popq	%rbx
	.cfi_def_cfa_offset 48
	popq	%r12
	.cfi_def_cfa_offset 40
	popq	%r13
	.cfi_def_cfa_offset 32
	popq	%r14
	.cfi_def_cfa_offset 24
	popq	%r15
	.cfi_def_cfa_offset 16
	popq	%rbp
	.cfi_def_cfa_offset 8
	retq
.LBB29_10:
	.cfi_def_cfa_offset 4160
.Ltmp249:
	movq	%rax, %r15
	cmpq	$512, %r13
	jbe	.LBB29_12
	shlq	$3, %rbx
	movl	$8, %edx
	movq	%r14, %rdi
	movq	%rbx, %rsi
	callq	*_RNvCsfLfy6EI15iL_7___rustc14___rust_dealloc@GOTPCREL(%rip)
.LBB29_12:
	movq	%r15, %rdi
	callq	_Unwind_Resume@PLT
.Lfunc_end29:
	.size	_ZN4core5slice4sort6stable14driftsort_main17h1a7478f9f2f3f94aE, .Lfunc_end29-_ZN4core5slice4sort6stable14driftsort_main17h1a7478f9f2f3f94aE
	.cfi_endproc
	.section	.gcc_except_table._ZN4core5slice4sort6stable14driftsort_main17h1a7478f9f2f3f94aE,"a",@progbits
	.p2align	2, 0x0
GCC_except_table29:
.Lexception9:
	.byte	255
	.byte	255
	.byte	1
	.uleb128 .Lcst_end9-.Lcst_begin9
.Lcst_begin9:
	.uleb128 .Lfunc_begin9-.Lfunc_begin9
	.uleb128 .Ltmp247-.Lfunc_begin9
	.byte	0
	.byte	0
	.uleb128 .Ltmp247-.Lfunc_begin9
	.uleb128 .Ltmp248-.Ltmp247
	.uleb128 .Ltmp249-.Lfunc_begin9
	.byte	0
	.uleb128 .Ltmp248-.Lfunc_begin9
	.uleb128 .Lfunc_end29-.Ltmp248
	.byte	0
	.byte	0
.Lcst_end9:
	.p2align	2, 0x0

	.section	.text._ZN4core5slice4sort6stable5drift4sort17hefe6c3538f6cd592E,"ax",@progbits
	.p2align	4
	.type	_ZN4core5slice4sort6stable5drift4sort17hefe6c3538f6cd592E,@function
_ZN4core5slice4sort6stable5drift4sort17hefe6c3538f6cd592E:
	.cfi_startproc
	pushq	%rbp
	.cfi_def_cfa_offset 16
	pushq	%r15
	.cfi_def_cfa_offset 24
	pushq	%r14
	.cfi_def_cfa_offset 32
	pushq	%r13
	.cfi_def_cfa_offset 40
	pushq	%r12
	.cfi_def_cfa_offset 48
	pushq	%rbx
	.cfi_def_cfa_offset 56
	subq	$728, %rsp
	.cfi_def_cfa_offset 784
	.cfi_offset %rbx, -56
	.cfi_offset %r12, -48
	.cfi_offset %r13, -40
	.cfi_offset %r14, -32
	.cfi_offset %r15, -24
	.cfi_offset %rbp, -16
	movabsq	$4611686018427387904, %rax
	movq	%rdx, %r15
	movl	%r8d, 68(%rsp)
	xorl	%edx, %edx
	divq	%rsi
	movq	%rdi, 16(%rsp)
	cmpq	$1, %rdx
	sbbq	$-1, %rax
	movq	%rax, 104(%rsp)
	movq	%rcx, 24(%rsp)
	movq	%rsi, 32(%rsp)
	cmpq	$4097, %rsi
	jae	.LBB30_1
	movq	%rsi, %rax
	shrq	%rax
	movq	%rsi, %rdx
	subq	%rax, %rdx
	movl	$64, %eax
	cmpq	$64, %rdx
	cmovbq	%rdx, %rax
	jmp	.LBB30_3
.LBB30_1:
	movq	%rsi, %rdi
	callq	*_RNvNtNtNtNtCsgEmfK2I1SDS_4core5slice4sort6stable5drift11sqrt_approx@GOTPCREL(%rip)
	movq	24(%rsp), %rcx
.LBB30_3:
	movq	%rax, (%rsp)
	movl	$1, %r14d
	xorl	%r13d, %r13d
	xorl	%r11d, %r11d
	movq	16(%rsp), %rax
	leaq	32(%rax), %rdx
	leaq	-32(%rax), %rsi
	addq	$-8, %rax
	movq	%rdx, 96(%rsp)
	movq	%rsi, 88(%rsp)
	movq	%rax, 72(%rsp)
	movq	%r15, 48(%rsp)
	.p2align	4
.LBB30_4:
	movq	16(%rsp), %rax
	movq	32(%rsp), %r10
	leaq	(%rax,%r13,8), %rax
	subq	%r13, %r10
	movq	%rax, 8(%rsp)
	jbe	.LBB30_5
	cmpq	(%rsp), %r10
	jae	.LBB30_23
.LBB30_21:
	cmpb	$0, 68(%rsp)
	je	.LBB30_22
	movq	8(%rsp), %rdi
	cmpq	$32, %r10
	movl	$32, %eax
	movq	%r15, %rdx
	movq	%r11, %rbx
	cmovaeq	%rax, %r10
	xorl	%r8d, %r8d
	xorl	%r9d, %r9d
	movq	%r10, %rsi
	movq	%r10, %r12
	vzeroupper
	callq	_ZN4core5slice4sort6stable9quicksort9quicksort17hd9000d69f302316eE
	movq	24(%rsp), %rcx
	movq	%rbx, %r11
	movl	%r12d, %eax
	jmp	.LBB30_43
	.p2align	4
.LBB30_5:
	movl	$1, %r10d
	xorl	%esi, %esi
	cmpq	$2, %r11
	jae	.LBB30_7
	jmp	.LBB30_61
	.p2align	4
.LBB30_23:
	cmpq	$2, %r10
	jae	.LBB30_25
	movq	%r10, %rax
.LBB30_43:
	leaq	1(%rax,%rax), %r10
	jmp	.LBB30_44
.LBB30_22:
	movq	(%rsp), %rax
	cmpq	%rax, %r10
	cmovaeq	%rax, %r10
	addq	%r10, %r10
.LBB30_44:
	movq	104(%rsp), %rdi
	movq	%r10, %rsi
	shrq	%rsi
	movq	%r14, %rax
	shrq	%rax
	leaq	(%r13,%r13), %rdx
	leaq	(%rsi,%r13,2), %r8
	subq	%rax, %rdx
	imulq	%rdi, %rdx
	imulq	%rdi, %r8
	xorq	%rdx, %r8
	lzcntq	%r8, %rsi
	cmpq	$2, %r11
	jb	.LBB30_61
.LBB30_7:
	movq	72(%rsp), %rax
	leaq	(%rax,%r13,8), %rax
	movq	%rax, 112(%rsp)
	movq	%r10, 40(%rsp)
	movq	%r13, 120(%rsp)
	movq	%rsi, 80(%rsp)
	jmp	.LBB30_8
.LBB30_53:
	movq	%r14, %r13
	movq	%r11, %rsi
.LBB30_54:
	subq	%rsi, %rdx
	movq	%r13, %rdi
	callq	*memcpy@GOTPCREL(%rip)
	movq	24(%rsp), %rcx
	movq	56(%rsp), %r11
	movq	40(%rsp), %r10
	movq	80(%rsp), %rsi
.LBB30_55:
	movq	48(%rsp), %r15
	movq	120(%rsp), %r13
	leaq	1(%rbx,%rbx), %r14
	cmpq	$1, %r11
	jbe	.LBB30_60
.LBB30_8:
	cmpb	%sil, 133(%rsp,%r11)
	jb	.LBB30_61
	decq	%r11
	movq	%r14, %rbp
	shrq	%rbp
	movq	200(%rsp,%r11,8), %rax
	movq	%rax, %r12
	shrq	%r12
	leaq	(%r12,%rbp), %rbx
	cmpq	%rcx, %rbx
	ja	.LBB30_11
	movl	%eax, %edx
	orl	%r14d, %edx
	andl	$1, %edx
	jne	.LBB30_11
	addq	%rbx, %rbx
	movq	%rbx, %r14
	cmpq	$1, %r11
	ja	.LBB30_8
	jmp	.LBB30_60
	.p2align	4
.LBB30_11:
	movq	16(%rsp), %rsi
	movq	%r13, %rdx
	subq	%rbx, %rdx
	movq	%r11, 56(%rsp)
	leaq	(%rsi,%rdx,8), %r13
	testb	$1, %al
	je	.LBB30_50
	testb	$1, %r14b
	je	.LBB30_57
.LBB30_13:
	movq	80(%rsp), %rsi
	testq	%rbp, %rbp
	jne	.LBB30_14
	jmp	.LBB30_55
	.p2align	4
.LBB30_50:
	movq	%r12, %rax
	orq	$1, %rax
	movq	%r13, %rdi
	movq	%r12, %rsi
	movq	%r15, %rdx
	xorl	%r9d, %r9d
	lzcntq	%rax, %r8
	addl	%r8d, %r8d
	xorl	$126, %r8d
	vzeroupper
	callq	_ZN4core5slice4sort6stable9quicksort9quicksort17hd9000d69f302316eE
	movq	40(%rsp), %r10
	movq	56(%rsp), %r11
	movq	24(%rsp), %rcx
	testb	$1, %r14b
	jne	.LBB30_13
.LBB30_57:
	movq	%rbp, %rax
	orq	$1, %rax
	leaq	(%r13,%r12,8), %rdi
	movq	%rbp, %rsi
	movq	%r15, %rdx
	xorl	%r9d, %r9d
	lzcntq	%rax, %r8
	addl	%r8d, %r8d
	xorl	$126, %r8d
	vzeroupper
	callq	_ZN4core5slice4sort6stable9quicksort9quicksort17hd9000d69f302316eE
	movq	40(%rsp), %r10
	movq	56(%rsp), %r11
	movq	24(%rsp), %rcx
	movq	80(%rsp), %rsi
	testq	%rbp, %rbp
	je	.LBB30_55
.LBB30_14:
	testq	%r12, %r12
	je	.LBB30_55
	cmpq	%r12, %rbp
	movq	%r12, %r15
	cmovbq	%rbp, %r15
	cmpq	%r15, %rcx
	jb	.LBB30_55
	movq	48(%rsp), %rdi
	leaq	(%r13,%r12,8), %r14
	cmpq	%rbp, %r12
	leaq	(,%r15,8), %rdx
	movq	%r13, %rsi
	cmovaq	%r14, %rsi
	vzeroupper
	callq	*memcpy@GOTPCREL(%rip)
	movq	48(%rsp), %r11
	leaq	(%r11,%r15,8), %rdx
	cmpq	%rbp, %r12
	jbe	.LBB30_17
	movq	112(%rsp), %rax
	.p2align	4
.LBB30_52:
	movq	-8(%rdx), %rcx
	movq	-8(%r14), %rsi
	xorl	%r9d, %r9d
	xorl	%r10d, %r10d
	movq	%rcx, %rdi
	sarq	$63, %rdi
	movq	%rsi, %r8
	sarq	$63, %r8
	shrq	%rdi
	shrq	%r8
	xorq	%rcx, %rdi
	xorq	%rsi, %r8
	cmpq	%r8, %rdi
	setl	%r9b
	setge	%r10b
	cmovlq	%rsi, %rcx
	leaq	-8(%r14,%r10,8), %r14
	leaq	-8(%rdx,%r9,8), %rdx
	movq	%rcx, (%rax)
	cmpq	%r13, %r14
	je	.LBB30_53
	addq	$-8, %rax
	cmpq	%r11, %rdx
	jne	.LBB30_52
	jmp	.LBB30_53
.LBB30_17:
	movq	%r11, %rsi
	movq	8(%rsp), %r11
	.p2align	4
.LBB30_18:
	movq	(%r14), %rdi
	movq	(%rsi), %r8
	xorl	%eax, %eax
	xorl	%r10d, %r10d
	movq	%rdi, %rcx
	sarq	$63, %rcx
	movq	%r8, %r9
	sarq	$63, %r9
	shrq	%rcx
	shrq	%r9
	xorq	%rdi, %rcx
	xorq	%r8, %r9
	cmpq	%r9, %rcx
	setge	%r10b
	cmovlq	%rdi, %r8
	setl	%cl
	leaq	(%rsi,%r10,8), %rsi
	movq	%r8, (%r13)
	addq	$8, %r13
	cmpq	%rdx, %rsi
	je	.LBB30_54
	movb	%cl, %al
	leaq	(%r14,%rax,8), %r14
	cmpq	%r11, %r14
	jne	.LBB30_18
	jmp	.LBB30_54
	.p2align	4
.LBB30_60:
	movl	$1, %r11d
.LBB30_61:
	movq	%r14, 200(%rsp,%r11,8)
	movb	%sil, 134(%rsp,%r11)
	cmpq	%r13, 32(%rsp)
	jbe	.LBB30_63
	movq	%r10, %rax
	shrq	%rax
	incq	%r11
	movq	%r10, %r14
	addq	%rax, %r13
	jmp	.LBB30_4
.LBB30_25:
	movq	8(%rsp), %rdx
	movq	8(%rdx), %rsi
	movq	(%rdx), %rax
	movq	%rsi, %r9
	sarq	$63, %r9
	movq	%rax, %rdx
	sarq	$63, %rdx
	shrq	%r9
	shrq	%rdx
	xorq	%rsi, %r9
	xorq	%rax, %rdx
	cmpq	%rdx, %r9
	jge	.LBB30_26
	cmpq	$2, %r10
	jne	.LBB30_30
	cmpq	$2, (%rsp)
	ja	.LBB30_21
	movl	$2, %eax
	movl	$1, %r10d
	jmp	.LBB30_37
.LBB30_26:
	movl	$2, %eax
	cmpq	$2, %r10
	jne	.LBB30_27
	cmpq	$2, (%rsp)
	ja	.LBB30_21
	jmp	.LBB30_43
	.p2align	4
.LBB30_27:
	movq	8(%rsp), %r8
	movq	%rsi, %rdi
	sarq	$63, %rdi
	shrq	%rdi
	xorq	%rsi, %rdi
	movq	(%r8,%rax,8), %rsi
	movq	%rsi, %r8
	sarq	$63, %r8
	shrq	%r8
	xorq	%rsi, %r8
	cmpq	%rdi, %r8
	jl	.LBB30_34
	incq	%rax
	cmpq	%rax, %r10
	jne	.LBB30_27
	jmp	.LBB30_33
.LBB30_30:
	movl	$2, %eax
	.p2align	4
.LBB30_31:
	movq	8(%rsp), %r8
	movq	%rsi, %rdi
	sarq	$63, %rdi
	shrq	%rdi
	xorq	%rsi, %rdi
	movq	(%r8,%rax,8), %rsi
	movq	%rsi, %r8
	sarq	$63, %r8
	shrq	%r8
	xorq	%rsi, %r8
	cmpq	%rdi, %r8
	jge	.LBB30_34
	incq	%rax
	cmpq	%rax, %r10
	jne	.LBB30_31
.LBB30_33:
	movq	%r10, %rax
.LBB30_34:
	cmpq	(%rsp), %rax
	jb	.LBB30_21
	cmpq	%rdx, %r9
	jge	.LBB30_43
	movq	%rax, %r10
	shrq	%r10
	je	.LBB30_43
.LBB30_37:
	cmpq	$8, %r10
	jae	.LBB30_45
	xorl	%edx, %edx
	jmp	.LBB30_48
.LBB30_45:
	movq	96(%rsp), %rdi
	leaq	(,%rax,8), %r8
	movq	%r10, %rdx
	andq	$-8, %rdx
	xorl	%r9d, %r9d
	movq	%rdx, %rsi
	negq	%rsi
	leaq	(%r8,%r13,8), %r8
	addq	88(%rsp), %r8
	leaq	(%rdi,%r13,8), %rdi
	.p2align	4
.LBB30_46:
	vpermpd	$27, (%r8,%r9,8), %ymm0
	vpermpd	$27, -32(%rdi), %ymm1
	vpermpd	$27, -32(%r8,%r9,8), %ymm2
	vmovups	%ymm0, -32(%rdi)
	vpermpd	$27, (%rdi), %ymm0
	vmovups	%ymm2, (%rdi)
	vmovups	%ymm1, (%r8,%r9,8)
	addq	$64, %rdi
	vmovups	%ymm0, -32(%r8,%r9,8)
	addq	$-8, %r9
	cmpq	%r9, %rsi
	jne	.LBB30_46
	cmpq	%rdx, %r10
	je	.LBB30_43
.LBB30_48:
	leaq	(,%r13,8), %rdi
	movq	%rdx, %rsi
	shlq	$3, %rdx
	negq	%r10
	negq	%rsi
	leaq	(%rdx,%r13,8), %rdx
	addq	16(%rsp), %rdx
	leaq	(%rdi,%rax,8), %rdi
	addq	72(%rsp), %rdi
	.p2align	4
.LBB30_49:
	vmovsd	(%rdx), %xmm0
	movq	(%rdi,%rsi,8), %r8
	movq	%r8, (%rdx)
	addq	$8, %rdx
	vmovsd	%xmm0, (%rdi,%rsi,8)
	decq	%rsi
	cmpq	%rsi, %r10
	jne	.LBB30_49
	jmp	.LBB30_43
.LBB30_63:
	testb	$1, %r14b
	jne	.LBB30_64
	movq	32(%rsp), %rsi
	movq	16(%rsp), %rdi
	movq	%r15, %rdx
	xorl	%r9d, %r9d
	movq	%rsi, %rax
	orq	$1, %rax
	lzcntq	%rax, %r8
	addl	%r8d, %r8d
	xorl	$126, %r8d
	addq	$728, %rsp
	.cfi_def_cfa_offset 56
	popq	%rbx
	.cfi_def_cfa_offset 48
	popq	%r12
	.cfi_def_cfa_offset 40
	popq	%r13
	.cfi_def_cfa_offset 32
	popq	%r14
	.cfi_def_cfa_offset 24
	popq	%r15
	.cfi_def_cfa_offset 16
	popq	%rbp
	.cfi_def_cfa_offset 8
	vzeroupper
	jmp	_ZN4core5slice4sort6stable9quicksort9quicksort17hd9000d69f302316eE
.LBB30_64:
	.cfi_def_cfa_offset 784
	addq	$728, %rsp
	.cfi_def_cfa_offset 56
	popq	%rbx
	.cfi_def_cfa_offset 48
	popq	%r12
	.cfi_def_cfa_offset 40
	popq	%r13
	.cfi_def_cfa_offset 32
	popq	%r14
	.cfi_def_cfa_offset 24
	popq	%r15
	.cfi_def_cfa_offset 16
	popq	%rbp
	.cfi_def_cfa_offset 8
	vzeroupper
	retq
.Lfunc_end30:
	.size	_ZN4core5slice4sort6stable5drift4sort17hefe6c3538f6cd592E, .Lfunc_end30-_ZN4core5slice4sort6stable5drift4sort17hefe6c3538f6cd592E
	.cfi_endproc

	.section	.text._ZN4core5slice4sort6stable9quicksort9quicksort17hd9000d69f302316eE,"ax",@progbits
	.p2align	4
	.type	_ZN4core5slice4sort6stable9quicksort9quicksort17hd9000d69f302316eE,@function
_ZN4core5slice4sort6stable9quicksort9quicksort17hd9000d69f302316eE:
.Lfunc_begin10:
	.cfi_startproc
	.cfi_personality 155, DW.ref.rust_eh_personality
	.cfi_lsda 27, .Lexception10
	pushq	%rbp
	.cfi_def_cfa_offset 16
	pushq	%r15
	.cfi_def_cfa_offset 24
	pushq	%r14
	.cfi_def_cfa_offset 32
	pushq	%r13
	.cfi_def_cfa_offset 40
	pushq	%r12
	.cfi_def_cfa_offset 48
	pushq	%rbx
	.cfi_def_cfa_offset 56
	subq	$72, %rsp
	.cfi_def_cfa_offset 128
	.cfi_offset %rbx, -56
	.cfi_offset %r12, -48
	.cfi_offset %r13, -40
	.cfi_offset %r14, -32
	.cfi_offset %r15, -24
	.cfi_offset %rbp, -16
	movq	%rcx, 8(%rsp)
	movq	%rdx, %rbx
	movq	%rsi, %r14
	movq	%rdi, %r13
	cmpq	$33, %rsi
	jae	.LBB31_7
.LBB31_1:
	movq	%r13, %rbp
.LBB31_2:
	cmpq	$2, %r14
	jb	.LBB31_11
	leaq	16(%r14), %rax
	cmpq	%rax, 8(%rsp)
	jb	.LBB31_50
	movq	%r14, %r13
	shrq	%r13
	cmpq	$15, %r14
	jbe	.LBB31_5
	leaq	(%rbx,%r14,8), %rdx
	movq	%rbp, %rdi
	movq	%rbx, %rsi
	vzeroupper
	callq	_ZN4core5slice4sort6shared9smallsort12sort8_stable17h6ef0fc9dfa2fbbbdE
	leaq	(%rbp,%r13,8), %rdi
	leaq	(%rbx,%r13,8), %rsi
	leaq	64(%rbx,%r14,8), %rdx
	callq	_ZN4core5slice4sort6shared9smallsort12sort8_stable17h6ef0fc9dfa2fbbbdE
	movl	$8, %ecx
	movq	%r14, %rdx
	subq	%r13, %rdx
	cmpq	%rcx, %r13
	ja	.LBB31_15
	jmp	.LBB31_30
.LBB31_7:
	movq	%r9, %r12
	movl	%r8d, 24(%rsp)
.LBB31_8:
	movl	24(%rsp), %ebp
	movq	%r14, %r15
	.p2align	4
.LBB31_9:
	testl	%ebp, %ebp
	je	.LBB31_10
	movq	%r15, %rcx
	shrq	$3, %rcx
	imulq	$56, %rcx, %rdx
	movq	%rcx, %rax
	shlq	$5, %rax
	addq	%r13, %rax
	addq	%r13, %rdx
	cmpq	$64, %r15
	jae	.LBB31_66
	movq	(%r13), %rcx
	movq	(%rax), %rsi
	movq	(%rdx), %r8
	movq	%rcx, %rdi
	sarq	$63, %rdi
	movq	%r8, %r9
	shrq	%rdi
	xorq	%rcx, %rdi
	movq	%rsi, %rcx
	sarq	$63, %rcx
	shrq	%rcx
	xorq	%rsi, %rcx
	cmpq	%rcx, %rdi
	setl	%sil
	sarq	$63, %r9
	shrq	%r9
	xorq	%r8, %r9
	cmpq	%r9, %rdi
	setl	%dil
	xorb	%sil, %dil
	cmpq	%r9, %rcx
	setl	%cl
	xorb	%sil, %cl
	cmovneq	%rdx, %rax
	testb	%dil, %dil
	cmovneq	%r13, %rax
	jmp	.LBB31_68
	.p2align	4
.LBB31_66:
	movq	%r13, %rdi
	movq	%rax, %rsi
	vzeroupper
	callq	_ZN4core5slice4sort6shared5pivot11median3_rec17hde37ea35567e1fabE
.LBB31_68:
	vmovq	(%rax), %xmm0
	decl	%ebp
	movl	%ebp, 24(%rsp)
	movq	%rax, %rbp
	subq	%r13, %rbp
	movq	%rbp, %rcx
	shrq	$3, %rcx
	movq	%rcx, 40(%rsp)
	vmovq	%xmm0, 64(%rsp)
	testq	%r12, %r12
	je	.LBB31_70
	movq	(%r12), %rcx
	vmovq	%xmm0, %rax
	movq	%rax, %rsi
	sarq	$63, %rsi
	shrq	%rsi
	xorq	%rax, %rsi
	movq	%rcx, %rdx
	sarq	$63, %rdx
	shrq	%rdx
	xorq	%rcx, %rdx
	cmpq	%rsi, %rdx
	jge	.LBB31_97
.LBB31_70:
	cmpq	%r15, 8(%rsp)
	jb	.LBB31_50
	movq	40(%rsp), %rdx
	leaq	(%rbx,%r15,8), %rax
	xorl	%r14d, %r14d
	movq	%r13, %rcx
	.p2align	4
.LBB31_72:
	movq	%rdx, %rsi
	subq	$3, %rsi
	movl	$0, %edi
	cmovaeq	%rsi, %rdi
	leaq	(%r13,%rdi,8), %rsi
	cmpq	%rsi, %rcx
	jae	.LBB31_75
	movq	(%r13,%rbp), %r8
	movq	%r8, %rdi
	sarq	$63, %rdi
	shrq	%rdi
	xorq	%r8, %rdi
	.p2align	4
.LBB31_74:
	movq	(%rcx), %r8
	leaq	-8(%rax), %r10
	xorl	%r11d, %r11d
	movq	%r8, %r9
	sarq	$63, %r9
	shrq	%r9
	xorq	%r8, %r9
	cmpq	%rdi, %r9
	cmovlq	%rbx, %r10
	setl	%r11b
	movq	%r8, (%r10,%r14,8)
	movq	8(%rcx), %r8
	addq	%r14, %r11
	leaq	-16(%rax), %r10
	xorl	%r14d, %r14d
	movq	%r8, %r9
	sarq	$63, %r9
	shrq	%r9
	xorq	%r8, %r9
	cmpq	%rdi, %r9
	cmovlq	%rbx, %r10
	setl	%r14b
	movq	%r8, (%r10,%r11,8)
	movq	16(%rcx), %r8
	addq	%r11, %r14
	leaq	-24(%rax), %r10
	xorl	%r11d, %r11d
	movq	%r8, %r9
	sarq	$63, %r9
	shrq	%r9
	xorq	%r8, %r9
	cmpq	%rdi, %r9
	cmovlq	%rbx, %r10
	setl	%r11b
	addq	$-32, %rax
	movq	%r8, (%r10,%r14,8)
	movq	24(%rcx), %r10
	addq	%r14, %r11
	xorl	%r14d, %r14d
	movq	%r10, %r9
	sarq	$63, %r9
	shrq	%r9
	xorq	%r10, %r9
	cmpq	%rdi, %r9
	movq	%rax, %r9
	setl	%r14b
	cmovlq	%rbx, %r9
	addq	$32, %rcx
	addq	%r11, %r14
	movq	%r10, (%r9,%r11,8)
	cmpq	%rsi, %rcx
	jb	.LBB31_74
.LBB31_75:
	leaq	(%r13,%rdx,8), %rsi
	cmpq	%rsi, %rcx
	jae	.LBB31_78
	movq	(%r13,%rbp), %r8
	movq	%r8, %rdi
	sarq	$63, %rdi
	shrq	%rdi
	xorq	%r8, %rdi
	.p2align	4
.LBB31_77:
	movq	(%rcx), %r9
	movq	%r14, %r8
	addq	$-8, %rax
	xorl	%r14d, %r14d
	movq	%r9, %r10
	sarq	$63, %r10
	shrq	%r10
	xorq	%r9, %r10
	cmpq	%rdi, %r10
	movq	%rax, %r10
	setl	%r14b
	cmovlq	%rbx, %r10
	addq	$8, %rcx
	addq	%r8, %r14
	movq	%r9, (%r10,%r8,8)
	cmpq	%rsi, %rcx
	jb	.LBB31_77
.LBB31_78:
	cmpq	%r15, %rdx
	je	.LBB31_80
	movq	(%rcx), %rdx
	addq	$8, %rcx
	movq	%rdx, -8(%rax,%r14,8)
	addq	$-8, %rax
	movq	%r15, %rdx
	jmp	.LBB31_72
	.p2align	4
.LBB31_80:
	leaq	(,%r14,8), %rdx
	movq	%r13, %rdi
	movq	%rbx, %rsi
	vzeroupper
	callq	*memcpy@GOTPCREL(%rip)
	movq	%r15, %rsi
	subq	%r14, %rsi
	je	.LBB31_94
	cmpq	$4, %rsi
	jae	.LBB31_83
	xorl	%eax, %eax
	jmp	.LBB31_92
	.p2align	4
.LBB31_83:
	cmpq	$16, %rsi
	jae	.LBB31_85
	xorl	%eax, %eax
	jmp	.LBB31_89
.LBB31_85:
	leaq	96(%r13), %rdx
	leaq	-32(%rbx), %rdi
	movq	%rsi, %rax
	andq	$-16, %rax
	xorl	%r8d, %r8d
	leaq	(%rdx,%r14,8), %rdx
	leaq	(%rdi,%r15,8), %rdi
	movq	%rax, %rcx
	negq	%rcx
	.p2align	4
.LBB31_86:
	vpermq	$27, (%rdi,%r8,8), %ymm0
	vpermq	$27, -32(%rdi,%r8,8), %ymm1
	vpermq	$27, -64(%rdi,%r8,8), %ymm2
	vpermq	$27, -96(%rdi,%r8,8), %ymm3
	addq	$-16, %r8
	vmovdqu	%ymm0, -96(%rdx)
	vmovdqu	%ymm1, -64(%rdx)
	vmovdqu	%ymm2, -32(%rdx)
	vmovdqu	%ymm3, (%rdx)
	subq	$-128, %rdx
	cmpq	%r8, %rcx
	jne	.LBB31_86
	cmpq	%rax, %rsi
	je	.LBB31_94
	testb	$12, %sil
	je	.LBB31_92
.LBB31_89:
	leaq	-32(%rbx), %rdi
	movq	%rax, %rcx
	leaq	(,%rcx,8), %r8
	leaq	(%r13,%r14,8), %rdx
	movq	%rsi, %rax
	andq	$-4, %rax
	leaq	(%rdi,%r15,8), %rdi
	subq	%r8, %rdi
	.p2align	4
.LBB31_90:
	vpermq	$27, (%rdi), %ymm0
	addq	$-32, %rdi
	vmovdqu	%ymm0, (%rdx,%rcx,8)
	addq	$4, %rcx
	cmpq	%rcx, %rax
	jne	.LBB31_90
	cmpq	%rax, %rsi
	je	.LBB31_94
.LBB31_92:
	leaq	-8(%rbx), %rdx
	leaq	(%rax,%r14), %rcx
	shlq	$3, %rax
	leaq	(%rdx,%r15,8), %rdx
	subq	%rax, %rdx
	.p2align	4
.LBB31_93:
	movq	(%rdx), %rax
	addq	$-8, %rdx
	movq	%rax, (%r13,%rcx,8)
	incq	%rcx
	cmpq	%rcx, %r15
	jne	.LBB31_93
.LBB31_94:
	testq	%r14, %r14
	je	.LBB31_97
	cmpq	%r14, %r15
	jb	.LBB31_124
	movl	24(%rsp), %ebp
	movq	8(%rsp), %rcx
	leaq	(%r13,%r14,8), %rdi
	leaq	64(%rsp), %r9
	movq	%rbx, %rdx
	movl	%ebp, %r8d
	vzeroupper
	callq	_ZN4core5slice4sort6stable9quicksort9quicksort17hd9000d69f302316eE
	movq	%r14, %r15
	cmpq	$33, %r14
	jae	.LBB31_9
	jmp	.LBB31_1
	.p2align	4
.LBB31_97:
	cmpq	%r15, 8(%rsp)
	jb	.LBB31_50
	movq	40(%rsp), %r14
	leaq	(%rbx,%r15,8), %rax
	xorl	%r12d, %r12d
	movq	%r13, %rcx
	.p2align	4
.LBB31_99:
	movq	%r14, %rdx
	subq	$3, %rdx
	movl	$0, %esi
	cmovaeq	%rdx, %rsi
	leaq	(%r13,%rsi,8), %rdx
	cmpq	%rdx, %rcx
	jae	.LBB31_102
	movq	(%r13,%rbp), %rdi
	movq	%rdi, %rsi
	sarq	$63, %rsi
	shrq	%rsi
	xorq	%rdi, %rsi
	.p2align	4
.LBB31_101:
	movq	(%rcx), %rdi
	leaq	-8(%rax), %r9
	xorl	%r10d, %r10d
	movq	%rdi, %r8
	sarq	$63, %r8
	shrq	%r8
	xorq	%rdi, %r8
	cmpq	%r8, %rsi
	cmovgeq	%rbx, %r9
	setge	%r10b
	xorl	%r11d, %r11d
	movq	%rdi, (%r9,%r12,8)
	movq	8(%rcx), %rdi
	addq	%r12, %r10
	leaq	-16(%rax), %r9
	movq	%rdi, %r8
	sarq	$63, %r8
	shrq	%r8
	xorq	%rdi, %r8
	cmpq	%r8, %rsi
	cmovgeq	%rbx, %r9
	setge	%r11b
	movq	%rdi, (%r9,%r10,8)
	movq	16(%rcx), %rdi
	addq	%r10, %r11
	leaq	-24(%rax), %r9
	xorl	%r10d, %r10d
	movq	%rdi, %r8
	sarq	$63, %r8
	shrq	%r8
	xorq	%rdi, %r8
	cmpq	%r8, %rsi
	cmovgeq	%rbx, %r9
	setge	%r10b
	addq	$-32, %rax
	xorl	%r12d, %r12d
	movq	%rdi, (%r9,%r11,8)
	movq	24(%rcx), %r9
	addq	%r11, %r10
	movq	%r9, %r8
	sarq	$63, %r8
	shrq	%r8
	xorq	%r9, %r8
	cmpq	%r8, %rsi
	movq	%rax, %r8
	setge	%r12b
	cmovgeq	%rbx, %r8
	addq	$32, %rcx
	addq	%r10, %r12
	movq	%r9, (%r8,%r10,8)
	cmpq	%rdx, %rcx
	jb	.LBB31_101
.LBB31_102:
	leaq	(%r13,%r14,8), %rdx
	cmpq	%rdx, %rcx
	jae	.LBB31_105
	movq	(%r13,%rbp), %rdi
	movq	%rdi, %rsi
	sarq	$63, %rsi
	shrq	%rsi
	xorq	%rdi, %rsi
	.p2align	4
.LBB31_104:
	movq	(%rcx), %r8
	movq	%r12, %rdi
	addq	$-8, %rax
	xorl	%r12d, %r12d
	movq	%r8, %r9
	sarq	$63, %r9
	shrq	%r9
	xorq	%r8, %r9
	cmpq	%r9, %rsi
	movq	%rax, %r9
	setge	%r12b
	cmovgeq	%rbx, %r9
	addq	$8, %rcx
	addq	%rdi, %r12
	movq	%r8, (%r9,%rdi,8)
	cmpq	%rdx, %rcx
	jb	.LBB31_104
.LBB31_105:
	cmpq	%r15, %r14
	je	.LBB31_107
	movq	(%rcx), %rdx
	addq	$-8, %rax
	addq	$8, %rcx
	movq	%r15, %r14
	movq	%rdx, (%rbx,%r12,8)
	incq	%r12
	jmp	.LBB31_99
	.p2align	4
.LBB31_107:
	leaq	(,%r12,8), %rdx
	movq	%r13, %rdi
	movq	%rbx, %rsi
	vzeroupper
	callq	*memcpy@GOTPCREL(%rip)
	movq	%r15, %r14
	subq	%r12, %r14
	je	.LBB31_11
	leaq	(%r13,%r12,8), %rbp
	cmpq	$4, %r14
	jae	.LBB31_110
	xorl	%eax, %eax
	jmp	.LBB31_119
.LBB31_110:
	cmpq	$16, %r14
	jae	.LBB31_112
	xorl	%eax, %eax
	jmp	.LBB31_116
.LBB31_112:
	leaq	-32(%rbx), %rsi
	movq	%r14, %rax
	andq	$-16, %rax
	leaq	96(%r13,%r12,8), %rdx
	xorl	%edi, %edi
	leaq	(%rsi,%r15,8), %rsi
	movq	%rax, %rcx
	negq	%rcx
	.p2align	4
.LBB31_113:
	vpermq	$27, (%rsi,%rdi,8), %ymm0
	vpermq	$27, -32(%rsi,%rdi,8), %ymm1
	vpermq	$27, -64(%rsi,%rdi,8), %ymm2
	vpermq	$27, -96(%rsi,%rdi,8), %ymm3
	addq	$-16, %rdi
	vmovdqu	%ymm0, -96(%rdx)
	vmovdqu	%ymm1, -64(%rdx)
	vmovdqu	%ymm2, -32(%rdx)
	vmovdqu	%ymm3, (%rdx)
	subq	$-128, %rdx
	cmpq	%rdi, %rcx
	jne	.LBB31_113
	cmpq	%rax, %r14
	je	.LBB31_121
	testb	$12, %r14b
	je	.LBB31_119
.LBB31_116:
	movq	%rax, %rcx
	leaq	(,%rcx,8), %rdx
	leaq	-32(%rbx), %rsi
	movq	%r14, %rax
	andq	$-4, %rax
	subq	%rdx, %rsi
	leaq	(%rsi,%r15,8), %rdx
	.p2align	4
.LBB31_117:
	vpermq	$27, (%rdx), %ymm0
	addq	$-32, %rdx
	vmovdqu	%ymm0, (%rbp,%rcx,8)
	addq	$4, %rcx
	cmpq	%rcx, %rax
	jne	.LBB31_117
	cmpq	%rax, %r14
	je	.LBB31_121
.LBB31_119:
	leaq	(%rax,%r12), %rcx
	shlq	$3, %rax
	leaq	-8(%rbx), %rdx
	subq	%rax, %rdx
	leaq	(%rdx,%r15,8), %rax
	.p2align	4
.LBB31_120:
	movq	(%rax), %rdx
	addq	$-8, %rax
	movq	%rdx, (%r13,%rcx,8)
	incq	%rcx
	cmpq	%rcx, %r15
	jne	.LBB31_120
.LBB31_121:
	cmpq	%r12, %r15
	jb	.LBB31_123
	movl	$0, %r12d
	movq	%rbp, %r13
	cmpq	$33, %r14
	jae	.LBB31_8
	jmp	.LBB31_2
.LBB31_10:
	movq	8(%rsp), %rcx
	movl	$1, %r8d
	movq	%r13, %rdi
	movq	%r15, %rsi
	movq	%rbx, %rdx
	vzeroupper
	callq	_ZN4core5slice4sort6stable5drift4sort17hefe6c3538f6cd592E
	jmp	.LBB31_11
.LBB31_5:
	cmpq	$7, %r14
	jbe	.LBB31_13
	vmovdqu	(%rbp), %xmm1
	vmovdqu	16(%rbp), %xmm2
	vpxor	%xmm0, %xmm0, %xmm0
	movl	$2, %r8d
	movq	%r13, 32(%rsp)
	movq	%rbp, 16(%rsp)
	vpcmpgtq	%xmm1, %xmm0, %xmm3
	vpsrlq	$1, %xmm3, %xmm3
	vpxor	%xmm1, %xmm3, %xmm1
	vpshufd	$238, %xmm1, %xmm3
	vpcmpgtq	%xmm3, %xmm1, %xmm1
	vmovq	%xmm1, %rax
	vpcmpgtq	%xmm2, %xmm0, %xmm1
	movl	%eax, %edi
	andl	$1, %edi
	vpsrlq	$1, %xmm1, %xmm1
	notb	%al
	movq	(%rbp,%rdi,8), %rcx
	vpxor	%xmm2, %xmm1, %xmm1
	movzbl	%al, %esi
	vpshufd	$238, %xmm1, %xmm2
	andl	$1, %esi
	vpcmpgtq	%xmm2, %xmm1, %xmm1
	movq	(%rbp,%rsi,8), %r11
	vmovd	%xmm1, %r10d
	subl	%r10d, %r8d
	andl	$1, %r10d
	movq	%rcx, %rdx
	sarq	$63, %rdx
	movq	(%rbp,%r8,8), %r9
	xorl	$3, %r10d
	movq	%rcx, %r15
	shrq	%rdx
	xorq	%rcx, %rdx
	movq	(%rbp,%r10,8), %rcx
	movq	%r11, %r13
	sarq	$63, %r13
	movl	%r8d, %ebp
	shrq	%r13
	xorq	%r11, %r13
	movq	%r9, %rax
	sarq	$63, %rax
	shrq	%rax
	movq	%rcx, %r12
	sarq	$63, %r12
	xorq	%r9, %rax
	shrq	%r12
	xorq	%rcx, %r12
	cmpq	%rdx, %rax
	cmovll	%esi, %ebp
	cmpq	%r13, %r12
	cmovlq	%r11, %rcx
	cmovgel	%esi, %r8d
	cmovll	%r10d, %ebp
	cmpq	%rdx, %rax
	movq	16(%rsp), %rax
	movq	%rcx, 8(%rsp)
	cmovll	%edi, %r8d
	cmovlq	%r9, %r15
	movq	%r15, 24(%rsp)
	movq	(%rax,%rbp,8), %rcx
	movq	16(%rsp), %rax
	movq	(%rax,%r8,8), %rax
	movq	%rcx, %rdx
	sarq	$63, %rdx
	shrq	%rdx
	xorq	%rcx, %rdx
	movq	%rax, %rdi
	sarq	$63, %rdi
	shrq	%rdi
	xorq	%rax, %rdi
	cmpq	%rdi, %rdx
	movq	%rax, %rdx
	cmovlq	%rcx, %rdx
	cmovlq	%rax, %rcx
	movq	32(%rsp), %rax
	movq	%rcx, 48(%rsp)
	movq	16(%rsp), %rcx
	movq	%rdx, 40(%rsp)
	movl	$2, %edx
	leaq	(%rcx,%rax,8), %rdi
	movq	32(%rsp), %rax
	movq	16(%rsp), %rcx
	vmovdqu	(%rcx,%rax,8), %xmm1
	movq	32(%rsp), %rax
	movq	16(%rsp), %rcx
	vmovdqu	16(%rcx,%rax,8), %xmm2
	vpcmpgtq	%xmm1, %xmm0, %xmm3
	vpsrlq	$1, %xmm3, %xmm3
	vpxor	%xmm1, %xmm3, %xmm1
	vpshufd	$238, %xmm1, %xmm3
	vpcmpgtq	%xmm3, %xmm1, %xmm1
	vpcmpgtq	%xmm2, %xmm0, %xmm0
	vmovq	%xmm1, %rax
	vpsrlq	$1, %xmm0, %xmm0
	movl	%eax, %r8d
	notb	%al
	andl	$1, %r8d
	vpxor	%xmm2, %xmm0, %xmm0
	movzbl	%al, %r13d
	movq	(%rdi,%r8,8), %r15
	vpshufd	$238, %xmm0, %xmm1
	andl	$1, %r13d
	vpcmpgtq	%xmm1, %xmm0, %xmm0
	movq	(%rdi,%r13,8), %rsi
	vmovd	%xmm0, %ebp
	subl	%ebp, %edx
	andl	$1, %ebp
	movq	(%rdi,%rdx,8), %rax
	xorl	$3, %ebp
	movq	%r15, %r9
	sarq	$63, %r9
	movl	%edx, %ecx
	movq	(%rdi,%rbp,8), %r10
	shrq	%r9
	xorq	%r15, %r9
	movq	%rax, %r12
	sarq	$63, %r12
	movq	%rax, 56(%rsp)
	shrq	%r12
	movq	%r10, %r11
	sarq	$63, %r11
	xorq	%rax, %r12
	movq	%rsi, %rax
	sarq	$63, %rax
	shrq	%r11
	shrq	%rax
	xorq	%r10, %r11
	xorq	%rsi, %rax
	cmpq	%r9, %r12
	cmovll	%r13d, %ecx
	cmpq	%rax, %r11
	cmovll	%ebp, %ecx
	cmovgel	%r13d, %edx
	cmovlq	%rsi, %r10
	cmpq	%r9, %r12
	movq	24(%rsp), %rsi
	movq	32(%rsp), %r13
	movq	16(%rsp), %rbp
	cmovll	%r8d, %edx
	movq	(%rdi,%rcx,8), %rax
	cmovlq	56(%rsp), %r15
	movq	(%rdi,%rdx,8), %rcx
	movq	40(%rsp), %rdi
	movq	%rsi, (%rbx)
	movq	48(%rsp), %rsi
	movq	%rax, %rdx
	sarq	$63, %rdx
	movq	%rcx, %r8
	sarq	$63, %r8
	movq	%rdi, 8(%rbx)
	movq	8(%rsp), %rdi
	shrq	%rdx
	shrq	%r8
	xorq	%rax, %rdx
	xorq	%rcx, %r8
	cmpq	%r8, %rdx
	movq	%rsi, 16(%rbx)
	movq	%rcx, %rsi
	cmovlq	%rax, %rsi
	cmovlq	%rcx, %rax
	movl	$4, %ecx
	movq	%rdi, 24(%rbx)
	movq	%r15, (%rbx,%r13,8)
	movq	%rsi, 8(%rbx,%r13,8)
	movq	%rax, 16(%rbx,%r13,8)
	movq	%r10, 24(%rbx,%r13,8)
	movq	%r14, %rdx
	subq	%r13, %rdx
	cmpq	%rcx, %r13
	ja	.LBB31_15
	jmp	.LBB31_30
.LBB31_13:
	movq	(%rbp), %rax
	movq	(%rbp,%r13,8), %rcx
	movq	%rax, (%rbx)
	movq	%rcx, (%rbx,%r13,8)
	movl	$1, %ecx
	movq	%r14, %rdx
	subq	%r13, %rdx
	cmpq	%rcx, %r13
	jbe	.LBB31_30
.LBB31_15:
	movl	%r13d, %edi
	subl	%ecx, %edi
	leaq	1(%rcx), %rsi
	movq	%rcx, %rax
	testb	$1, %dil
	je	.LBB31_24
	leal	(,%rcx,8), %edi
	movq	(%rbp,%rdi), %rax
	movq	%rax, (%rbx,%rdi)
	movq	-8(%rbx,%rdi), %r8
	movq	%rax, %rdi
	sarq	$63, %rdi
	shrq	%rdi
	xorq	%rax, %rdi
	movq	%r8, %r9
	sarq	$63, %r9
	shrq	%r9
	xorq	%r8, %r9
	cmpq	%r9, %rdi
	jge	.LBB31_23
	leal	(,%rcx,8), %r9d
	.p2align	4
.LBB31_18:
	movq	%r8, (%rbx,%r9)
	cmpq	$8, %r9
	je	.LBB31_19
	movq	-16(%rbx,%r9), %r8
	addq	$-8, %r9
	movq	%r8, %r10
	sarq	$63, %r10
	shrq	%r10
	xorq	%r8, %r10
	cmpq	%r10, %rdi
	jl	.LBB31_18
	addq	%rbx, %r9
	jmp	.LBB31_22
.LBB31_19:
	movq	%rbx, %r9
.LBB31_22:
	movq	%rax, (%r9)
.LBB31_23:
	movq	%rsi, %rax
.LBB31_24:
	cmpq	%rsi, %r13
	jne	.LBB31_25
.LBB31_30:
	leaq	(%rbx,%r13,8), %rax
	cmpq	%rdx, %rcx
	jae	.LBB31_40
	leaq	(%rbp,%r13,8), %rsi
	leal	(,%rcx,8), %edi
	jmp	.LBB31_32
	.p2align	4
.LBB31_35:
	movq	%rax, %r11
.LBB31_38:
	movq	%r8, (%r11)
.LBB31_39:
	incq	%rcx
	addq	$8, %rdi
	cmpq	%rdx, %rcx
	je	.LBB31_40
.LBB31_32:
	movq	(%rsi,%rcx,8), %r8
	movq	%r8, (%rax,%rcx,8)
	movq	%r8, %r9
	sarq	$63, %r9
	movq	-8(%rax,%rcx,8), %r10
	shrq	%r9
	xorq	%r8, %r9
	movq	%r10, %r11
	sarq	$63, %r11
	shrq	%r11
	xorq	%r10, %r11
	cmpq	%r11, %r9
	jge	.LBB31_39
	movq	%rdi, %r11
	.p2align	4
.LBB31_34:
	movq	%r10, (%rax,%r11)
	cmpq	$8, %r11
	je	.LBB31_35
	movq	-16(%rax,%r11), %r10
	addq	$-8, %r11
	movq	%r10, %r12
	sarq	$63, %r12
	shrq	%r12
	xorq	%r10, %r12
	cmpq	%r12, %r9
	jl	.LBB31_34
	addq	%rax, %r11
	jmp	.LBB31_38
.LBB31_40:
	leaq	-8(%rbp,%r14,8), %r9
	leaq	-8(%rbx,%r14,8), %rdx
	leaq	-8(%rax), %r11
	cmpq	$1, %r13
	jne	.LBB31_51
	movq	%rbp, %r8
	movq	%rbx, %r10
	jmp	.LBB31_43
.LBB31_51:
	movl	%r13d, %ecx
	andl	$30, %ecx
	movq	%r13, 32(%rsp)
	movq	%rbp, 16(%rsp)
	movq	%rbp, %r8
	movq	%rbx, %r10
	.p2align	4
.LBB31_52:
	movq	(%rax), %r12
	movq	(%r10), %r15
	movq	%r10, 8(%rsp)
	xorl	%esi, %esi
	xorl	%r10d, %r10d
	movq	%r12, %rbp
	sarq	$63, %rbp
	movq	%r15, %r13
	sarq	$63, %r13
	shrq	%rbp
	shrq	%r13
	xorq	%r12, %rbp
	xorq	%r15, %r13
	cmpq	%r13, %rbp
	cmovlq	%r12, %r15
	movq	(%r11), %r12
	setge	%sil
	setl	%r10b
	xorl	%edi, %edi
	movq	%r15, (%r8)
	movq	(%rdx), %r15
	movq	%rsi, 24(%rsp)
	xorl	%esi, %esi
	movq	%r12, %rbp
	sarq	$63, %rbp
	movq	%r15, %r13
	sarq	$63, %r13
	shrq	%rbp
	shrq	%r13
	xorq	%r12, %rbp
	xorq	%r15, %r13
	cmpq	%rbp, %r13
	movq	8(%rsp), %rbp
	cmovlq	%r12, %r15
	setl	%sil
	setge	%dil
	xorl	%r12d, %r12d
	xorl	%r13d, %r13d
	movq	%r15, (%r9)
	movq	24(%rsp), %r15
	shll	$3, %edi
	shll	$3, %esi
	subq	%rsi, %r11
	subq	%rdi, %rdx
	movq	(%rax,%r10,8), %rsi
	leaq	(%rax,%r10,8), %rax
	movq	8(%rsp), %r10
	movq	%r11, 48(%rsp)
	movq	(%rbp,%r15,8), %rdi
	movq	%rsi, %r15
	sarq	$63, %r15
	shrq	%r15
	xorq	%rsi, %r15
	movq	%rdi, %rbp
	sarq	$63, %rbp
	shrq	%rbp
	xorq	%rdi, %rbp
	cmpq	%rbp, %r15
	cmovlq	%rsi, %rdi
	movq	(%rdx), %rsi
	setl	%r13b
	setge	%r12b
	movq	%rdi, 8(%r8)
	movq	(%r11), %rdi
	addq	$16, %r8
	movq	%rcx, %r11
	movq	%r9, %rcx
	xorl	%r9d, %r9d
	leaq	(%rax,%r13,8), %rax
	movq	%r8, 40(%rsp)
	xorl	%r8d, %r8d
	movq	%rsi, %r15
	sarq	$63, %r15
	movq	%rdi, %rbp
	sarq	$63, %rbp
	shrq	%r15
	shrq	%rbp
	xorq	%rsi, %r15
	xorq	%rdi, %rbp
	cmpq	%rbp, %r15
	movq	24(%rsp), %r15
	setge	%r9b
	cmovlq	%rdi, %rsi
	setl	%r8b
	shll	$3, %r9d
	movq	%rsi, -8(%rcx)
	shll	$3, %r8d
	subq	%r9, %rdx
	movq	%rcx, %r9
	movq	%r11, %rcx
	movq	48(%rsp), %r11
	addq	$-16, %r9
	leaq	(%r10,%r15,8), %r10
	leaq	(%r10,%r12,8), %r10
	subq	%r8, %r11
	movq	40(%rsp), %r8
	addq	$-2, %rcx
	jne	.LBB31_52
	testb	$1, 32(%rsp)
	movq	16(%rsp), %rbp
	jne	.LBB31_43
	addq	$8, %r11
	addq	$8, %rdx
	testb	$1, %r14b
	jne	.LBB31_45
.LBB31_46:
	cmpq	%rdx, %rax
	setne	%al
	cmpq	%r11, %r10
	jne	.LBB31_47
.LBB31_48:
	testb	%al, %al
	jne	.LBB31_49
.LBB31_11:
	addq	$72, %rsp
	.cfi_def_cfa_offset 56
	popq	%rbx
	.cfi_def_cfa_offset 48
	popq	%r12
	.cfi_def_cfa_offset 40
	popq	%r13
	.cfi_def_cfa_offset 32
	popq	%r14
	.cfi_def_cfa_offset 24
	popq	%r15
	.cfi_def_cfa_offset 16
	popq	%rbp
	.cfi_def_cfa_offset 8
	vzeroupper
	retq
.LBB31_43:
	.cfi_def_cfa_offset 128
	movq	%r9, %rdi
	movq	(%rax), %r9
	movq	%r10, %rsi
	movq	(%r10), %r10
	xorl	%r12d, %r12d
	xorl	%r13d, %r13d
	movq	%r9, %rcx
	sarq	$63, %rcx
	movq	%r10, %r15
	sarq	$63, %r15
	shrq	%rcx
	shrq	%r15
	xorq	%r9, %rcx
	xorq	%r10, %r15
	cmpq	%r15, %rcx
	cmovlq	%r9, %r10
	movq	(%rdx), %r9
	setge	%r12b
	setl	%r13b
	movq	%r10, (%r8)
	movq	(%r11), %r10
	leaq	(%rax,%r13,8), %rax
	leaq	(%rsi,%r12,8), %rsi
	addq	$8, %r8
	xorl	%r12d, %r12d
	xorl	%r13d, %r13d
	movq	%r9, %rcx
	sarq	$63, %rcx
	movq	%r10, %r15
	sarq	$63, %r15
	shrq	%rcx
	shrq	%r15
	xorq	%r9, %rcx
	xorq	%r10, %r15
	cmpq	%r15, %rcx
	setl	%r12b
	setge	%r13b
	cmovlq	%r10, %r9
	movq	%rsi, %r10
	shll	$3, %r13d
	shll	$3, %r12d
	movq	%r9, (%rdi)
	subq	%r13, %rdx
	subq	%r12, %r11
	addq	$8, %r11
	addq	$8, %rdx
	testb	$1, %r14b
	je	.LBB31_46
.LBB31_45:
	xorl	%esi, %esi
	xorl	%edi, %edi
	cmpq	%r11, %r10
	movq	%r8, %r9
	movq	%rax, %r8
	cmovbq	%r10, %r8
	setae	%sil
	setb	%dil
	movq	(%r8), %r8
	leaq	(%r10,%rdi,8), %r10
	leaq	(%rax,%rsi,8), %rax
	movq	%r8, (%r9)
	cmpq	%rdx, %rax
	setne	%al
	cmpq	%r11, %r10
	je	.LBB31_48
.LBB31_47:
	movb	$1, %al
	testb	%al, %al
	je	.LBB31_11
.LBB31_49:
.Ltmp250:
	vzeroupper
	callq	*_RNvNtNtNtNtCsgEmfK2I1SDS_4core5slice4sort6shared9smallsort22panic_on_ord_violation@GOTPCREL(%rip)
.Ltmp251:
.LBB31_50:
	ud2
.LBB31_25:
	leal	(,%rax,8), %esi
	leaq	8(%rsi,%rbx), %rdi
	jmp	.LBB31_26
	.p2align	4
.LBB31_60:
	movq	%rbx, %r11
.LBB31_63:
	movq	%r8, (%r11)
.LBB31_64:
	addq	$2, %rax
	addq	$16, %rsi
	addq	$16, %rdi
	cmpq	%r13, %rax
	je	.LBB31_30
.LBB31_26:
	movq	(%rbp,%rax,8), %r8
	movq	%r8, (%rbx,%rax,8)
	movq	%r8, %r9
	sarq	$63, %r9
	movq	-8(%rbx,%rax,8), %r10
	shrq	%r9
	xorq	%r8, %r9
	movq	%r10, %r11
	sarq	$63, %r11
	shrq	%r11
	xorq	%r10, %r11
	cmpq	%r11, %r9
	jge	.LBB31_57
	movq	%rsi, %r11
	.p2align	4
.LBB31_28:
	movq	%r10, (%rbx,%r11)
	cmpq	$8, %r11
	je	.LBB31_29
	movq	-16(%rbx,%r11), %r10
	addq	$-8, %r11
	movq	%r10, %r15
	sarq	$63, %r15
	shrq	%r15
	xorq	%r10, %r15
	cmpq	%r15, %r9
	jl	.LBB31_28
	addq	%rbx, %r11
	jmp	.LBB31_56
	.p2align	4
.LBB31_29:
	movq	%rbx, %r11
.LBB31_56:
	movq	%r8, (%r11)
.LBB31_57:
	movq	8(%rbp,%rax,8), %r8
	movq	%r8, 8(%rbx,%rax,8)
	movq	%r8, %r9
	sarq	$63, %r9
	movq	(%rbx,%rax,8), %r10
	shrq	%r9
	xorq	%r8, %r9
	movq	%r10, %r11
	sarq	$63, %r11
	shrq	%r11
	xorq	%r10, %r11
	cmpq	%r11, %r9
	jge	.LBB31_64
	xorl	%r11d, %r11d
	.p2align	4
.LBB31_59:
	movq	%r10, (%rdi,%r11)
	movq	%rsi, %r10
	addq	%r11, %r10
	je	.LBB31_60
	movq	-16(%rdi,%r11), %r10
	addq	$-8, %r11
	movq	%r10, %r15
	sarq	$63, %r15
	shrq	%r15
	xorq	%r10, %r15
	cmpq	%r15, %r9
	jl	.LBB31_59
	addq	%rdi, %r11
	jmp	.LBB31_63
.LBB31_124:
	leaq	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.66(%rip), %rdi
	leaq	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.68(%rip), %rdx
	movl	$19, %esi
	vzeroupper
	callq	*_RNvNtCsgEmfK2I1SDS_4core9panicking9panic_fmt@GOTPCREL(%rip)
.LBB31_123:
	leaq	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.69(%rip), %rcx
	movq	%r12, %rdi
	movq	%r15, %rsi
	movq	%r15, %rdx
	vzeroupper
	callq	*_RNvNtNtCsgEmfK2I1SDS_4core5slice5index16slice_index_fail@GOTPCREL(%rip)
.LBB31_53:
.Ltmp252:
	shlq	$3, %r14
	movq	%rax, %r15
	movq	%rbp, %rdi
	movq	%rbx, %rsi
	movq	%r14, %rdx
	callq	*memcpy@GOTPCREL(%rip)
	movq	%r15, %rdi
	callq	_Unwind_Resume@PLT
.Lfunc_end31:
	.size	_ZN4core5slice4sort6stable9quicksort9quicksort17hd9000d69f302316eE, .Lfunc_end31-_ZN4core5slice4sort6stable9quicksort9quicksort17hd9000d69f302316eE
	.cfi_endproc
	.section	.gcc_except_table._ZN4core5slice4sort6stable9quicksort9quicksort17hd9000d69f302316eE,"a",@progbits
	.p2align	2, 0x0
GCC_except_table31:
.Lexception10:
	.byte	255
	.byte	255
	.byte	1
	.uleb128 .Lcst_end10-.Lcst_begin10
.Lcst_begin10:
	.uleb128 .Lfunc_begin10-.Lfunc_begin10
	.uleb128 .Ltmp250-.Lfunc_begin10
	.byte	0
	.byte	0
	.uleb128 .Ltmp250-.Lfunc_begin10
	.uleb128 .Ltmp251-.Ltmp250
	.uleb128 .Ltmp252-.Lfunc_begin10
	.byte	0
	.uleb128 .Ltmp251-.Lfunc_begin10
	.uleb128 .Lfunc_end31-.Ltmp251
	.byte	0
	.byte	0
.Lcst_end10:
	.p2align	2, 0x0

	.section	.text.unlikely._ZN4core9panicking13assert_failed17h37cf44da67905b27E,"ax",@progbits
	.type	_ZN4core9panicking13assert_failed17h37cf44da67905b27E,@function
_ZN4core9panicking13assert_failed17h37cf44da67905b27E:
	.cfi_startproc
	subq	$40, %rsp
	.cfi_def_cfa_offset 48
	leaq	24(%rsp), %rax
	leaq	32(%rsp), %rcx
	movq	%rdx, 8(%rsp)
	leaq	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.75(%rip), %rdx
	xorl	%r9d, %r9d
	movq	%rdi, (%rax)
	movq	%rsi, (%rcx)
	xorl	%edi, %edi
	movq	%rax, %rsi
	movq	%rdx, %r8
	callq	*_RNvNtCsgEmfK2I1SDS_4core9panicking19assert_failed_inner@GOTPCREL(%rip)
.Lfunc_end32:
	.size	_ZN4core9panicking13assert_failed17h37cf44da67905b27E, .Lfunc_end32-_ZN4core9panicking13assert_failed17h37cf44da67905b27E
	.cfi_endproc

	.section	".text.unlikely._ZN5alloc7raw_vec19RawVec$LT$T$C$A$GT$8grow_one17h53c42511abd71e68E","ax",@progbits
	.p2align	4
	.type	_ZN5alloc7raw_vec19RawVec$LT$T$C$A$GT$8grow_one17h53c42511abd71e68E,@function
_ZN5alloc7raw_vec19RawVec$LT$T$C$A$GT$8grow_one17h53c42511abd71e68E:
	.cfi_startproc
	pushq	%r14
	.cfi_def_cfa_offset 16
	pushq	%rbx
	.cfi_def_cfa_offset 24
	subq	$24, %rsp
	.cfi_def_cfa_offset 48
	.cfi_offset %rbx, -24
	.cfi_offset %r14, -16
	movq	(%rdi), %rsi
	movq	8(%rdi), %rdx
	movl	$4, %r14d
	movl	$8, %r8d
	movl	$48, %r9d
	movq	%rdi, %rbx
	movq	%rsp, %rdi
	leaq	(%rsi,%rsi), %rax
	cmpq	$5, %rax
	cmovaeq	%rax, %r14
	movq	%r14, %rcx
	callq	_ZN5alloc7raw_vec20RawVecInner$LT$A$GT$11finish_grow17h27139e1aef2978e9E
	cmpl	$1, (%rsp)
	je	.LBB33_2
	movq	8(%rsp), %rax
	movq	%rax, 8(%rbx)
	movq	%r14, (%rbx)
	addq	$24, %rsp
	.cfi_def_cfa_offset 24
	popq	%rbx
	.cfi_def_cfa_offset 16
	popq	%r14
	.cfi_def_cfa_offset 8
	retq
.LBB33_2:
	.cfi_def_cfa_offset 48
	movq	8(%rsp), %rdi
	movq	16(%rsp), %rsi
	callq	*_RNvNtCslNYArtu3iFV_5alloc7raw_vec12handle_error@GOTPCREL(%rip)
.Lfunc_end33:
	.size	_ZN5alloc7raw_vec19RawVec$LT$T$C$A$GT$8grow_one17h53c42511abd71e68E, .Lfunc_end33-_ZN5alloc7raw_vec19RawVec$LT$T$C$A$GT$8grow_one17h53c42511abd71e68E
	.cfi_endproc

	.section	".text.unlikely._ZN5alloc7raw_vec19RawVec$LT$T$C$A$GT$8grow_one17h6e5426fe02c91381E","ax",@progbits
	.p2align	4
	.type	_ZN5alloc7raw_vec19RawVec$LT$T$C$A$GT$8grow_one17h6e5426fe02c91381E,@function
_ZN5alloc7raw_vec19RawVec$LT$T$C$A$GT$8grow_one17h6e5426fe02c91381E:
	.cfi_startproc
	pushq	%r14
	.cfi_def_cfa_offset 16
	pushq	%rbx
	.cfi_def_cfa_offset 24
	subq	$24, %rsp
	.cfi_def_cfa_offset 48
	.cfi_offset %rbx, -24
	.cfi_offset %r14, -16
	movq	(%rdi), %rsi
	movq	8(%rdi), %rdx
	movl	$4, %r14d
	movl	$8, %r8d
	movl	$16, %r9d
	movq	%rdi, %rbx
	movq	%rsp, %rdi
	leaq	(%rsi,%rsi), %rax
	cmpq	$5, %rax
	cmovaeq	%rax, %r14
	movq	%r14, %rcx
	callq	_ZN5alloc7raw_vec20RawVecInner$LT$A$GT$11finish_grow17h27139e1aef2978e9E
	cmpl	$1, (%rsp)
	je	.LBB34_2
	movq	8(%rsp), %rax
	movq	%rax, 8(%rbx)
	movq	%r14, (%rbx)
	addq	$24, %rsp
	.cfi_def_cfa_offset 24
	popq	%rbx
	.cfi_def_cfa_offset 16
	popq	%r14
	.cfi_def_cfa_offset 8
	retq
.LBB34_2:
	.cfi_def_cfa_offset 48
	movq	8(%rsp), %rdi
	movq	16(%rsp), %rsi
	callq	*_RNvNtCslNYArtu3iFV_5alloc7raw_vec12handle_error@GOTPCREL(%rip)
.Lfunc_end34:
	.size	_ZN5alloc7raw_vec19RawVec$LT$T$C$A$GT$8grow_one17h6e5426fe02c91381E, .Lfunc_end34-_ZN5alloc7raw_vec19RawVec$LT$T$C$A$GT$8grow_one17h6e5426fe02c91381E
	.cfi_endproc

	.section	".text.unlikely._ZN5alloc7raw_vec20RawVecInner$LT$A$GT$11finish_grow17h27139e1aef2978e9E","ax",@progbits
	.p2align	4
	.type	_ZN5alloc7raw_vec20RawVecInner$LT$A$GT$11finish_grow17h27139e1aef2978e9E,@function
_ZN5alloc7raw_vec20RawVecInner$LT$A$GT$11finish_grow17h27139e1aef2978e9E:
	.cfi_startproc
	pushq	%r15
	.cfi_def_cfa_offset 16
	pushq	%r14
	.cfi_def_cfa_offset 24
	pushq	%r12
	.cfi_def_cfa_offset 32
	pushq	%rbx
	.cfi_def_cfa_offset 40
	pushq	%rax
	.cfi_def_cfa_offset 48
	.cfi_offset %rbx, -40
	.cfi_offset %r12, -32
	.cfi_offset %r14, -24
	.cfi_offset %r15, -16
	movq	%r8, %r15
	movq	%rdx, %r8
	movq	%r9, %rax
	mulq	%rcx
	movabsq	$-9223372036854775808, %rcx
	movl	$1, %r12d
	movq	%rdi, %rbx
	seto	%dl
	subq	%r15, %rcx
	movq	%rax, %r14
	cmpq	%rcx, %r14
	seta	%cl
	orb	%dl, %cl
	je	.LBB35_2
	movl	$8, %eax
	xorl	%r14d, %r14d
	jmp	.LBB35_10
.LBB35_2:
	testq	%rsi, %rsi
	je	.LBB35_4
	imulq	%rsi, %r9
	movq	%r8, %rdi
	movq	%r15, %rdx
	movq	%r14, %rcx
	movq	%r9, %rsi
	callq	*_RNvCsfLfy6EI15iL_7___rustc14___rust_realloc@GOTPCREL(%rip)
	testq	%rax, %rax
	jne	.LBB35_6
	jmp	.LBB35_9
.LBB35_4:
	testq	%r14, %r14
	je	.LBB35_5
	callq	*_RNvCsfLfy6EI15iL_7___rustc35___rust_no_alloc_shim_is_unstable_v2@GOTPCREL(%rip)
	movq	%r14, %rdi
	movq	%r15, %rsi
	callq	*_RNvCsfLfy6EI15iL_7___rustc12___rust_alloc@GOTPCREL(%rip)
	testq	%rax, %rax
	jne	.LBB35_6
.LBB35_9:
	movl	$16, %eax
	movq	%r15, 8(%rbx)
	jmp	.LBB35_10
.LBB35_5:
	movq	%r15, %rax
.LBB35_6:
	movq	%rax, 8(%rbx)
	movl	$16, %eax
	xorl	%r12d, %r12d
.LBB35_10:
	movq	%r14, (%rbx,%rax)
	movq	%r12, (%rbx)
	addq	$8, %rsp
	.cfi_def_cfa_offset 40
	popq	%rbx
	.cfi_def_cfa_offset 32
	popq	%r12
	.cfi_def_cfa_offset 24
	popq	%r14
	.cfi_def_cfa_offset 16
	popq	%r15
	.cfi_def_cfa_offset 8
	retq
.Lfunc_end35:
	.size	_ZN5alloc7raw_vec20RawVecInner$LT$A$GT$11finish_grow17h27139e1aef2978e9E, .Lfunc_end35-_ZN5alloc7raw_vec20RawVecInner$LT$A$GT$11finish_grow17h27139e1aef2978e9E
	.cfi_endproc

	.section	".text.unlikely._ZN5alloc7raw_vec20RawVecInner$LT$A$GT$7reserve21do_reserve_and_handle17h5eeef80eb9b0c257E","ax",@progbits
	.p2align	4
	.type	_ZN5alloc7raw_vec20RawVecInner$LT$A$GT$7reserve21do_reserve_and_handle17h5eeef80eb9b0c257E,@function
_ZN5alloc7raw_vec20RawVecInner$LT$A$GT$7reserve21do_reserve_and_handle17h5eeef80eb9b0c257E:
	.cfi_startproc
	pushq	%r14
	.cfi_def_cfa_offset 16
	pushq	%rbx
	.cfi_def_cfa_offset 24
	subq	$24, %rsp
	.cfi_def_cfa_offset 48
	.cfi_offset %rbx, -24
	.cfi_offset %r14, -16
	addq	%rdx, %rsi
	jb	.LBB36_1
	movq	(%rdi), %rax
	movq	%r8, %r9
	movq	%rcx, %r8
	movq	8(%rdi), %rdx
	movq	%rdi, %rbx
	movq	%rsp, %rdi
	leaq	(%rax,%rax), %r14
	cmpq	%r14, %rsi
	cmovaq	%rsi, %r14
	xorl	%ecx, %ecx
	cmpq	$1, %r9
	movq	%rax, %rsi
	sete	%cl
	leaq	4(,%rcx,4), %rcx
	cmpq	%rcx, %r14
	cmovbeq	%rcx, %r14
	movq	%r14, %rcx
	callq	_ZN5alloc7raw_vec20RawVecInner$LT$A$GT$11finish_grow17h27139e1aef2978e9E
	cmpl	$1, (%rsp)
	je	.LBB36_3
	movq	8(%rsp), %rax
	movq	%rax, 8(%rbx)
	movq	%r14, (%rbx)
	addq	$24, %rsp
	.cfi_def_cfa_offset 24
	popq	%rbx
	.cfi_def_cfa_offset 16
	popq	%r14
	.cfi_def_cfa_offset 8
	retq
.LBB36_1:
	.cfi_def_cfa_offset 48
	xorl	%edi, %edi
	callq	*_RNvNtCslNYArtu3iFV_5alloc7raw_vec12handle_error@GOTPCREL(%rip)
.LBB36_3:
	movq	8(%rsp), %rdi
	movq	16(%rsp), %rsi
	callq	*_RNvNtCslNYArtu3iFV_5alloc7raw_vec12handle_error@GOTPCREL(%rip)
.Lfunc_end36:
	.size	_ZN5alloc7raw_vec20RawVecInner$LT$A$GT$7reserve21do_reserve_and_handle17h5eeef80eb9b0c257E, .Lfunc_end36-_ZN5alloc7raw_vec20RawVecInner$LT$A$GT$7reserve21do_reserve_and_handle17h5eeef80eb9b0c257E
	.cfi_endproc

	.section	.rodata.cst32,"aM",@progbits,32
	.p2align	5, 0x0
.LCPI37_0:
	.quad	0
	.quad	8
	.quad	0
	.quad	-9223372036854775808
	.section	.text.unlikely._ZN5prost5error11DecodeError3new17h19d861c2f6bb76b8E,"ax",@progbits
	.p2align	4
	.type	_ZN5prost5error11DecodeError3new17h19d861c2f6bb76b8E,@function
_ZN5prost5error11DecodeError3new17h19d861c2f6bb76b8E:
.Lfunc_begin11:
	.cfi_startproc
	.cfi_personality 155, DW.ref.rust_eh_personality
	.cfi_lsda 27, .Lexception11
	pushq	%rbx
	.cfi_def_cfa_offset 16
	subq	$48, %rsp
	.cfi_def_cfa_offset 64
	.cfi_offset %rbx, -16
	vmovaps	.LCPI37_0(%rip), %ymm0
	movq	%rdi, 32(%rsp)
	movq	%rsi, 40(%rsp)
	vmovups	%ymm0, (%rsp)
	vzeroupper
	callq	*_RNvCsfLfy6EI15iL_7___rustc35___rust_no_alloc_shim_is_unstable_v2@GOTPCREL(%rip)
	movl	$48, %edi
	movl	$8, %esi
	callq	*_RNvCsfLfy6EI15iL_7___rustc12___rust_alloc@GOTPCREL(%rip)
	testq	%rax, %rax
	je	.LBB37_1
	vmovups	16(%rsp), %ymm1
	vmovups	(%rsp), %ymm0
	vmovups	%ymm1, 16(%rax)
	vmovups	%ymm0, (%rax)
	addq	$48, %rsp
	.cfi_def_cfa_offset 16
	popq	%rbx
	.cfi_def_cfa_offset 8
	vzeroupper
	retq
.LBB37_1:
	.cfi_def_cfa_offset 64
.Ltmp253:
	movl	$8, %edi
	movl	$48, %esi
	callq	*_RNvNtCslNYArtu3iFV_5alloc5alloc18handle_alloc_error@GOTPCREL(%rip)
.Ltmp254:
	ud2
.LBB37_3:
.Ltmp255:
	movq	%rsp, %rdi
	movq	%rax, %rbx
	callq	_ZN4core3ptr40drop_in_place$LT$prost..error..Inner$GT$17h5eac8c554bcc6811E
	movq	%rbx, %rdi
	callq	_Unwind_Resume@PLT
.Lfunc_end37:
	.size	_ZN5prost5error11DecodeError3new17h19d861c2f6bb76b8E, .Lfunc_end37-_ZN5prost5error11DecodeError3new17h19d861c2f6bb76b8E
	.cfi_endproc
	.section	.gcc_except_table._ZN5prost5error11DecodeError3new17h19d861c2f6bb76b8E,"a",@progbits
	.p2align	2, 0x0
GCC_except_table37:
.Lexception11:
	.byte	255
	.byte	255
	.byte	1
	.uleb128 .Lcst_end11-.Lcst_begin11
.Lcst_begin11:
	.uleb128 .Ltmp253-.Lfunc_begin11
	.uleb128 .Ltmp254-.Ltmp253
	.uleb128 .Ltmp255-.Lfunc_begin11
	.byte	0
	.uleb128 .Ltmp254-.Lfunc_begin11
	.uleb128 .Lfunc_end37-.Ltmp254
	.byte	0
	.byte	0
.Lcst_end11:
	.p2align	2, 0x0

	.section	.text.unlikely._ZN5prost5error11DecodeError3new17hdeede837cadf4d68E,"ax",@progbits
	.p2align	4
	.type	_ZN5prost5error11DecodeError3new17hdeede837cadf4d68E,@function
_ZN5prost5error11DecodeError3new17hdeede837cadf4d68E:
.Lfunc_begin12:
	.cfi_startproc
	.cfi_personality 155, DW.ref.rust_eh_personality
	.cfi_lsda 27, .Lexception12
	pushq	%rbx
	.cfi_def_cfa_offset 16
	subq	$48, %rsp
	.cfi_def_cfa_offset 64
	.cfi_offset %rbx, -16
	vmovups	(%rdi), %xmm0
	movq	16(%rdi), %rax
	movq	%rax, 40(%rsp)
	vmovups	%xmm0, 24(%rsp)
	movq	$0, (%rsp)
	movq	$8, 8(%rsp)
	movq	$0, 16(%rsp)
	callq	*_RNvCsfLfy6EI15iL_7___rustc35___rust_no_alloc_shim_is_unstable_v2@GOTPCREL(%rip)
	movl	$48, %edi
	movl	$8, %esi
	callq	*_RNvCsfLfy6EI15iL_7___rustc12___rust_alloc@GOTPCREL(%rip)
	testq	%rax, %rax
	je	.LBB38_1
	vmovups	16(%rsp), %ymm1
	vmovups	(%rsp), %ymm0
	vmovups	%ymm1, 16(%rax)
	vmovups	%ymm0, (%rax)
	addq	$48, %rsp
	.cfi_def_cfa_offset 16
	popq	%rbx
	.cfi_def_cfa_offset 8
	vzeroupper
	retq
.LBB38_1:
	.cfi_def_cfa_offset 64
.Ltmp256:
	movl	$8, %edi
	movl	$48, %esi
	callq	*_RNvNtCslNYArtu3iFV_5alloc5alloc18handle_alloc_error@GOTPCREL(%rip)
.Ltmp257:
	ud2
.LBB38_3:
.Ltmp258:
	movq	%rsp, %rdi
	movq	%rax, %rbx
	callq	_ZN4core3ptr40drop_in_place$LT$prost..error..Inner$GT$17h5eac8c554bcc6811E
	movq	%rbx, %rdi
	callq	_Unwind_Resume@PLT
.Lfunc_end38:
	.size	_ZN5prost5error11DecodeError3new17hdeede837cadf4d68E, .Lfunc_end38-_ZN5prost5error11DecodeError3new17hdeede837cadf4d68E
	.cfi_endproc
	.section	.gcc_except_table._ZN5prost5error11DecodeError3new17hdeede837cadf4d68E,"a",@progbits
	.p2align	2, 0x0
GCC_except_table38:
.Lexception12:
	.byte	255
	.byte	255
	.byte	1
	.uleb128 .Lcst_end12-.Lcst_begin12
.Lcst_begin12:
	.uleb128 .Ltmp256-.Lfunc_begin12
	.uleb128 .Ltmp257-.Ltmp256
	.uleb128 .Ltmp258-.Lfunc_begin12
	.byte	0
	.uleb128 .Ltmp257-.Lfunc_begin12
	.uleb128 .Lfunc_end38-.Ltmp257
	.byte	0
	.byte	0
.Lcst_end12:
	.p2align	2, 0x0

	.section	.text._ZN5prost7message7Message6decode17h7e463807ec2b546fE,"ax",@progbits
	.p2align	4
	.type	_ZN5prost7message7Message6decode17h7e463807ec2b546fE,@function
_ZN5prost7message7Message6decode17h7e463807ec2b546fE:
.Lfunc_begin13:
	.cfi_startproc
	.cfi_personality 155, DW.ref.rust_eh_personality
	.cfi_lsda 27, .Lexception13
	pushq	%rbp
	.cfi_def_cfa_offset 16
	pushq	%r15
	.cfi_def_cfa_offset 24
	pushq	%r14
	.cfi_def_cfa_offset 32
	pushq	%r13
	.cfi_def_cfa_offset 40
	pushq	%r12
	.cfi_def_cfa_offset 48
	pushq	%rbx
	.cfi_def_cfa_offset 56
	subq	$152, %rsp
	.cfi_def_cfa_offset 208
	.cfi_offset %rbx, -56
	.cfi_offset %r12, -48
	.cfi_offset %r13, -40
	.cfi_offset %r14, -32
	.cfi_offset %r15, -24
	.cfi_offset %rbp, -16
	leaq	56(%rsp), %r15
	movl	$8, %r12d
	leaq	144(%rsp), %r13
	movq	%rsi, 56(%rsp)
	movq	$0, 80(%rsp)
	movq	$8, 88(%rsp)
	xorl	%ebp, %ebp
	movq	%rdi, 136(%rsp)
	movq	%rdx, 64(%rsp)
	movq	$0, 96(%rsp)
	movq	%r15, 144(%rsp)
	jmp	.LBB39_1
	.p2align	4
.LBB39_70:
.Ltmp265:
	movzbl	%al, %edi
	movl	$100, %ecx
	movl	%edx, %esi
	movq	%r13, %rdx
	callq	_ZN5prost8encoding10skip_field17hecfc4b3cbbe23b00E
	movq	%rax, %rdx
.Ltmp266:
	testq	%rdx, %rdx
	jne	.LBB39_53
.LBB39_1:
	cmpq	$0, 64(%rsp)
	je	.LBB39_74
.Ltmp259:
	movq	%r15, %rdi
	callq	_ZN5prost8encoding6varint13decode_varint17h5757c281c4676585E
.Ltmp260:
	testb	$1, %al
	jne	.LBB39_53
	movq	%rdx, %rax
	shrq	$32, %rax
	movq	%rdx, 104(%rsp)
	jne	.LBB39_8
	movl	%edx, %eax
	andl	$7, %eax
	movq	%rax, (%rsp)
	cmpq	$6, %rax
	jae	.LBB39_6
	shrl	$3, %edx
	je	.LBB39_11
	cmpl	$1, %edx
	jne	.LBB39_70
	movb	$2, (%rsp)
	movb	%al, 40(%rsp)
	cmpl	$2, %eax
	jne	.LBB39_14
.Ltmp271:
	movq	%r15, %rdi
	callq	_ZN5prost8encoding6varint13decode_varint17h5757c281c4676585E
.Ltmp272:
	testb	$1, %al
	je	.LBB39_18
	testq	%rdx, %rdx
	jne	.LBB39_51
	xorl	%r14d, %r14d
	vxorps	%xmm0, %xmm0, %xmm0
	vmovsd	%xmm0, 72(%rsp)
	jmp	.LBB39_65
	.p2align	4
.LBB39_18:
	movq	64(%rsp), %rbx
	subq	%rdx, %rbx
	jb	.LBB39_19
	vxorps	%xmm0, %xmm0, %xmm0
	xorl	%r14d, %r14d
	vmovsd	%xmm0, 72(%rsp)
	jmp	.LBB39_22
	.p2align	4
.LBB39_37:
.Ltmp297:
	movzbl	%al, %edi
	movl	$99, %ecx
	movl	%edx, %esi
	movq	%r13, %rdx
	callq	_ZN5prost8encoding10skip_field17hecfc4b3cbbe23b00E
	movq	%rax, %rdx
.Ltmp298:
	testq	%rdx, %rdx
	jne	.LBB39_51
.LBB39_22:
	cmpq	%rbx, 64(%rsp)
	jbe	.LBB39_23
.Ltmp273:
	movq	%r15, %rdi
	callq	_ZN5prost8encoding6varint13decode_varint17h5757c281c4676585E
.Ltmp274:
	testb	$1, %al
	jne	.LBB39_51
	movq	%rdx, %rax
	shrq	$32, %rax
	movq	%rdx, 104(%rsp)
	jne	.LBB39_31
	movl	%edx, %eax
	andl	$7, %eax
	movq	%rax, (%rsp)
	cmpq	$6, %rax
	jae	.LBB39_29
	shrl	$3, %edx
	je	.LBB39_34
	cmpl	$1, %edx
	je	.LBB39_38
	cmpl	$2, %edx
	jne	.LBB39_37
	movb	$1, (%rsp)
	movb	%al, 40(%rsp)
	cmpl	$1, %eax
	jne	.LBB39_47
	movq	64(%rsp), %rax
	cmpq	$7, %rax
	jbe	.LBB39_56
	movq	56(%rsp), %rcx
	addq	$-8, %rax
	vmovsd	(%rcx), %xmm0
	addq	$8, %rcx
	movq	%rcx, 56(%rsp)
	movq	%rax, 64(%rsp)
	vmovsd	%xmm0, 72(%rsp)
	xorl	%edx, %edx
	testq	%rdx, %rdx
	je	.LBB39_22
	jmp	.LBB39_51
	.p2align	4
.LBB39_38:
	movb	$0, (%rsp)
	movb	%al, 40(%rsp)
	testq	%rax, %rax
	jne	.LBB39_39
.Ltmp292:
	movq	%r15, %rdi
	callq	_ZN5prost8encoding6varint13decode_varint17h5757c281c4676585E
.Ltmp293:
	testb	$1, %al
	je	.LBB39_44
	testq	%rdx, %rdx
	jne	.LBB39_41
	xorl	%edx, %edx
	testq	%rdx, %rdx
	je	.LBB39_22
	jmp	.LBB39_51
.LBB39_44:
	movl	%edx, %r14d
	xorl	%edx, %edx
	testq	%rdx, %rdx
	je	.LBB39_22
	jmp	.LBB39_51
.LBB39_23:
	jne	.LBB39_24
.LBB39_65:
	cmpq	80(%rsp), %rbp
	jne	.LBB39_68
.Ltmp311:
	leaq	80(%rsp), %rdi
	callq	_ZN5alloc7raw_vec19RawVec$LT$T$C$A$GT$8grow_one17h6e5426fe02c91381E
.Ltmp312:
	movq	88(%rsp), %r12
.LBB39_68:
	vmovsd	72(%rsp), %xmm0
	movq	%rbp, %rax
	shlq	$4, %rax
	incq	%rbp
	xorl	%edx, %edx
	vmovsd	%xmm0, (%r12,%rax)
	movl	%r14d, 8(%r12,%rax)
	movq	%rbp, 96(%rsp)
	testq	%rdx, %rdx
	je	.LBB39_1
	jmp	.LBB39_53
.LBB39_74:
	vmovups	80(%rsp), %xmm0
	movq	136(%rsp), %rax
	vmovups	%xmm0, (%rax)
	movq	%rbp, 16(%rax)
	jmp	.LBB39_75
.LBB39_31:
	movq	_RNvXsd_NtNtNtCsgEmfK2I1SDS_4core3fmt3num3impyNtB9_7Display3fmt@GOTPCREL(%rip), %rcx
	leaq	104(%rsp), %rax
	movq	%rax, 8(%rsp)
	movq	%rcx, 16(%rsp)
.Ltmp302:
	leaq	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.78(%rip), %rsi
	leaq	112(%rsp), %rdi
	leaq	8(%rsp), %rdx
	callq	*_RNvNvNtCslNYArtu3iFV_5alloc3fmt6format12format_inner@GOTPCREL(%rip)
.Ltmp303:
.Ltmp304:
	leaq	112(%rsp), %rdi
	callq	_ZN5prost5error11DecodeError3new17hdeede837cadf4d68E
	movq	%rax, %rdx
.Ltmp305:
	jmp	.LBB39_51
.LBB39_34:
.Ltmp300:
	leaq	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.77(%rip), %rdi
	movl	$20, %esi
	callq	_ZN5prost5error11DecodeError3new17h19d861c2f6bb76b8E
	movq	%rax, %rdx
.Ltmp301:
	jmp	.LBB39_51
.LBB39_8:
	movq	_RNvXsd_NtNtNtCsgEmfK2I1SDS_4core3fmt3num3impyNtB9_7Display3fmt@GOTPCREL(%rip), %rcx
	leaq	104(%rsp), %rax
	movq	%rax, 8(%rsp)
	movq	%rcx, 16(%rsp)
.Ltmp316:
	leaq	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.78(%rip), %rsi
	leaq	112(%rsp), %rdi
	leaq	8(%rsp), %rdx
	callq	*_RNvNvNtCslNYArtu3iFV_5alloc3fmt6format12format_inner@GOTPCREL(%rip)
.Ltmp317:
.Ltmp318:
	leaq	112(%rsp), %rdi
	callq	_ZN5prost5error11DecodeError3new17hdeede837cadf4d68E
	movq	%rax, %rdx
.Ltmp319:
	jmp	.LBB39_53
.LBB39_11:
.Ltmp314:
	leaq	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.77(%rip), %rdi
	movl	$20, %esi
	callq	_ZN5prost5error11DecodeError3new17h19d861c2f6bb76b8E
	movq	%rax, %rdx
.Ltmp315:
	jmp	.LBB39_53
.LBB39_39:
	leaq	40(%rsp), %rax
	leaq	_ZN73_$LT$prost..encoding..wire_type..WireType$u20$as$u20$core..fmt..Debug$GT$3fmt17hb528d417acae5beaE(%rip), %r8
	movq	%rsp, %rcx
	movq	%rax, 8(%rsp)
	movq	%r8, 16(%rsp)
	movq	%rcx, 24(%rsp)
	movq	%r8, 32(%rsp)
.Ltmp288:
	leaq	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.84(%rip), %rsi
	leaq	112(%rsp), %rdi
	leaq	8(%rsp), %rdx
	callq	*_RNvNvNtCslNYArtu3iFV_5alloc3fmt6format12format_inner@GOTPCREL(%rip)
.Ltmp289:
.Ltmp290:
	leaq	112(%rsp), %rdi
	callq	_ZN5prost5error11DecodeError3new17hdeede837cadf4d68E
	movq	%rax, %rdx
.Ltmp291:
.LBB39_41:
	movq	%rdx, 8(%rsp)
.Ltmp294:
	leaq	8(%rsp), %r14
	leaq	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.96(%rip), %rsi
	leaq	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.97(%rip), %rcx
	movl	$11, %edx
	movl	$2, %r8d
	movq	%r14, %rdi
	callq	*_ZN5prost5error11DecodeError4push17h91b95da0c6bbbd16E@GOTPCREL(%rip)
.Ltmp295:
.LBB39_50:
	movq	8(%rsp), %rdx
.LBB39_51:
	movq	%rdx, 8(%rsp)
.Ltmp308:
	leaq	8(%rsp), %r14
	leaq	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.99(%rip), %rsi
	leaq	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.100(%rip), %rcx
	movl	$15, %edx
	movl	$7, %r8d
	movq	%r14, %rdi
	callq	*_ZN5prost5error11DecodeError4push17h91b95da0c6bbbd16E@GOTPCREL(%rip)
.Ltmp309:
	movq	8(%rsp), %rdx
.LBB39_53:
	movq	136(%rsp), %rcx
	movq	80(%rsp), %rsi
	movq	88(%rsp), %rdi
	movabsq	$-9223372036854775808, %rax
	movq	%rdx, 8(%rcx)
	movq	%rax, (%rcx)
	testq	%rsi, %rsi
	je	.LBB39_75
	shlq	$4, %rsi
	movl	$8, %edx
	callq	*_RNvCsfLfy6EI15iL_7___rustc14___rust_dealloc@GOTPCREL(%rip)
.LBB39_75:
	addq	$152, %rsp
	.cfi_def_cfa_offset 56
	popq	%rbx
	.cfi_def_cfa_offset 48
	popq	%r12
	.cfi_def_cfa_offset 40
	popq	%r13
	.cfi_def_cfa_offset 32
	popq	%r14
	.cfi_def_cfa_offset 24
	popq	%r15
	.cfi_def_cfa_offset 16
	popq	%rbp
	.cfi_def_cfa_offset 8
	retq
.LBB39_47:
	.cfi_def_cfa_offset 208
	leaq	40(%rsp), %rax
	leaq	_ZN73_$LT$prost..encoding..wire_type..WireType$u20$as$u20$core..fmt..Debug$GT$3fmt17hb528d417acae5beaE(%rip), %r8
	movq	%rsp, %rcx
	movq	%rax, 8(%rsp)
	movq	%r8, 16(%rsp)
	movq	%rcx, 24(%rsp)
	movq	%r8, 32(%rsp)
.Ltmp279:
	leaq	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.84(%rip), %rsi
	leaq	112(%rsp), %rdi
	leaq	8(%rsp), %rdx
	callq	*_RNvNvNtCslNYArtu3iFV_5alloc3fmt6format12format_inner@GOTPCREL(%rip)
.Ltmp280:
.Ltmp281:
	leaq	112(%rsp), %rdi
	callq	_ZN5prost5error11DecodeError3new17hdeede837cadf4d68E
.Ltmp282:
	jmp	.LBB39_49
.LBB39_56:
.Ltmp283:
	leaq	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.80(%rip), %rdi
	movl	$16, %esi
	callq	_ZN5prost5error11DecodeError3new17h19d861c2f6bb76b8E
.Ltmp284:
.LBB39_49:
	movq	%rax, 8(%rsp)
.Ltmp285:
	leaq	8(%rsp), %r14
	leaq	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.96(%rip), %rsi
	leaq	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.98(%rip), %rcx
	movl	$11, %edx
	movl	$5, %r8d
	movq	%r14, %rdi
	callq	*_ZN5prost5error11DecodeError4push17h91b95da0c6bbbd16E@GOTPCREL(%rip)
.Ltmp286:
	jmp	.LBB39_50
.LBB39_14:
	leaq	40(%rsp), %rax
	leaq	_ZN73_$LT$prost..encoding..wire_type..WireType$u20$as$u20$core..fmt..Debug$GT$3fmt17hb528d417acae5beaE(%rip), %r8
	movq	%rsp, %rcx
	movq	%rax, 8(%rsp)
	movq	%r8, 16(%rsp)
	movq	%rcx, 24(%rsp)
	movq	%r8, 32(%rsp)
.Ltmp267:
	leaq	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.84(%rip), %rsi
	leaq	112(%rsp), %rdi
	leaq	8(%rsp), %rdx
	callq	*_RNvNvNtCslNYArtu3iFV_5alloc3fmt6format12format_inner@GOTPCREL(%rip)
.Ltmp268:
.Ltmp269:
	leaq	112(%rsp), %rdi
	callq	_ZN5prost5error11DecodeError3new17hdeede837cadf4d68E
	movq	%rax, %rdx
.Ltmp270:
	jmp	.LBB39_51
.LBB39_29:
	movq	_RNvXsd_NtNtNtCsgEmfK2I1SDS_4core3fmt3num3impyNtB9_7Display3fmt@GOTPCREL(%rip), %rcx
	movq	%rsp, %rax
	movq	%rax, 40(%rsp)
	movq	%rcx, 48(%rsp)
.Ltmp275:
	leaq	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.101(%rip), %rsi
	leaq	8(%rsp), %rdi
	leaq	40(%rsp), %rdx
	callq	*_RNvNvNtCslNYArtu3iFV_5alloc3fmt6format12format_inner@GOTPCREL(%rip)
.Ltmp276:
.Ltmp277:
	leaq	8(%rsp), %rdi
	callq	_ZN5prost5error11DecodeError3new17hdeede837cadf4d68E
	movq	%rax, %rdx
.Ltmp278:
	jmp	.LBB39_51
.LBB39_19:
	movl	$16, %esi
	leaq	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.80(%rip), %rdi
	jmp	.LBB39_20
.LBB39_6:
	movq	_RNvXsd_NtNtNtCsgEmfK2I1SDS_4core3fmt3num3impyNtB9_7Display3fmt@GOTPCREL(%rip), %rcx
	movq	%rsp, %rax
	movq	%rax, 40(%rsp)
	movq	%rcx, 48(%rsp)
.Ltmp261:
	leaq	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.101(%rip), %rsi
	leaq	8(%rsp), %rdi
	leaq	40(%rsp), %rdx
	callq	*_RNvNvNtCslNYArtu3iFV_5alloc3fmt6format12format_inner@GOTPCREL(%rip)
.Ltmp262:
.Ltmp263:
	leaq	8(%rsp), %rdi
	callq	_ZN5prost5error11DecodeError3new17hdeede837cadf4d68E
	movq	%rax, %rdx
.Ltmp264:
	jmp	.LBB39_53
.LBB39_24:
	movl	$25, %esi
	leaq	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.79(%rip), %rdi
.LBB39_20:
.Ltmp306:
	callq	_ZN5prost5error11DecodeError3new17h19d861c2f6bb76b8E
	movq	%rax, %rdx
.Ltmp307:
	jmp	.LBB39_51
.LBB39_58:
.Ltmp287:
	jmp	.LBB39_59
.LBB39_73:
.Ltmp320:
	jmp	.LBB39_77
.LBB39_57:
.Ltmp296:
	jmp	.LBB39_59
.LBB39_71:
.Ltmp310:
.LBB39_59:
	movq	%r14, %rdi
	movq	%rax, %rbx
	callq	_ZN4core3ptr46drop_in_place$LT$prost..error..DecodeError$GT$17h6452743001005f2fE
	jmp	.LBB39_78
.LBB39_72:
.Ltmp313:
	jmp	.LBB39_77
.LBB39_76:
.Ltmp299:
.LBB39_77:
	movq	%rax, %rbx
.LBB39_78:
	movq	80(%rsp), %rsi
	testq	%rsi, %rsi
	je	.LBB39_80
	movq	88(%rsp), %rdi
	shlq	$4, %rsi
	movl	$8, %edx
	callq	*_RNvCsfLfy6EI15iL_7___rustc14___rust_dealloc@GOTPCREL(%rip)
.LBB39_80:
	movq	%rbx, %rdi
	callq	_Unwind_Resume@PLT
.Lfunc_end39:
	.size	_ZN5prost7message7Message6decode17h7e463807ec2b546fE, .Lfunc_end39-_ZN5prost7message7Message6decode17h7e463807ec2b546fE
	.cfi_endproc
	.section	.gcc_except_table._ZN5prost7message7Message6decode17h7e463807ec2b546fE,"a",@progbits
	.p2align	2, 0x0
GCC_except_table39:
.Lexception13:
	.byte	255
	.byte	255
	.byte	1
	.uleb128 .Lcst_end13-.Lcst_begin13
.Lcst_begin13:
	.uleb128 .Ltmp265-.Lfunc_begin13
	.uleb128 .Ltmp272-.Ltmp265
	.uleb128 .Ltmp313-.Lfunc_begin13
	.byte	0
	.uleb128 .Ltmp297-.Lfunc_begin13
	.uleb128 .Ltmp293-.Ltmp297
	.uleb128 .Ltmp299-.Lfunc_begin13
	.byte	0
	.uleb128 .Ltmp311-.Lfunc_begin13
	.uleb128 .Ltmp312-.Ltmp311
	.uleb128 .Ltmp313-.Lfunc_begin13
	.byte	0
	.uleb128 .Ltmp302-.Lfunc_begin13
	.uleb128 .Ltmp291-.Ltmp302
	.uleb128 .Ltmp320-.Lfunc_begin13
	.byte	0
	.uleb128 .Ltmp294-.Lfunc_begin13
	.uleb128 .Ltmp295-.Ltmp294
	.uleb128 .Ltmp296-.Lfunc_begin13
	.byte	0
	.uleb128 .Ltmp308-.Lfunc_begin13
	.uleb128 .Ltmp309-.Ltmp308
	.uleb128 .Ltmp310-.Lfunc_begin13
	.byte	0
	.uleb128 .Ltmp279-.Lfunc_begin13
	.uleb128 .Ltmp284-.Ltmp279
	.uleb128 .Ltmp320-.Lfunc_begin13
	.byte	0
	.uleb128 .Ltmp285-.Lfunc_begin13
	.uleb128 .Ltmp286-.Ltmp285
	.uleb128 .Ltmp287-.Lfunc_begin13
	.byte	0
	.uleb128 .Ltmp267-.Lfunc_begin13
	.uleb128 .Ltmp307-.Ltmp267
	.uleb128 .Ltmp320-.Lfunc_begin13
	.byte	0
	.uleb128 .Ltmp307-.Lfunc_begin13
	.uleb128 .Lfunc_end39-.Ltmp307
	.byte	0
	.byte	0
.Lcst_end13:
	.p2align	2, 0x0

	.section	.text._ZN5prost8encoding10skip_field17hecfc4b3cbbe23b00E,"ax",@progbits
	.p2align	4
	.type	_ZN5prost8encoding10skip_field17hecfc4b3cbbe23b00E,@function
_ZN5prost8encoding10skip_field17hecfc4b3cbbe23b00E:
	.cfi_startproc
	pushq	%rbp
	.cfi_def_cfa_offset 16
	pushq	%r15
	.cfi_def_cfa_offset 24
	pushq	%r14
	.cfi_def_cfa_offset 32
	pushq	%rbx
	.cfi_def_cfa_offset 40
	subq	$88, %rsp
	.cfi_def_cfa_offset 128
	.cfi_offset %rbx, -40
	.cfi_offset %r14, -32
	.cfi_offset %r15, -24
	.cfi_offset %rbp, -16
	testl	%ecx, %ecx
	je	.LBB40_1
	movq	%rdx, %rbx
	movzbl	%dil, %eax
	leaq	.LJTI40_0(%rip), %rdx
	movl	%ecx, %ebp
	movslq	(%rdx,%rax,4), %rax
	addq	%rdx, %rax
	jmpq	*%rax
.LBB40_4:
	movq	(%rbx), %r14
	movq	%r14, %rdi
	callq	_ZN5prost8encoding6varint13decode_varint17h5757c281c4676585E
	movq	%rax, %rcx
	testb	$1, %cl
	je	.LBB40_23
	movq	%rdx, %rax
	jmp	.LBB40_20
.LBB40_6:
	movq	(%rbx), %r14
	decl	%ebp
	movl	%esi, %r15d
	.p2align	4
.LBB40_7:
	movq	%r14, %rdi
	callq	_ZN5prost8encoding6varint13decode_varint17h5757c281c4676585E
	movq	%rax, %rcx
	movq	%rdx, %rax
	testb	$1, %cl
	jne	.LBB40_20
	movq	%rax, %rcx
	shrq	$32, %rcx
	movq	%rax, 8(%rsp)
	jne	.LBB40_11
	movl	%eax, %edi
	andl	$7, %edi
	movq	%rdi, 16(%rsp)
	cmpq	$6, %rdi
	jae	.LBB40_10
	shrl	$3, %eax
	je	.LBB40_14
	cmpl	$4, %edi
	je	.LBB40_22
	movl	%eax, %esi
	movq	%rbx, %rdx
	movl	%ebp, %ecx
	callq	_ZN5prost8encoding10skip_field17hecfc4b3cbbe23b00E
	testq	%rax, %rax
	je	.LBB40_7
	jmp	.LBB40_20
.LBB40_19:
	movq	(%rbx), %rdi
	callq	_ZN5prost8encoding6varint13decode_varint17h5757c281c4676585E
	movq	%rax, %rcx
	movq	%rdx, %rax
	testb	$1, %cl
	je	.LBB40_17
	jmp	.LBB40_20
.LBB40_16:
	movl	$8, %eax
	jmp	.LBB40_17
.LBB40_15:
	movl	$4, %eax
.LBB40_17:
	movq	(%rbx), %r14
	movq	8(%r14), %rcx
	cmpq	%rcx, %rax
	jbe	.LBB40_18
	leaq	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.80(%rip), %rdi
	movl	$16, %esi
	jmp	.LBB40_2
.LBB40_22:
	cmpl	%r15d, %eax
	jne	.LBB40_25
.LBB40_23:
	movq	8(%r14), %rcx
	xorl	%eax, %eax
.LBB40_18:
	addq	%rax, (%r14)
	subq	%rax, %rcx
	xorl	%eax, %eax
	movq	%rcx, 8(%r14)
.LBB40_20:
	addq	$88, %rsp
	.cfi_def_cfa_offset 40
	popq	%rbx
	.cfi_def_cfa_offset 32
	popq	%r14
	.cfi_def_cfa_offset 24
	popq	%r15
	.cfi_def_cfa_offset 16
	popq	%rbp
	.cfi_def_cfa_offset 8
	retq
.LBB40_11:
	.cfi_def_cfa_offset 128
	movq	_RNvXsd_NtNtNtCsgEmfK2I1SDS_4core3fmt3num3impyNtB9_7Display3fmt@GOTPCREL(%rip), %rcx
	leaq	8(%rsp), %rax
	leaq	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.78(%rip), %rsi
	leaq	64(%rsp), %rbx
	leaq	40(%rsp), %rdx
	movq	%rax, 40(%rsp)
	movq	%rcx, 48(%rsp)
.LBB40_12:
	movq	%rbx, %rdi
	callq	*_RNvNvNtCslNYArtu3iFV_5alloc3fmt6format12format_inner@GOTPCREL(%rip)
	movq	%rbx, %rdi
	callq	_ZN5prost5error11DecodeError3new17hdeede837cadf4d68E
	jmp	.LBB40_20
.LBB40_14:
	leaq	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.77(%rip), %rdi
	movl	$20, %esi
	callq	_ZN5prost5error11DecodeError3new17h19d861c2f6bb76b8E
	jmp	.LBB40_20
.LBB40_1:
	leaq	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.81(%rip), %rdi
	movl	$23, %esi
	jmp	.LBB40_2
.LBB40_10:
	movq	_RNvXsd_NtNtNtCsgEmfK2I1SDS_4core3fmt3num3impyNtB9_7Display3fmt@GOTPCREL(%rip), %rcx
	leaq	16(%rsp), %rax
	leaq	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.101(%rip), %rsi
	leaq	40(%rsp), %rbx
	leaq	24(%rsp), %rdx
	movq	%rax, 24(%rsp)
	movq	%rcx, 32(%rsp)
	jmp	.LBB40_12
.LBB40_25:
	leaq	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.82(%rip), %rdi
	movl	$24, %esi
.LBB40_2:
	addq	$88, %rsp
	.cfi_def_cfa_offset 40
	popq	%rbx
	.cfi_def_cfa_offset 32
	popq	%r14
	.cfi_def_cfa_offset 24
	popq	%r15
	.cfi_def_cfa_offset 16
	popq	%rbp
	.cfi_def_cfa_offset 8
	jmp	_ZN5prost5error11DecodeError3new17h19d861c2f6bb76b8E
.Lfunc_end40:
	.size	_ZN5prost8encoding10skip_field17hecfc4b3cbbe23b00E, .Lfunc_end40-_ZN5prost8encoding10skip_field17hecfc4b3cbbe23b00E
	.cfi_endproc
	.section	.rodata._ZN5prost8encoding10skip_field17hecfc4b3cbbe23b00E,"a",@progbits
	.p2align	2, 0x0
.LJTI40_0:
	.long	.LBB40_4-.LJTI40_0
	.long	.LBB40_16-.LJTI40_0
	.long	.LBB40_19-.LJTI40_0
	.long	.LBB40_6-.LJTI40_0
	.long	.LBB40_25-.LJTI40_0
	.long	.LBB40_15-.LJTI40_0

	.section	.text._ZN5prost8encoding6varint13decode_varint17h5757c281c4676585E,"ax",@progbits
	.p2align	4
	.type	_ZN5prost8encoding6varint13decode_varint17h5757c281c4676585E,@function
_ZN5prost8encoding6varint13decode_varint17h5757c281c4676585E:
	.cfi_startproc
	pushq	%rax
	.cfi_def_cfa_offset 16
	movq	8(%rdi), %rax
	testq	%rax, %rax
	je	.LBB41_26
	movq	(%rdi), %rcx
	movsbq	(%rcx), %rdx
	testq	%rdx, %rdx
	js	.LBB41_3
	decq	%rax
	incq	%rcx
	jmp	.LBB41_25
.LBB41_3:
	cmpq	$10, %rax
	ja	.LBB41_5
	cmpb	$0, -1(%rcx,%rax)
	js	.LBB41_27
.LBB41_5:
	movzbl	1(%rcx), %esi
	movzbl	%dl, %edx
	movl	%esi, %r8d
	shll	$7, %esi
	leal	-128(%rdx,%rsi), %edx
	testb	%r8b, %r8b
	js	.LBB41_7
	movl	$2, %esi
	movl	%edx, %edx
	jmp	.LBB41_24
.LBB41_7:
	movzbl	2(%rcx), %esi
	movl	%esi, %r8d
	shll	$14, %esi
	leal	-16384(%rdx,%rsi), %edx
	testb	%r8b, %r8b
	js	.LBB41_9
	movl	%edx, %edx
	movl	$3, %esi
	jmp	.LBB41_24
.LBB41_9:
	movzbl	3(%rcx), %esi
	movl	%esi, %r8d
	shll	$21, %esi
	leal	-2097152(%rdx,%rsi), %esi
	testb	%r8b, %r8b
	js	.LBB41_11
	movl	%esi, %edx
	movl	$4, %esi
	jmp	.LBB41_24
.LBB41_11:
	movzbl	4(%rcx), %edx
	addl	$-268435456, %esi
	testb	%dl, %dl
	js	.LBB41_13
	movl	%edx, %edx
	shlq	$28, %rdx
	addq	%rsi, %rdx
	movl	$5, %esi
	jmp	.LBB41_24
.LBB41_13:
	movzbl	5(%rcx), %r8d
	movl	%r8d, %r9d
	shll	$7, %r8d
	leal	-128(%rdx,%r8), %edx
	testb	%r9b, %r9b
	js	.LBB41_15
	movl	%edx, %edx
	shlq	$28, %rdx
	addq	%rsi, %rdx
	movl	$6, %esi
	jmp	.LBB41_24
.LBB41_15:
	movzbl	6(%rcx), %r8d
	movl	%r8d, %r9d
	shll	$14, %r8d
	leal	-16384(%rdx,%r8), %edx
	testb	%r9b, %r9b
	js	.LBB41_17
	movl	%edx, %edx
	shlq	$28, %rdx
	addq	%rsi, %rdx
	movl	$7, %esi
	jmp	.LBB41_24
.LBB41_17:
	movzbl	7(%rcx), %r8d
	movl	%r8d, %r9d
	shll	$21, %r8d
	leal	-2097152(%rdx,%r8), %r8d
	testb	%r9b, %r9b
	js	.LBB41_19
	movl	%r8d, %edx
	shlq	$28, %rdx
	addq	%rsi, %rdx
	movl	$8, %esi
	jmp	.LBB41_24
.LBB41_19:
	movsbq	8(%rcx), %rdx
	addl	$-268435456, %r8d
	shlq	$28, %r8
	addq	%rsi, %r8
	testq	%rdx, %rdx
	js	.LBB41_21
	shlq	$56, %rdx
	movl	$9, %esi
	jmp	.LBB41_23
.LBB41_21:
	movzbl	9(%rcx), %esi
	cmpb	$2, %sil
	jae	.LBB41_26
	andb	$127, %dl
	shlb	$7, %sil
	orb	%dl, %sil
	movzbl	%sil, %edx
	movl	$10, %esi
	shlq	$56, %rdx
.LBB41_23:
	addq	%r8, %rdx
.LBB41_24:
	subq	%rsi, %rax
	addq	%rsi, %rcx
.LBB41_25:
	movq	%rcx, (%rdi)
	movq	%rax, 8(%rdi)
	xorl	%eax, %eax
	popq	%rcx
	.cfi_def_cfa_offset 8
	retq
.LBB41_26:
	.cfi_def_cfa_offset 16
	leaq	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.83(%rip), %rdi
	movl	$14, %esi
	callq	_ZN5prost5error11DecodeError3new17h19d861c2f6bb76b8E
	movq	%rax, %rdx
	movl	$1, %eax
	popq	%rcx
	.cfi_def_cfa_offset 8
	retq
.LBB41_27:
	.cfi_def_cfa_offset 16
	callq	_ZN5prost8encoding6varint18decode_varint_slow17h9575c3dfb9f83bceE
	popq	%rcx
	.cfi_def_cfa_offset 8
	retq
.Lfunc_end41:
	.size	_ZN5prost8encoding6varint13decode_varint17h5757c281c4676585E, .Lfunc_end41-_ZN5prost8encoding6varint13decode_varint17h5757c281c4676585E
	.cfi_endproc

	.section	.text._ZN5prost8encoding6varint13encode_varint17hfef3876196e789c2E,"ax",@progbits
	.p2align	4
	.type	_ZN5prost8encoding6varint13encode_varint17hfef3876196e789c2E,@function
_ZN5prost8encoding6varint13encode_varint17hfef3876196e789c2E:
	.cfi_startproc
	pushq	%r15
	.cfi_def_cfa_offset 16
	pushq	%r14
	.cfi_def_cfa_offset 24
	pushq	%rbx
	.cfi_def_cfa_offset 32
	.cfi_offset %rbx, -32
	.cfi_offset %r14, -24
	.cfi_offset %r15, -16
	movq	(%rsi), %r9
	movq	16(%rsi), %rax
	cmpq	$128, %rdi
	jb	.LBB42_12
	movl	%edi, %ebx
	orb	$-128, %bl
	cmpq	%rax, %r9
	je	.LBB42_15
.LBB42_2:
	movq	8(%rsi), %rcx
	movb	%bl, (%rcx,%rax)
	movq	%rdi, %rbx
	incq	%rax
	shrq	$7, %rbx
	movq	%rax, 16(%rsi)
	cmpq	$16384, %rdi
	jb	.LBB42_3
	orb	$-128, %bl
	cmpq	%rax, %r9
	je	.LBB42_16
.LBB42_5:
	movb	%bl, (%rcx,%rax)
	movq	%rdi, %rbx
	incq	%rax
	shrq	$14, %rbx
	movq	%rax, 16(%rsi)
	cmpq	$2097152, %rdi
	jb	.LBB42_3
	orb	$-128, %bl
	cmpq	%rax, %r9
	je	.LBB42_17
.LBB42_8:
	movb	%bl, (%rcx,%rax)
	incq	%rax
	movq	%rdi, %rbx
	shrq	$21, %rbx
	movq	%rax, 16(%rsi)
	movq	(%rsi), %r9
	cmpq	$268435456, %rdi
	jae	.LBB42_10
.LBB42_3:
	movq	%rbx, %rdi
.LBB42_12:
	cmpq	%rax, %r9
	je	.LBB42_14
.LBB42_13:
	movq	8(%rsi), %rcx
	movb	%dil, (%rcx,%rax)
	incq	%rax
	movq	%rax, 16(%rsi)
	popq	%rbx
	.cfi_def_cfa_offset 24
	popq	%r14
	.cfi_def_cfa_offset 16
	popq	%r15
	.cfi_def_cfa_offset 8
	retq
.LBB42_10:
	.cfi_def_cfa_offset 32
	orb	$-128, %bl
	cmpq	%rax, %r9
	je	.LBB42_18
.LBB42_11:
	movq	8(%rsi), %rcx
	shrq	$28, %rdi
	movb	%bl, (%rcx,%rax)
	incq	%rax
	movq	%rax, 16(%rsi)
	cmpq	%rax, %r9
	jne	.LBB42_13
.LBB42_14:
	movl	$1, %edx
	movl	$1, %ecx
	movl	$1, %r8d
	movq	%rdi, %rbx
	movq	%rsi, %rdi
	movq	%rsi, %r14
	movq	%r9, %rsi
	callq	_ZN5alloc7raw_vec20RawVecInner$LT$A$GT$7reserve21do_reserve_and_handle17h5eeef80eb9b0c257E
	movq	16(%r14), %rax
	movq	%rbx, %rdi
	movq	%r14, %rsi
	jmp	.LBB42_13
.LBB42_15:
	movl	$1, %edx
	movl	$1, %ecx
	movl	$1, %r8d
	movq	%rdi, %r14
	movq	%rsi, %rdi
	movq	%rsi, %r15
	movq	%rax, %rsi
	callq	_ZN5alloc7raw_vec20RawVecInner$LT$A$GT$7reserve21do_reserve_and_handle17h5eeef80eb9b0c257E
	movq	(%r15), %r9
	movq	16(%r15), %rax
	movq	%r14, %rdi
	movq	%r15, %rsi
	jmp	.LBB42_2
.LBB42_16:
	movl	$1, %edx
	movl	$1, %ecx
	movl	$1, %r8d
	movq	%rdi, %r14
	movq	%rsi, %rdi
	movq	%rsi, %r15
	movq	%r9, %rsi
	callq	_ZN5alloc7raw_vec20RawVecInner$LT$A$GT$7reserve21do_reserve_and_handle17h5eeef80eb9b0c257E
	movq	16(%r15), %rax
	movq	(%r15), %r9
	movq	8(%r15), %rcx
	movq	%r14, %rdi
	movq	%r15, %rsi
	jmp	.LBB42_5
.LBB42_17:
	movl	$1, %edx
	movl	$1, %ecx
	movl	$1, %r8d
	movq	%rdi, %r14
	movq	%rsi, %rdi
	movq	%rsi, %r15
	movq	%r9, %rsi
	callq	_ZN5alloc7raw_vec20RawVecInner$LT$A$GT$7reserve21do_reserve_and_handle17h5eeef80eb9b0c257E
	movq	8(%r15), %rcx
	movq	16(%r15), %rax
	movq	%r14, %rdi
	movq	%r15, %rsi
	jmp	.LBB42_8
.LBB42_18:
	movl	$1, %edx
	movl	$1, %ecx
	movl	$1, %r8d
	movq	%rdi, %r14
	movq	%rsi, %rdi
	movq	%rsi, %r15
	movq	%rax, %rsi
	callq	_ZN5alloc7raw_vec20RawVecInner$LT$A$GT$7reserve21do_reserve_and_handle17h5eeef80eb9b0c257E
	movq	(%r15), %r9
	movq	16(%r15), %rax
	movq	%r14, %rdi
	movq	%r15, %rsi
	jmp	.LBB42_11
.Lfunc_end42:
	.size	_ZN5prost8encoding6varint13encode_varint17hfef3876196e789c2E, .Lfunc_end42-_ZN5prost8encoding6varint13encode_varint17hfef3876196e789c2E
	.cfi_endproc

	.section	.text.unlikely._ZN5prost8encoding6varint18decode_varint_slow17h9575c3dfb9f83bceE,"ax",@progbits
	.p2align	4
	.type	_ZN5prost8encoding6varint18decode_varint_slow17h9575c3dfb9f83bceE,@function
_ZN5prost8encoding6varint18decode_varint_slow17h9575c3dfb9f83bceE:
	.cfi_startproc
	pushq	%rax
	.cfi_def_cfa_offset 16
	movq	8(%rdi), %rax
	movl	$10, %esi
	cmpq	$10, %rax
	cmovbq	%rax, %rsi
	testq	%rax, %rax
	je	.LBB43_3
	movq	(%rdi), %r8
	decq	%rsi
	xorl	%ecx, %ecx
	xorl	%r9d, %r9d
	xorl	%edx, %edx
	.p2align	4
.LBB43_5:
	movq	%rdx, %r11
	movzbl	(%r8,%rcx), %edx
	movl	%edx, %r10d
	andl	$127, %edx
	shlxq	%r9, %rdx, %rdx
	orq	%r11, %rdx
	testb	%r10b, %r10b
	jns	.LBB43_7
	cmpq	%rcx, %rsi
	je	.LBB43_2
	incq	%rcx
	addq	$7, %r9
	movq	%rax, %r10
	subq	%rcx, %r10
	jne	.LBB43_5
	addq	%rcx, %r8
	movq	%r8, (%rdi)
	movq	%r10, 8(%rdi)
	leaq	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.76(%rip), %rdi
	callq	*_ZN5bytes13panic_advance17hbb273e3b3bca4d8fE@GOTPCREL(%rip)
.LBB43_7:
	leaq	1(%r8,%rcx), %rsi
	movq	%rcx, %r8
	notq	%r8
	addq	%rax, %r8
	cmpq	$9, %rcx
	sete	%al
	cmpb	$2, %r10b
	movq	%rsi, (%rdi)
	movq	%r8, 8(%rdi)
	setae	%cl
	testb	%cl, %al
	jne	.LBB43_3
	xorl	%eax, %eax
	popq	%rcx
	.cfi_def_cfa_offset 8
	retq
.LBB43_2:
	.cfi_def_cfa_offset 16
	leaq	1(%r8,%rcx), %rdx
	notq	%rcx
	addq	%rax, %rcx
	movq	%rdx, (%rdi)
	movq	%rcx, 8(%rdi)
.LBB43_3:
	leaq	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.83(%rip), %rdi
	movl	$14, %esi
	callq	_ZN5prost5error11DecodeError3new17h19d861c2f6bb76b8E
	movq	%rax, %rdx
	movl	$1, %eax
	popq	%rcx
	.cfi_def_cfa_offset 8
	retq
.Lfunc_end43:
	.size	_ZN5prost8encoding6varint18decode_varint_slow17h9575c3dfb9f83bceE, .Lfunc_end43-_ZN5prost8encoding6varint18decode_varint_slow17h9575c3dfb9f83bceE
	.cfi_endproc

	.section	".text._ZN62_$LT$prost..error..EncodeError$u20$as$u20$core..fmt..Debug$GT$3fmt17hd2504d7eb9dd6f24E","ax",@progbits
	.p2align	4
	.type	_ZN62_$LT$prost..error..EncodeError$u20$as$u20$core..fmt..Debug$GT$3fmt17hd2504d7eb9dd6f24E,@function
_ZN62_$LT$prost..error..EncodeError$u20$as$u20$core..fmt..Debug$GT$3fmt17hd2504d7eb9dd6f24E:
	.cfi_startproc
	pushq	%r14
	.cfi_def_cfa_offset 16
	pushq	%rbx
	.cfi_def_cfa_offset 24
	pushq	%rax
	.cfi_def_cfa_offset 32
	.cfi_offset %rbx, -24
	.cfi_offset %r14, -16
	leaq	8(%rdi), %rcx
	movq	%rsi, %rax
	movq	%rdi, %r9
	movq	%rcx, (%rsp)
	subq	$8, %rsp
	.cfi_adjust_cfa_offset 8
	leaq	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.86(%rip), %r10
	leaq	8(%rsp), %r11
	leaq	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.89(%rip), %rbx
	leaq	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.85(%rip), %r14
	leaq	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.87(%rip), %rsi
	leaq	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.88(%rip), %rcx
	movl	$11, %edx
	movl	$8, %r8d
	movq	%rax, %rdi
	pushq	%r10
	.cfi_adjust_cfa_offset 8
	pushq	%r11
	.cfi_adjust_cfa_offset 8
	pushq	$9
	.cfi_adjust_cfa_offset 8
	pushq	%rbx
	.cfi_adjust_cfa_offset 8
	pushq	%r14
	.cfi_adjust_cfa_offset 8
	callq	*_RNvMsa_NtCsgEmfK2I1SDS_4core3fmtNtB5_9Formatter26debug_struct_field2_finish@GOTPCREL(%rip)
	addq	$56, %rsp
	.cfi_adjust_cfa_offset -56
	popq	%rbx
	.cfi_def_cfa_offset 16
	popq	%r14
	.cfi_def_cfa_offset 8
	retq
.Lfunc_end44:
	.size	_ZN62_$LT$prost..error..EncodeError$u20$as$u20$core..fmt..Debug$GT$3fmt17hd2504d7eb9dd6f24E, .Lfunc_end44-_ZN62_$LT$prost..error..EncodeError$u20$as$u20$core..fmt..Debug$GT$3fmt17hd2504d7eb9dd6f24E
	.cfi_endproc

	.section	".text._ZN73_$LT$prost..encoding..wire_type..WireType$u20$as$u20$core..fmt..Debug$GT$3fmt17hb528d417acae5beaE","ax",@progbits
	.p2align	4
	.type	_ZN73_$LT$prost..encoding..wire_type..WireType$u20$as$u20$core..fmt..Debug$GT$3fmt17hb528d417acae5beaE,@function
_ZN73_$LT$prost..encoding..wire_type..WireType$u20$as$u20$core..fmt..Debug$GT$3fmt17hb528d417acae5beaE:
	.cfi_startproc
	movzbl	(%rdi), %ecx
	leaq	.Lswitch.table._ZN73_$LT$prost..encoding..wire_type..WireType$u20$as$u20$core..fmt..Debug$GT$3fmt17hb528d417acae5beaE.168.rel(%rip), %rdi
	leaq	.Lswitch.table._ZN73_$LT$prost..encoding..wire_type..WireType$u20$as$u20$core..fmt..Debug$GT$3fmt17hb528d417acae5beaE(%rip), %rdx
	movq	%rsi, %rax
	movslq	(%rdi,%rcx,4), %rsi
	movq	(%rdx,%rcx,8), %rdx
	addq	%rdi, %rsi
	movq	%rax, %rdi
	jmpq	*_RNvMsa_NtCsgEmfK2I1SDS_4core3fmtNtB5_9Formatter9write_str@GOTPCREL(%rip)
.Lfunc_end45:
	.size	_ZN73_$LT$prost..encoding..wire_type..WireType$u20$as$u20$core..fmt..Debug$GT$3fmt17hb528d417acae5beaE, .Lfunc_end45-_ZN73_$LT$prost..encoding..wire_type..WireType$u20$as$u20$core..fmt..Debug$GT$3fmt17hb528d417acae5beaE
	.cfi_endproc

	.section	.text.main,"ax",@progbits
	.globl	main
	.p2align	4
	.type	main,@function
main:
	.cfi_startproc
	pushq	%rax
	.cfi_def_cfa_offset 16
	movq	%rsi, %rcx
	movslq	%edi, %rdx
	leaq	_ZN21protobuf_decode_bench4main17ha1ffaea49e936ebfE(%rip), %rax
	leaq	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.65(%rip), %rsi
	movq	%rsp, %rdi
	xorl	%r8d, %r8d
	movq	%rax, (%rsp)
	callq	*_RNvNtCsjrHSEGnQ3l9_3std2rt19lang_start_internal@GOTPCREL(%rip)
	popq	%rcx
	.cfi_def_cfa_offset 8
	retq
.Lfunc_end46:
	.size	main, .Lfunc_end46-main
	.cfi_endproc

	.type	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.0,@object
	.section	.rodata..Lanon.e78093d818c3c2c2ebd9e44bad9f3825.0,"a",@progbits
.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.0:
	.ascii	"NotPresent"
	.size	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.0, 10

	.type	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.1,@object
	.section	.data.rel.ro..Lanon.e78093d818c3c2c2ebd9e44bad9f3825.1,"aw",@progbits
	.p2align	3, 0x0
.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.1:
	.asciz	"\000\000\000\000\000\000\000\000\b\000\000\000\000\000\000\000\b\000\000\000\000\000\000"
	.quad	_ZN42_$LT$$RF$T$u20$as$u20$core..fmt..Debug$GT$3fmt17h2676d877de11e1abE
	.size	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.1, 32

	.type	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.2,@object
	.section	.rodata..Lanon.e78093d818c3c2c2ebd9e44bad9f3825.2,"a",@progbits
.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.2:
	.ascii	"NotUnicode"
	.size	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.2, 10

	.type	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.3,@object
	.section	.rodata..Lanon.e78093d818c3c2c2ebd9e44bad9f3825.3,"a",@progbits
.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.3:
	.ascii	"BENCH_RUN_LABEL"
	.size	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.3, 15

	.type	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.4,@object
	.section	.rodata..Lanon.e78093d818c3c2c2ebd9e44bad9f3825.4,"a",@progbits
.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.4:
	.ascii	"BENCH_COMMIT"
	.size	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.4, 12

	.type	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.5,@object
	.section	.rodata..Lanon.e78093d818c3c2c2ebd9e44bad9f3825.5,"a",@progbits
.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.5:
	.ascii	"BENCH_SOURCE_SHA256"
	.size	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.5, 19

	.type	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.6,@object
	.section	.rodata..Lanon.e78093d818c3c2c2ebd9e44bad9f3825.6,"a",@progbits
.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.6:
	.ascii	"BENCH_RUSTC"
	.size	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.6, 11

	.type	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.7,@object
	.section	.rodata..Lanon.e78093d818c3c2c2ebd9e44bad9f3825.7,"a",@progbits
.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.7:
	.ascii	"BENCH_AFFINITY"
	.size	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.7, 14

	.type	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.8,@object
	.section	.rodata..Lanon.e78093d818c3c2c2ebd9e44bad9f3825.8,"a",@progbits
.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.8:
	.ascii	"BENCH_WORKTREE_STATUS"
	.size	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.8, 21

	.type	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.9,@object
	.section	.rodata..Lanon.e78093d818c3c2c2ebd9e44bad9f3825.9,"a",@progbits
.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.9:
	.ascii	"BENCH_RUSTFLAGS"
	.size	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.9, 15

	.type	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.10,@object
	.section	.rodata..Lanon.e78093d818c3c2c2ebd9e44bad9f3825.10,"a",@progbits
.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.10:
	.ascii	"BENCH_EFFECTIVE_RUSTC_FLAGS"
	.size	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.10, 27

	.type	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.11,@object
	.section	.rodata.str1.1,"aMS",@progbits,1
.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.11:
	.asciz	"\034required benchmark metadata \300\t is unset"
	.size	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.11, 41

	.type	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.12,@object
.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.12:
	.asciz	"src/bin/protobuf_decode_bench.rs"
	.size	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.12, 33

	.type	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.13,@object
	.section	.data.rel.ro..Lanon.e78093d818c3c2c2ebd9e44bad9f3825.13,"aw",@progbits
	.p2align	3, 0x0
.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.13:
	.quad	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.12
	.asciz	" \000\000\000\000\000\000\000D\000\000\000%\000\000"
	.size	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.13, 24

	.type	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.14,@object
	.section	.data.rel.ro..Lanon.e78093d818c3c2c2ebd9e44bad9f3825.14,"aw",@progbits
	.p2align	3, 0x0
.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.14:
	.quad	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.12
	.asciz	" \000\000\000\000\000\000\000\237\000\000\000\n\000\000"
	.size	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.14, 24

	.type	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.15,@object
	.section	.rodata..Lanon.e78093d818c3c2c2ebd9e44bad9f3825.15,"a",@progbits
.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.15:
	.ascii	"sysconf(_SC_PAGESIZE) did not return a positive page size"
	.size	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.15, 57

	.type	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.16,@object
	.section	.data.rel.ro..Lanon.e78093d818c3c2c2ebd9e44bad9f3825.16,"aw",@progbits
	.p2align	3, 0x0
.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.16:
	.quad	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.12
	.asciz	" \000\000\000\000\000\000\000\216\000\000\000+\000\000"
	.size	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.16, 24

	.type	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.17,@object
	.section	.rodata..Lanon.e78093d818c3c2c2ebd9e44bad9f3825.17,"a",@progbits
.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.17:
	.ascii	"encode benchmark input"
	.size	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.17, 22

	.type	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.18,@object
	.section	.data.rel.ro..Lanon.e78093d818c3c2c2ebd9e44bad9f3825.18,"aw",@progbits
	.p2align	3, 0x0
.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.18:
	.quad	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.12
	.asciz	" \000\000\000\000\000\000\000!\001\000\000\036\000\000"
	.size	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.18, 24

	.type	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.19,@object
	.section	.rodata..Lanon.e78093d818c3c2c2ebd9e44bad9f3825.19,"a",@progbits
.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.19:
	.ascii	"prefault benchmark input"
	.size	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.19, 24

	.type	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.20,@object
	.section	.data.rel.ro..Lanon.e78093d818c3c2c2ebd9e44bad9f3825.20,"aw",@progbits
	.p2align	3, 0x0
.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.20:
	.quad	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.12
	.asciz	" \000\000\000\000\000\000\000\"\001\000\000\034\000\000"
	.size	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.20, 24

	.type	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.21,@object
	.section	.rodata..Lanon.e78093d818c3c2c2ebd9e44bad9f3825.21,"a",@progbits
.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.21:
	.ascii	"decode validation input"
	.size	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.21, 23

	.type	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.22,@object
	.section	.data.rel.ro..Lanon.e78093d818c3c2c2ebd9e44bad9f3825.22,"aw",@progbits
	.p2align	3, 0x0
.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.22:
	.quad	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.12
	.asciz	" \000\000\000\000\000\000\000&\001\000\0007\000\000"
	.size	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.22, 24

	.type	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.23,@object
	.section	.data.rel.ro..Lanon.e78093d818c3c2c2ebd9e44bad9f3825.23,"aw",@progbits
	.p2align	3, 0x0
.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.23:
	.quad	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.12
	.asciz	" \000\000\000\000\000\000\000'\001\000\000\005\000\000"
	.size	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.23, 24

	.type	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.24,@object
	.section	.data.rel.ro..Lanon.e78093d818c3c2c2ebd9e44bad9f3825.24,"aw",@progbits
	.p2align	3, 0x0
.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.24:
	.quad	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.12
	.asciz	" \000\000\000\000\000\000\000/\001\000\000\005\000\000"
	.size	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.24, 24

	.type	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.25,@object
	.section	.data.rel.ro..Lanon.e78093d818c3c2c2ebd9e44bad9f3825.25,"aw",@progbits
	.p2align	3, 0x0
.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.25:
	.asciz	"\000\000\000\000\000\000\000\000\b\000\000\000\000\000\000\000\b\000\000\000\000\000\000"
	.quad	_ZN4core3ops8function6FnOnce40call_once$u7b$$u7b$vtable.shim$u7d$$u7d$17h5021214c16a0eb18E
	.quad	_ZN21protobuf_decode_bench4main28_$u7b$$u7b$closure$u7d$$u7d$17h8393f7a28191b620E
	.quad	_ZN21protobuf_decode_bench4main28_$u7b$$u7b$closure$u7d$$u7d$17h8393f7a28191b620E
	.size	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.25, 48

	.type	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.26,@object
	.section	.rodata..Lanon.e78093d818c3c2c2ebd9e44bad9f3825.26,"a",@progbits
.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.26:
	.ascii	"Value-Only"
	.size	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.26, 10

	.type	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.27,@object
	.section	.rodata..Lanon.e78093d818c3c2c2ebd9e44bad9f3825.27,"a",@progbits
.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.27:
	.ascii	"Prost decode into owned Vec<ProtoRecord> + sum(value)"
	.size	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.27, 53

	.type	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.28,@object
	.section	.data.rel.ro..Lanon.e78093d818c3c2c2ebd9e44bad9f3825.28,"aw",@progbits
	.p2align	3, 0x0
.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.28:
	.asciz	"\000\000\000\000\000\000\000\000\b\000\000\000\000\000\000\000\b\000\000\000\000\000\000"
	.quad	_ZN4core3ops8function6FnOnce40call_once$u7b$$u7b$vtable.shim$u7d$$u7d$17hec0797272d3fcec1E
	.quad	_ZN21protobuf_decode_bench4main28_$u7b$$u7b$closure$u7d$$u7d$17he017dc1e830bdeb0E
	.quad	_ZN21protobuf_decode_bench4main28_$u7b$$u7b$closure$u7d$$u7d$17he017dc1e830bdeb0E
	.size	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.28, 48

	.type	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.29,@object
	.section	.rodata..Lanon.e78093d818c3c2c2ebd9e44bad9f3825.29,"a",@progbits
.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.29:
	.ascii	"Full-Record"
	.size	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.29, 11

	.type	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.30,@object
	.section	.rodata..Lanon.e78093d818c3c2c2ebd9e44bad9f3825.30,"a",@progbits
.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.30:
	.ascii	"Prost decode into owned Vec<ProtoRecord> + sum(id as f64 + value)"
	.size	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.30, 65

	.type	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.31,@object
	.section	.rodata.str1.1,"aMS",@progbits,1
.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.31:
	.asciz	"\300"
	.size	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.31, 2

	.type	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.32,@object
	.section	.rodata..Lanon.e78093d818c3c2c2ebd9e44bad9f3825.32,"a",@progbits
.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.32:
	.ascii	"BENCH_REPORT_PATH"
	.size	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.32, 17

	.type	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.33,@object
	.section	.rodata..Lanon.e78093d818c3c2c2ebd9e44bad9f3825.33,"a",@progbits
.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.33:
	.ascii	"required benchmark metadata BENCH_REPORT_PATH is unset"
	.size	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.33, 54

	.type	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.34,@object
	.section	.data.rel.ro..Lanon.e78093d818c3c2c2ebd9e44bad9f3825.34,"aw",@progbits
	.p2align	3, 0x0
.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.34:
	.quad	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.12
	.asciz	" \000\000\000\000\000\000\000~\001\000\000\n\000\000"
	.size	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.34, 24

	.type	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.35,@object
	.section	.rodata..Lanon.e78093d818c3c2c2ebd9e44bad9f3825.35,"a",@progbits
.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.35:
	.ascii	"write raw benchmark report"
	.size	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.35, 26

	.type	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.36,@object
	.section	.data.rel.ro..Lanon.e78093d818c3c2c2ebd9e44bad9f3825.36,"aw",@progbits
	.p2align	3, 0x0
.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.36:
	.quad	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.12
	.asciz	" \000\000\000\000\000\000\000\177\001\000\000&\000\000"
	.size	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.36, 24

	.type	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.37,@object
	.section	.rodata..Lanon.e78093d818c3c2c2ebd9e44bad9f3825.37,"a",@progbits
.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.37:
	.ascii	"migration retry limit reached"
	.size	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.37, 29

	.type	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.38,@object
	.section	.data.rel.ro..Lanon.e78093d818c3c2c2ebd9e44bad9f3825.38,"aw",@progbits
	.p2align	3, 0x0
.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.38:
	.quad	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.12
	.asciz	" \000\000\000\000\000\000\000l\001\000\000\025\000\000"
	.size	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.38, 24

	.type	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.39,@object
	.section	.rodata..Lanon.e78093d818c3c2c2ebd9e44bad9f3825.39,"a",@progbits
.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.39:
	.ascii	"decode benchmark input"
	.size	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.39, 22

	.type	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.40,@object
	.section	.data.rel.ro..Lanon.e78093d818c3c2c2ebd9e44bad9f3825.40,"aw",@progbits
	.p2align	3, 0x0
.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.40:
	.quad	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.12
	.asciz	" \000\000\000\000\000\000\000@\001\000\000=\000\000"
	.size	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.40, 24

	.type	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.41,@object
	.section	.data.rel.ro..Lanon.e78093d818c3c2c2ebd9e44bad9f3825.41,"aw",@progbits
	.p2align	3, 0x0
.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.41:
	.quad	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.12
	.asciz	" \000\000\000\000\000\000\000L\001\000\000=\000\000"
	.size	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.41, 24

	.type	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.42,@object
	.section	.rodata..Lanon.e78093d818c3c2c2ebd9e44bad9f3825.42,"a",@progbits
.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.42:
	.ascii	"assertion failed: !samples.is_empty()"
	.size	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.42, 37

	.type	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.43,@object
	.section	.data.rel.ro..Lanon.e78093d818c3c2c2ebd9e44bad9f3825.43,"aw",@progbits
	.p2align	3, 0x0
.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.43:
	.quad	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.12
	.asciz	" \000\000\000\000\000\000\000\246\000\000\000\005\000\000"
	.size	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.43, 24

	.type	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.44,@object
	.section	.rodata..Lanon.e78093d818c3c2c2ebd9e44bad9f3825.44,"a",@progbits
.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.44:
	.ascii	"unavailable-for-at-least-one-sample"
	.size	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.44, 35

	.type	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.45,@object
	.section	.rodata..Lanon.e78093d818c3c2c2ebd9e44bad9f3825.45,"a",@progbits
.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.45:
	.ascii	"available"
	.size	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.45, 9

	.type	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.46,@object
	.section	.rodata..Lanon.e78093d818c3c2c2ebd9e44bad9f3825.46,"a",@progbits
.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.46:
	.ascii	"# isolated Prost decode raw benchmark report\n"
	.size	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.46, 45

	.type	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.47,@object
	.section	.rodata..Lanon.e78093d818c3c2c2ebd9e44bad9f3825.47,"a",@progbits
.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.47:
	.ascii	"run_label"
	.size	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.47, 9

	.type	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.48,@object
	.section	.rodata..Lanon.e78093d818c3c2c2ebd9e44bad9f3825.48,"a",@progbits
.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.48:
	.ascii	"commit"
	.size	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.48, 6

	.type	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.49,@object
	.section	.rodata..Lanon.e78093d818c3c2c2ebd9e44bad9f3825.49,"a",@progbits
.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.49:
	.ascii	"source_sha256"
	.size	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.49, 13

	.type	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.50,@object
	.section	.rodata..Lanon.e78093d818c3c2c2ebd9e44bad9f3825.50,"a",@progbits
.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.50:
	.ascii	"rustc"
	.size	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.50, 5

	.type	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.51,@object
	.section	.rodata.cst8,"aM",@progbits,8
.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.51:
	.ascii	"affinity"
	.size	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.51, 8

	.type	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.52,@object
	.section	.rodata..Lanon.e78093d818c3c2c2ebd9e44bad9f3825.52,"a",@progbits
.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.52:
	.ascii	"worktree_status"
	.size	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.52, 15

	.type	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.53,@object
	.section	.rodata..Lanon.e78093d818c3c2c2ebd9e44bad9f3825.53,"a",@progbits
.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.53:
	.ascii	"rustflags"
	.size	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.53, 9

	.type	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.54,@object
	.section	.rodata..Lanon.e78093d818c3c2c2ebd9e44bad9f3825.54,"a",@progbits
.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.54:
	.ascii	"effective_rustc_flags"
	.size	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.54, 21

	.type	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.55,@object
	.section	.rodata.cst8,"aM",@progbits,8
	.p2align	3, 0x0
.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.55:
	.asciz	"@B\017\000\000\000\000"
	.size	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.55, 8

	.type	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.56,@object
	.p2align	3, 0x0
.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.56:
	.asciz	"\005\000\000\000\000\000\000"
	.size	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.56, 8

	.type	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.57,@object
	.p2align	3, 0x0
.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.57:
	.asciz	"\036\000\000\000\000\000\000"
	.size	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.57, 8

	.type	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.58,@object
	.section	.rodata.str1.1,"aMS",@progbits,1
.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.58:
	.asciz	"\023cpu_identification=\300\t\nrecords=\300\016\nwarm_up_runs=\300\017\nmeasured_runs=\300\200\260\002\nexecution_order=deterministic-balanced-two-target-rotation; Value-Only position-0 on even rounds and Full-Record position-0 on odd rounds\ninput=one-prefaulted-Protobuf-byte-buffer-reused-for-all-samples\ntimed_operation=decode-into-owned-Vec<ProtoRecord>; numeric-sum-reduction; local-decoded-value-is-dropped-before-closure-return\ntiming_windows=Instant starts after TSC start and ends before TSC end; diagnostic TSC rate is ticks divided by Instant duration and is not active-core-frequency\nblack_box=returned-numeric-aggregation-is-made-opaque-before-end-timestamp; barrier-overhead-is-timed\nquantiles=floor-index: floor(n*p), clamped to n-1\nstandard_deviation=population: divide by N\n"
	.size	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.58, 757

	.type	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.59,@object
	.section	.rodata..Lanon.e78093d818c3c2c2ebd9e44bad9f3825.59,"a",@progbits
.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.59:
	.ascii	"kind,workload,operation,accepted_samples,rejected_migration_samples,median_ms,mean_ms,population_stddev_ms,p5_ms,p95_ms,min_ms,max_ms,median_diagnostic_tsc_rate_ghz\n"
	.size	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.59, 165

	.type	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.60,@object
	.section	.rodata..Lanon.e78093d818c3c2c2ebd9e44bad9f3825.60,"a",@progbits
.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.60:
	.ascii	"kind,workload,operation,round,position,duration_ns,tsc_ticks,diagnostic_tsc_rate_ghz,cpu_before,cpu_after,migrated\n"
	.size	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.60, 115

	.type	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.61,@object
	.section	.rodata..Lanon.e78093d818c3c2c2ebd9e44bad9f3825.61,"a",@progbits
.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.61:
	.ascii	"unavailable"
	.size	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.61, 11

	.type	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.62,@object
	.section	.rodata..Lanon.e78093d818c3c2c2ebd9e44bad9f3825.62,"a",@progbits
.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.62:
	.asciz	"\007sample,\300\001,\300\001,\300\001,\300\001,\301 \000\000p\001,\300\001,\305 \000\000p\t\000\001,\300\001,\300\007,false\n"
	.size	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.62, 52

	.type	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.63,@object
	.section	.rodata..Lanon.e78093d818c3c2c2ebd9e44bad9f3825.63,"a",@progbits
.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.63:
	.asciz	"\bsummary,\300\001,\300\001,\300\001,\300\001,\305 \000\000p\t\000\001,\305 \000\000p\t\000\001,\305 \000\000p\t\000\001,\305 \000\000p\t\000\001,\305 \000\000p\t\000\001,\305 \000\000p\t\000\001,\305 \000\000p\t\000\001,\305 \000\000p\t\000\001\n"
	.size	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.63, 94

	.type	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.64,@object
	.section	.rodata.str1.1,"aMS",@progbits,1
.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.64:
	.asciz	"\300\001=\300\001\n"
	.size	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.64, 7

	.type	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.65,@object
	.section	.data.rel.ro..Lanon.e78093d818c3c2c2ebd9e44bad9f3825.65,"aw",@progbits
	.p2align	3, 0x0
.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.65:
	.asciz	"\000\000\000\000\000\000\000\000\b\000\000\000\000\000\000\000\b\000\000\000\000\000\000"
	.quad	_ZN4core3ops8function6FnOnce40call_once$u7b$$u7b$vtable.shim$u7d$$u7d$17hd73245c67b3b00b0E
	.quad	_ZN3std2rt10lang_start28_$u7b$$u7b$closure$u7d$$u7d$17h5c9919aee196cda5E
	.quad	_ZN3std2rt10lang_start28_$u7b$$u7b$closure$u7d$$u7d$17h5c9919aee196cda5E
	.size	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.65, 48

	.type	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.66,@object
	.section	.rodata..Lanon.e78093d818c3c2c2ebd9e44bad9f3825.66,"a",@progbits
.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.66:
	.ascii	"mid > len"
	.size	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.66, 9

	.type	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.67,@object
	.section	.rodata.str1.1,"aMS",@progbits,1
.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.67:
	.asciz	"/rustc/59807616e1fa2540724bfbac14d7976d7e4a3860/library/core/src/slice/sort/stable/quicksort.rs"
	.size	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.67, 96

	.type	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.68,@object
	.section	.data.rel.ro..Lanon.e78093d818c3c2c2ebd9e44bad9f3825.68,"aw",@progbits
	.p2align	3, 0x0
.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.68:
	.quad	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.67
	.asciz	"_\000\000\000\000\000\000\000M\000\000\000\037\000\000"
	.size	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.68, 24

	.type	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.69,@object
	.section	.data.rel.ro..Lanon.e78093d818c3c2c2ebd9e44bad9f3825.69,"aw",@progbits
	.p2align	3, 0x0
.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.69:
	.quad	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.67
	.asciz	"_\000\000\000\000\000\000\000G\000\000\000\027\000\000"
	.size	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.69, 24

	.type	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.70,@object
	.section	.data.rel.ro..Lanon.e78093d818c3c2c2ebd9e44bad9f3825.70,"aw",@progbits
	.p2align	3, 0x0
.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.70:
	.asciz	"\000\000\000\000\000\000\000\000\020\000\000\000\000\000\000\000\b\000\000\000\000\000\000"
	.quad	_ZN62_$LT$prost..error..EncodeError$u20$as$u20$core..fmt..Debug$GT$3fmt17hd2504d7eb9dd6f24E
	.size	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.70, 32

	.type	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.71,@object
	.section	.data.rel.ro..Lanon.e78093d818c3c2c2ebd9e44bad9f3825.71,"aw",@progbits
	.p2align	3, 0x0
.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.71:
	.asciz	"\000\000\000\000\000\000\000\000\020\000\000\000\000\000\000\000\b\000\000\000\000\000\000"
	.quad	_ZN42_$LT$$RF$T$u20$as$u20$core..fmt..Debug$GT$3fmt17h9cb5016884c992d4E
	.size	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.71, 32

	.type	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.72,@object
	.section	.data.rel.ro..Lanon.e78093d818c3c2c2ebd9e44bad9f3825.72,"aw",@progbits
	.p2align	3, 0x0
.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.72:
	.quad	_ZN4core3ptr39drop_in_place$LT$std..env..VarError$GT$17hd6eda7ca964ee990E
	.asciz	"\030\000\000\000\000\000\000\000\b\000\000\000\000\000\000"
	.quad	_RNvXsk_NtCsjrHSEGnQ3l9_3std3envNtB5_8VarErrorNtNtCsgEmfK2I1SDS_4core3fmt5Debug3fmt
	.size	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.72, 32

	.type	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.73,@object
	.section	.data.rel.ro..Lanon.e78093d818c3c2c2ebd9e44bad9f3825.73,"aw",@progbits
	.p2align	3, 0x0
.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.73:
	.quad	_ZN4core3ptr46drop_in_place$LT$prost..error..DecodeError$GT$17h6452743001005f2fE
	.asciz	"\b\000\000\000\000\000\000\000\b\000\000\000\000\000\000"
	.quad	_ZN62_$LT$prost..error..DecodeError$u20$as$u20$core..fmt..Debug$GT$3fmt17hf8167a64aff360a5E
	.size	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.73, 32

	.type	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.74,@object
	.section	.data.rel.ro..Lanon.e78093d818c3c2c2ebd9e44bad9f3825.74,"aw",@progbits
	.p2align	3, 0x0
.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.74:
	.quad	_ZN4core3ptr42drop_in_place$LT$std..io..error..Error$GT$17hcedecc219889b047E
	.asciz	"\b\000\000\000\000\000\000\000\b\000\000\000\000\000\000"
	.quad	_RNvXNtNtCsjrHSEGnQ3l9_3std2io5errorNtB2_5ErrorNtNtCsgEmfK2I1SDS_4core3fmt5Debug3fmt
	.size	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.74, 32

	.type	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.75,@object
	.section	.data.rel.ro..Lanon.e78093d818c3c2c2ebd9e44bad9f3825.75,"aw",@progbits
	.p2align	3, 0x0
.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.75:
	.asciz	"\000\000\000\000\000\000\000\000\b\000\000\000\000\000\000\000\b\000\000\000\000\000\000"
	.quad	_ZN42_$LT$$RF$T$u20$as$u20$core..fmt..Debug$GT$3fmt17h0fab7b41fe56f931E
	.size	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.75, 32

	.type	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.76,@object
	.section	.rodata.cst16,"aM",@progbits,16
	.p2align	3, 0x0
.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.76:
	.asciz	"\001\000\000\000\000\000\000\000\000\000\000\000\000\000\000"
	.size	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.76, 16

	.type	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.77,@object
	.section	.rodata..Lanon.e78093d818c3c2c2ebd9e44bad9f3825.77,"a",@progbits
.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.77:
	.ascii	"invalid tag value: 0"
	.size	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.77, 20

	.type	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.78,@object
	.section	.rodata.str1.1,"aMS",@progbits,1
.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.78:
	.asciz	"\023invalid key value: \300"
	.size	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.78, 22

	.type	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.79,@object
	.section	.rodata..Lanon.e78093d818c3c2c2ebd9e44bad9f3825.79,"a",@progbits
.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.79:
	.ascii	"delimited length exceeded"
	.size	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.79, 25

	.type	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.80,@object
	.section	.rodata.cst16,"aM",@progbits,16
.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.80:
	.ascii	"buffer underflow"
	.size	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.80, 16

	.type	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.81,@object
	.section	.rodata..Lanon.e78093d818c3c2c2ebd9e44bad9f3825.81,"a",@progbits
.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.81:
	.ascii	"recursion limit reached"
	.size	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.81, 23

	.type	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.82,@object
	.section	.rodata..Lanon.e78093d818c3c2c2ebd9e44bad9f3825.82,"a",@progbits
.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.82:
	.ascii	"unexpected end group tag"
	.size	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.82, 24

	.type	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.83,@object
	.section	.rodata..Lanon.e78093d818c3c2c2ebd9e44bad9f3825.83,"a",@progbits
.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.83:
	.ascii	"invalid varint"
	.size	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.83, 14

	.type	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.84,@object
	.section	.rodata.str1.1,"aMS",@progbits,1
.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.84:
	.asciz	"\023invalid wire type: \300\013 (expected \300\001)"
	.size	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.84, 37

	.type	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.85,@object
	.section	.data.rel.ro..Lanon.e78093d818c3c2c2ebd9e44bad9f3825.85,"aw",@progbits
	.p2align	3, 0x0
.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.85:
	.asciz	"\000\000\000\000\000\000\000\000\b\000\000\000\000\000\000\000\b\000\000\000\000\000\000"
	.quad	_RNvXsZ_NtNtCsgEmfK2I1SDS_4core3fmt3numjNtB7_5Debug3fmt
	.size	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.85, 32

	.type	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.86,@object
	.section	.data.rel.ro..Lanon.e78093d818c3c2c2ebd9e44bad9f3825.86,"aw",@progbits
	.p2align	3, 0x0
.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.86:
	.asciz	"\000\000\000\000\000\000\000\000\b\000\000\000\000\000\000\000\b\000\000\000\000\000\000"
	.quad	_ZN42_$LT$$RF$T$u20$as$u20$core..fmt..Debug$GT$3fmt17hbd12bc66adbbe4b9E
	.size	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.86, 32

	.type	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.87,@object
	.section	.rodata..Lanon.e78093d818c3c2c2ebd9e44bad9f3825.87,"a",@progbits
.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.87:
	.ascii	"EncodeError"
	.size	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.87, 11

	.type	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.88,@object
	.section	.rodata.cst8,"aM",@progbits,8
.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.88:
	.ascii	"required"
	.size	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.88, 8

	.type	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.89,@object
	.section	.rodata..Lanon.e78093d818c3c2c2ebd9e44bad9f3825.89,"a",@progbits
.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.89:
	.ascii	"remaining"
	.size	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.89, 9

	.type	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.90,@object
	.section	.rodata..Lanon.e78093d818c3c2c2ebd9e44bad9f3825.90,"a",@progbits
.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.90:
	.ascii	"Varint"
	.size	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.90, 6

	.type	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.91,@object
	.section	.rodata..Lanon.e78093d818c3c2c2ebd9e44bad9f3825.91,"a",@progbits
.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.91:
	.ascii	"SixtyFourBit"
	.size	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.91, 12

	.type	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.92,@object
	.section	.rodata..Lanon.e78093d818c3c2c2ebd9e44bad9f3825.92,"a",@progbits
.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.92:
	.ascii	"LengthDelimited"
	.size	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.92, 15

	.type	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.93,@object
	.section	.rodata..Lanon.e78093d818c3c2c2ebd9e44bad9f3825.93,"a",@progbits
.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.93:
	.ascii	"StartGroup"
	.size	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.93, 10

	.type	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.94,@object
	.section	.rodata.cst8,"aM",@progbits,8
.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.94:
	.ascii	"EndGroup"
	.size	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.94, 8

	.type	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.95,@object
	.section	.rodata..Lanon.e78093d818c3c2c2ebd9e44bad9f3825.95,"a",@progbits
.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.95:
	.ascii	"ThirtyTwoBit"
	.size	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.95, 12

	.type	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.96,@object
	.section	.rodata..Lanon.e78093d818c3c2c2ebd9e44bad9f3825.96,"a",@progbits
.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.96:
	.ascii	"ProtoRecord"
	.size	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.96, 11

	.type	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.97,@object
	.section	.rodata..Lanon.e78093d818c3c2c2ebd9e44bad9f3825.97,"a",@progbits
.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.97:
	.ascii	"id"
	.size	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.97, 2

	.type	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.98,@object
	.section	.rodata..Lanon.e78093d818c3c2c2ebd9e44bad9f3825.98,"a",@progbits
.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.98:
	.ascii	"value"
	.size	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.98, 5

	.type	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.99,@object
	.section	.rodata..Lanon.e78093d818c3c2c2ebd9e44bad9f3825.99,"a",@progbits
.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.99:
	.ascii	"ProtoRecordList"
	.size	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.99, 15

	.type	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.100,@object
	.section	.rodata..Lanon.e78093d818c3c2c2ebd9e44bad9f3825.100,"a",@progbits
.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.100:
	.ascii	"records"
	.size	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.100, 7

	.type	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.101,@object
	.section	.rodata.str1.1,"aMS",@progbits,1
.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.101:
	.asciz	"\031invalid wire type value: \300"
	.size	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.101, 28

	.type	.Lswitch.table._ZN73_$LT$prost..encoding..wire_type..WireType$u20$as$u20$core..fmt..Debug$GT$3fmt17hb528d417acae5beaE,@object
	.section	".rodata..Lswitch.table._ZN73_$LT$prost..encoding..wire_type..WireType$u20$as$u20$core..fmt..Debug$GT$3fmt17hb528d417acae5beaE","a",@progbits
	.p2align	3, 0x0
.Lswitch.table._ZN73_$LT$prost..encoding..wire_type..WireType$u20$as$u20$core..fmt..Debug$GT$3fmt17hb528d417acae5beaE:
	.quad	6
	.quad	12
	.quad	15
	.quad	10
	.quad	8
	.quad	12
	.size	.Lswitch.table._ZN73_$LT$prost..encoding..wire_type..WireType$u20$as$u20$core..fmt..Debug$GT$3fmt17hb528d417acae5beaE, 48

	.type	.Lswitch.table._ZN73_$LT$prost..encoding..wire_type..WireType$u20$as$u20$core..fmt..Debug$GT$3fmt17hb528d417acae5beaE.168.rel,@object
	.section	".rodata..Lswitch.table._ZN73_$LT$prost..encoding..wire_type..WireType$u20$as$u20$core..fmt..Debug$GT$3fmt17hb528d417acae5beaE.168.rel","a",@progbits
	.p2align	2, 0x0
.Lswitch.table._ZN73_$LT$prost..encoding..wire_type..WireType$u20$as$u20$core..fmt..Debug$GT$3fmt17hb528d417acae5beaE.168.rel:
	.long	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.90-.Lswitch.table._ZN73_$LT$prost..encoding..wire_type..WireType$u20$as$u20$core..fmt..Debug$GT$3fmt17hb528d417acae5beaE.168.rel
	.long	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.91-.Lswitch.table._ZN73_$LT$prost..encoding..wire_type..WireType$u20$as$u20$core..fmt..Debug$GT$3fmt17hb528d417acae5beaE.168.rel
	.long	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.92-.Lswitch.table._ZN73_$LT$prost..encoding..wire_type..WireType$u20$as$u20$core..fmt..Debug$GT$3fmt17hb528d417acae5beaE.168.rel
	.long	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.93-.Lswitch.table._ZN73_$LT$prost..encoding..wire_type..WireType$u20$as$u20$core..fmt..Debug$GT$3fmt17hb528d417acae5beaE.168.rel
	.long	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.94-.Lswitch.table._ZN73_$LT$prost..encoding..wire_type..WireType$u20$as$u20$core..fmt..Debug$GT$3fmt17hb528d417acae5beaE.168.rel
	.long	.Lanon.e78093d818c3c2c2ebd9e44bad9f3825.95-.Lswitch.table._ZN73_$LT$prost..encoding..wire_type..WireType$u20$as$u20$core..fmt..Debug$GT$3fmt17hb528d417acae5beaE.168.rel
	.size	.Lswitch.table._ZN73_$LT$prost..encoding..wire_type..WireType$u20$as$u20$core..fmt..Debug$GT$3fmt17hb528d417acae5beaE.168.rel, 24

	.hidden	DW.ref.rust_eh_personality
	.weak	DW.ref.rust_eh_personality
	.section	.data.DW.ref.rust_eh_personality,"awG",@progbits,DW.ref.rust_eh_personality,comdat
	.p2align	3, 0x0
	.type	DW.ref.rust_eh_personality,@object
	.size	DW.ref.rust_eh_personality, 8
DW.ref.rust_eh_personality:
	.quad	rust_eh_personality
	.ident	"rustc version 1.95.0 (59807616e 2026-04-14)"
	.section	".note.GNU-stack","",@progbits
