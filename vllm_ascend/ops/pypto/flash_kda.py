# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS FILE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root directory of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# Vendored delivery: keep the original DSL source and import order unchanged.
# ruff: noqa: E501, E702, F401, F841, I001, SIM300
# fmt: off


"""flash_kda — fused chunked KDA (Flash Kimi Delta Attention) operator.

TND accepts packed rank-3 inputs and explicit sequence boundaries. Its kernels
use padded 64-token chunks internally and keep each sequence's FP32 state
independent. Dense BSND/BNSD keep their original kernels.

Stage1 and Stage2 each have an independent host wrapper.  The legacy
``flash_kda_wrapper`` remains as a composition convenience API.

  stage1 (flash_kda_stage1_kernel) — flat (chunk, head) software-pipelined
    ring processing the previous work's cube/vector phases before the
    current work's next phase:
      M1 gate/qn/kn (vec) -> M2 cs (cube) -> M3 qd/kr/gam (vec)
      M4 KKT/Mqk (BF16 cube, Acc->Vec delivery)
      M6 Neumann inversion (FP16 double-doubling on-chip seed) + fused
      W/Upre (BF16 operands, FP32 accumulate; W BF16, Upre FP32 stores)
    Chunk constants (sl_mask / L_tril / eye64) are built on-chip from INT32
    lane indices — the kernels take no GM mask/constant inputs.
    GM outputs consumed by stage2: qd/kr/gam/mqk, w, upre.

  stage2 (flash_kda_stage2_kernel) — task-local P4 state chain
    (head-outer, chunk-inner, DV_SPLIT slices per task):
      cubeA: ud^T = snap^T @ W^T (snap from the dual-view L1 slot)
      vecA : u = bf16(ud^T + upre^T) -> u dual-view L1 insert
      cubeB: delta^T = u^T @ Kr ; o accumulation (mqk @ u)
      vecB : S^T update (gam scale + delta) -> snap insert (MTE3 ->
             MTE1 handoff, event 12) -> out / final_state stores
    S is resident in UB per task (no GM round trip); each task's chunk-0
    snapshot is initialized on-chip at the task prologue.

Public inference outputs (module_interfaces.yaml): ``out`` [B,S,Nv,D]-like
bf16 layout-native and ``final_state`` [B,Nv,D,D] fp32.
Stage1's qn/kn and BF16 high/low v_beta scratch live in one ``workspace`` byte
buffer（训练态 cache 已移除，本算子为推理专用）。

Key invariants:
  * W/Upre matmul operands stay BF16 and accumulate in FP32. Upre keeps
    the BF16 rounding residual of beta*V through a second matmul, and stays
    FP32 until Stage2 forms u = bf16(Upre - W@snap). State stays FP32.
  * Cross-core handoffs use set/wait_cross_core only (AIC<->AIV); AIV
    intra-core pipe pairs (MTE2/V/MTE3) use sync_src/sync_dst — the
    cross_core form deadlocks for same-engine pairs on this build.
  * Seed/constant staging UB [0xC000,0xE000) aliases P3_UB_XY / kd_nz /
    cs_g: the staging tile groups share mutex 16 so the auto-mutex chain
    carries the cross-pipe ordering (no manual syncs there).
  * cs_g is a single-slot WB buffer: M5's VF read is published with the
    per-work ``_M5_CS_DONE`` event before M2's next delivery may overwrite.
  * gate/qn/kn/cs handoffs ride on-chip UB/L1 slots (no GM round trip);
    slot parity follows the per-core work ordinal.
  * Event ids are phase-scoped and reused only after full-core boundaries.
"""

import os

import torch
import torch_npu  # noqa: F401  (NPU device init)

import pypto_pro.language as pl
from pypto_pro.language import Vf as vf  # noqa: N813
from pypto_pro.runtime.platform import get_platform_info
from pypto_pro.runtime.tilingkey import TilingKeyField

CHUNK_C = 64
HEAD_D = 128
L2_EPS = 1e-6
LANES = 64

DV_SPLIT = 8
DV = HEAD_D // DV_SPLIT

class _Stage2TilingKey:
    """V rows per task; each key has its own physical buffer sizes."""

    Stage2VRows = TilingKeyField(bits=2, values=[16, 64, 128])


class _Stage2TndTilingKey:
    """Packed tails reserve additional UB and do not use the 128-row key."""

    Stage2VRows = TilingKeyField(bits=1, values=[16, 64])


def _stage2_v_rows(heads, total_heads, cube_cores, *, tnd=False):
    """Pick the Stage2 V-slice width (DV) for the dense/TND schedules.

    The choice trades two effects against each other:

      * per-task GM traffic: each (b, hv, dv) task re-reads the whole
        W/Kr/Qd/Mqk of its head, so a wider DV (fewer slices S = 128/DV) reads
        less from GM (MTE2);
      * core balance: the grid runs ``ceil(total_heads * S / cube_cores)`` waves,
        so a finer split can lower the per-core wave ceiling at the cost of the
        S-times traffic.

    Measured on Ascend 950PR (28 cube cores) with the s4096_h32 / s8192_h32
    cases the per-head cost model ``Tc/S + Tm`` with ``Tm/Tc ~= 0.24`` reproduces
    the wave-count break-even: DV=64 wins only when the total head count falls
    just above the core count (the DV=128 schedule needs 2 waves for barely more
    than one core-width of work).  A 5% margin keeps DV=128 unless DV=64 is
    clearly better.

    The same model also drives the small-head (``heads < 32``) dense path, where
    wave quantization dominates: e.g. ``total_heads=8`` runs 3 waves at DV=16
    (40.1us) but a single wave at DV=64 (24.9us), and ``total_heads=64`` at
    B=16/S=512 drops 116.5us -> 58.5us.  TND stays on its 16/64 key set.
    """
    def _cost(dv):
        s = HEAD_D // dv
        waves = -(-(total_heads * s) // cube_cores)   # ceil
        return waves * (1.0 / s + 0.24)               # Tc/S + Tm, in Tc units

    if tnd:
        # TND tiling key supports only DV=16/64 (packed tail needs extra UB).
        # Each (b, hv, dv) task re-reads the whole W/Kr/Qd/Mqk of its head, so
        # DV=16 costs 4x the GM traffic of DV=64 for the same arithmetic and
        # saturates AIC MTE2 (measured 0.77-0.84 vs 0.53 of DV=64).  The finer
        # split only pays off when its 8x task grid still fits in a single wave
        # (8*total_heads <= cube_cores), where its lower per-task cost wins
        # without adding waves.  Kept SoC-derived so the crossover moves with
        # the core count (A5 28-core: t<=3; a 32-core part: t<=4).
        return 16 if total_heads * 8 <= cube_cores else 64
    if heads < 32:
        # Small-head dense: the wave quantization ``ceil(total_heads*S/cores)``
        # often dominates, so a wider slice that finishes in fewer waves can beat
        # the finer DV=16 split (e.g. total_heads=8: DV=64 -> 1 wave vs DV=16 -> 3).
        return min((16, 64, 128), key=_cost)
    if total_heads < cube_cores:
        return 64

    # Dense, heads >= 32: choose between DV=128 (S=1) and DV=64 (S=2).
    if _cost(64) < 0.95 * _cost(128):
        return 64
    return 128


ROW_BLOCK = 16
NV_MAX = 96
NK_MAX = 64
B_MAX = 64
BB = 32
HS = CHUNK_C // 2
BLOCK_STRIDE_ND = (CHUNK_C >> 1) | 1
REPEAT_STRIDE_ND = 1

_GATE_FWD_IDS = (0, 1)
_GATE_BWD_IDS = (2, 3)

_M6_UPRE_EVENT = 4
_M7_SNAP_EVENT = 12
_M7_UD_EVENTS = (13, 16)
_M5_CS_DONE = 14
_M7_U_EVENT = 14
_M7_D_EVENTS = (15, 17)
_P4A_SNAP_BANK_EVENT = 8

VA_X_B = 0x00000
VA_TMP_F = 0x0C000
VA_QN_B = 0x14000
VA_KN_B = 0x18000
VA_GATE_F = 0x1C000
VA_GATE_NZ = 0x24000
VA_A_LOG = 0x2C000
VA_LB = 0x2C100
VA_DT_B = 0x2C200
VA_SC = 0x2C400
VA_CS0 = 0x2C600
VA_CS1 = 0x34600
VA_M4_SCR_B16 = 0x26000

VA_M5_KKT = 0x8000
VA_M5_MQK = 0x10000
VA_M5_BETA_F = VA_QN_B
VA_M5_VB = VA_KN_B
VA_M5_VF = VA_GATE_NZ
VA_M5_KNF = 0x4000

VA_M7_S = 0x00000
VA_M7_DEL = 0x10000
VA_M7_UD = 0x20000
VA_M7_UB = 0x30000
VA_M7_UTNZ = 0x32000
VA_M7_SNAP = 0x34000
P4A_UB_UPRE0 = 0x28000
P4A_UB_UPRE1 = 0x2C000
P4A_UB_GAM0 = 0x3C000
P4A_UB_GAM1 = 0x3C200

L1_L = 0x00000
L1_G0 = 0x04000
L1_G1 = 0x0C000
L6_OP = 0x66000

L6_OP2 = 0x6E000
L6_LA_VBT = 0x5000
L7_SNAP = 0x38000
L7_QD = 0x2C000
L7_U = 0x30000
P4A_L1_W0 = 0x20000
P4A_L1_W1 = 0x24000
P4A_L1_KR0 = 0x28000
P4A_L1_KR1 = 0x2C000
P4A_L1_QD0 = 0x44000
P4A_L1_QD1 = 0x48000
P4A_L1_MQK0 = 0x4C000
P4A_L1_MQK1 = 0x4E000
P4A_L1_SNAPIN0 = 0x50000
P4A_L1_SNAPIN1 = 0x52000
P4A_L1_UNZ0 = 0x54000
P4A_L1_UNZ1 = 0x55000
L9_MQK = 0x40000

LA0 = 0x0
RA0 = 0x0
CA0 = 0x0
L6_LB_OP = 0x0
L6_LB_OP2 = 0xC000

L6_LC_W = 0x10000

L6_LC_U = 0x18000

P3_UB_X = 0x0A000
P3_UB_XY = 0x0C000

P3_L1_Y0 = 0x24000
P3_L1_Y1 = 0x28000
P3_L1_X = 0x2C000
L7_LC_UD = 0x0
L7_LC_DEL = 0x10000

@pl.vector_function
def m1_gate_vf(x_b, dtb, a_row, out_f, lower_bound):
    """gate = lb * sigmoid(exp(a_log[hv]) * (g + dt_bias[hv]))."""
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    pregbf = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_BF16)
    a_b = vf.load_align(a_row, 0, dist=pl.LoadDist.BRC_B32)
    a_h = vf.exp(a_b, preg)
    zero = vf.full(0.0, preg, dtype=pl.DT_FP32)
    one = vf.full(1.0, preg, dtype=pl.DT_FP32)
    dr0 = vf.load_align(dtb, 0)
    dr1 = vf.load_align(dtb, LANES)
    for m in pl.range(0, HS):
        base = m * HEAD_D
        bg = vf.load_align(x_b, base, dtype=pl.DT_BF16)
        e = vf.astype(bg, pregbf, dtype=pl.DT_FP32)
        o_ = vf.astype(bg, pregbf, dtype=pl.DT_FP32,
                       layout=pl.CastLayout.ONE)
        gr0, gr1 = vf.interleave(e, o_)
        sm = vf.add(gr0, dr0, preg)
        tm = vf.mul(sm, a_h, preg)
        em = vf.exp_sub(zero, tm, preg)
        dn = vf.adds(em, 1.0, preg)
        sg = vf.div(one, dn, preg)
        gl = vf.muls(sg, lower_bound, preg)
        vf.store_align(out_f + base, gl, preg)
        sm = vf.add(gr1, dr1, preg)
        tm = vf.mul(sm, a_h, preg)
        em = vf.exp_sub(zero, tm, preg)
        dn = vf.adds(em, 1.0, preg)
        sg = vf.div(one, dn, preg)
        gl = vf.muls(sg, lower_bound, preg)
        vf.store_align(out_f + base + LANES, gl, preg)

@pl.vector_function
def m1_l2norm_vf(x_b, out_b, eps):
    """Row-L2 div norm, bf16 in -> bf16 out (both casts fused in-VF)."""
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    pregbf = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_BF16)
    for m in pl.range(0, HS):
        base = m * HEAD_D
        bx = vf.load_align(x_b, base, dtype=pl.DT_BF16)
        e = vf.astype(bx, pregbf, dtype=pl.DT_FP32)
        o_ = vf.astype(bx, pregbf, dtype=pl.DT_FP32,
                       layout=pl.CastLayout.ONE)
        x0, x1 = vf.interleave(e, o_)
        s0 = vf.mul(x0, x0, preg)
        p0 = vf.reduce_sum(s0, preg)
        s1 = vf.mul(x1, x1, preg)
        p1 = vf.reduce_sum(s1, preg)
        rsum = vf.add(p0, p1, preg)
        denom = vf.full(rsum, preg)
        denom = vf.sqrt(denom, preg)
        denom = vf.adds(denom, eps, preg)
        y0 = vf.div(x0, denom, preg)
        y1 = vf.div(x1, denom, preg)
        b0 = vf.astype(y0, pregbf, dtype=pl.DT_BF16,
                       round_mode=pl.VFRoundMode.CAST_ROUND)
        b1 = vf.astype(y1, pregbf, dtype=pl.DT_BF16,
                       round_mode=pl.VFRoundMode.CAST_ROUND)
        d0, d1 = vf.de_interleave(b0, b1)
        vf.store_align(out_b + base, d0, pregbf)

@pl.vector_function
def m3_qd_vf(qn_b, cs_f, scale, out_b, r0):
    """qd = qn * exp(cs) * scale — bf16 in -> bf16 out (both casts fused in-VF)."""
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    pregbf = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_BF16)
    for m in pl.range(0, HS):
        base = m * HEAD_D
        cbase = (r0 + m) * HEAD_D
        bq = vf.load_align(qn_b, base, dtype=pl.DT_BF16)
        e = vf.astype(bq, pregbf, dtype=pl.DT_FP32)
        o_ = vf.astype(bq, pregbf, dtype=pl.DT_FP32,
                       layout=pl.CastLayout.ONE)
        q0, q1 = vf.interleave(e, o_)
        c0 = vf.load_align(cs_f, cbase)
        c1 = vf.load_align(cs_f, cbase + LANES)
        e0 = vf.exp(c0, preg)
        e1 = vf.exp(c1, preg)
        t0 = vf.mul(q0, e0, preg)
        t1 = vf.mul(q1, e1, preg)
        d0v = vf.muls(t0, scale, preg)
        d1v = vf.muls(t1, scale, preg)
        b0 = vf.astype(d0v, pregbf, dtype=pl.DT_BF16,
                       round_mode=pl.VFRoundMode.CAST_ROUND)
        b1 = vf.astype(d1v, pregbf, dtype=pl.DT_BF16,
                       round_mode=pl.VFRoundMode.CAST_ROUND)
        drow, d1_ = vf.de_interleave(b0, b1)
        vf.store_align(out_b + base, drow, pregbf)

@pl.vector_function
def m3_kr_vf(kn_b, cs_f, csl_f, out_b, r0):
    """kr = kn * exp(cs_last - cs) — bf16 in -> bf16 out (both casts fused in-VF)."""
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    pregbf = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_BF16)
    l0 = vf.load_align(csl_f, 0)
    l1 = vf.load_align(csl_f, LANES)
    for m in pl.range(0, HS):
        base = m * HEAD_D
        cbase = (r0 + m) * HEAD_D
        bk = vf.load_align(kn_b, base, dtype=pl.DT_BF16)
        e = vf.astype(bk, pregbf, dtype=pl.DT_FP32)
        o_ = vf.astype(bk, pregbf, dtype=pl.DT_FP32,
                       layout=pl.CastLayout.ONE)
        k0, k1 = vf.interleave(e, o_)
        c0 = vf.load_align(cs_f, cbase)
        c1 = vf.load_align(cs_f, cbase + LANES)
        t0_e = vf.exp_sub(l0, c0, preg)
        t0 = vf.mul(k0, t0_e, preg)
        t1_e = vf.exp_sub(l1, c1, preg)
        t1 = vf.mul(k1, t1_e, preg)
        b0 = vf.astype(t0, pregbf, dtype=pl.DT_BF16,
                       round_mode=pl.VFRoundMode.CAST_ROUND)
        b1 = vf.astype(t1, pregbf, dtype=pl.DT_BF16,
                       round_mode=pl.VFRoundMode.CAST_ROUND)
        drow, d1_ = vf.de_interleave(b0, b1)
        vf.store_align(out_b + base, drow, pregbf)

@pl.vector_function
def m3_gam_vf(csl_f, gam_f):
    """gam = exp(cs_last).  May legitimately underflow to 0 (no clamp)."""
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    for r in pl.range(0, 2):
        off = r * LANES
        l_r = vf.load_align(csl_f, off)
        e_r = vf.exp(l_r, preg)
        vf.store_align(gam_f + off, e_r, preg)

@pl.vector_function
def m4_ref_row_vf(cs_f, ref_f, row):
    """Extract cs row ``row`` (16*rb row-block base) into a [1,128] ref tile."""
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    for r in pl.range(0, 2):
        off = row * HEAD_D + r * LANES
        v = vf.load_align(cs_f, off)
        vf.store_align(ref_f + r * LANES, v, preg)

@pl.vector_function
def m4_lop_vf(src_f, cs_f, ref_f, out_b, r0):
    """l_op = src[r0:r0+16] * exp(cs[r0:r0+16] - ref) -> out rows 0..15 (compacted)."""
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    pregbf = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_BF16)
    rf0 = vf.load_align(ref_f, 0)
    rf1 = vf.load_align(ref_f, LANES)
    for m in pl.range(0, ROW_BLOCK):
        sb = (r0 + m) * HEAD_D
        ob = m * HEAD_D
        sb16 = vf.load_align(src_f, sb, dtype=pl.DT_BF16)
        se0 = vf.astype(sb16, pregbf, dtype=pl.DT_FP32)
        se1 = vf.astype(sb16, pregbf, dtype=pl.DT_FP32,
                        layout=pl.CastLayout.ONE)
        s0, s1 = vf.interleave(se0, se1)
        c0 = vf.load_align(cs_f, sb)
        c1 = vf.load_align(cs_f, sb + LANES)
        t0_e = vf.exp_sub(c0, rf0, preg)
        t0 = vf.mul(s0, t0_e, preg)
        t1_e = vf.exp_sub(c1, rf1, preg)
        t1 = vf.mul(s1, t1_e, preg)
        b0 = vf.astype(t0, pregbf, dtype=pl.DT_BF16,
                       round_mode=pl.VFRoundMode.CAST_ROUND)
        b1 = vf.astype(t1, pregbf, dtype=pl.DT_BF16,
                       round_mode=pl.VFRoundMode.CAST_ROUND)
        drow, d1_ = vf.de_interleave(b0, b1)
        vf.store_align(out_b + ob, drow, pregbf)

@pl.vector_function
def m4_rop_chunk_vf(src_f, cs_f, ref_f, out_b, r0c):
    """r_op 16-row chunk: out rows 0..15 = src[r0c..r0c+16) * exp(ref - cs[same])."""
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    pregbf = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_BF16)
    rf0 = vf.load_align(ref_f, 0)
    rf1 = vf.load_align(ref_f, LANES)
    for m in pl.range(0, ROW_BLOCK):
        sb = (r0c + m) * HEAD_D
        ob = m * HEAD_D
        sb16 = vf.load_align(src_f, sb, dtype=pl.DT_BF16)
        se0 = vf.astype(sb16, pregbf, dtype=pl.DT_FP32)
        se1 = vf.astype(sb16, pregbf, dtype=pl.DT_FP32,
                        layout=pl.CastLayout.ONE)
        s0, s1 = vf.interleave(se0, se1)
        c0 = vf.load_align(cs_f, sb)
        c1 = vf.load_align(cs_f, sb + LANES)
        t0_e = vf.exp_sub(rf0, c0, preg)
        t0 = vf.mul(s0, t0_e, preg)
        t1_e = vf.exp_sub(rf1, c1, preg)
        t1 = vf.mul(s1, t1_e, preg)
        b0 = vf.astype(t0, pregbf, dtype=pl.DT_BF16,
                       round_mode=pl.VFRoundMode.CAST_ROUND)
        b1 = vf.astype(t1, pregbf, dtype=pl.DT_BF16,
                       round_mode=pl.VFRoundMode.CAST_ROUND)
        drow, d1_ = vf.de_interleave(b0, b1)
        vf.store_align(out_b + ob, drow, pregbf)

@pl.vector_function
def m4_lqop_vf(src_f, cs_f, ref_f, out_b, r0, scale):
    """lq_op = src[r0:r0+16] * exp(cs[block] - ref) * scale -> out rows 0..15."""
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    pregbf = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_BF16)
    rf0 = vf.load_align(ref_f, 0)
    rf1 = vf.load_align(ref_f, LANES)
    for m in pl.range(0, ROW_BLOCK):
        sb = (r0 + m) * HEAD_D
        ob = m * HEAD_D
        sb16 = vf.load_align(src_f, sb, dtype=pl.DT_BF16)
        se0 = vf.astype(sb16, pregbf, dtype=pl.DT_FP32)
        se1 = vf.astype(sb16, pregbf, dtype=pl.DT_FP32,
                        layout=pl.CastLayout.ONE)
        s0, s1 = vf.interleave(se0, se1)
        c0 = vf.load_align(cs_f, sb)
        c1 = vf.load_align(cs_f, sb + LANES)
        t0_e = vf.exp_sub(c0, rf0, preg)
        t0 = vf.mul(s0, t0_e, preg)
        t1_e = vf.exp_sub(c1, rf1, preg)
        t1 = vf.mul(s1, t1_e, preg)
        t0 = vf.muls(t0, scale, preg)
        t1 = vf.muls(t1, scale, preg)
        b0 = vf.astype(t0, pregbf, dtype=pl.DT_BF16,
                       round_mode=pl.VFRoundMode.CAST_ROUND)
        b1 = vf.astype(t1, pregbf, dtype=pl.DT_BF16,
                       round_mode=pl.VFRoundMode.CAST_ROUND)
        drow, d1_ = vf.de_interleave(b0, b1)
        vf.store_align(out_b + ob, drow, pregbf)

@pl.vector_function
def m5_tmqk_vf(kkt_f, mqk_f, beta_blk_f, tb_out_f, mqk_out_b,
               off_base, step, r0):
    """Fused pair-VF: t_beta and mqk row finishes in one pass."""
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    pregbf = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_BF16)
    zero = vf.full(0.0, preg, dtype=pl.DT_FP32)
    one = vf.full(1.0, preg, dtype=pl.DT_FP32)
    neg1 = vf.full(-1.0, preg, dtype=pl.DT_FP32)
    pi32 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_INT32)
    lane_i = vf.arange(0, dtype=pl.DT_INT32)
    for p in pl.range(0, HS // 2):
        gr0 = r0 + 2 * p
        gr1 = gr0 + 1
        base0 = 2 * p * CHUNK_C
        base1 = (2 * p + 1) * CHUNK_C
        brow = vf.load_align(beta_blk_f, off_base + gr0 * step,
                             dist=pl.LoadDist.BRC_B32)
        e = vf.exp_sub(zero, brow, preg)
        dn = vf.adds(e, 1.0, preg)
        sg0 = vf.div(one, dn, preg)
        rv0 = vf.full(gr0, dtype=pl.DT_INT32)
        keep_full = vf.le(lane_i, rv0, pi32)      # j <= row
        keep_strict = vf.lt(lane_i, rv0, pi32)    # j <  row
        mq0 = vf.load_align(mqk_f, base0)
        mq0 = vf.select(mq0, zero, keep_full)
        krow = vf.load_align(kkt_f, base0)
        t = vf.mul(krow, sg0, preg)
        t = vf.mul(t, neg1, preg)
        t = vf.select(t, zero, keep_strict)
        vf.store_align(tb_out_f + base0, t, preg)
        brow = vf.load_align(beta_blk_f, off_base + gr1 * step,
                             dist=pl.LoadDist.BRC_B32)
        e = vf.exp_sub(zero, brow, preg)
        dn = vf.adds(e, 1.0, preg)
        sg1 = vf.div(one, dn, preg)
        rv1 = vf.full(gr1, dtype=pl.DT_INT32)
        keep_full1 = vf.le(lane_i, rv1, pi32)
        keep_strict1 = vf.lt(lane_i, rv1, pi32)
        mq1 = vf.load_align(mqk_f, base1)
        mq1 = vf.select(mq1, zero, keep_full1)
        krow = vf.load_align(kkt_f, base1)
        t = vf.mul(krow, sg1, preg)
        t = vf.mul(t, neg1, preg)
        t = vf.select(t, zero, keep_strict1)
        vf.store_align(tb_out_f + base1, t, preg)
        b0 = vf.astype(mq0, pregbf, dtype=pl.DT_BF16,
                       round_mode=pl.VFRoundMode.CAST_ROUND)
        b1 = vf.astype(mq1, pregbf, dtype=pl.DT_BF16,
                       round_mode=pl.VFRoundMode.CAST_ROUND)
        drow, d1_ = vf.de_interleave(b0, b1)
        vf.store_align(mqk_out_b + p * 2 * CHUNK_C, drow, pregbf)

@pl.vector_function
def m5_vkn_vf(v_b, kn_b, cs_f, beta_blk_f, v_out_b, kd_out_b,
              off_base, step, r0):
    """Split beta*V into BF16 high/low parts; also produce BF16 -beta*K*exp(cs).

    Reuse each consumed row of v_b for the low part. The caller stores both
    parts before the M6 handoff; no extra UB tile or mutex is needed.
    """
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    pregbf = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_BF16)
    zero = vf.full(0.0, preg, dtype=pl.DT_FP32)
    one = vf.full(1.0, preg, dtype=pl.DT_FP32)
    neg1 = vf.full(-1.0, preg, dtype=pl.DT_FP32)
    for m in pl.range(0, HS):
        base = m * HEAD_D
        brow = vf.load_align(beta_blk_f, off_base + (r0 + m) * step,
                             dist=pl.LoadDist.BRC_B32)
        e = vf.exp_sub(zero, brow, preg)
        dn = vf.adds(e, 1.0, preg)
        sg = vf.div(one, dn, preg)
        bv = vf.load_align(v_b, base, dtype=pl.DT_BF16)
        e0 = vf.astype(bv, pregbf, dtype=pl.DT_FP32)
        e1 = vf.astype(bv, pregbf, dtype=pl.DT_FP32,
                       layout=pl.CastLayout.ONE)
        vr0, vr1 = vf.interleave(e0, e1)
        t0 = vf.mul(vr0, sg, preg)
        t1 = vf.mul(vr1, sg, preg)
        b0 = vf.astype(t0, pregbf, dtype=pl.DT_BF16,
                       round_mode=pl.VFRoundMode.CAST_ROUND)
        b1 = vf.astype(t1, pregbf, dtype=pl.DT_BF16,
                       round_mode=pl.VFRoundMode.CAST_ROUND)
        packed, unused = vf.de_interleave(b0, b1)
        vf.store_align(v_out_b + base, packed, pregbf)
        # Keep the rounding residual while t0/t1 still hold beta*V in FP32.
        hi0 = vf.astype(b0, pregbf, dtype=pl.DT_FP32)
        hi1 = vf.astype(b1, pregbf, dtype=pl.DT_FP32)
        lo0 = vf.sub(t0, hi0, preg)
        lo1 = vf.sub(t1, hi1, preg)
        lb0 = vf.astype(lo0, pregbf, dtype=pl.DT_BF16,
                        round_mode=pl.VFRoundMode.CAST_ROUND)
        lb1 = vf.astype(lo1, pregbf, dtype=pl.DT_BF16,
                        round_mode=pl.VFRoundMode.CAST_ROUND)
        packed_lo, unused_lo = vf.de_interleave(lb0, lb1)
        vf.store_align(v_b + base, packed_lo, pregbf)
        bk = vf.load_align(kn_b, base, dtype=pl.DT_BF16)
        e0 = vf.astype(bk, pregbf, dtype=pl.DT_FP32)
        e1 = vf.astype(bk, pregbf, dtype=pl.DT_FP32,
                       layout=pl.CastLayout.ONE)
        k0, k1 = vf.interleave(e0, e1)
        c0 = vf.load_align(cs_f, (r0 + m) * HEAD_D)
        c1 = vf.load_align(cs_f, (r0 + m) * HEAD_D + LANES)
        e0 = vf.exp(c0, preg)
        e1 = vf.exp(c1, preg)
        t0 = vf.mul(k0, e0, preg)
        t1 = vf.mul(k1, e1, preg)
        t0 = vf.mul(t0, sg, preg)
        t1 = vf.mul(t1, sg, preg)
        t0 = vf.mul(t0, neg1, preg)
        t1 = vf.mul(t1, neg1, preg)
        b0 = vf.astype(t0, pregbf, dtype=pl.DT_BF16,
                       round_mode=pl.VFRoundMode.CAST_ROUND)
        b1 = vf.astype(t1, pregbf, dtype=pl.DT_BF16,
                       round_mode=pl.VFRoundMode.CAST_ROUND)
        packed, unused = vf.de_interleave(b0, b1)
        vf.store_align(kd_out_b + base, packed, pregbf)

@pl.vector_function
def m7_ut_add_upre_bf16_vf(ud_t, up_f, out_b, row0, rows):
    """Add FP32 ud/upre, then round u once to BF16 for the next matmul."""
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    pregbf = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_BF16)
    for m2 in pl.range(0, (rows // 2) // 2):
        off = m2 * (2 * CHUNK_C)
        r0 = vf.load_align(up_f, off)
        r1 = vf.load_align(up_f, off + CHUNK_C)
        a0 = vf.load_align(ud_t, (row0 + 2 * m2) * CHUNK_C)
        a1 = vf.load_align(ud_t, (row0 + 2 * m2 + 1) * CHUNK_C)
        s0 = vf.add(a0, r0, preg)
        s1 = vf.add(a1, r1, preg)
        b0 = vf.astype(s0, pregbf, dtype=pl.DT_BF16,
                       round_mode=pl.VFRoundMode.CAST_ROUND)
        b1 = vf.astype(s1, pregbf, dtype=pl.DT_BF16,
                       round_mode=pl.VFRoundMode.CAST_ROUND)
        drow, d1_ = vf.de_interleave(b0, b1)
        vf.store_align(out_b + off, drow, pregbf)

@pl.vector_function
def m7_scale_t_vf(s_t, gam_f, row0, rows):
    """S^T *= gam — the early half of the state update."""
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    g = vf.load_align(gam_f, row0)
    for j2 in pl.range(0, rows // 2):
        off = j2 * (2 * CHUNK_C)
        sv0 = vf.load_align(s_t, off)
        sv1 = vf.load_align(s_t, off + CHUNK_C)
        m0 = vf.mul(sv0, g, preg)
        m1 = vf.mul(sv1, g, preg)
        vf.store_align(s_t + off, m0, preg)
        vf.store_align(s_t + off + CHUNK_C, m1, preg)

@pl.vector_function
def m7_state_add_snap_vf(s_t, d_t, out_t, out_b, rows):
    """S^T += delta^T (late half; S already gam-scaled), emit the FP32
    snapshot out_t and the BF16 de-interleaved out_b for the L1 insert."""
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    pregbf = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_BF16)
    for j2 in pl.range(0, rows // 2):
        off = j2 * (2 * CHUNK_C)
        sv0 = vf.load_align(s_t, off)
        sv1 = vf.load_align(s_t, off + CHUNK_C)
        dv0 = vf.load_align(d_t, off)
        dv1 = vf.load_align(d_t, off + CHUNK_C)
        t0 = vf.add(sv0, dv0, preg)
        t1 = vf.add(sv1, dv1, preg)
        vf.store_align(out_t + off, t0, preg)
        vf.store_align(out_t + off + CHUNK_C, t1, preg)
        b0 = vf.astype(t0, pregbf, dtype=pl.DT_BF16,
                       round_mode=pl.VFRoundMode.CAST_ROUND)
        b1 = vf.astype(t1, pregbf, dtype=pl.DT_BF16,
                       round_mode=pl.VFRoundMode.CAST_ROUND)
        drow, d1_ = vf.de_interleave(b0, b1)
        vf.store_align(out_b + off, drow, pregbf)

@pl.vector_function
def nd2nz_sg(tile, out, m):
    """ND→NZ row re-assembly into the P3 XY mirror buffer."""
    preg_all = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    for i in pl.range(m):
        inreg = vf.load_align(tile, i * CHUNK_C)
        vf.store_align(out, inreg, preg_all,
                       block_stride=BLOCK_STRIDE_ND, repeat_stride=REPEAT_STRIDE_ND,
                       data_copy_mode=pl.DataCopyMode.DATA_BLOCK_COPY, post_update=True)

@pl.vector_function
def ltril_rows_vf(out, row0):
    """Lower-triangular (diag kept) rows [row0, row0+HS) into plain UB scratch.

    Replaces the host-built L_tril GM tensor: each row is derived from the
    INT32 lane index (j <= row), matching the sl_mask lane-keep idiom.
    """
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    pi32 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_INT32)
    lane_i = vf.arange(0, dtype=pl.DT_INT32)
    one = vf.full(1.0, preg, dtype=pl.DT_FP32)
    zero = vf.full(0.0, preg, dtype=pl.DT_FP32)
    for i in pl.range(HS):
        rv = vf.full(row0 + i, dtype=pl.DT_INT32)
        keep = vf.le(lane_i, rv, pi32)
        row = vf.select(one, zero, keep)
        vf.store_align(out + i * CHUNK_C, row, preg)

@pl.vector_function
def eye_rows_vf(out, row0):
    """Identity rows [row0, row0+HS) into plain UB scratch (Neumann seed).

    Replaces the host-built eye64 GM tensor: row r is 1.0 on lane r only.
    """
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    pi32 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_INT32)
    lane_i = vf.arange(0, dtype=pl.DT_INT32)
    one = vf.full(1.0, preg, dtype=pl.DT_FP32)
    zero = vf.full(0.0, preg, dtype=pl.DT_FP32)
    for i in pl.range(HS):
        rv = vf.full(row0 + i, dtype=pl.DT_INT32)
        hit = vf.eq(lane_i, rv, pi32)
        row = vf.select(one, zero, hit)
        vf.store_align(out + i * CHUNK_C, row, preg)

def m1_prefetch_v(q_v, k_v, g_v, b, hk, hv, s0, x_b_g, xk_g, xg_g):
    """Preload: issue the NEXT work's q/k/g GM loads ahead of the pipeline."""
    sub = pl.get_subblock_idx()
    r0 = sub * HS
    xq = x_b_g.next()
    xk = xk_g.next()
    xg = xg_g.next()
    pl.load(xq, q_v, [b, hk, s0 + r0, 0], order=[2, 3])
    pl.load(xk, k_v, [b, hk, s0 + r0, 0], order=[2, 3])
    pl.load(xg, g_v, [b, hv, s0 + r0, 0], order=[2, 3])

def m1_gate_norm_v(qn_o, kn_o, a_view, lower_bound, dt_bias,
                   gate_l1, gate_slot_idx, b, hv, hk, s0, have_p2,
                   x_b_g, xk_g, xg_g, qn_h_g, kn_h_g, gate_f_g, gate_nz_g,
                   a_g, d_g):
    """Module-1 vector stage: gate/qn/kn for one chunk of one (b, hv)."""
    sub = pl.get_subblock_idx()
    r0 = sub * HS
    if have_p2:
        pl.system.wait_cross_core(pipe=pl.PipeType.V, event_id=3)
    xq = x_b_g.current()
    xk = xk_g.current()
    xg = xg_g.current()
    a_s = a_g.next()
    pl.set_validshape(a_s, [1, 1])
    pl.load(a_s, a_view, [hv, 0])
    d_s = d_g.next()
    pl.load(d_s, dt_bias, [hv, 0])
    gatel = gate_f_g.next()
    m1_gate_vf(xg, d_s, a_s, gatel, lower_bound)
    gate_nz = gate_nz_g.next()
    pl.set_validshape(gate_nz, [HS, HEAD_D])
    pl.set_validshape(gatel, [HS, HEAD_D])
    pl.move(gate_nz, gatel)
    gate_slot = gate_l1[gate_slot_idx]
    pl.insert(gate_slot, gate_nz, [r0, 0])
    pl.system.set_cross_core(pipe=pl.PipeType.MTE3, event_id=0)
    qnb = qn_h_g[gate_slot_idx]
    m1_l2norm_vf(xq, qnb, L2_EPS)
    pl.store(qn_o, qnb, [b, hk, s0 + r0, 0], order=[2, 3])
    knb = kn_h_g[gate_slot_idx]
    m1_l2norm_vf(xk, knb, L2_EPS)
    pl.store(kn_o, knb, [b, hk, s0 + r0, 0], order=[2, 3])

def m2_cumsum_c(gate_l1, cs_g, gate_slot_idx,
                have_prev, l_pre, L_l0a_g, g_l0b_g, acc_g):
    """Module-2 cube body: cs = L_tril @ gate."""
    pl.system.wait_cross_core(pipe=pl.PipeType.MTE1, event_id=0)
    if have_prev:
        pl.system.wait_cross_core(pipe=pl.PipeType.MTE1, event_id=7)
    gate_slot = gate_l1[gate_slot_idx]
    l_l1 = l_pre
    l_a = L_l0a_g.next()
    pl.move(l_a, l_l1)
    g_r = g_l0b_g.next()
    pl.move(g_r, gate_slot)
    acc = acc_g.next()
    pl.matmul(acc, l_a, g_r,
              phase=pl.AccPhase.Final)
    pl.set_validshape(acc, [CHUNK_C, HEAD_D])
    cs_slot = cs_g[0]
    pl.set_validshape(cs_slot, [CHUNK_C, HEAD_D])
    if have_prev:
        pl.system.wait_cross_core(pipe=pl.PipeType.FIX, event_id=_M5_CS_DONE)
    pl.move(cs_slot, acc, acc_to_vec_mode=pl.AccToVecMode.SingleModeVec0,
            phase=pl.STPhase.Partial)
    pl.move(cs_slot, acc, acc_to_vec_mode=pl.AccToVecMode.SingleModeVec1,
            phase=pl.STPhase.Final)
    pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=2)
    pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=3)
    pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=11)

def m3_qd_kr_gam_v(qn_o, kn_o, qd_o, kr_o, gam_o, cs_g, scale,
                   c, b, hv, hk, s0, gate_slot_idx, Nv, nc,
                   qn_h_g, kn_h_g, m3_out_b_g, m3_csl_g,
                   m4_kn_b_g, m4_qn_b_g,
                   m4_opscr_b16_g, m4_ref_g,
                   lk0_g, lk1_g, lk2_g, lk3_g,
                   lq0_g, lq1_g, lq2_g, lq3_g,
                   rW0_g, rW1_g, rW2_g, rW3_g,
                   opnz_g):
    """Module-3 vector stage: qd/kr/gam + M5 operand pre-pass."""
    pl.system.wait_cross_core(pipe=pl.PipeType.V, event_id=2)
    cs_slot = cs_g[0]
    pl.set_validshape(cs_slot, [CHUNK_C, HEAD_D])
    sub_m3 = pl.get_subblock_idx()
    r0 = sub_m3 * HS
    csl = m3_csl_g.next()
    pl.set_validshape(csl, [1, HEAD_D])
    pl.move(csl, cs_slot, offset=[CHUNK_C - 1, 0])
    qnb = qn_h_g[gate_slot_idx]
    qd_b = m3_out_b_g.next()
    m3_qd_vf(qnb, cs_slot, scale, qd_b, r0)
    pl.store(qd_o, qd_b, [b, hv, s0 + r0, 0], order=[2, 3])

    knb = kn_h_g[gate_slot_idx]
    kr_b = m3_out_b_g.next()
    m3_kr_vf(knb, cs_slot, csl, kr_b, r0)
    pl.store(kr_o, kr_b, [b, hv, s0 + r0, 0], order=[2, 3])

    m3_gam_vf(csl, csl)
    pl.store(gam_o, csl, [b, hv, c, 0], order=[2, 3])

    knb4 = m4_kn_b_g.next()
    pl.load(knb4, kn_o, [b, hk, s0, 0], order=[2, 3])
    kn4 = knb4
    qnb4 = m4_qn_b_g.next()
    pl.load(qnb4, qn_o, [b, hk, s0, 0], order=[2, 3])
    qn4 = qnb4
    sub_m3 = pl.get_subblock_idx()
    ref0 = m4_ref_g.next()
    ref1 = m4_ref_g.next()
    ref2 = m4_ref_g.next()
    ref3 = m4_ref_g.next()
    scrb0 = m4_opscr_b16_g[0]
    scrb1 = m4_opscr_b16_g[1]
    scrb2 = m4_opscr_b16_g[0]
    if sub_m3 == 0:
        r0v = 0 * ROW_BLOCK
        pl.set_validshape(ref0, [1, HEAD_D])
        m4_ref_row_vf(cs_slot, ref0, r0v)
        m4_lop_vf(kn4, cs_slot, ref0, scrb0, r0v)

        snz = opnz_g[0]
        pl.set_validshape(snz, [ROW_BLOCK, HEAD_D])
        pl.set_validshape(scrb0, [ROW_BLOCK, HEAD_D])
        pl.move(snz, scrb0)
        dst_s = lk0_g[gate_slot_idx]
        pl.insert(dst_s, snz, [0, 0])
        m4_rop_chunk_vf(kn4, cs_slot, ref0, scrb1, 0 * ROW_BLOCK)
        snz = opnz_g[0]
        pl.set_validshape(snz, [ROW_BLOCK, HEAD_D])
        pl.set_validshape(scrb1, [ROW_BLOCK, HEAD_D])
        pl.move(snz, scrb1)
        dst_s = rW0_g[gate_slot_idx]
        pl.insert(dst_s, snz, [0 * ROW_BLOCK, 0])
        m4_lqop_vf(qn4, cs_slot, ref0, scrb2, r0v, scale)
        snz = opnz_g[0]
        pl.set_validshape(snz, [ROW_BLOCK, HEAD_D])
        pl.set_validshape(scrb2, [ROW_BLOCK, HEAD_D])
        pl.move(snz, scrb2)
        dst_s = lq0_g[gate_slot_idx]
        pl.insert(dst_s, snz, [0, 0])
        r1v = 1 * ROW_BLOCK
        pl.set_validshape(ref1, [1, HEAD_D])
        m4_ref_row_vf(cs_slot, ref1, r1v)
        m4_lop_vf(kn4, cs_slot, ref1, scrb0, r1v)
        snz = opnz_g[0]
        pl.set_validshape(snz, [ROW_BLOCK, HEAD_D])
        pl.set_validshape(scrb0, [ROW_BLOCK, HEAD_D])
        pl.move(snz, scrb0)
        dst_s = lk1_g[gate_slot_idx]
        pl.insert(dst_s, snz, [0, 0])
        m4_rop_chunk_vf(kn4, cs_slot, ref1, scrb1, 0 * ROW_BLOCK)
        snz = opnz_g[0]
        pl.set_validshape(snz, [ROW_BLOCK, HEAD_D])
        pl.set_validshape(scrb1, [ROW_BLOCK, HEAD_D])
        pl.move(snz, scrb1)
        dst_s = rW1_g[gate_slot_idx]
        pl.insert(dst_s, snz, [0 * ROW_BLOCK, 0])
        m4_rop_chunk_vf(kn4, cs_slot, ref1, scrb2, 1 * ROW_BLOCK)
        snz = opnz_g[0]
        pl.set_validshape(snz, [ROW_BLOCK, HEAD_D])
        pl.set_validshape(scrb2, [ROW_BLOCK, HEAD_D])
        pl.move(snz, scrb2)
        dst_s = rW1_g[gate_slot_idx]
        pl.insert(dst_s, snz, [1 * ROW_BLOCK, 0])
        m4_lqop_vf(qn4, cs_slot, ref1, scrb0, r1v, scale)
        snz = opnz_g[0]
        pl.set_validshape(snz, [ROW_BLOCK, HEAD_D])
        pl.set_validshape(scrb0, [ROW_BLOCK, HEAD_D])
        pl.move(snz, scrb0)
        dst_s = lq1_g[gate_slot_idx]
        pl.insert(dst_s, snz, [0, 0])
    else:
        r2v = 2 * ROW_BLOCK
        pl.set_validshape(ref2, [1, HEAD_D])
        m4_ref_row_vf(cs_slot, ref2, r2v)
        m4_lop_vf(kn4, cs_slot, ref2, scrb1, r2v)
        snz = opnz_g[0]
        pl.set_validshape(snz, [ROW_BLOCK, HEAD_D])
        pl.set_validshape(scrb1, [ROW_BLOCK, HEAD_D])
        pl.move(snz, scrb1)
        dst_s = lk2_g[gate_slot_idx]
        pl.insert(dst_s, snz, [0, 0])
        m4_rop_chunk_vf(kn4, cs_slot, ref2, scrb2, 0 * ROW_BLOCK)
        snz = opnz_g[0]
        pl.set_validshape(snz, [ROW_BLOCK, HEAD_D])
        pl.set_validshape(scrb2, [ROW_BLOCK, HEAD_D])
        pl.move(snz, scrb2)
        dst_s = rW2_g[gate_slot_idx]
        pl.insert(dst_s, snz, [0 * ROW_BLOCK, 0])
        m4_rop_chunk_vf(kn4, cs_slot, ref2, scrb0, 1 * ROW_BLOCK)
        snz = opnz_g[0]
        pl.set_validshape(snz, [ROW_BLOCK, HEAD_D])
        pl.set_validshape(scrb0, [ROW_BLOCK, HEAD_D])
        pl.move(snz, scrb0)
        dst_s = rW2_g[gate_slot_idx]
        pl.insert(dst_s, snz, [1 * ROW_BLOCK, 0])
        m4_rop_chunk_vf(kn4, cs_slot, ref2, scrb1, 2 * ROW_BLOCK)
        snz = opnz_g[0]
        pl.set_validshape(snz, [ROW_BLOCK, HEAD_D])
        pl.set_validshape(scrb1, [ROW_BLOCK, HEAD_D])
        pl.move(snz, scrb1)
        dst_s = rW2_g[gate_slot_idx]
        pl.insert(dst_s, snz, [2 * ROW_BLOCK, 0])
        m4_lqop_vf(qn4, cs_slot, ref2, scrb2, r2v, scale)
        snz = opnz_g[0]
        pl.set_validshape(snz, [ROW_BLOCK, HEAD_D])
        pl.set_validshape(scrb2, [ROW_BLOCK, HEAD_D])
        pl.move(snz, scrb2)
        dst_s = lq2_g[gate_slot_idx]
        pl.insert(dst_s, snz, [0, 0])
        r3v = 3 * ROW_BLOCK
        pl.set_validshape(ref3, [1, HEAD_D])
        m4_ref_row_vf(cs_slot, ref3, r3v)
        m4_lop_vf(kn4, cs_slot, ref3, scrb0, r3v)
        snz = opnz_g[0]
        pl.set_validshape(snz, [ROW_BLOCK, HEAD_D])
        pl.set_validshape(scrb0, [ROW_BLOCK, HEAD_D])
        pl.move(snz, scrb0)
        dst_s = lk3_g[gate_slot_idx]
        pl.insert(dst_s, snz, [0, 0])
        m4_rop_chunk_vf(kn4, cs_slot, ref3, scrb1, 0 * ROW_BLOCK)
        snz = opnz_g[0]
        pl.set_validshape(snz, [ROW_BLOCK, HEAD_D])
        pl.set_validshape(scrb1, [ROW_BLOCK, HEAD_D])
        pl.move(snz, scrb1)
        dst_s = rW3_g[gate_slot_idx]
        pl.insert(dst_s, snz, [0 * ROW_BLOCK, 0])
        m4_rop_chunk_vf(kn4, cs_slot, ref3, scrb2, 1 * ROW_BLOCK)
        snz = opnz_g[0]
        pl.set_validshape(snz, [ROW_BLOCK, HEAD_D])
        pl.set_validshape(scrb2, [ROW_BLOCK, HEAD_D])
        pl.move(snz, scrb2)
        dst_s = rW3_g[gate_slot_idx]
        pl.insert(dst_s, snz, [1 * ROW_BLOCK, 0])
        m4_rop_chunk_vf(kn4, cs_slot, ref3, scrb0, 2 * ROW_BLOCK)
        snz = opnz_g[0]
        pl.set_validshape(snz, [ROW_BLOCK, HEAD_D])
        pl.set_validshape(scrb0, [ROW_BLOCK, HEAD_D])
        pl.move(snz, scrb0)
        dst_s = rW3_g[gate_slot_idx]
        pl.insert(dst_s, snz, [2 * ROW_BLOCK, 0])
        m4_rop_chunk_vf(kn4, cs_slot, ref3, scrb1, 3 * ROW_BLOCK)
        snz = opnz_g[0]
        pl.set_validshape(snz, [ROW_BLOCK, HEAD_D])
        pl.set_validshape(scrb1, [ROW_BLOCK, HEAD_D])
        pl.move(snz, scrb1)
        dst_s = rW3_g[gate_slot_idx]
        pl.insert(dst_s, snz, [3 * ROW_BLOCK, 0])
        m4_lqop_vf(qn4, cs_slot, ref3, scrb2, r3v, scale)
        snz = opnz_g[0]
        pl.set_validshape(snz, [ROW_BLOCK, HEAD_D])
        pl.set_validshape(scrb2, [ROW_BLOCK, HEAD_D])
        pl.move(snz, scrb2)
        dst_s = lq3_g[gate_slot_idx]
        pl.insert(dst_s, snz, [0, 0])
    pl.system.set_cross_core(pipe=pl.PipeType.MTE3, event_id=7)

    pl.system.set_cross_core(pipe=pl.PipeType.MTE3, event_id=10)

def m4_kkt_mqk_blocks(kt_g, mq_g, pslot,
                      lk0_g, lk1_g, lk2_g, lk3_g,
                      lq0_g, lq1_g, lq2_g, lq3_g,
                      rZ0_g, rZ1_g, rZ2_g, rZ3_g,
                      m4_la_g, m4_b_g, m4_c_g):
    """Module-4 cube body: KKT/Mqk 16-row blocks + Acc→Vec delivery."""
    pl.system.wait_cross_core(pipe=pl.PipeType.MTE1, event_id=10)
    pl.system.set_mm_layout_transform(enabled=True)
    kt_t = kt_g[0]
    mq_t = mq_g[0]
    pl.set_validshape(kt_t, [HS, CHUNK_C])
    pl.set_validshape(mq_t, [HS, CHUNK_C])
    for rb in pl.range(0, 4):
        a_t = m4_la_g.next()
        lk_s = lk0_g[pslot] if rb == 0 else (lk1_g[pslot] if rb == 1 else (lk2_g[pslot] if rb == 2 else lk3_g[pslot]))
        lq_s = lq0_g[pslot] if rb == 0 else (lq1_g[pslot] if rb == 1 else (lq2_g[pslot] if rb == 2 else lq3_g[pslot]))
        b_full = rZ0_g[pslot] if rb == 0 else (rZ1_g[pslot] if rb == 1 else (rZ2_g[pslot] if rb == 2 else rZ3_g[pslot]))
        rs = (rb % 2) * ROW_BLOCK
        pl.move(a_t, lk_s)
        for j in pl.range(0, rb + 1):
            b_t = m4_b_g.next()
            c_t = m4_c_g.next()
            pl.move(b_t, b_full, [0, j * ROW_BLOCK])
            pl.matmul(c_t, a_t, b_t, phase=pl.AccPhase.Final)
            if rb < 2:
                pl.move(kt_t[rs:rs + ROW_BLOCK, j * ROW_BLOCK:(j + 1) * ROW_BLOCK],
                        c_t, acc_to_vec_mode=pl.AccToVecMode.SingleModeVec0,
                        phase=pl.STPhase.Final)
            else:
                pl.move(kt_t[rs:rs + ROW_BLOCK, j * ROW_BLOCK:(j + 1) * ROW_BLOCK],
                        c_t, acc_to_vec_mode=pl.AccToVecMode.SingleModeVec1,
                        phase=pl.STPhase.Final)

        a_t = m4_la_g.next()
        pl.move(a_t, lq_s)
        for j in pl.range(0, rb + 1):
            b_t = m4_b_g.next()
            c_t = m4_c_g.next()
            pl.move(b_t, b_full, [0, j * ROW_BLOCK])
            pl.matmul(c_t, a_t, b_t, phase=pl.AccPhase.Final)
            if rb < 2:
                pl.move(mq_t[rs:rs + ROW_BLOCK, j * ROW_BLOCK:(j + 1) * ROW_BLOCK],
                        c_t, acc_to_vec_mode=pl.AccToVecMode.SingleModeVec0,
                        phase=pl.STPhase.Final)
            else:
                pl.move(mq_t[rs:rs + ROW_BLOCK, j * ROW_BLOCK:(j + 1) * ROW_BLOCK],
                        c_t, acc_to_vec_mode=pl.AccToVecMode.SingleModeVec1,
                        phase=pl.STPhase.Final)
    pl.system.set_mm_layout_transform(enabled=False)
    pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=9)

def t71_prepare_neumann_v(m5_mqk_g, t71_n, t71_eye, t71_zero,
                          t71_eye_bf16, t71_nd_h, t71_nz_h,
                          t71_nd_b, t71_nz_b, x_ub_g, reload_mask):
    """Prepare the Neumann seed tiles and publish event 13 early."""
    t71_sub = pl.get_subblock_idx()
    t71_off = t71_sub * HS
    pl.set_validshape(m5_mqk_g[0], [HS, CHUNK_C])
    pl.set_validshape(t71_nd_h, [HS, CHUNK_C])
    pl.set_validshape(t71_nz_h, [HS, CHUNK_C])
    pl.set_validshape(t71_nd_b, [HS, CHUNK_C])
    pl.set_validshape(t71_nz_b, [HS, CHUNK_C])
    pl.cast(t71_nd_h, m5_mqk_g[0], mode=pl.RoundMode.CAST_RINT)
    pl.move(t71_nz_h, t71_nd_h)
    pl.insert(t71_n, t71_nz_h, [t71_off, 0])
    if reload_mask:
        pl.set_validshape(x_ub_g.current(), [HS, CHUNK_C])
        eye_rows_vf(x_ub_g.current(), t71_off)
        pl.cast(t71_nd_h, x_ub_g.current(), mode=pl.RoundMode.CAST_RINT)
        pl.move(t71_nz_h, t71_nd_h)
        pl.insert(t71_eye, t71_nz_h, [t71_off, 0])
        pl.cast(t71_nd_b, x_ub_g.current(), mode=pl.RoundMode.CAST_RINT)
        pl.move(t71_nz_b, t71_nd_b)
        pl.insert(t71_eye_bf16, t71_nz_b, [t71_off, 0])
        pl.expands(t71_nd_h, 0.0)
        pl.move(t71_nz_h, t71_nd_h)
        pl.insert(t71_zero, t71_nz_h, [t71_off, 0])
    pl.system.set_cross_core(pipe=pl.PipeType.MTE3, event_id=13)

def m5_finish_v(beta, v_v, kn_slot,
                mqk_o, v_beta_bf16_o, v_beta_lo_o,
                layout_code, cs_g, reload_mask,
                c, b, hv, s0, Nv,
                y0_l1, snz_stg_g, snz_nz_g, kd_nz_g,
                kd_l1,
                m5_beta_raw_b_g, m5_beta_blk_f_g, m5_v_b_g,
                m5_kkt_g, m5_mqk_g, m5_mqk_b_g, m5_v_operand_g, m5_kd_operand_g,
                t71_n, t71_eye, t71_zero, t71_eye_bf16,
                t71_nd_h, t71_nz_h, t71_nd_b, t71_nz_b, x_ub_g):
    """M5 finish, issued one work behind the merged loop (lagged)."""
    h0 = (hv // 64) * 64
    hv_local = hv - h0
    off_base = hv_local * (1 - layout_code) + hv_local * 64 * layout_code
    step = 64 * (1 - layout_code) + 1 * layout_code
    brow_b = s0 * (1 - layout_code) + h0 * layout_code
    bcol_b = h0 * (1 - layout_code) + s0 * layout_code
    n_rows = 64 * (1 - layout_code) + pl.min(64, Nv - h0) * layout_code
    n_cols = pl.min(64, Nv - h0) * (1 - layout_code) + 64 * layout_code
    sub_m5 = pl.get_subblock_idx()
    r0 = sub_m5 * HS
    bb_raw = m5_beta_raw_b_g.next()
    pl.set_validshape(bb_raw, [n_rows, n_cols])
    pl.load(bb_raw, beta, [b, brow_b, bcol_b], order=[1, 2])
    blk_f = m5_beta_blk_f_g.next()
    pl.set_validshape(blk_f, [n_rows, n_cols])
    pl.cast(blk_f, bb_raw)
    vb = m5_v_b_g.next()
    pl.load(vb, v_v, [b, hv, s0 + r0, 0], order=[2, 3])

    pl.system.wait_cross_core(pipe=pl.PipeType.V, event_id=11)
    cs_slot = cs_g[0]
    kb = kn_slot
    vf_t = m5_v_operand_g.next()
    kf = m5_kd_operand_g.next()
    m5_vkn_vf(vb, kb, cs_slot, blk_f, vf_t, kf, off_base, step, r0)
    pl.system.set_cross_core(pipe=pl.PipeType.V, event_id=_M5_CS_DONE)
    pl.store(v_beta_bf16_o, vf_t, [b, hv, s0 + r0, 0], order=[2, 3])
    pl.store(v_beta_lo_o, vb, [b, hv, s0 + r0, 0], order=[2, 3])

    pl.system.wait_cross_core(pipe=pl.PipeType.V, event_id=9)
    kt = m5_kkt_g[0]
    mq = m5_mqk_g[0]
    mqb = m5_mqk_b_g.next()
    m5_tmqk_vf(kt, mq, blk_f, mq, mqb, off_base, step, r0)
    pl.set_validshape(mq, [HS, CHUNK_C])
    nd2nz_sg(mq, snz_stg_g.current(), HS)
    pl.set_validshape(snz_nz_g.current(), [HS, CHUNK_C])
    pl.insert(y0_l1, snz_nz_g.current(), [r0, 0])
    pl.store(mqk_o, mqb, [b, hv, c, r0, 0])

    t71_prepare_neumann_v(m5_mqk_g, t71_n, t71_eye, t71_zero,
                          t71_eye_bf16, t71_nd_h, t71_nz_h,
                          t71_nd_b, t71_nz_b, x_ub_g, reload_mask)
    kd_nz = kd_nz_g.next()
    pl.set_validshape(kd_nz, [HS, HEAD_D])
    pl.move(kd_nz, kf)
    pl.insert(kd_l1, kd_nz, [r0, 0])

    # _M5M6_EVENT had no consumer in the current M6 chain; do not emit a
    # per-work cross-core token that can accumulate and exhaust the event queue.
    pl.system.set_cross_core(pipe=pl.PipeType.MTE3, event_id=_M6_UPRE_EVENT)

def t71_next_l0_fp16(a_g, b_g, ad0_g, ad1_g, ad2_g, ad3_g, a10_g, a32_g,
                     bd0_g, bd1_g, bd2_g, bd3_g, b10_g, b32_g,
                     bt0_g, bt1_g, bl0_g, bl1_g):
    a_g.next(); b_g.next()
    ad0_g.next(); ad1_g.next(); ad2_g.next(); ad3_g.next()
    a10_g.next(); a32_g.next()
    bd0_g.next(); bd1_g.next(); bd2_g.next(); bd3_g.next()
    b10_g.next(); b32_g.next()
    bt0_g.next(); bt1_g.next(); bl0_g.next(); bl1_g.next()

def t71_next_l0_bf16(a_g, b_g):
    a_g.next(); b_g.next()

def t71_next_l0c16(c_g, d0_g, d1_g, d2_g, d3_g):
    c_g.next(); d0_g.next(); d1_g.next(); d2_g.next(); d3_g.next()

def m6_neumann_cube(
        t71_a_g, t71_b_g, t71_ad0_g, t71_ad1_g,
        t71_ad2_g, t71_ad3_g, t71_a10_g, t71_a32_g,
        t71_bd0_g, t71_bd1_g, t71_bd2_g, t71_bd3_g,
        t71_b10_g, t71_b32_g, t71_bt0_g, t71_bt1_g,
        t71_bl0_g, t71_bl1_g, t71_zero, t71_eye,
        t71_aux_g, t71_auxd0_g, t71_auxd1_g, t71_auxd2_g,
        t71_auxd3_g, t71_a_bf16_g, t71_b_bf16_g, t71_eye_bf16,
        v_beta_bf16_o, v_beta_lo_o, w_o, upre_o, t71_c_g,
        t71_n_db_g, p_gsi, pb, phv,
        ps0, t71_scratch_g, t71_tmpd0_g, t71_tmpd1_g,
        t71_tmpd2_g, t71_tmpd3_g, t71_p_g, t71_tmp_g,
        t71_inv_bf16_g, x_ub_g, m6_inv_l1_g, m6_lb128_g,
        m6_op_l1_g, m6_la_g, m6_w_lc_g, m6_vbt_l1_g,
        m6_la_vbt_g, m6_lb_invt_g, m6_invt_zn_g, m6_u_lc_g):
    """m6 Neumann cube chain for one work: seed, 6 double-doubling rounds, fused W/Upre tail."""
    with pl.section_cube():
        pl.system.wait_cross_core(pipe=pl.PipeType.MTE1, event_id=13)
        t71_next_l0_fp16(t71_a_g, t71_b_g, t71_ad0_g, t71_ad1_g, t71_ad2_g, t71_ad3_g, t71_a10_g, t71_a32_g, t71_bd0_g, t71_bd1_g, t71_bd2_g, t71_bd3_g, t71_b10_g, t71_b32_g, t71_bt0_g, t71_bt1_g, t71_bl0_g, t71_bl1_g)
        t71_c_g.next()
        pl.move(t71_a_g.current(), t71_zero)
        pl.move(t71_b_g.current(), t71_zero)
        pl.move(t71_a_g.current(), t71_n_db_g[p_gsi])
        pl.move(t71_b_g.current(), t71_eye)
        pl.matmul(t71_c_g.current(), t71_a_g.current(), t71_b_g.current())
        t71_next_l0_fp16(t71_a_g, t71_b_g, t71_ad0_g, t71_ad1_g, t71_ad2_g, t71_ad3_g, t71_a10_g, t71_a32_g, t71_bd0_g, t71_bd1_g, t71_bd2_g, t71_bd3_g, t71_b10_g, t71_b32_g, t71_bt0_g, t71_bt1_g, t71_bl0_g, t71_bl1_g)
        pl.move(t71_a_g.current(), t71_zero)
        pl.move(t71_b_g.current(), t71_zero)
        pl.move(t71_a_g.current(), t71_eye)
        pl.move(t71_b_g.current(), t71_eye)
        pl.matmul_acc(t71_c_g.current(), t71_c_g.current(), t71_a_g.current(), t71_b_g.current())
        pl.move(t71_scratch_g.current(), t71_c_g.current())
        t71_next_l0_fp16(t71_a_g, t71_b_g, t71_ad0_g, t71_ad1_g, t71_ad2_g, t71_ad3_g, t71_a10_g, t71_a32_g, t71_bd0_g, t71_bd1_g, t71_bd2_g, t71_bd3_g, t71_b10_g, t71_b32_g, t71_bt0_g, t71_bt1_g, t71_bl0_g, t71_bl1_g)
        t71_c_g.next()
        t71_next_l0c16(t71_aux_g, t71_auxd0_g, t71_auxd1_g, t71_auxd2_g, t71_auxd3_g)
        pl.move(t71_a_g.current(), t71_zero)
        pl.move(t71_b_g.current(), t71_zero)
        pl.move(t71_ad0_g.current(), t71_n_db_g[p_gsi], [0, 0])
        pl.move(t71_bd0_g.current(), t71_n_db_g[p_gsi], [0, 0])
        pl.move(t71_ad1_g.current(), t71_n_db_g[p_gsi], [16, 16])
        pl.move(t71_bd1_g.current(), t71_n_db_g[p_gsi], [16, 16])
        pl.move(t71_ad2_g.current(), t71_n_db_g[p_gsi], [32, 32])
        pl.move(t71_bd2_g.current(), t71_n_db_g[p_gsi], [32, 32])
        pl.move(t71_ad3_g.current(), t71_n_db_g[p_gsi], [48, 48])
        pl.move(t71_bd3_g.current(), t71_n_db_g[p_gsi], [48, 48])
        pl.matmul(t71_auxd0_g.current(), t71_ad0_g.current(), t71_bd0_g.current())
        pl.matmul(t71_auxd1_g.current(), t71_ad1_g.current(), t71_bd1_g.current())
        pl.matmul(t71_auxd2_g.current(), t71_ad2_g.current(), t71_bd2_g.current())
        pl.matmul(t71_auxd3_g.current(), t71_ad3_g.current(), t71_bd3_g.current())
        pl.move(t71_tmpd0_g.current(), t71_auxd0_g.current())
        pl.move(t71_tmpd1_g.current(), t71_auxd1_g.current())
        pl.move(t71_tmpd2_g.current(), t71_auxd2_g.current())
        pl.move(t71_tmpd3_g.current(), t71_auxd3_g.current())
        t71_next_l0_fp16(t71_a_g, t71_b_g, t71_ad0_g, t71_ad1_g, t71_ad2_g, t71_ad3_g, t71_a10_g, t71_a32_g, t71_bd0_g, t71_bd1_g, t71_bd2_g, t71_bd3_g, t71_b10_g, t71_b32_g, t71_bt0_g, t71_bt1_g, t71_bl0_g, t71_bl1_g)
        pl.move(t71_a_g.current(), t71_zero)
        pl.move(t71_b_g.current(), t71_zero)
        pl.move(t71_ad0_g.current(), t71_scratch_g.current(), [0, 0])
        pl.move(t71_bd0_g.current(), t71_eye, [0, 0])
        pl.move(t71_ad1_g.current(), t71_scratch_g.current(), [16, 16])
        pl.move(t71_bd1_g.current(), t71_eye, [16, 16])
        pl.move(t71_ad2_g.current(), t71_scratch_g.current(), [32, 32])
        pl.move(t71_bd2_g.current(), t71_eye, [32, 32])
        pl.move(t71_ad3_g.current(), t71_scratch_g.current(), [48, 48])
        pl.move(t71_bd3_g.current(), t71_eye, [48, 48])
        pl.matmul(t71_c_g.current(), t71_a_g.current(), t71_b_g.current())
        t71_next_l0_fp16(t71_a_g, t71_b_g, t71_ad0_g, t71_ad1_g, t71_ad2_g, t71_ad3_g, t71_a10_g, t71_a32_g, t71_bd0_g, t71_bd1_g, t71_bd2_g, t71_bd3_g, t71_b10_g, t71_b32_g, t71_bt0_g, t71_bt1_g, t71_bl0_g, t71_bl1_g)
        pl.move(t71_ad0_g.current(), t71_scratch_g.current(), [0, 0])
        pl.move(t71_bd0_g.current(), t71_tmpd0_g.current())
        pl.move(t71_ad1_g.current(), t71_scratch_g.current(), [16, 16])
        pl.move(t71_bd1_g.current(), t71_tmpd1_g.current())
        pl.move(t71_ad2_g.current(), t71_scratch_g.current(), [32, 32])
        pl.move(t71_bd2_g.current(), t71_tmpd2_g.current())
        pl.move(t71_ad3_g.current(), t71_scratch_g.current(), [48, 48])
        pl.move(t71_bd3_g.current(), t71_tmpd3_g.current())
        pl.matmul_acc(t71_c_g.current(), t71_c_g.current(), t71_a_g.current(), t71_b_g.current())
        pl.move(t71_p_g.current(), t71_c_g.current())
        t71_next_l0_fp16(t71_a_g, t71_b_g, t71_ad0_g, t71_ad1_g, t71_ad2_g, t71_ad3_g, t71_a10_g, t71_a32_g, t71_bd0_g, t71_bd1_g, t71_bd2_g, t71_bd3_g, t71_b10_g, t71_b32_g, t71_bt0_g, t71_bt1_g, t71_bl0_g, t71_bl1_g)
        t71_next_l0c16(t71_aux_g, t71_auxd0_g, t71_auxd1_g, t71_auxd2_g, t71_auxd3_g)
        pl.move(t71_ad0_g.current(), t71_tmp_g.current(), [0, 0])
        pl.move(t71_bd0_g.current(), t71_tmpd0_g.current())
        pl.move(t71_ad1_g.current(), t71_tmp_g.current(), [16, 16])
        pl.move(t71_bd1_g.current(), t71_tmpd1_g.current())
        pl.move(t71_ad2_g.current(), t71_tmp_g.current(), [32, 32])
        pl.move(t71_bd2_g.current(), t71_tmpd2_g.current())
        pl.move(t71_ad3_g.current(), t71_tmp_g.current(), [48, 48])
        pl.move(t71_bd3_g.current(), t71_tmpd3_g.current())
        pl.matmul(t71_auxd0_g.current(), t71_ad0_g.current(), t71_bd0_g.current())
        pl.matmul(t71_auxd1_g.current(), t71_ad1_g.current(), t71_bd1_g.current())
        pl.matmul(t71_auxd2_g.current(), t71_ad2_g.current(), t71_bd2_g.current())
        pl.matmul(t71_auxd3_g.current(), t71_ad3_g.current(), t71_bd3_g.current())
        pl.move(t71_tmpd0_g.current(), t71_auxd0_g.current())
        pl.move(t71_tmpd1_g.current(), t71_auxd1_g.current())
        pl.move(t71_tmpd2_g.current(), t71_auxd2_g.current())
        pl.move(t71_tmpd3_g.current(), t71_auxd3_g.current())
        t71_next_l0_fp16(t71_a_g, t71_b_g, t71_ad0_g, t71_ad1_g, t71_ad2_g, t71_ad3_g, t71_a10_g, t71_a32_g, t71_bd0_g, t71_bd1_g, t71_bd2_g, t71_bd3_g, t71_b10_g, t71_b32_g, t71_bt0_g, t71_bt1_g, t71_bl0_g, t71_bl1_g)
        pl.move(t71_a_g.current(), t71_zero)
        pl.move(t71_b_g.current(), t71_zero)
        pl.move(t71_ad0_g.current(), t71_p_g.current(), [0, 0])
        pl.move(t71_bd0_g.current(), t71_tmpd0_g.current())
        pl.move(t71_ad1_g.current(), t71_p_g.current(), [16, 16])
        pl.move(t71_bd1_g.current(), t71_tmpd1_g.current())
        pl.move(t71_ad2_g.current(), t71_p_g.current(), [32, 32])
        pl.move(t71_bd2_g.current(), t71_tmpd2_g.current())
        pl.move(t71_ad3_g.current(), t71_p_g.current(), [48, 48])
        pl.move(t71_bd3_g.current(), t71_tmpd3_g.current())
        pl.matmul_acc(t71_c_g.current(), t71_c_g.current(), t71_a_g.current(), t71_b_g.current())
        pl.move(t71_p_g.current(), t71_c_g.current())
        t71_next_l0_fp16(t71_a_g, t71_b_g, t71_ad0_g, t71_ad1_g, t71_ad2_g, t71_ad3_g, t71_a10_g, t71_a32_g, t71_bd0_g, t71_bd1_g, t71_bd2_g, t71_bd3_g, t71_b10_g, t71_b32_g, t71_bt0_g, t71_bt1_g, t71_bl0_g, t71_bl1_g)
        t71_next_l0c16(t71_aux_g, t71_auxd0_g, t71_auxd1_g, t71_auxd2_g, t71_auxd3_g)
        pl.move(t71_ad0_g.current(), t71_tmp_g.current(), [0, 0])
        pl.move(t71_bd0_g.current(), t71_tmpd0_g.current())
        pl.move(t71_ad1_g.current(), t71_tmp_g.current(), [16, 16])
        pl.move(t71_bd1_g.current(), t71_tmpd1_g.current())
        pl.move(t71_ad2_g.current(), t71_tmp_g.current(), [32, 32])
        pl.move(t71_bd2_g.current(), t71_tmpd2_g.current())
        pl.move(t71_ad3_g.current(), t71_tmp_g.current(), [48, 48])
        pl.move(t71_bd3_g.current(), t71_tmpd3_g.current())
        pl.matmul(t71_auxd0_g.current(), t71_ad0_g.current(), t71_bd0_g.current())
        pl.matmul(t71_auxd1_g.current(), t71_ad1_g.current(), t71_bd1_g.current())
        pl.matmul(t71_auxd2_g.current(), t71_ad2_g.current(), t71_bd2_g.current())
        pl.matmul(t71_auxd3_g.current(), t71_ad3_g.current(), t71_bd3_g.current())
        pl.move(t71_tmpd0_g.current(), t71_auxd0_g.current())
        pl.move(t71_tmpd1_g.current(), t71_auxd1_g.current())
        pl.move(t71_tmpd2_g.current(), t71_auxd2_g.current())
        pl.move(t71_tmpd3_g.current(), t71_auxd3_g.current())
        t71_next_l0_fp16(t71_a_g, t71_b_g, t71_ad0_g, t71_ad1_g, t71_ad2_g, t71_ad3_g, t71_a10_g, t71_a32_g, t71_bd0_g, t71_bd1_g, t71_bd2_g, t71_bd3_g, t71_b10_g, t71_b32_g, t71_bt0_g, t71_bt1_g, t71_bl0_g, t71_bl1_g)
        pl.move(t71_a_g.current(), t71_zero)
        pl.move(t71_b_g.current(), t71_zero)
        pl.move(t71_ad0_g.current(), t71_p_g.current(), [0, 0])
        pl.move(t71_bd0_g.current(), t71_tmpd0_g.current())
        pl.move(t71_ad1_g.current(), t71_p_g.current(), [16, 16])
        pl.move(t71_bd1_g.current(), t71_tmpd1_g.current())
        pl.move(t71_ad2_g.current(), t71_p_g.current(), [32, 32])
        pl.move(t71_bd2_g.current(), t71_tmpd2_g.current())
        pl.move(t71_ad3_g.current(), t71_p_g.current(), [48, 48])
        pl.move(t71_bd3_g.current(), t71_tmpd3_g.current())
        pl.matmul_acc(t71_c_g.current(), t71_c_g.current(), t71_a_g.current(), t71_b_g.current())
        pl.move(t71_p_g.current(), t71_c_g.current())
        t71_next_l0_fp16(t71_a_g, t71_b_g, t71_ad0_g, t71_ad1_g, t71_ad2_g, t71_ad3_g, t71_a10_g, t71_a32_g, t71_bd0_g, t71_bd1_g, t71_bd2_g, t71_bd3_g, t71_b10_g, t71_b32_g, t71_bt0_g, t71_bt1_g, t71_bl0_g, t71_bl1_g)
        t71_next_l0c16(t71_aux_g, t71_auxd0_g, t71_auxd1_g, t71_auxd2_g, t71_auxd3_g)
        pl.move(t71_a_g.current(), t71_zero)
        pl.move(t71_b_g.current(), t71_zero)
        pl.move(t71_a10_g.current(), t71_scratch_g.current(), [16, 0])
        pl.move(t71_a32_g.current(), t71_scratch_g.current(), [48, 32])
        pl.move(t71_bd0_g.current(), t71_p_g.current(), [0, 0])
        pl.move(t71_bd2_g.current(), t71_p_g.current(), [32, 32])
        pl.matmul(t71_aux_g.current(), t71_a_g.current(), t71_b_g.current())
        pl.move(t71_tmp_g.current(), t71_aux_g.current())
        t71_next_l0_fp16(t71_a_g, t71_b_g, t71_ad0_g, t71_ad1_g, t71_ad2_g, t71_ad3_g, t71_a10_g, t71_a32_g, t71_bd0_g, t71_bd1_g, t71_bd2_g, t71_bd3_g, t71_b10_g, t71_b32_g, t71_bt0_g, t71_bt1_g, t71_bl0_g, t71_bl1_g)
        pl.move(t71_a_g.current(), t71_zero)
        pl.move(t71_b_g.current(), t71_zero)
        pl.move(t71_ad1_g.current(), t71_p_g.current(), [16, 16])
        pl.move(t71_ad3_g.current(), t71_p_g.current(), [48, 48])
        pl.move(t71_b10_g.current(), t71_tmp_g.current(), [16, 0])
        pl.move(t71_b32_g.current(), t71_tmp_g.current(), [48, 32])
        pl.matmul_acc(t71_c_g.current(), t71_c_g.current(), t71_a_g.current(), t71_b_g.current())
        pl.move(t71_p_g.current(), t71_c_g.current())
        t71_next_l0_fp16(t71_a_g, t71_b_g, t71_ad0_g, t71_ad1_g, t71_ad2_g, t71_ad3_g, t71_a10_g, t71_a32_g, t71_bd0_g, t71_bd1_g, t71_bd2_g, t71_bd3_g, t71_b10_g, t71_b32_g, t71_bt0_g, t71_bt1_g, t71_bl0_g, t71_bl1_g)
        t71_next_l0c16(t71_aux_g, t71_auxd0_g, t71_auxd1_g, t71_auxd2_g, t71_auxd3_g)
        pl.move(t71_a_g.current(), t71_scratch_g.current())
        pl.move(t71_b_g.current(), t71_zero)
        pl.move(t71_bt0_g.current(), t71_p_g.current(), [0, 0])
        pl.move(t71_bt1_g.current(), t71_p_g.current(), [16, 0])
        pl.matmul(t71_aux_g.current(), t71_a_g.current(), t71_b_g.current())
        pl.move(t71_tmp_g.current(), t71_aux_g.current())
        t71_next_l0_fp16(t71_a_g, t71_b_g, t71_ad0_g, t71_ad1_g, t71_ad2_g, t71_ad3_g, t71_a10_g, t71_a32_g, t71_bd0_g, t71_bd1_g, t71_bd2_g, t71_bd3_g, t71_b10_g, t71_b32_g, t71_bt0_g, t71_bt1_g, t71_bl0_g, t71_bl1_g)
        pl.move(t71_a_g.current(), t71_p_g.current())
        pl.move(t71_b_g.current(), t71_zero)
        pl.move(t71_bl0_g.current(), t71_tmp_g.current(), [32, 0])
        pl.move(t71_bl1_g.current(), t71_tmp_g.current(), [48, 0])
        pl.matmul_acc(t71_c_g.current(), t71_c_g.current(), t71_a_g.current(), t71_b_g.current())
        pl.move(t71_inv_bf16_g.current(), t71_c_g.current())
        t71_next_l0_bf16(t71_a_bf16_g, t71_b_bf16_g)
        t71_next_l0c16(t71_aux_g, t71_auxd0_g, t71_auxd1_g, t71_auxd2_g, t71_auxd3_g)
        pl.move(t71_a_bf16_g.current(), t71_inv_bf16_g.current())
        pl.move(t71_b_bf16_g.current(), t71_eye_bf16)
        pl.matmul(t71_aux_g.current(), t71_a_bf16_g.current(), t71_b_bf16_g.current(), phase=pl.AccPhase.Final)
        pl.set_validshape(x_ub_g.current(), [HS, CHUNK_C])
        pl.move(x_ub_g.current(), t71_aux_g.current(), acc_to_vec_mode=pl.AccToVecMode.DualModeSplitM, phase=pl.STPhase.Final)
        pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=12)
    with pl.section_vector():
        pl.system.wait_cross_core(pipe=pl.PipeType.V, event_id=12)
        pl.set_validshape(x_ub_g.current(), [HS, CHUNK_C])
    with pl.section_cube():
        pl.move(m6_inv_l1_g.next(), t71_c_g.current())
        pl.move(m6_lb128_g.next(), m6_op_l1_g.current())
        pl.move(m6_la_g.next(), m6_inv_l1_g.current())
        pl.matmul(m6_w_lc_g.next(), m6_la_g.current(), m6_lb128_g.current(),
                  phase=pl.AccPhase.Final)
        pl.system.wait_cross_core(pipe=pl.PipeType.MTE2, event_id=_M6_UPRE_EVENT)
        pl.load(m6_vbt_l1_g.next(), v_beta_bf16_o, [pb, phv, ps0, 0], order=[3, 2])
        pl.move(m6_la_vbt_g.next(), m6_vbt_l1_g.current())
        pl.move(m6_lb_invt_g.next(), m6_invt_zn_g.current())
        pl.matmul(m6_u_lc_g.next(), m6_la_vbt_g.current(), m6_lb_invt_g.current(),
                  phase=pl.AccPhase.Partial)
        # Accumulate the low product before the FP32 Upre store.
        pl.load(m6_vbt_l1_g.next(), v_beta_lo_o, [pb, phv, ps0, 0], order=[3, 2])
        pl.move(m6_la_vbt_g.next(), m6_vbt_l1_g.current())
        pl.matmul_acc(m6_u_lc_g.current(), m6_u_lc_g.current(),
                      m6_la_vbt_g.current(), m6_lb_invt_g.current(),
                      phase=pl.AccPhase.Final)
        pl.store(w_o, m6_w_lc_g.current(), [pb, phv, ps0, 0], order=[2, 3],
                 phase=pl.STPhase.Final)
        pl.store(upre_o, m6_u_lc_g.current(), [pb, phv, 0, ps0], order=[2, 3],
                 phase=pl.STPhase.Final)


@pl.jit(auto_mutex=True)
def flash_kda_stage1_kernel(
    # 输入
    q: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16],
    k: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16],
    g: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16],
    beta: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16],
    v: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16],
    a_log: pl.Tensor[[pl.DYNAMIC], pl.DT_FP32],
    dt_bias: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    lower_bound: pl.DT_FP32,
    scale: pl.DT_FP32,
    layout_code: pl.DT_INT64,
    # 输出
    qd_out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16],
    kr_out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16],
    gam_out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    mqk_out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16],
    w_out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16],
    upre_out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    # workspace
    workspace: pl.Ptr[pl.DT_UINT8],
):
    core_id = pl.get_block_idx() // pl.get_subblock_num()
    num_cores = pl.get_block_num()

    B = g.shape[0]
    S = g.shape[1] * (1 - layout_code) + g.shape[2] * layout_code
    Nk = q.shape[2] * (1 - layout_code) + q.shape[1] * layout_code
    Nv = g.shape[2] * (1 - layout_code) + g.shape[1] * layout_code
    nq = Nv // Nk
    nc = S // CHUNK_C
    total = B * Nv
    tt_b = pl.TileType(shape=[CHUNK_C, HEAD_D], dtype=pl.DT_BF16,
                       target_memory=pl.MemorySpace.Vec)
    tt_f32 = pl.TileType(shape=[HS, HEAD_D], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Vec)
    tt_b32 = pl.TileType(shape=[HS, HEAD_D], dtype=pl.DT_BF16,
                         target_memory=pl.MemorySpace.Vec)
    tt_c64h = pl.TileType(shape=[HS, CHUNK_C], dtype=pl.DT_FP32,
                          target_memory=pl.MemorySpace.Vec)
    tt_c64bh = pl.TileType(shape=[HS, CHUNK_C], dtype=pl.DT_BF16,
                           target_memory=pl.MemorySpace.Vec)
    tt_a = pl.TileType(shape=[CHUNK_C, 1], dtype=pl.DT_FP32,
                       target_memory=pl.MemorySpace.Vec,
                       valid_shape=[-1, -1])
    tt_d = pl.TileType(shape=[1, HEAD_D], dtype=pl.DT_FP32,
                       target_memory=pl.MemorySpace.Vec)
    tt_c64 = pl.TileType(shape=[CHUNK_C, CHUNK_C], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Vec)
    tt_c64b = pl.TileType(shape=[CHUNK_C, CHUNK_C], dtype=pl.DT_BF16,
                          target_memory=pl.MemorySpace.Vec)

    gate_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[CHUNK_C, HEAD_D], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Mat, layout=pl.NZ),
        addrs=[L1_G0, L1_G1], mutex_ids=[11, 12],
        fwd_ids=_GATE_FWD_IDS, bwd_ids=_GATE_BWD_IDS,
    )

    cs_g = pl.make_tile_group(
        type=pl.TileType(shape=[CHUNK_C, HEAD_D], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Vec),
        addrs=[VA_CS0], mutex_ids=[16],
        fwd_ids=(4,), bwd_ids=(6,),
    )

    x_b_g = pl.make_tile_group(type=tt_b32, addrs=[VA_X_B], mutex_ids=[0])
    qn_h_g = pl.make_tile_group(
        type=tt_b32, addrs=[0x34800, 0x36800], mutex_ids=[10, 13])
    kn_h_g = pl.make_tile_group(
        type=tt_b32, addrs=[0x38800, 0x3A800], mutex_ids=[14, 15])
    xk_g = pl.make_tile_group(type=tt_b32, addrs=[0x21000], mutex_ids=[17])

    xg_g = pl.make_tile_group(type=tt_b32, addrs=[0x2000], mutex_ids=[18])
    gate_f_g = pl.make_tile_group(type=tt_f32, addrs=[VA_GATE_F], mutex_ids=[5])
    gate_nz_g = pl.make_tile_group(
        type=pl.TileType(shape=[HS, HEAD_D], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Vec, layout=pl.NZ,
                         valid_shape=[-1, -1], compact=1),
        addrs=[VA_GATE_NZ], mutex_ids=[6])
    a_g = pl.make_tile_group(type=tt_a, addrs=[VA_A_LOG], mutex_ids=[8])
    d_g = pl.make_tile_group(type=tt_d, addrs=[VA_DT_B], mutex_ids=[9])

    m3_out_b_g = pl.make_tile_group(type=tt_b32, addrs=[VA_QN_B], mutex_ids=[3])
    m3_csl_g = pl.make_tile_group(
        type=pl.TileType(shape=[1, HEAD_D], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Vec),
        addrs=[0x20800], mutex_ids=[21])

    m4_kn_b_g = pl.make_tile_group(type=tt_b, addrs=[VA_KN_B], mutex_ids=[4])
    m4_qn_b_g = pl.make_tile_group(type=tt_b, addrs=[VA_QN_B], mutex_ids=[3])
    m4_opscr_b16_g = pl.make_tile_group(
        type=pl.TileType(shape=[ROW_BLOCK, HEAD_D], dtype=pl.DT_BF16,
                         target_memory=pl.MemorySpace.Vec),
        addrs=[VA_M4_SCR_B16, VA_M4_SCR_B16 + 0x1000], mutex_ids=[6, 19])
    m4_ref_g = pl.make_tile_group(
        type=pl.TileType(shape=[1, HEAD_D], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Vec),
        addrs=[0x20A00], mutex_ids=[30])

    _tt_op16 = pl.TileType(shape=[ROW_BLOCK, HEAD_D], dtype=pl.DT_BF16,
                           target_memory=pl.MemorySpace.Mat, layout=pl.NZ,
                           valid_shape=[-1, -1], compact=1)

    lk0_g = pl.make_tile_group(type=_tt_op16, addrs=[0x14000, 0x15000], mutex_ids=[31, 27])
    lk1_g = pl.make_tile_group(type=_tt_op16, addrs=[0x16000, 0x17000], mutex_ids=[28, 29])
    lk2_g = pl.make_tile_group(type=_tt_op16, addrs=[0x18000, 0x19000], mutex_ids=[30, 31])
    lk3_g = pl.make_tile_group(type=_tt_op16, addrs=[0x1A000, 0x1B000], mutex_ids=[27, 28])
    lq0_g = pl.make_tile_group(type=_tt_op16, addrs=[0x1C000, 0x1D000], mutex_ids=[29, 30])
    lq1_g = pl.make_tile_group(type=_tt_op16, addrs=[0x1E000, 0x1F000], mutex_ids=[31, 27])
    lq2_g = pl.make_tile_group(type=_tt_op16, addrs=[0x20000, 0x21000], mutex_ids=[28, 29])
    lq3_g = pl.make_tile_group(type=_tt_op16, addrs=[0x22000, 0x23000], mutex_ids=[30, 31])
    _tt_rw = pl.TileType(shape=[CHUNK_C, HEAD_D], dtype=pl.DT_BF16,
                         target_memory=pl.MemorySpace.Mat, layout=pl.NZ,
                         valid_shape=[-1, -1], compact=1)
    _tt_rz64 = pl.TileType(shape=[HEAD_D, CHUNK_C], dtype=pl.DT_BF16,
                           target_memory=pl.MemorySpace.Mat, layout=pl.ZN,
                           valid_shape=[-1, -1], compact=1)
    rW0_g = pl.make_tile_group(type=_tt_rw, addrs=[0x34000, 0x56000], mutex_ids=[27, 28])
    rW1_g = pl.make_tile_group(type=_tt_rw, addrs=[0x38000, 0x5A000], mutex_ids=[29, 30])
    rW2_g = pl.make_tile_group(type=_tt_rw, addrs=[0x3C000, 0x5E000], mutex_ids=[31, 27])
    rW3_g = pl.make_tile_group(type=_tt_rw, addrs=[0x40000, 0x62000], mutex_ids=[28, 29])
    rZ0_g = pl.make_tile_group(type=_tt_rz64, addrs=[0x34000, 0x56000], mutex_ids=[27, 28])
    rZ1_g = pl.make_tile_group(type=_tt_rz64, addrs=[0x38000, 0x5A000], mutex_ids=[29, 30])
    rZ2_g = pl.make_tile_group(type=_tt_rz64, addrs=[0x3C000, 0x5E000], mutex_ids=[31, 27])
    rZ3_g = pl.make_tile_group(type=_tt_rz64, addrs=[0x40000, 0x62000], mutex_ids=[28, 29])

    # kn_h_g[1] ends at 0x3C800; the lagged M5 still needs its last rows.
    opnz_g = pl.make_tile_group(
        type=pl.TileType(shape=[ROW_BLOCK, HEAD_D], dtype=pl.DT_BF16,
                         target_memory=pl.MemorySpace.Vec, layout=pl.NZ,
                         valid_shape=[-1, -1], compact=1),
        addrs=[0x3C800],
        mutex_ids=[2])
    m4_la_g = pl.make_tile_group(
        type=pl.TileType(shape=[ROW_BLOCK, HEAD_D], dtype=pl.DT_BF16,
                         target_memory=pl.MemorySpace.Left, layout=pl.NZ),
        addrs=[0x4000, 0xD000], mutex_ids=[13, 23])
    m4_b_g = pl.make_tile_group(
        type=pl.TileType(shape=[HEAD_D, ROW_BLOCK], dtype=pl.DT_BF16,
                         target_memory=pl.MemorySpace.Right, layout=pl.ZN),
        addrs=[0x8000, 0x9000, 0xA000, 0xB000],
        mutex_ids=[14, 16, 17, 18])
    m4_c_g = pl.make_tile_group(
        type=pl.TileType(shape=[ROW_BLOCK, ROW_BLOCK], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Acc, layout=pl.NZ,
                         fractal=1024),
        addrs=[0x8000, 0x8400, 0x8800, 0x8C00],
        mutex_ids=[15, 19, 20, 21])

    l_pre = pl.make_tile(
        pl.TileType(shape=[CHUNK_C, CHUNK_C], dtype=pl.DT_FP32,
                    target_memory=pl.MemorySpace.Mat, layout=pl.NZ),
        addr=L1_L)
    L_l0a_g = pl.make_tile_group(
        type=pl.TileType(shape=[CHUNK_C, CHUNK_C], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Left, layout=pl.NZ),
        addrs=[LA0], mutex_ids=[13])
    g_l0b_g = pl.make_tile_group(
        type=pl.TileType(shape=[CHUNK_C, HEAD_D], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Right, layout=pl.ZN),
        addrs=[RA0], mutex_ids=[14])
    acc_g = pl.make_tile_group(
        type=pl.TileType(shape=[CHUNK_C, HEAD_D], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Acc, layout=pl.NZ,
                         fractal=1024),
        addrs=[CA0], mutex_ids=[15])

    m5_kkt_g = pl.make_tile_group(
        type=tt_c64h, addrs=[VA_M5_KKT], mutex_ids=[1])
    m5_mqk_g = pl.make_tile_group(
        type=tt_c64h, addrs=[VA_M5_MQK], mutex_ids=[2])
    m5_mqk_b_g = pl.make_tile_group(
        type=tt_c64bh, addrs=[0xE100], mutex_ids=[2])

    m5_beta_raw_b_g = pl.make_tile_group(
        type=tt_c64b, addrs=[0x12000], mutex_ids=[2])
    m5_beta_blk_f_g = pl.make_tile_group(
        type=tt_c64, addrs=[VA_M5_BETA_F], mutex_ids=[3])
    m5_v_b_g = pl.make_tile_group(
        type=tt_b32, addrs=[VA_M5_VB], mutex_ids=[4])
    m5_v_operand_g = pl.make_tile_group(
        type=tt_b32, addrs=[VA_M5_VF], mutex_ids=[6])
    m5_kd_operand_g = pl.make_tile_group(
        type=tt_b32, addrs=[VA_M5_KNF], mutex_ids=[5])

    m5_kd_nz_g = pl.make_tile_group(
        type=pl.TileType(shape=[HS, HEAD_D], dtype=pl.DT_BF16,
                         target_memory=pl.MemorySpace.Vec, layout=pl.NZ,
                         valid_shape=[-1, -1], compact=1),
        addrs=[P3_UB_XY], mutex_ids=[16])

    tt_m6_l64 = pl.TileType(shape=[CHUNK_C, CHUNK_C], dtype=pl.DT_FP32,
                            target_memory=pl.MemorySpace.Mat, layout=pl.NZ,
                            valid_shape=[-1, -1])
    tt_m6_lc128 = pl.TileType(shape=[CHUNK_C, HEAD_D], dtype=pl.DT_FP32,
                              target_memory=pl.MemorySpace.Acc, layout=pl.NZ,
                              fractal=1024)
    tt_m6_vhs = pl.TileType(shape=[HS, CHUNK_C], dtype=pl.DT_FP32,
                            target_memory=pl.MemorySpace.Vec,
                            pad=pl.TilePad.zero, valid_shape=[-1, -1])
    tt_m6_vhs33 = pl.TileType(shape=[HS + 1, CHUNK_C], dtype=pl.DT_FP32,
                              target_memory=pl.MemorySpace.Vec,
                              pad=pl.TilePad.zero, valid_shape=[-1, -1])
    tt_m6_nz = pl.TileType(shape=[HS + 1, CHUNK_C], dtype=pl.DT_FP32,
                           target_memory=pl.MemorySpace.Vec, layout=pl.NZ)
    x_ub_g = pl.make_tile_group(type=tt_m6_vhs, addrs=[P3_UB_X], mutex_ids=[15])
    xy_ub_g = pl.make_tile_group(type=tt_m6_vhs33, addrs=[P3_UB_XY], mutex_ids=[16])

    snz_ub_g = pl.make_tile_group(type=tt_m6_nz, addrs=[P3_UB_XY], mutex_ids=[16])
    y_l1_g = pl.make_tile_group(type=tt_m6_l64, addrs=[P3_L1_Y0, P3_L1_Y1],
                                mutex_ids=[0, 1])
    _t71_l1_fp16 = pl.TileType(shape=[64, 64], dtype=pl.DT_FP16,
                               target_memory=pl.MemorySpace.Mat, layout=pl.NZ,
                               fractal=512, compact=0)
    t71_n_db_g = pl.make_tile_group(type=_t71_l1_fp16,
                                    addrs=[0x76000, 0x78000],
                                    mutex_ids=[24, 25])
    t71_eye = pl.make_tile(_t71_l1_fp16, addr=0x7A000)
    t71_zero = pl.make_tile(_t71_l1_fp16, addr=0x7C000)
    t71_scratch_g = pl.make_tile_group(type=_t71_l1_fp16, addrs=172032, mutex_ids=[26])
    t71_p_g = pl.make_tile_group(type=_t71_l1_fp16, addrs=180224, mutex_ids=[27])
    t71_tmp_g = pl.make_tile_group(type=_t71_l1_fp16, addrs=188416, mutex_ids=[28])
    _t71_l1_fp16_d = pl.TileType(shape=[16, 16], dtype=pl.DT_FP16,
                                 target_memory=pl.MemorySpace.Mat, layout=pl.NZ,
                                 fractal=512, compact=0)
    t71_tmpd0_g = pl.make_tile_group(type=_t71_l1_fp16_d, addrs=188416, mutex_ids=[28])
    t71_tmpd1_g = pl.make_tile_group(type=_t71_l1_fp16_d, addrs=188416 + 0xA00, mutex_ids=[29])
    t71_tmpd2_g = pl.make_tile_group(type=_t71_l1_fp16_d, addrs=188416 + 0x1400, mutex_ids=[30])
    t71_tmpd3_g = pl.make_tile_group(type=_t71_l1_fp16_d, addrs=188416 + 0x1E00, mutex_ids=[31])
    _t71_l1_bf16 = pl.TileType(shape=[64, 64], dtype=pl.DT_BF16,
                               target_memory=pl.MemorySpace.Mat, layout=pl.NZ,
                               fractal=512, compact=0)
    t71_inv_bf16_g = pl.make_tile_group(type=_t71_l1_bf16, addrs=147456, mutex_ids=[29])
    t71_eye_bf16 = pl.make_tile(_t71_l1_bf16, addr=0x7E000)

    _t71_a = pl.TileType(shape=[64, 64], dtype=pl.DT_FP16,
                         target_memory=pl.MemorySpace.Left, layout=pl.NZ,
                         fractal=512, compact=0)
    _t71_b = pl.TileType(shape=[64, 64], dtype=pl.DT_FP16,
                         target_memory=pl.MemorySpace.Right, layout=pl.ZN,
                         fractal=512, compact=0)
    t71_a_g = pl.make_tile_group(type=_t71_a, addrs=[0x5000, 0x7000], mutex_ids=[18, 19])
    t71_b_g = pl.make_tile_group(type=_t71_b, addrs=[0xC000, 0xE000], mutex_ids=[20, 21])
    _t71_ad = pl.TileType(shape=[16, 16], dtype=pl.DT_FP16,
                          target_memory=pl.MemorySpace.Left, layout=pl.NZ,
                          fractal=512, compact=0)
    _t71_bd = pl.TileType(shape=[16, 16], dtype=pl.DT_FP16,
                          target_memory=pl.MemorySpace.Right, layout=pl.ZN,
                          fractal=512, compact=0)
    t71_ad0_g = pl.make_tile_group(type=_t71_ad, addrs=[0x5000, 0x7000], mutex_ids=[18, 19])
    t71_ad1_g = pl.make_tile_group(type=_t71_ad, addrs=[0x5A00, 0x7A00], mutex_ids=[0, 1])
    t71_ad2_g = pl.make_tile_group(type=_t71_ad, addrs=[0x6400, 0x8400], mutex_ids=[2, 3])
    t71_ad3_g = pl.make_tile_group(type=_t71_ad, addrs=[0x6E00, 0x8E00], mutex_ids=[4, 5])
    t71_a10_g = pl.make_tile_group(type=_t71_ad, addrs=[0x5200, 0x7200], mutex_ids=[18, 19])
    t71_a32_g = pl.make_tile_group(type=_t71_ad, addrs=[0x6600, 0x8600], mutex_ids=[18, 19])
    t71_bd0_g = pl.make_tile_group(type=_t71_bd, addrs=[0xC000, 0xE000], mutex_ids=[20, 21])
    t71_bd1_g = pl.make_tile_group(type=_t71_bd, addrs=[0xCA00, 0xEA00], mutex_ids=[6, 7])
    t71_bd2_g = pl.make_tile_group(type=_t71_bd, addrs=[0xD400, 0xF400], mutex_ids=[8, 9])
    t71_bd3_g = pl.make_tile_group(type=_t71_bd, addrs=[0xDE00, 0xFE00], mutex_ids=[10, 11])
    t71_b10_g = pl.make_tile_group(type=_t71_bd, addrs=[0xC800, 0xE800], mutex_ids=[20, 21])
    t71_b32_g = pl.make_tile_group(type=_t71_bd, addrs=[0xDC00, 0xFC00], mutex_ids=[20, 21])
    _t71_bt = pl.TileType(shape=[16, 32], dtype=pl.DT_FP16,
                          target_memory=pl.MemorySpace.Right, layout=pl.ZN,
                          fractal=512, compact=0)
    t71_bt0_g = pl.make_tile_group(type=_t71_bt, addrs=[0xC000, 0xE000], mutex_ids=[20, 21])
    t71_bt1_g = pl.make_tile_group(type=_t71_bt, addrs=[0xC800, 0xE800], mutex_ids=[20, 21])
    t71_bl0_g = pl.make_tile_group(type=_t71_bt, addrs=[0xD000, 0xF000], mutex_ids=[20, 21])
    t71_bl1_g = pl.make_tile_group(type=_t71_bt, addrs=[0xD800, 0xF800], mutex_ids=[20, 21])

    _t71_c = pl.TileType(shape=[64, 64], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Acc, layout=pl.NZ,
                         fractal=1024, compact=0)
    t71_c_g = pl.make_tile_group(type=_t71_c, addrs=[0x9000, 0x11000], mutex_ids=[22, 23])
    _t71_c16 = pl.TileType(shape=[16, 16], dtype=pl.DT_FP32,
                           target_memory=pl.MemorySpace.Acc, layout=pl.NZ,
                           fractal=1024, compact=0)

    t71_auxd0_g = pl.make_tile_group(type=_t71_c16, addrs=[0xD000, 0x15000], mutex_ids=[24, 25])
    t71_auxd1_g = pl.make_tile_group(type=_t71_c16, addrs=[0xE400, 0x16400], mutex_ids=[26, 27])
    t71_auxd2_g = pl.make_tile_group(type=_t71_c16, addrs=[0xF800, 0x17800], mutex_ids=[28, 29])
    t71_auxd3_g = pl.make_tile_group(type=_t71_c16, addrs=[0x10C00, 0x18C00], mutex_ids=[30, 31])
    t71_aux_g = pl.make_tile_group(type=_t71_c, addrs=[0xD000, 0x15000], mutex_ids=[24, 25])

    _t71_a_bf16 = pl.TileType(shape=[64, 64], dtype=pl.DT_BF16,
                              target_memory=pl.MemorySpace.Left, layout=pl.NZ,
                              fractal=512, compact=0)
    _t71_b_bf16 = pl.TileType(shape=[64, 64], dtype=pl.DT_BF16,
                              target_memory=pl.MemorySpace.Right, layout=pl.ZN,
                              fractal=512, compact=0)
    t71_a_bf16_g = pl.make_tile_group(type=_t71_a_bf16, addrs=[0x5000, 0x7000], mutex_ids=[18, 19])
    t71_b_bf16_g = pl.make_tile_group(type=_t71_b_bf16, addrs=[0xC000, 0xE000], mutex_ids=[20, 21])

    _t71_st_nd_h = pl.TileType(shape=[32, 64], dtype=pl.DT_FP16,
                               target_memory=pl.MemorySpace.Vec,
                               valid_shape=[-1, -1], compact=0)
    _t71_st_nz_h = pl.TileType(shape=[32, 64], dtype=pl.DT_FP16,
                               target_memory=pl.MemorySpace.Vec, layout=pl.NZ,
                               fractal=512, valid_shape=[-1, -1], compact=1)
    _t71_st_nd_b = pl.TileType(shape=[32, 64], dtype=pl.DT_BF16,
                               target_memory=pl.MemorySpace.Vec,
                               valid_shape=[-1, -1], compact=0)
    _t71_st_nz_b = pl.TileType(shape=[32, 64], dtype=pl.DT_BF16,
                               target_memory=pl.MemorySpace.Vec, layout=pl.NZ,
                               fractal=512, valid_shape=[-1, -1], compact=1)
    t71_nd_h_g = pl.make_tile_group(type=_t71_st_nd_h, addrs=[49152], mutex_ids=[16])
    t71_nz_h_g = pl.make_tile_group(type=_t71_st_nz_h, addrs=[53248], mutex_ids=[16])
    t71_nd_b_g = pl.make_tile_group(type=_t71_st_nd_b, addrs=[49152], mutex_ids=[16])
    t71_nz_b_g = pl.make_tile_group(type=_t71_st_nz_b, addrs=[53248], mutex_ids=[16])
    tt_m6_l64 = pl.TileType(shape=[CHUNK_C, CHUNK_C], dtype=pl.DT_BF16,
                            target_memory=pl.MemorySpace.Mat, layout=pl.NZ,
                            valid_shape=[-1, -1])
    tt_m6_l128 = pl.TileType(shape=[CHUNK_C, HEAD_D], dtype=pl.DT_BF16,
                             target_memory=pl.MemorySpace.Mat)
    tt_m6_la = pl.TileType(shape=[CHUNK_C, CHUNK_C], dtype=pl.DT_BF16,
                           target_memory=pl.MemorySpace.Left, layout=pl.NZ)
    tt_m6_lb128 = pl.TileType(shape=[CHUNK_C, HEAD_D], dtype=pl.DT_BF16,
                              target_memory=pl.MemorySpace.Right, layout=pl.ZN)
    m6_la_g = pl.make_tile_group(type=tt_m6_la, addrs=[LA0], mutex_ids=[18])
    m6_inv_l1_g = pl.make_tile_group(type=tt_m6_l64, addrs=[P3_L1_X], mutex_ids=[31])
    m6_op_l1_g = pl.make_tile_group(type=tt_m6_l128, addrs=[L6_OP], mutex_ids=[19])

    tt_m6_hd64 = pl.TileType(shape=[HEAD_D, CHUNK_C], dtype=pl.DT_BF16,
                             target_memory=pl.MemorySpace.Mat)
    tt_m6_la_hd64 = pl.TileType(shape=[HEAD_D, CHUNK_C], dtype=pl.DT_BF16,
                                target_memory=pl.MemorySpace.Left, layout=pl.NZ)
    tt_m6_lb64 = pl.TileType(shape=[CHUNK_C, CHUNK_C], dtype=pl.DT_BF16,
                             target_memory=pl.MemorySpace.Right, layout=pl.ZN)
    tt_m6_lc_ut = pl.TileType(shape=[HEAD_D, CHUNK_C], dtype=pl.DT_FP32,
                              target_memory=pl.MemorySpace.Acc, layout=pl.NZ,
                              fractal=1024)
    m6_vbt_l1_g = pl.make_tile_group(type=tt_m6_hd64, addrs=[L6_OP2],
                                     mutex_ids=[24])
    m6_la_vbt_g = pl.make_tile_group(type=tt_m6_la_hd64, addrs=[L6_LA_VBT],
                                     mutex_ids=[25])
    tt_m6_l64_zn = pl.TileType(shape=[CHUNK_C, CHUNK_C], dtype=pl.DT_BF16,
                               target_memory=pl.MemorySpace.Mat, layout=pl.ZN,
                               valid_shape=[-1, -1])
    m6_invt_zn_g = pl.make_tile_group(type=tt_m6_l64_zn, addrs=[P3_L1_X],
                                      mutex_ids=[31])
    m6_lb_invt_g = pl.make_tile_group(type=tt_m6_lb64, addrs=[L6_LB_OP2],
                                      mutex_ids=[25])

    m6_lb128_g = pl.make_tile_group(type=tt_m6_lb128, addrs=[L6_LB_OP], mutex_ids=[20])
    m6_w_lc_g = pl.make_tile_group(type=tt_m6_lc128, addrs=[L6_LC_W], mutex_ids=[21])
    m6_u_lc_g = pl.make_tile_group(type=tt_m6_lc_ut, addrs=[L6_LC_U], mutex_ids=[22])

    str1 = HEAD_D * (1 - layout_code) + S * HEAD_D * layout_code
    str_qk = (Nk * HEAD_D) * (1 - layout_code) + HEAD_D * layout_code
    str_vv = (Nv * HEAD_D) * (1 - layout_code) + HEAD_D * layout_code
    q_v = pl.make_tensor(q, [B, Nk, S, HEAD_D], [Nk * S * HEAD_D, str1, str_qk, 1])
    k_v = pl.make_tensor(k, [B, Nk, S, HEAD_D], [Nk * S * HEAD_D, str1, str_qk, 1])
    g_v = pl.make_tensor(g, [B, Nv, S, HEAD_D], [Nv * S * HEAD_D, str1, str_vv, 1])

    ws_nk_bytes = B * Nk * S * HEAD_D * 2
    qn_o = pl.make_tensor(workspace, [B, Nk, S, HEAD_D],
                          [Nk * S * HEAD_D, str1, str_qk, 1], dtype=pl.DT_BF16)
    kn_o = pl.make_tensor(workspace + ws_nk_bytes, [B, Nk, S, HEAD_D],
                          [Nk * S * HEAD_D, str1, str_qk, 1], dtype=pl.DT_BF16)
    qd_o = pl.make_tensor(qd_out, [B, Nv, S, HEAD_D], [Nv * S * HEAD_D, str1, str_vv, 1])
    kr_o = pl.make_tensor(kr_out, [B, Nv, S, HEAD_D], [Nv * S * HEAD_D, str1, str_vv, 1])
    v_v = pl.make_tensor(v, [B, Nv, S, HEAD_D], [Nv * S * HEAD_D, str1, str_vv, 1])
    w_o = pl.make_tensor(w_out, [B, Nv, S, HEAD_D], [Nv * S * HEAD_D, str1, str_vv, 1])
    upre_o = pl.make_tensor(upre_out, [B, Nv, HEAD_D, S],
                            [Nv * HEAD_D * S, HEAD_D * S, S, 1])
    gam_o = pl.make_tensor(gam_out, [B, Nv, nc, HEAD_D],
                           [Nv * nc * HEAD_D, nc * HEAD_D, HEAD_D, 1])
    mqk_o = pl.make_tensor(mqk_out, [B, Nv, nc, CHUNK_C, CHUNK_C],
                           [Nv * nc * CHUNK_C * CHUNK_C, nc * CHUNK_C * CHUNK_C,
                            CHUNK_C * CHUNK_C, CHUNK_C, 1])
    v_beta_bf16_o = pl.make_tensor(workspace + ws_nk_bytes * 2, [B, Nv, S, HEAD_D],
                                   [Nv * S * HEAD_D, str1, str_vv, 1], dtype=pl.DT_BF16)
    v_beta_lo_o = pl.make_tensor(workspace + ws_nk_bytes * 2 + B * Nv * S * HEAD_D * 2,
                                [B, Nv, S, HEAD_D],
                                [Nv * S * HEAD_D, str1, str_vv, 1], dtype=pl.DT_BF16)
    a_view = pl.make_tensor(a_log, [Nv, 1], [1, 1])

    with pl.section_vector():
        sub_l = pl.get_subblock_idx()
        r0_l = sub_l * HS
        ltril_rows_vf(x_ub_g.current(), r0_l)
        nd2nz_sg(x_ub_g.current(), xy_ub_g.current(), HS)
        pl.set_validshape(snz_ub_g.current(), [HS, CHUNK_C])
        pl.insert(l_pre, snz_ub_g.current(), [r0_l, 0])
        w_first = core_id
        if w_first < nc * total:
            c_f = w_first % nc
            bh_f = w_first // nc
            b_f = bh_f // Nv
            hv_f = bh_f % Nv
            hk_f = hv_f // nq
            m1_prefetch_v(q_v, k_v, g_v, b_f, hk_f, hv_f,
                          c_f * CHUNK_C, x_b_g, xk_g, xg_g)

    pw_valid = 0
    pc = 0
    pb = 0
    phv = 0
    ps0 = 0
    p_gsi = 0
    for work in pl.range(core_id, nc * total, num_cores):
        c = work % nc
        bh = work // nc
        b = bh // Nv
        hv = bh % Nv
        hk = hv // nq
        s0 = c * CHUNK_C
        w_seq = (work - core_id) // num_cores
        gate_slot_idx = w_seq % 2
        if pw_valid:
            with pl.section_cube():
                m4_kkt_mqk_blocks(m5_kkt_g, m5_mqk_g,
                                  p_gsi,
                                  lk0_g, lk1_g, lk2_g, lk3_g,
                                  lq0_g, lq1_g, lq2_g, lq3_g,
                                  rZ0_g, rZ1_g, rZ2_g, rZ3_g,
                                  m4_la_g, m4_b_g, m4_c_g)
        with pl.section_vector():
            m1_gate_norm_v(qn_o, kn_o, a_view, lower_bound,
                           dt_bias, gate_l1, gate_slot_idx, b, hv, hk, s0,
                           1 if w_seq >= 2 else 0,
                           x_b_g, xk_g, xg_g, qn_h_g, kn_h_g, gate_f_g,
                           gate_nz_g, a_g, d_g)
        if pw_valid:
            with pl.section_vector():
                m5_finish_v(beta, v_v, kn_h_g[p_gsi],
                            mqk_o, v_beta_bf16_o, v_beta_lo_o,
                            layout_code, cs_g,
                            1 if w_seq == 1 else 0,
                            pc, pb, phv, ps0, Nv,
                            y_l1_g[0], xy_ub_g, snz_ub_g, m5_kd_nz_g,
                            m6_op_l1_g[0],
                            m5_beta_raw_b_g, m5_beta_blk_f_g, m5_v_b_g,
                            m5_kkt_g, m5_mqk_g, m5_mqk_b_g,
                            m5_v_operand_g, m5_kd_operand_g,
                            t71_n_db_g[p_gsi], t71_eye, t71_zero, t71_eye_bf16,
                            t71_nd_h_g[0], t71_nz_h_g[0], t71_nd_b_g[0], t71_nz_b_g[0],
                            x_ub_g)
        with pl.section_cube():
            m2_cumsum_c(gate_l1, cs_g, gate_slot_idx,
                        pw_valid, l_pre, L_l0a_g, g_l0b_g, acc_g)
        with pl.section_vector():
            m3_qd_kr_gam_v(qn_o, kn_o, qd_o, kr_o, gam_o, cs_g, scale,
                           c, b, hv, hk, s0, gate_slot_idx, Nv, nc,
                           qn_h_g, kn_h_g, m3_out_b_g, m3_csl_g,
                           m4_kn_b_g, m4_qn_b_g,
                           m4_opscr_b16_g, m4_ref_g,
                           lk0_g, lk1_g, lk2_g, lk3_g,
                           lq0_g, lq1_g, lq2_g, lq3_g,
                           rW0_g, rW1_g, rW2_g, rW3_g,
                           opnz_g)

        if pw_valid:
            m6_neumann_cube(
                t71_a_g, t71_b_g, t71_ad0_g, t71_ad1_g,
                t71_ad2_g, t71_ad3_g, t71_a10_g, t71_a32_g,
                t71_bd0_g, t71_bd1_g, t71_bd2_g, t71_bd3_g,
                t71_b10_g, t71_b32_g, t71_bt0_g, t71_bt1_g,
                t71_bl0_g, t71_bl1_g, t71_zero, t71_eye,
                t71_aux_g, t71_auxd0_g, t71_auxd1_g, t71_auxd2_g,
                t71_auxd3_g, t71_a_bf16_g, t71_b_bf16_g, t71_eye_bf16,
                v_beta_bf16_o, v_beta_lo_o, w_o, upre_o, t71_c_g,
                t71_n_db_g, p_gsi, pb, phv,
                ps0, t71_scratch_g, t71_tmpd0_g, t71_tmpd1_g,
                t71_tmpd2_g, t71_tmpd3_g, t71_p_g, t71_tmp_g,
                t71_inv_bf16_g, x_ub_g, m6_inv_l1_g, m6_lb128_g,
                m6_op_l1_g, m6_la_g, m6_w_lc_g, m6_vbt_l1_g,
                m6_la_vbt_g, m6_lb_invt_g, m6_invt_zn_g, m6_u_lc_g)

        nw = pl.min(work + num_cores, nc * total - 1)
        c_n = nw % nc
        bh_n = nw // nc
        with pl.section_vector():
            m1_prefetch_v(q_v, k_v, g_v, bh_n // Nv, (bh_n % Nv) // nq,
                          bh_n % Nv, c_n * CHUNK_C, x_b_g, xk_g, xg_g)
        pw_valid = 1
        pc = c
        pb = b
        phv = hv
        ps0 = s0
        p_gsi = gate_slot_idx

    n_works = (nc * total - core_id + num_cores - 1) // num_cores
    if pw_valid:
        with pl.section_cube():
            m4_kkt_mqk_blocks(m5_kkt_g, m5_mqk_g,
                              p_gsi,
                              lk0_g, lk1_g, lk2_g, lk3_g,
                              lq0_g, lq1_g, lq2_g, lq3_g,
                              rZ0_g, rZ1_g, rZ2_g, rZ3_g,
                              m4_la_g, m4_b_g, m4_c_g)
        with pl.section_vector():
            m5_finish_v(beta, v_v, kn_h_g[p_gsi],
                        mqk_o, v_beta_bf16_o, v_beta_lo_o,
                        layout_code, cs_g,
                        1 if n_works == 1 else 0,
                        pc, pb, phv, ps0, Nv,
                        y_l1_g[0], xy_ub_g, snz_ub_g, m5_kd_nz_g,
                        m6_op_l1_g[0],
                        m5_beta_raw_b_g, m5_beta_blk_f_g, m5_v_b_g,
                        m5_kkt_g, m5_mqk_g, m5_mqk_b_g,
                        m5_v_operand_g, m5_kd_operand_g,
                        t71_n_db_g[p_gsi], t71_eye, t71_zero, t71_eye_bf16,
                        t71_nd_h_g[0], t71_nz_h_g[0], t71_nd_b_g[0], t71_nz_b_g[0],
                        x_ub_g)
    if n_works >= 1:
        with pl.section_vector():
            pass
            for _ in pl.range(0, pl.min(n_works, 2)):
                pl.system.wait_cross_core(pipe=pl.PipeType.V, event_id=3)
    if pw_valid:
        m6_neumann_cube(
            t71_a_g, t71_b_g, t71_ad0_g, t71_ad1_g,
            t71_ad2_g, t71_ad3_g, t71_a10_g, t71_a32_g,
            t71_bd0_g, t71_bd1_g, t71_bd2_g, t71_bd3_g,
            t71_b10_g, t71_b32_g, t71_bt0_g, t71_bt1_g,
            t71_bl0_g, t71_bl1_g, t71_zero, t71_eye,
            t71_aux_g, t71_auxd0_g, t71_auxd1_g, t71_auxd2_g,
            t71_auxd3_g, t71_a_bf16_g, t71_b_bf16_g, t71_eye_bf16,
            v_beta_bf16_o, v_beta_lo_o, w_o, upre_o, t71_c_g,
            t71_n_db_g, p_gsi, pb, phv,
            ps0, t71_scratch_g, t71_tmpd0_g, t71_tmpd1_g,
            t71_tmpd2_g, t71_tmpd3_g, t71_p_g, t71_tmp_g,
            t71_inv_bf16_g, x_ub_g, m6_inv_l1_g, m6_lb128_g,
            m6_op_l1_g, m6_la_g, m6_w_lc_g, m6_vbt_l1_g,
            m6_la_vbt_g, m6_lb_invt_g, m6_invt_zn_g, m6_u_lc_g)

@pl.jit(auto_mutex=True, tiling_key=_Stage2TilingKey)
def flash_kda_stage2_kernel(
    # 输入
    layout_code: pl.DT_INT64,
    qd_out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16],
    kr_out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16],
    gam_out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    mqk_out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16],
    w_out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16],
    upre_out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    initial_state: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    # 输出
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16],
    final_state: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
):
    DV = Stage2VRows  # noqa: F821 - compile-time TilingKey field
    DV_SPLIT = HEAD_D // DV
    core_id = pl.get_block_idx() // pl.get_subblock_num()
    num_cores = pl.get_block_num()

    B = qd_out.shape[0]
    S = qd_out.shape[1] * (1 - layout_code) + qd_out.shape[2] * layout_code
    Nv = qd_out.shape[2] * (1 - layout_code) + qd_out.shape[1] * layout_code
    nc = S // CHUNK_C
    total = B * Nv

    tt_m7_lw = pl.TileType(shape=[CHUNK_C, HEAD_D], dtype=pl.DT_BF16,
                           target_memory=pl.MemorySpace.Left, layout=pl.NZ)
    tt_m7_lw2 = pl.TileType(shape=[CHUNK_C, CHUNK_C], dtype=pl.DT_BF16,
                            target_memory=pl.MemorySpace.Left, layout=pl.NZ)
    tt_m7_rb64 = pl.TileType(shape=[CHUNK_C, HEAD_D], dtype=pl.DT_BF16,
                             target_memory=pl.MemorySpace.Right, layout=pl.ZN)
    tt_m7_la_snapT = pl.TileType(shape=[DV, HEAD_D], dtype=pl.DT_BF16,
                                 target_memory=pl.MemorySpace.Left, layout=pl.NZ)
    tt_m7_rb_wt = pl.TileType(shape=[HEAD_D, CHUNK_C], dtype=pl.DT_BF16,
                              target_memory=pl.MemorySpace.Right, layout=pl.ZN)
    tt_m7_la_ut = pl.TileType(shape=[DV, CHUNK_C], dtype=pl.DT_BF16,
                              target_memory=pl.MemorySpace.Left, layout=pl.NZ)
    tt_m7_lc_udT = pl.TileType(shape=[DV, CHUNK_C], dtype=pl.DT_FP32,
                               target_memory=pl.MemorySpace.Acc, layout=pl.NZ,
                               fractal=1024)
    tt_m7_lc_delT = pl.TileType(shape=[DV, HEAD_D], dtype=pl.DT_FP32,
                                target_memory=pl.MemorySpace.Acc, layout=pl.NZ,
                                fractal=1024)
    tt_m7_st_f = pl.TileType(shape=[DV, CHUNK_C], dtype=pl.DT_FP32,
                             target_memory=pl.MemorySpace.Vec)
    tt_m7_st_b = pl.TileType(shape=[DV, CHUNK_C], dtype=pl.DT_BF16,
                             target_memory=pl.MemorySpace.Vec)
    tt_m7_ut_b = pl.TileType(shape=[DV // 2, CHUNK_C], dtype=pl.DT_BF16,
                             target_memory=pl.MemorySpace.Vec)
    if DV == 16:
        m7_sT_g = pl.make_tile_group(
            type=tt_m7_st_f, addrs=[VA_M7_S, VA_M7_S + 0x1000], mutex_ids=[0, 8])
        m7_delT_g = pl.make_tile_group(
            type=tt_m7_st_f, addrs=[VA_M7_DEL, VA_M7_DEL + 0x2000], mutex_ids=[1, 9])
        tt_m7_udT_half = pl.TileType(shape=[DV // 2, CHUNK_C], dtype=pl.DT_FP32,
                                     target_memory=pl.MemorySpace.Vec)
        m7_udT_g = pl.make_tile_group(
            type=tt_m7_udT_half, addrs=[VA_M7_UD, VA_M7_UD + 0x1000], mutex_ids=[2, 10])

        tt_m7_ut_b_c = pl.TileType(shape=[DV // 2, CHUNK_C], dtype=pl.DT_BF16,
                                   target_memory=pl.MemorySpace.Vec,
                                   valid_shape=[-1, -1], compact=1)
        tt_m7_ut_nz = pl.TileType(shape=[DV // 2, CHUNK_C], dtype=pl.DT_BF16,
                                  target_memory=pl.MemorySpace.Vec, layout=pl.NZ,
                                  valid_shape=[-1, -1], compact=1)
        m7_ut_b_g = pl.make_tile_group(type=tt_m7_ut_b_c,
                                       addrs=[VA_M7_UB, VA_M7_UB + 0x400],
                                       mutex_ids=[4, 11])
        m7_ut_nz_g = pl.make_tile_group(type=tt_m7_ut_nz, addrs=[VA_M7_UTNZ],
                                        mutex_ids=[3])
        tt_m7_snapb_c = pl.TileType(shape=[DV, CHUNK_C], dtype=pl.DT_BF16,
                                    target_memory=pl.MemorySpace.Vec,
                                    valid_shape=[-1, -1], compact=1)
        tt_m7_snapnz = pl.TileType(shape=[DV, CHUNK_C], dtype=pl.DT_BF16,
                                   target_memory=pl.MemorySpace.Vec, layout=pl.NZ,
                                   valid_shape=[-1, -1], compact=1)
        m7_snapb_c_g = pl.make_tile_group(type=tt_m7_snapb_c, addrs=[VA_M7_UB + 0x1000],
                                          mutex_ids=[6])
        m7_snapnz_g = pl.make_tile_group(type=tt_m7_snapnz, addrs=[VA_M7_UB + 0x2000],
                                         mutex_ids=[7])
        m7_snapT_b_g = pl.make_tile_group(type=tt_m7_st_b, addrs=[VA_M7_SNAP],
                                          mutex_ids=[5])

        # Each Upre bank holds (DV // 2) * 64 FP32 values.
        upre_f4_g = pl.make_tile_group(
            type=tt_m7_udT_half, addrs=[P4A_UB_UPRE0, P4A_UB_UPRE1], mutex_ids=[28, 29])
        gam_f4_g = pl.make_tile_group(
            type=pl.TileType(shape=[1, HEAD_D], dtype=pl.DT_FP32,
                             target_memory=pl.MemorySpace.Vec),
            addrs=[P4A_UB_GAM0, P4A_UB_GAM1], mutex_ids=[30, 31])
        wt_l1_g = pl.make_tile_group(
            type=pl.TileType(shape=[HEAD_D, CHUNK_C], dtype=pl.DT_BF16,
                             target_memory=pl.MemorySpace.Mat),
            addrs=[P4A_L1_W0, P4A_L1_W1], mutex_ids=[24, 25])
        kr_l1_g = pl.make_tile_group(
            type=pl.TileType(shape=[CHUNK_C, HEAD_D], dtype=pl.DT_BF16,
                             target_memory=pl.MemorySpace.Mat),
            addrs=[P4A_L1_KR0, P4A_L1_KR1], mutex_ids=[26, 27])
        qd_l1_g = pl.make_tile_group(
            type=pl.TileType(shape=[CHUNK_C, HEAD_D], dtype=pl.DT_BF16,
                             target_memory=pl.MemorySpace.Mat),
            addrs=[P4A_L1_QD0, P4A_L1_QD1], mutex_ids=[22, 23])
        mqk_l1_g = pl.make_tile_group(
            type=pl.TileType(shape=[CHUNK_C, CHUNK_C], dtype=pl.DT_BF16,
                             target_memory=pl.MemorySpace.Mat),
            addrs=[P4A_L1_MQK0, P4A_L1_MQK1], mutex_ids=[20, 21])
        tt_m7_snapin = pl.TileType(shape=[DV, HEAD_D], dtype=pl.DT_BF16,
                                   target_memory=pl.MemorySpace.Mat,
                                   layout=pl.NZ,
                                   valid_shape=[-1, -1], compact=1)
        snapin_l1_g = pl.make_tile_group(
            type=tt_m7_snapin, addrs=[P4A_L1_SNAPIN0, P4A_L1_SNAPIN1],
            mutex_ids=[28, 19])
        snapin_zn_l1_g = pl.make_tile_group(
            type=pl.TileType(shape=[HEAD_D, DV], dtype=pl.DT_BF16,
                             target_memory=pl.MemorySpace.Mat,
                             layout=pl.ZN,
                             valid_shape=[-1, -1], compact=1),
            addrs=[P4A_L1_SNAPIN0, P4A_L1_SNAPIN1],
            mutex_ids=[28, 19])
        tt_m7_unz = pl.TileType(shape=[DV, CHUNK_C], dtype=pl.DT_BF16,
                               target_memory=pl.MemorySpace.Mat,
                               layout=pl.NZ,
                               valid_shape=[-1, -1], compact=1)
        u_nz_l1_g = pl.make_tile_group(
            type=tt_m7_unz, addrs=[P4A_L1_UNZ0, P4A_L1_UNZ1],
            mutex_ids=[22, 23])
        u_zn_l1_g = pl.make_tile_group(
            type=pl.TileType(shape=[CHUNK_C, DV], dtype=pl.DT_BF16,
                             target_memory=pl.MemorySpace.Mat,
                             layout=pl.ZN,
                             valid_shape=[-1, -1], compact=1),
            addrs=[P4A_L1_UNZ0, P4A_L1_UNZ1],
            mutex_ids=[22, 23])
        m7_la_snapT_g = pl.make_tile_group(type=tt_m7_la_snapT, addrs=[LA0],
                                           mutex_ids=[13])
        m7_la_ut_g = pl.make_tile_group(type=tt_m7_la_ut, addrs=[0x2000],
                                        mutex_ids=[13])
        m7_la_qd_g = pl.make_tile_group(type=tt_m7_lw, addrs=[0x4000],
                                        mutex_ids=[13])
        m7_la_mqk_g = pl.make_tile_group(type=tt_m7_lw2, addrs=[0x8000],
                                         mutex_ids=[13])

        m7_rb_wt_g = pl.make_tile_group(type=tt_m7_rb_wt, addrs=[0x0],
                                        mutex_ids=[14])
        m7_rb_kr_g = pl.make_tile_group(type=tt_m7_rb64, addrs=[0x4000],
                                        mutex_ids=[14])
        m7_rb_snapt_g = pl.make_tile_group(
            type=pl.TileType(shape=[HEAD_D, DV], dtype=pl.DT_BF16,
                             target_memory=pl.MemorySpace.Right, layout=pl.ZN),
            addrs=[0x8000], mutex_ids=[14])
        m7_rb_uot_g = pl.make_tile_group(
            type=pl.TileType(shape=[CHUNK_C, DV], dtype=pl.DT_BF16,
                             target_memory=pl.MemorySpace.Right, layout=pl.ZN),
            addrs=[0x8800], mutex_ids=[14])
        m7_lc_udT_g = pl.make_tile_group(type=tt_m7_lc_udT, addrs=[L7_LC_UD],
                                         mutex_ids=[15])
        m7_lc_delT_g = pl.make_tile_group(type=tt_m7_lc_delT, addrs=[L7_LC_DEL],
                                          mutex_ids=[17])
        m7_lc_o_g = pl.make_tile_group(
            type=pl.TileType(shape=[CHUNK_C, DV], dtype=pl.DT_FP32,
                             target_memory=pl.MemorySpace.Acc),
            addrs=[0x20000], mutex_ids=[18])
    else:
        m7_sT_g = pl.make_tile_group(
            type=tt_m7_st_f, addrs=[0, DV * 256], mutex_ids=[0, 8])
        m7_delT_g = pl.make_tile_group(
            type=tt_m7_st_f, addrs=[DV * 512, DV * 768], mutex_ids=[1, 9])
        tt_m7_udT_half = pl.TileType(shape=[DV // 2, CHUNK_C], dtype=pl.DT_FP32,
                                     target_memory=pl.MemorySpace.Vec)
        m7_udT_g = pl.make_tile_group(
            type=tt_m7_udT_half, addrs=[DV * 1024, DV * 1152], mutex_ids=[2, 10])

        tt_m7_ut_b_c = pl.TileType(shape=[DV // 2, CHUNK_C], dtype=pl.DT_BF16,
                                   target_memory=pl.MemorySpace.Vec,
                                   valid_shape=[-1, -1], compact=1)
        tt_m7_ut_nz = pl.TileType(shape=[DV // 2, CHUNK_C], dtype=pl.DT_BF16,
                                  target_memory=pl.MemorySpace.Vec, layout=pl.NZ,
                                  valid_shape=[-1, -1], compact=1)
        m7_ut_b_g = pl.make_tile_group(type=tt_m7_ut_b_c,
                                       addrs=[DV * 1536],
                                       mutex_ids=[4])
        m7_ut_nz_g = pl.make_tile_group(type=tt_m7_ut_nz, addrs=[DV * 1600],
                                        mutex_ids=[3])
        tt_m7_snapb_c = pl.TileType(shape=[DV, CHUNK_C], dtype=pl.DT_BF16,
                                    target_memory=pl.MemorySpace.Vec,
                                    valid_shape=[-1, -1], compact=1)
        tt_m7_snapnz = pl.TileType(shape=[DV, CHUNK_C], dtype=pl.DT_BF16,
                                   target_memory=pl.MemorySpace.Vec, layout=pl.NZ,
                                   valid_shape=[-1, -1], compact=1)
        m7_snapb_c_g = pl.make_tile_group(type=tt_m7_snapb_c, addrs=[DV * 1664],
                                          mutex_ids=[6])
        m7_snapnz_g = pl.make_tile_group(type=tt_m7_snapnz, addrs=[DV * 1792],
                                         mutex_ids=[7])
        m7_snapT_b_g = m7_snapb_c_g

        # Each Upre bank holds (DV // 2) * 64 FP32 values.
        upre_f4_g = pl.make_tile_group(
            type=tt_m7_udT_half, addrs=[DV * 1280, DV * 1408], mutex_ids=[28, 29])
        gam_f4_g = pl.make_tile_group(
            type=pl.TileType(shape=[1, HEAD_D], dtype=pl.DT_FP32,
                             target_memory=pl.MemorySpace.Vec),
            addrs=[DV * 1920, DV * 1920 + 512], mutex_ids=[30, 31])
        wt_l1_g = pl.make_tile_group(
            type=pl.TileType(shape=[HEAD_D, CHUNK_C], dtype=pl.DT_BF16,
                             target_memory=pl.MemorySpace.Mat),
            addrs=[P4A_L1_W0, P4A_L1_W1], mutex_ids=[24, 25])
        kr_l1_g = pl.make_tile_group(
            type=pl.TileType(shape=[CHUNK_C, HEAD_D], dtype=pl.DT_BF16,
                             target_memory=pl.MemorySpace.Mat),
            addrs=[P4A_L1_KR0, P4A_L1_KR1], mutex_ids=[26, 27])
        qd_l1_g = pl.make_tile_group(
            type=pl.TileType(shape=[CHUNK_C, HEAD_D], dtype=pl.DT_BF16,
                             target_memory=pl.MemorySpace.Mat),
            addrs=[P4A_L1_QD0, P4A_L1_QD1], mutex_ids=[22, 23])
        mqk_l1_g = pl.make_tile_group(
            type=pl.TileType(shape=[CHUNK_C, CHUNK_C], dtype=pl.DT_BF16,
                             target_memory=pl.MemorySpace.Mat),
            addrs=[P4A_L1_MQK0, P4A_L1_MQK1], mutex_ids=[20, 21])
        tt_m7_snapin = pl.TileType(shape=[DV, HEAD_D], dtype=pl.DT_BF16,
                                   target_memory=pl.MemorySpace.Mat,
                                   layout=pl.NZ,
                                   valid_shape=[-1, -1], compact=1)
        snapin_l1_g = pl.make_tile_group(
            type=tt_m7_snapin, addrs=[0x50000, 0x50000 + DV * 256],
            mutex_ids=[28, 19])
        snapin_zn_l1_g = pl.make_tile_group(
            type=pl.TileType(shape=[HEAD_D, DV], dtype=pl.DT_BF16,
                             target_memory=pl.MemorySpace.Mat,
                             layout=pl.ZN,
                             valid_shape=[-1, -1], compact=1),
            addrs=[0x50000, 0x50000 + DV * 256],
            mutex_ids=[28, 19])
        tt_m7_unz = pl.TileType(shape=[DV, CHUNK_C], dtype=pl.DT_BF16,
                               target_memory=pl.MemorySpace.Mat,
                               layout=pl.NZ,
                               valid_shape=[-1, -1], compact=1)
        u_nz_l1_g = pl.make_tile_group(
            type=tt_m7_unz, addrs=[0x50000 + DV * 512, 0x50000 + DV * 640],
            mutex_ids=[22, 23])
        u_zn_l1_g = pl.make_tile_group(
            type=pl.TileType(shape=[CHUNK_C, DV], dtype=pl.DT_BF16,
                             target_memory=pl.MemorySpace.Mat,
                             layout=pl.ZN,
                             valid_shape=[-1, -1], compact=1),
            addrs=[0x50000 + DV * 512, 0x50000 + DV * 640],
            mutex_ids=[22, 23])
        m7_la_snapT_g = pl.make_tile_group(type=tt_m7_la_snapT, addrs=[LA0],
                                           mutex_ids=[13])
        m7_la_ut_g = pl.make_tile_group(type=tt_m7_la_ut, addrs=[0],
                                        mutex_ids=[13])
        m7_la_qd_g = pl.make_tile_group(type=tt_m7_lw, addrs=[0x8000],
                                        mutex_ids=[13])
        m7_la_mqk_g = pl.make_tile_group(type=tt_m7_lw2, addrs=[0xC000],
                                         mutex_ids=[13])

        m7_rb_wt_g = pl.make_tile_group(type=tt_m7_rb_wt, addrs=[0x0],
                                        mutex_ids=[14])
        m7_rb_kr_g = pl.make_tile_group(type=tt_m7_rb64, addrs=[0x4000],
                                        mutex_ids=[14])
        m7_rb_snapt_g = pl.make_tile_group(
            type=pl.TileType(shape=[HEAD_D, DV], dtype=pl.DT_BF16,
                             target_memory=pl.MemorySpace.Right, layout=pl.ZN),
            addrs=[0x8000], mutex_ids=[14])
        m7_rb_uot_g = pl.make_tile_group(
            type=pl.TileType(shape=[CHUNK_C, DV], dtype=pl.DT_BF16,
                             target_memory=pl.MemorySpace.Right, layout=pl.ZN),
            addrs=[0x8000], mutex_ids=[14])
        m7_lc_udT_g = pl.make_tile_group(type=tt_m7_lc_udT, addrs=[L7_LC_UD],
                                         mutex_ids=[15])
        m7_lc_delT_g = pl.make_tile_group(type=tt_m7_lc_delT, addrs=[L7_LC_DEL],
                                          mutex_ids=[17])
        m7_lc_o_g = pl.make_tile_group(
            type=pl.TileType(shape=[CHUNK_C, DV], dtype=pl.DT_FP32,
                             target_memory=pl.MemorySpace.Acc),
            addrs=[0x20000], mutex_ids=[18])
    str1 = HEAD_D * (1 - layout_code) + S * HEAD_D * layout_code
    str_vv = (Nv * HEAD_D) * (1 - layout_code) + HEAD_D * layout_code

    qd_o = pl.make_tensor(qd_out, [B, Nv, S, HEAD_D], [Nv * S * HEAD_D, str1, str_vv, 1])
    kr_o = pl.make_tensor(kr_out, [B, Nv, S, HEAD_D], [Nv * S * HEAD_D, str1, str_vv, 1])
    w_o = pl.make_tensor(w_out, [B, Nv, S, HEAD_D], [Nv * S * HEAD_D, str1, str_vv, 1])
    upre_o = pl.make_tensor(upre_out, [B, Nv, HEAD_D, S],
                            [Nv * HEAD_D * S, HEAD_D * S, S, 1])
    out_o = pl.make_tensor(out, [B, Nv, S, HEAD_D],
                           [Nv * S * HEAD_D, str1, str_vv, 1])
    final_state_o = pl.make_tensor(final_state, [B, Nv, HEAD_D, HEAD_D],
                                   [Nv * HEAD_D * HEAD_D, HEAD_D * HEAD_D, HEAD_D, 1])
    gam_o = pl.make_tensor(gam_out, [B, Nv, nc, HEAD_D],
                           [Nv * nc * HEAD_D, nc * HEAD_D, HEAD_D, 1])
    mqk_o = pl.make_tensor(mqk_out, [B, Nv, nc, CHUNK_C, CHUNK_C],
                           [Nv * nc * CHUNK_C * CHUNK_C, nc * CHUNK_C * CHUNK_C,
                            CHUNK_C * CHUNK_C, CHUNK_C, 1])

    sub = pl.get_subblock_idx()

    total_dv = total * DV_SPLIT
    first_m7t = core_id
    fb8 = first_m7t // (Nv * DV_SPLIT)
    frem = first_m7t % (Nv * DV_SPLIT)
    fhv8 = frem // DV_SPLIT
    fco = (frem % DV_SPLIT) * DV
    with pl.section_vector():
        st_init = m7_sT_g.next()
        pl.load(st_init, initial_state, [fb8, fhv8, fco, sub * (HEAD_D // 2)])
        sbt_init = m7_snapT_b_g.next()
        pl.cast(sbt_init, st_init, mode=pl.RoundMode.CAST_ROUND)
        snz_init = m7_snapnz_g.next()
        pl.set_validshape(snz_init, [DV, CHUNK_C])
        pl.set_validshape(sbt_init, [DV, CHUNK_C])
        pl.move(snz_init, sbt_init)
        pl.insert(snapin_l1_g[0], snz_init, [0, sub * (HEAD_D // 2)])
        pl.system.set_cross_core(pipe=pl.PipeType.MTE3,
                                 event_id=_M7_SNAP_EVENT)

    for m7t in pl.range(core_id, total_dv, num_cores):
        task_ord = (m7t - core_id) // num_cores
        base_slot = (task_ord * (nc % 2)) % 2
        st_task = m7_sT_g.current()
        b8 = m7t // (Nv * DV_SPLIT)
        mrem = m7t % (Nv * DV_SPLIT)
        hv8 = mrem // DV_SPLIT
        co = (mrem % DV_SPLIT) * DV
        with pl.section_cube():
            pl.system.wait_cross_core(pipe=pl.PipeType.MTE1,
                                      event_id=_M7_SNAP_EVENT)
            pl.load(wt_l1_g[base_slot], w_o, [b8, hv8, 0, 0], order=[3, 2])
            pl.load(kr_l1_g[base_slot], kr_o, [b8, hv8, 0, 0])
            pl.load(qd_l1_g[base_slot], qd_o, [b8, hv8, 0, 0])
            pl.load(mqk_l1_g[base_slot], mqk_o, [b8, hv8, 0, 0, 0])
        with pl.section_vector():
            pl.load(upre_f4_g[base_slot], upre_o,
                    [b8, hv8, co + sub * (DV // 2), 0])
            pl.load(gam_f4_g[base_slot], gam_o, [b8, hv8, 0, 0])
        for c8 in pl.range(0, nc):
            s8 = c8 * CHUNK_C
            is_last = (c8 == nc - 1)
            sl = (base_slot + c8) % 2
            ps = (base_slot + c8 + 1) % 2
            with pl.section_cube():
                if c8 + 1 < nc:
                    pl.load(wt_l1_g[ps], w_o, [b8, hv8, s8 + CHUNK_C, 0],
                            order=[3, 2])
                    pl.load(kr_l1_g[ps], kr_o, [b8, hv8, s8 + CHUNK_C, 0])
                    pl.load(qd_l1_g[ps], qd_o, [b8, hv8, s8 + CHUNK_C, 0])
                    pl.load(mqk_l1_g[ps], mqk_o, [b8, hv8, c8 + 1, 0, 0])
                wt_rb = m7_rb_wt_g.next()
                pl.move(wt_rb, wt_l1_g[sl])
                qd_la = m7_la_qd_g.next()
                pl.move(qd_la, qd_l1_g[sl])
                if c8 > 0:
                    pl.system.wait_cross_core(pipe=pl.PipeType.MTE1,
                                              event_id=_P4A_SNAP_BANK_EVENT)
                    snap_la = m7_la_snapT_g.next()
                    pl.move(snap_la, snapin_l1_g[sl])
                else:
                    snap_la = m7_la_snapT_g.next()
                    pl.move(snap_la, snapin_l1_g[base_slot])
                ud_lc = m7_lc_udT_g.next()
                pl.matmul(ud_lc, snap_la, wt_rb)
                ud_t = m7_udT_g.next()
                pl.move(ud_t, ud_lc,
                        acc_to_vec_mode=pl.AccToVecMode.DualModeSplitM)
                pl.system.set_cross_core(pipe=pl.PipeType.FIX,
                                         event_id=_M7_UD_EVENTS[c8 % 2])
                snapt_rb = m7_rb_snapt_g.next()
                if c8 > 0:
                    pl.move(snapt_rb, snapin_zn_l1_g[sl])
                else:
                    pl.move(snapt_rb, snapin_zn_l1_g[base_slot])
                o_lc = m7_lc_o_g.next()
                pl.matmul(o_lc, qd_la, snapt_rb,
                        phase=pl.AccPhase.Partial)
            with pl.section_vector():
                if c8 + 1 < nc:
                    pl.load(upre_f4_g[ps], upre_o,
                            [b8, hv8, co + sub * (DV // 2), s8 + CHUNK_C])
                    pl.load(gam_f4_g[ps], gam_o, [b8, hv8, c8 + 1, 0])
                pl.system.wait_cross_core(pipe=pl.PipeType.V,
                                               event_id=_M7_UD_EVENTS[c8 % 2])
                ud = m7_udT_g.next()
                u_b = m7_ut_b_g.next()
                m7_ut_add_upre_bf16_vf(ud, upre_f4_g[sl], u_b, 0, DV)
                unz = m7_ut_nz_g.next()
                pl.set_validshape(unz, [DV // 2, CHUNK_C])
                pl.set_validshape(u_b, [DV // 2, CHUNK_C])
                pl.move(unz, u_b)
                pl.insert(u_nz_l1_g[sl], unz, [sub * (DV // 2), 0])
                pl.system.set_cross_core(pipe=pl.PipeType.MTE3, event_id=_M7_U_EVENT)
                st = st_task
                m7_scale_t_vf(st, gam_f4_g[sl], sub * (HEAD_D // 2), DV)
            with pl.section_cube():
                kr_rb = m7_rb_kr_g.next()
                pl.move(kr_rb, kr_l1_g[sl])
                mqk_la = m7_la_mqk_g.next()
                pl.move(mqk_la, mqk_l1_g[sl])
                pl.system.wait_cross_core(pipe=pl.PipeType.MTE1, event_id=_M7_U_EVENT)
                u_la = m7_la_ut_g.next()
                pl.move(u_la, u_nz_l1_g[sl])
                del_lc = m7_lc_delT_g.next()
                pl.matmul(del_lc, u_la, kr_rb)
                de_t = m7_delT_g.next()
                pl.move(de_t, del_lc,
                        acc_to_vec_mode=pl.AccToVecMode.DualModeSplitN)
                pl.system.set_cross_core(pipe=pl.PipeType.FIX,
                                         event_id=_M7_D_EVENTS[c8 % 2])
                uot_rb = m7_rb_uot_g.next()
                pl.move(uot_rb, u_zn_l1_g[sl])
                pl.matmul_acc(o_lc, o_lc, mqk_la, uot_rb,
                             phase=pl.AccPhase.Final)
                pl.store(out_o, o_lc, [b8, hv8, s8, co], order=[2, 3],
                         phase=pl.STPhase.Final)
            with pl.section_vector():
                pl.system.wait_cross_core(pipe=pl.PipeType.V,
                                               event_id=_M7_D_EVENTS[c8 % 2])
                if is_last:
                    next_m7t = m7t + num_cores
                    if next_m7t < total_dv:
                        nb8 = next_m7t // (Nv * DV_SPLIT)
                        nrem = next_m7t % (Nv * DV_SPLIT)
                        nhv8 = nrem // DV_SPLIT
                        nco = (nrem % DV_SPLIT) * DV
                        next_ord = (next_m7t - core_id) // num_cores
                        next_slot = (next_ord * (nc % 2)) % 2
                        st_next = m7_sT_g.next()
                        pl.load(st_next, initial_state, [nb8, nhv8, nco, sub * (HEAD_D // 2)])
                        sbt_next = m7_snapT_b_g.next()
                        pl.cast(sbt_next, st_next, mode=pl.RoundMode.CAST_ROUND)
                        snz_next = m7_snapnz_g.next()
                        pl.set_validshape(snz_next, [DV, CHUNK_C])
                        pl.set_validshape(sbt_next, [DV, CHUNK_C])
                        pl.move(snz_next, sbt_next)
                        pl.insert(snapin_l1_g[next_slot], snz_next,
                                  [0, sub * (HEAD_D // 2)])
                        pl.system.set_cross_core(pipe=pl.PipeType.MTE3,
                                                 event_id=_M7_SNAP_EVENT)
                st = st_task
                de = m7_delT_g.next()
                sbc = m7_snapb_c_g.next()
                m7_state_add_snap_vf(st, de, st, sbc, DV)
                if is_last:
                    pl.store(final_state_o, st, [b8, hv8, co, sub * (HEAD_D // 2)])
                if c8 < nc - 1:
                    snz = m7_snapnz_g.next()
                    pl.set_validshape(snz, [DV, CHUNK_C])
                    pl.set_validshape(sbc, [DV, CHUNK_C])
                    pl.move(snz, sbc)
                    pl.insert(snapin_l1_g[ps], snz, [0, sub * (HEAD_D // 2)])
                    pl.system.set_cross_core(pipe=pl.PipeType.MTE3,
                                             event_id=_P4A_SNAP_BANK_EVENT)

_PLATFORM_CFG_CACHE = {}
def _get_platform_config():
    """Cube/vector core counts from the PyPTO-Pro platform info (framework
    interface, same as the deepseek_v4_1 operators), cached per process.

    Replaces the previous hand-rolled platform .ini lookup: that path resolved
    to ``<pypto_pro pkg>/pypto/lib/configs`` which does not exist in this
    install layout, so it always fell through to the device probe anyway.

    Fail-fast by design: no static fallback core counts — a broken or empty
    platform query must surface here instead of silently degrading the launch
    configuration (e.g. 28-core defaults on a 32-core SoC, or a block_dim the
    SoC cannot honor, which would mis-align the Stage2 DV decision).
    """
    cached = _PLATFORM_CFG_CACHE.get("cfg")
    if cached is not None:
        return cached
    try:
        info = get_platform_info()
        cube = int(info.cube_core_num)
        vec = int(info.vector_core_num)
    except Exception as exc:  # fail-fast: no static fallback
        raise AssertionError(f"get_platform_info() failed: {exc}") from exc
    assert cube > 0 and vec > 0, \
        f"invalid platform core counts: cube={cube}, vec={vec}"
    _PLATFORM_CFG_CACHE["cfg"] = (cube, vec)
    return cube, vec


def flash_kda_stage1_wrapper(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g: torch.Tensor,
    beta: torch.Tensor, scale: float, a_log: torch.Tensor,
    dt_bias: torch.Tensor, lower_bound: float,
    layout_qkv: str = "BSND", cu_seqlens=None,
):
    """Run Stage1 (dense BSND/BNSD; TND is dispatched to the kernel-level path)."""
    if layout_qkv == "TND":
        return _flash_kda_stage1_tnd(q, k, v, g, beta, scale, a_log, dt_bias,
                                    lower_bound, cu_seqlens)
    if cu_seqlens is not None:
        raise ValueError("flash_kda does not support cu_seqlens (must be None)")
    if layout_qkv not in ("BSND", "BNSD"):
        raise ValueError(f"layout_qkv must be BSND or BNSD, got {layout_qkv!r}")
    if not isinstance(scale, (int, float)) or not isinstance(lower_bound, (int, float)):
        raise TypeError("scale and lower_bound must be Python scalars")
    if not (-5.0 <= float(lower_bound) <= 0.0):
        raise ValueError("lower_bound must be in [-5, 0]")
    for name, t in (("q", q), ("k", k), ("g", g), ("v", v), ("beta", beta)):
        if t.dtype != torch.bfloat16:
            raise TypeError(f"{name} must be bfloat16, got {t.dtype}")
    for name, t in (("a_log", a_log), ("dt_bias", dt_bias)):
        if t.dtype != torch.float32:
            raise TypeError(f"{name} must be float32, got {t.dtype}")
    if layout_qkv == "BSND":
        B, S, Nk, D = q.shape
        Nv = g.shape[2]
    else:
        B, Nk, S, D = q.shape
        Nv = g.shape[1]
    if D != HEAD_D:
        raise ValueError(f"flash_kda supports only D={HEAD_D}, got {D}")
    if S < CHUNK_C or S % CHUNK_C:
        raise ValueError(f"S must be a multiple of {CHUNK_C} and at least {CHUNK_C}")
    if Nv % Nk:
        raise ValueError(f"GQA requires Nv % Nk == 0, got Nv={Nv}, Nk={Nk}")
    if not (1 <= B <= B_MAX and 1 <= Nk <= NK_MAX and 1 <= Nv <= NV_MAX):
        raise ValueError(f"dims out of SPEC bounds: B={B}, Nk={Nk}, Nv={Nv}")
    if k.shape != q.shape or v.shape != g.shape:
        raise ValueError("q/k and g/v must have matching shapes")
    beta_shape = (B, S, Nv) if layout_qkv == "BSND" else (B, Nv, S)
    if beta.shape != beta_shape:
        raise ValueError(f"beta must have layout-native shape {beta_shape}")
    if a_log.shape != (Nv,) or dt_bias.shape != (Nv, D):
        raise ValueError("a_log must be [Nv] and dt_bias must be [Nv,D]")

    cshp = (B, Nv, S, D) if layout_qkv == "BNSD" else (B, S, Nv, D)
    ws_nk_bytes = B * Nk * S * HEAD_D * 2
    ws_nv_bytes = B * Nv * S * HEAD_D * 2
    workspace = torch.empty(2 * ws_nk_bytes + 2 * ws_nv_bytes,
                            dtype=torch.uint8, device=q.device)
    qd = torch.empty(cshp, dtype=torch.bfloat16, device=q.device)
    kr = torch.empty(cshp, dtype=torch.bfloat16, device=q.device)
    nc = S // CHUNK_C
    gam = torch.empty((B, Nv, nc, D), dtype=torch.float32, device=q.device)
    mqk = torch.empty((B, Nv, nc, CHUNK_C, CHUNK_C), dtype=torch.bfloat16, device=q.device)
    w = torch.empty(cshp, dtype=torch.bfloat16, device=q.device)
    upre = torch.empty(cshp, dtype=torch.float32, device=q.device)
    layout_code = 0 if layout_qkv == "BSND" else 1
    tasks = B * Nv
    cube_cores, _ = _get_platform_config()
    num_cores_stage1 = min(cube_cores, nc * tasks)
    flash_kda_stage1_kernel[None, num_cores_stage1](
        q, k, g, beta, v, a_log, dt_bias, float(lower_bound), float(scale),
        layout_code,
        qd, kr, gam, mqk, w, upre,
        workspace)
    return qd, kr, gam, mqk, w, upre


def flash_kda_stage2_wrapper(
    qd: torch.Tensor, kr: torch.Tensor, gam: torch.Tensor,
    mqk: torch.Tensor, w: torch.Tensor, upre: torch.Tensor,
    initial_state: torch.Tensor, layout_qkv: str = "BSND",
    cu_seqlens=None,
):
    """Run Stage2 (dense BSND/BNSD; TND is dispatched to the kernel-level path)."""
    if layout_qkv == "TND":
        return _flash_kda_stage2_tnd(qd, kr, gam, mqk, w, upre, initial_state,
                                    cu_seqlens)
    if cu_seqlens is not None:
        raise ValueError("flash_kda does not support cu_seqlens (must be None)")
    if layout_qkv not in ("BSND", "BNSD"):
        raise ValueError(f"layout_qkv must be BSND or BNSD, got {layout_qkv!r}")
    for name, t in (("qd", qd), ("kr", kr), ("mqk", mqk), ("w", w)):
        if t.dtype != torch.bfloat16:
            raise TypeError(f"{name} must be bfloat16, got {t.dtype}")
    for name, t in (("gam", gam), ("upre", upre), ("initial_state", initial_state)):
        if t.dtype != torch.float32:
            raise TypeError(f"{name} must be float32, got {t.dtype}")
    if layout_qkv == "BSND":
        B, S, Nv, D = qd.shape
    else:
        B, Nv, S, D = qd.shape
    if D != HEAD_D or S < CHUNK_C or S % CHUNK_C:
        raise ValueError("Stage2 requires D=128 and S a positive multiple of 64")
    if not (1 <= B <= B_MAX and 1 <= Nv <= NV_MAX):
        raise ValueError(f"dims out of SPEC bounds: B={B}, Nv={Nv}")
    if any(t.shape != qd.shape for t in (kr, w, upre)):
        raise ValueError("qd, kr, w and upre must share the layout-native shape")
    nc = S // CHUNK_C
    if gam.shape != (B, Nv, nc, D) or mqk.shape != (B, Nv, nc, CHUNK_C, CHUNK_C):
        raise ValueError("gam/mqk shape does not match Stage2 contract")
    if initial_state.shape != (B, Nv, D, D):
        raise ValueError("initial_state must have shape [B,N,128,128]")
    layout_code = 0 if layout_qkv == "BSND" else 1
    cube_cores, _ = _get_platform_config()
    v_rows = _stage2_v_rows(Nv, B * Nv, cube_cores)
    num_cores_stage2 = min(cube_cores, B * Nv * (HEAD_D // v_rows))
    out = torch.empty_like(qd)
    final_state = torch.empty((B, Nv, HEAD_D, HEAD_D), dtype=torch.float32, device=qd.device)
    flash_kda_stage2_kernel[None, num_cores_stage2, {"Stage2VRows": v_rows}](
        layout_code, qd, kr, gam, mqk, w, upre, initial_state.contiguous(),
        out, final_state)
    return out, final_state


def flash_kda_wrapper(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g: torch.Tensor,
    beta: torch.Tensor, scale: float, initial_state: torch.Tensor,
    a_log: torch.Tensor, dt_bias: torch.Tensor, lower_bound: float,
    layout_qkv: str = "BSND", cu_seqlens=None,
):
    """flash_kda public entry: dense BSND/BNSD + kernel-level TND varlen.

    Dense (``cu_seqlens=None``): BSND/BNSD, S a positive multiple of 64,
    returns ``(out [B,S,Nv,D], final_state [B,Nv,D,D])``.

    Varlen (``cu_seqlens`` given, or ``layout_qkv="TND"``): packed token-major
    inputs — 4-D ``[1,T,H,D]`` (BSND packed) or 3-D ``[T,Nk,D]`` (TND).  Each
    sequence is padded to 64-token chunks **inside the kernels** (on-chip
    zero-filled tails; no GM padding copies), and every sequence's FP32 state
    stays independent: two kernel launches total regardless of the number of
    sequences.  Returns ``(out [1,T,Nv,D] bf16, final_state [N,Nv,D,D] fp32)``.
    """
    if cu_seqlens is not None or layout_qkv == "TND":
        return _flash_kda_varlen_tnd(
            q, k, v, g, beta, scale, initial_state, a_log, dt_bias,
            lower_bound, layout_qkv, cu_seqlens)
    s1 = flash_kda_stage1_wrapper(q, k, v, g, beta, scale, a_log, dt_bias,
                                  lower_bound, layout_qkv)
    return flash_kda_stage2_wrapper(*s1[:6], initial_state, layout_qkv)


def _flash_kda_varlen_tnd(q, k, v, g, beta, scale, initial_state, a_log,
                          dt_bias, lower_bound, layout_qkv, cu_seqlens):
    """TND varlen host entry: contract checks + the two TND kernel calls.

    ``cu_seqlens`` stays device-resident end to end (zero D2H): both kernels
    derive the per-sequence chunk spans on-chip from the passed tensor.
    Boundaries are validated on host only when cu_seqlens lives on CPU (free);
    a device-resident tensor is trusted — violating cu[0]==0, non-decreasing
    or cu[-1]==T reads/writes out of bounds.  Empty sequences are legal.
    """
    if layout_qkv == "TND":
        if q.ndim != 3:
            raise ValueError("TND q/k/v/g must have rank 3")
        if beta.ndim != 2:
            raise ValueError("beta must be [T,Nv] for TND")
        T = q.shape[0]
    else:
        if layout_qkv != "BSND":
            raise ValueError(
                f"flash_kda varlen requires layout_qkv='BSND' (packed) or 'TND', "
                f"got {layout_qkv!r}")
        if q.dim() != 4 or q.shape[0] != 1:
            raise ValueError("flash_kda varlen requires packed batch size 1: q [1,T,Nk,D]")
        if beta.dim() != 3 or beta.shape[0] != 1:
            raise ValueError("beta must be [1,T,Nv] for varlen")
        T = q.shape[1]
        q, k, v, g, beta = (t[0] for t in (q, k, v, g, beta))
    if not isinstance(cu_seqlens, torch.Tensor):
        raise TypeError("TND requires a 1-D int32/int64 cu_seqlens tensor")
    if cu_seqlens.dim() != 1 or cu_seqlens.numel() < 2:
        raise ValueError("cu_seqlens must be 1-D [N+1] with N >= 1")
    if cu_seqlens.dtype not in (torch.int32, torch.int64):
        raise TypeError(f"cu_seqlens must be int32/int64, got {cu_seqlens.dtype}")
    if cu_seqlens.device != q.device and cu_seqlens.device.type != "cpu":
        raise ValueError("cu_seqlens must be on CPU or on the same device as q")
    if cu_seqlens.device.type == "cpu":
        # Host-resident boundaries validate for free (no D2H synchronisation).
        cu_list = cu_seqlens.detach().tolist()
        if cu_list[0] != 0:
            raise ValueError("cu_seqlens[0] must be 0")
        if any(b < a for a, b in zip(cu_list, cu_list[1:])):
            raise ValueError("cu_seqlens must be non-decreasing")
        if cu_list[-1] != T:
            raise ValueError(f"cu_seqlens[-1] must equal T={T}, got {cu_list[-1]}")
    R = cu_seqlens.numel() - 1
    if initial_state is None:
        initial_state = torch.zeros(R, v.shape[1], HEAD_D, HEAD_D,
                                    dtype=torch.float32, device=q.device)
    if initial_state.dim() != 4 or initial_state.shape[0] != R:
        raise ValueError(f"initial_state must be [N={R},Nv,D,D] for varlen")
    if initial_state.dtype != torch.float32:
        raise TypeError("initial_state must be float32")
    initial_state = initial_state.contiguous()
    cu = cu_seqlens.detach().to(device=q.device, dtype=torch.int64)
    s1 = _flash_kda_stage1_tnd(q, k, v, g, beta, scale, a_log, dt_bias,
                               lower_bound, cu)
    out, final_state = _flash_kda_stage2_tnd(*s1[:6], initial_state, cu, T)
    return out.unsqueeze(0), final_state


_TND_OUT_EVENT = 6


@pl.vector_function
def tnd_zero_bf16_vf(tile, elements):
    mask = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_BF16)
    zero = vf.full(0.0, mask, dtype=pl.DT_BF16)
    for off in pl.range(0, elements, 128):
        vf.store_align(tile + off, zero, mask)


@pl.vector_function
def tnd_zero_gate_tail_vf(tile, valid_rows):
    mask = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    zero = vf.full(0.0, mask, dtype=pl.DT_FP32)
    for row in pl.range(valid_rows, HS):
        vf.store_align(tile + row * HEAD_D, zero, mask)
        vf.store_align(tile + row * HEAD_D + LANES, zero, mask)


def tnd_load_half(tile, tensor, head, src, valid):
    """Load this subblock's half-chunk rows; zero-pad beyond the tail."""
    sub = pl.get_subblock_idx()
    rows = pl.max(0, pl.min(HS, valid - sub * HS))
    # LOAD leaves the remaining physical rows untouched; initialise them first.
    pl.set_validshape(tile, [HS, HEAD_D])
    if rows < HS:
        tnd_zero_bf16_vf(tile, HS * HEAD_D)
    if rows > 0:
        pl.set_validshape(tile, [rows, HEAD_D])
        pl.load(tile, tensor, [0, head, src + sub * HS, 0], order=[2, 3])
    pl.set_validshape(tile, [HS, HEAD_D])


def tnd_locate_chunk(cu_seqlens, chunk, num_seqs):
    """Map a global padded-chunk index to (token start, valid tokens).

    On-chip replacement for the host-built chunk_starts/chunk_lengths tables:
    scan the R+1 boundaries and take the sequence whose padded chunk range
    covers ``chunk``.  Chunks past the total count (the launch grid uses the
    worst-case J = T//64 + R and never shrinks to the exact J) fall through as
    (0, 0): they run zero-filled and their outputs land in the padded region
    stage2 never reads.
    """
    src = 0
    valid = 0
    acc = 0
    for s in pl.range(0, num_seqs):
        begin = cu_seqlens[s]
        end = cu_seqlens[s + 1]
        cnt = (end - begin + CHUNK_C - 1) // CHUNK_C
        local = chunk - acc
        if local >= 0:
            if local < cnt:
                src = begin + local * CHUNK_C
                valid = pl.min(CHUNK_C, end - src)
        acc = acc + cnt
    return src, valid


def tnd_seq_span(cu_seqlens, seq, num_seqs):
    """Prefix-scan the boundaries: (first padded chunk, chunk count) of ``seq``."""
    first_chunk = 0
    count = 0
    for s in pl.range(0, num_seqs):
        begin = cu_seqlens[s]
        end = cu_seqlens[s + 1]
        cnt = (end - begin + CHUNK_C - 1) // CHUNK_C
        if s < seq:
            first_chunk = first_chunk + cnt
        if s == seq:
            count = cnt
    return first_chunk, count


def tnd_m1_prefetch_v(q_v, k_v, g_v, b, hk, hv, src, valid, x_b_g, xk_g, xg_g):
    xq = x_b_g.next()
    xk = xk_g.next()
    xg = xg_g.next()
    tnd_load_half(xq, q_v, hk, src, valid)
    tnd_load_half(xk, k_v, hk, src, valid)
    tnd_load_half(xg, g_v, hv, src, valid)


def tnd_m1_gate_norm_v(qn_o, kn_o, a_view, lower_bound, dt_bias,
                   gate_l1, gate_slot_idx, b, hv, hk, s0, have_p2,
                   x_b_g, xk_g, xg_g, qn_h_g, kn_h_g, gate_f_g, gate_nz_g,
                   a_g, d_g, valid):
    """Module-1 vector stage: gate/qn/kn for one chunk of one (b, hv)."""
    sub = pl.get_subblock_idx()
    r0 = sub * HS
    if have_p2:
        pl.system.wait_cross_core(pipe=pl.PipeType.V, event_id=3)
    xq = x_b_g.current()
    xk = xk_g.current()
    xg = xg_g.current()
    a_s = a_g.next()
    pl.set_validshape(a_s, [1, 1])
    pl.load(a_s, a_view, [hv, 0])
    d_s = d_g.next()
    pl.load(d_s, dt_bias, [hv, 0])
    gatel = gate_f_g.next()
    m1_gate_vf(xg, d_s, a_s, gatel, lower_bound)
    tnd_zero_gate_tail_vf(gatel, pl.max(0, pl.min(HS, valid - r0)))
    gate_nz = gate_nz_g.next()
    pl.set_validshape(gate_nz, [HS, HEAD_D])
    pl.set_validshape(gatel, [HS, HEAD_D])
    pl.move(gate_nz, gatel)
    gate_slot = gate_l1[gate_slot_idx]
    pl.insert(gate_slot, gate_nz, [r0, 0])
    pl.system.set_cross_core(pipe=pl.PipeType.MTE3, event_id=0)
    qnb = qn_h_g[gate_slot_idx]
    m1_l2norm_vf(xq, qnb, L2_EPS)
    pl.store(qn_o, qnb, [b, hk, s0 + r0, 0], order=[2, 3])
    knb = kn_h_g[gate_slot_idx]
    m1_l2norm_vf(xk, knb, L2_EPS)
    pl.store(kn_o, knb, [b, hk, s0 + r0, 0], order=[2, 3])


@pl.vector_function
def tnd_m5_tmqk_vf(kkt_f, mqk_f, beta_blk_f, tb_out_f, mqk_out_b,
               off_base, step, r0, valid):
    """Fused pair-VF: t_beta and mqk row finishes in one pass."""
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    pregbf = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_BF16)
    zero = vf.full(0.0, preg, dtype=pl.DT_FP32)
    one = vf.full(1.0, preg, dtype=pl.DT_FP32)
    neg1 = vf.full(-1.0, preg, dtype=pl.DT_FP32)
    pi32 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_INT32)
    lane_i = vf.arange(0, dtype=pl.DT_INT32)
    for p in pl.range(0, HS // 2):
        gr0 = r0 + 2 * p
        gr1 = gr0 + 1
        base0 = 2 * p * CHUNK_C
        base1 = (2 * p + 1) * CHUNK_C
        brow = vf.load_align(beta_blk_f, off_base + gr0 * step,
                             dist=pl.LoadDist.BRC_B32)
        e = vf.exp_sub(zero, brow, preg)
        dn = vf.adds(e, 1.0, preg)
        sg0 = vf.div(one, dn, preg)   # 尾行 T_raw 已为 0，无需置 0（VF 保持无分支）
        rv0 = vf.full(gr0, dtype=pl.DT_INT32)
        keep_full = vf.le(lane_i, rv0, pi32)      # j <= row
        keep_strict = vf.lt(lane_i, rv0, pi32)    # j <  row
        mq0 = vf.load_align(mqk_f, base0)
        mq0 = vf.select(mq0, zero, keep_full)
        krow = vf.load_align(kkt_f, base0)
        t = vf.mul(krow, sg0, preg)
        t = vf.mul(t, neg1, preg)
        t = vf.select(t, zero, keep_strict)
        vf.store_align(tb_out_f + base0, t, preg)
        brow = vf.load_align(beta_blk_f, off_base + gr1 * step,
                             dist=pl.LoadDist.BRC_B32)
        e = vf.exp_sub(zero, brow, preg)
        dn = vf.adds(e, 1.0, preg)
        sg1 = vf.div(one, dn, preg)
        rv1 = vf.full(gr1, dtype=pl.DT_INT32)
        keep_full1 = vf.le(lane_i, rv1, pi32)
        keep_strict1 = vf.lt(lane_i, rv1, pi32)
        mq1 = vf.load_align(mqk_f, base1)
        mq1 = vf.select(mq1, zero, keep_full1)
        krow = vf.load_align(kkt_f, base1)
        t = vf.mul(krow, sg1, preg)
        t = vf.mul(t, neg1, preg)
        t = vf.select(t, zero, keep_strict1)
        vf.store_align(tb_out_f + base1, t, preg)
        b0 = vf.astype(mq0, pregbf, dtype=pl.DT_BF16,
                       round_mode=pl.VFRoundMode.CAST_ROUND)
        b1 = vf.astype(mq1, pregbf, dtype=pl.DT_BF16,
                       round_mode=pl.VFRoundMode.CAST_ROUND)
        drow, d1_ = vf.de_interleave(b0, b1)
        vf.store_align(mqk_out_b + p * 2 * CHUNK_C, drow, pregbf)


@pl.vector_function
def tnd_m5_vkn_vf(v_b, kn_b, cs_f, beta_blk_f, v_out_b, kd_out_b,
              off_base, step, r0, valid):
    """Split beta*V into BF16 high/low parts; also produce BF16 -beta*K*exp(cs).

    Reuse each consumed row of v_b for the low part. The caller stores both
    parts before the M6 handoff; no extra UB tile or mutex is needed.
    """
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    pregbf = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_BF16)
    zero = vf.full(0.0, preg, dtype=pl.DT_FP32)
    one = vf.full(1.0, preg, dtype=pl.DT_FP32)
    neg1 = vf.full(-1.0, preg, dtype=pl.DT_FP32)
    for m in pl.range(0, HS):
        base = m * HEAD_D
        brow = vf.load_align(beta_blk_f, off_base + (r0 + m) * step,
                             dist=pl.LoadDist.BRC_B32)
        e = vf.exp_sub(zero, brow, preg)
        dn = vf.adds(e, 1.0, preg)
        sg = vf.div(one, dn, preg)
        # 尾行无需额外置 0：kn/v 在核内已补零，vβ/kd 尾行恒为 0（保持 VF 无分支）
        bv = vf.load_align(v_b, base, dtype=pl.DT_BF16)
        e0 = vf.astype(bv, pregbf, dtype=pl.DT_FP32)
        e1 = vf.astype(bv, pregbf, dtype=pl.DT_FP32,
                       layout=pl.CastLayout.ONE)
        vr0, vr1 = vf.interleave(e0, e1)
        t0 = vf.mul(vr0, sg, preg)
        t1 = vf.mul(vr1, sg, preg)
        b0 = vf.astype(t0, pregbf, dtype=pl.DT_BF16,
                       round_mode=pl.VFRoundMode.CAST_ROUND)
        b1 = vf.astype(t1, pregbf, dtype=pl.DT_BF16,
                       round_mode=pl.VFRoundMode.CAST_ROUND)
        packed, unused = vf.de_interleave(b0, b1)
        vf.store_align(v_out_b + base, packed, pregbf)
        # Keep the rounding residual while t0/t1 still hold beta*V in FP32.
        hi0 = vf.astype(b0, pregbf, dtype=pl.DT_FP32)
        hi1 = vf.astype(b1, pregbf, dtype=pl.DT_FP32)
        lo0 = vf.sub(t0, hi0, preg)
        lo1 = vf.sub(t1, hi1, preg)
        lb0 = vf.astype(lo0, pregbf, dtype=pl.DT_BF16,
                        round_mode=pl.VFRoundMode.CAST_ROUND)
        lb1 = vf.astype(lo1, pregbf, dtype=pl.DT_BF16,
                        round_mode=pl.VFRoundMode.CAST_ROUND)
        packed_lo, unused_lo = vf.de_interleave(lb0, lb1)
        vf.store_align(v_b + base, packed_lo, pregbf)
        bk = vf.load_align(kn_b, base, dtype=pl.DT_BF16)
        e0 = vf.astype(bk, pregbf, dtype=pl.DT_FP32)
        e1 = vf.astype(bk, pregbf, dtype=pl.DT_FP32,
                       layout=pl.CastLayout.ONE)
        k0, k1 = vf.interleave(e0, e1)
        c0 = vf.load_align(cs_f, (r0 + m) * HEAD_D)
        c1 = vf.load_align(cs_f, (r0 + m) * HEAD_D + LANES)
        e0 = vf.exp(c0, preg)
        e1 = vf.exp(c1, preg)
        t0 = vf.mul(k0, e0, preg)
        t1 = vf.mul(k1, e1, preg)
        t0 = vf.mul(t0, sg, preg)
        t1 = vf.mul(t1, sg, preg)
        t0 = vf.mul(t0, neg1, preg)
        t1 = vf.mul(t1, neg1, preg)
        b0 = vf.astype(t0, pregbf, dtype=pl.DT_BF16,
                       round_mode=pl.VFRoundMode.CAST_ROUND)
        b1 = vf.astype(t1, pregbf, dtype=pl.DT_BF16,
                       round_mode=pl.VFRoundMode.CAST_ROUND)
        packed, unused = vf.de_interleave(b0, b1)
        vf.store_align(kd_out_b + base, packed, pregbf)


def tnd_m5_finish_v(beta, v_v, kn_slot,
                mqk_o, v_beta_bf16_o, v_beta_lo_o,
                layout_code, cs_g, reload_mask,
                c, b, hv, s0, Nv, src, valid,
                y0_l1, snz_stg_g, snz_nz_g, kd_nz_g,
                kd_l1,
                m5_beta_raw_b_g, m5_beta_blk_f_g, m5_v_b_g,
                m5_kkt_g, m5_mqk_g, m5_mqk_b_g, m5_v_operand_g, m5_kd_operand_g,
                t71_n, t71_eye, t71_zero, t71_eye_bf16,
                t71_nd_h, t71_nz_h, t71_nd_b, t71_nz_b, x_ub_g):
    """M5 finish, issued one work behind the merged loop (lagged)."""
    h0 = (hv // 64) * 64
    hv_local = hv - h0
    off_base = hv_local * (1 - layout_code) + hv_local * 64 * layout_code
    step = 64 * (1 - layout_code) + 1 * layout_code
    brow_b = src * (1 - layout_code) + h0 * layout_code
    bcol_b = h0 * (1 - layout_code) + src * layout_code
    n_rows = pl.max(1, valid) * (1 - layout_code) + pl.min(64, Nv - h0) * layout_code
    n_cols = pl.min(64, Nv - h0) * (1 - layout_code) + 64 * layout_code
    sub_m5 = pl.get_subblock_idx()
    r0 = sub_m5 * HS
    bb_raw = m5_beta_raw_b_g.next()
    pl.set_validshape(bb_raw, [64, 64])
    if valid < CHUNK_C:
        tnd_zero_bf16_vf(bb_raw, 64 * 64)
    pl.set_validshape(bb_raw, [n_rows, n_cols])
    pl.load(bb_raw, beta, [b, brow_b, bcol_b], order=[1, 2])
    blk_f = m5_beta_blk_f_g.next()
    pl.set_validshape(bb_raw, [64, n_cols])
    pl.set_validshape(blk_f, [64, n_cols])
    pl.cast(blk_f, bb_raw)
    vb = m5_v_b_g.next()
    tnd_load_half(vb, v_v, hv, src, valid)

    pl.system.wait_cross_core(pipe=pl.PipeType.V, event_id=11)
    cs_slot = cs_g[0]
    kb = kn_slot
    vf_t = m5_v_operand_g.next()
    kf = m5_kd_operand_g.next()
    tnd_m5_vkn_vf(vb, kb, cs_slot, blk_f, vf_t, kf, off_base, step, r0, valid)
    pl.system.set_cross_core(pipe=pl.PipeType.V, event_id=_M5_CS_DONE)
    pl.store(v_beta_bf16_o, vf_t, [b, hv, s0 + r0, 0], order=[2, 3])
    pl.store(v_beta_lo_o, vb, [b, hv, s0 + r0, 0], order=[2, 3])

    pl.system.wait_cross_core(pipe=pl.PipeType.V, event_id=9)
    kt = m5_kkt_g[0]
    mq = m5_mqk_g[0]
    mqb = m5_mqk_b_g.next()
    tnd_m5_tmqk_vf(kt, mq, blk_f, mq, mqb, off_base, step, r0, valid)
    pl.set_validshape(mq, [HS, CHUNK_C])
    nd2nz_sg(mq, snz_stg_g.current(), HS)
    pl.set_validshape(snz_nz_g.current(), [HS, CHUNK_C])
    pl.insert(y0_l1, snz_nz_g.current(), [r0, 0])
    pl.store(mqk_o, mqb, [b, hv, c, r0, 0])

    t71_prepare_neumann_v(m5_mqk_g, t71_n, t71_eye, t71_zero,
                          t71_eye_bf16, t71_nd_h, t71_nz_h,
                          t71_nd_b, t71_nz_b, x_ub_g, reload_mask)
    kd_nz = kd_nz_g.next()
    pl.set_validshape(kd_nz, [HS, HEAD_D])
    pl.move(kd_nz, kf)
    pl.insert(kd_l1, kd_nz, [r0, 0])

    # _M5M6_EVENT had no consumer in the current M6 chain; do not emit a
    # per-work cross-core token that can accumulate and exhaust the event queue.
    pl.system.set_cross_core(pipe=pl.PipeType.MTE3, event_id=_M6_UPRE_EVENT)


@pl.jit(auto_mutex=True)
def flash_kda_stage1_tnd_kernel(
    cu_seqlens: pl.Tensor[[pl.DYNAMIC], pl.DT_INT64],
    # 输入
    q: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16],
    k: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16],
    g: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16],
    beta: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16],
    v: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16],
    a_log: pl.Tensor[[pl.DYNAMIC], pl.DT_FP32],
    dt_bias: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    lower_bound: pl.DT_FP32,
    scale: pl.DT_FP32,
    layout_code: pl.DT_INT64,
    # 输出
    qd_out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16],
    kr_out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16],
    gam_out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    mqk_out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16],
    w_out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16],
    upre_out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    # workspace
    workspace: pl.Ptr[pl.DT_UINT8],
):
    core_id = pl.get_block_idx() // pl.get_subblock_num()
    num_cores = pl.get_block_num()

    B = g.shape[0]
    T = g.shape[1]
    S = qd_out.shape[1]
    Nk = q.shape[2] * (1 - layout_code) + q.shape[1] * layout_code
    Nv = g.shape[2] * (1 - layout_code) + g.shape[1] * layout_code
    nq = Nv // Nk
    nc = S // CHUNK_C
    R = cu_seqlens.shape[0] - 1
    total = B * Nv
    tt_b = pl.TileType(shape=[CHUNK_C, HEAD_D], dtype=pl.DT_BF16,
                       target_memory=pl.MemorySpace.Vec)
    tt_f32 = pl.TileType(shape=[HS, HEAD_D], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Vec)
    tt_b32 = pl.TileType(shape=[HS, HEAD_D], dtype=pl.DT_BF16,
                         target_memory=pl.MemorySpace.Vec)
    tt_c64h = pl.TileType(shape=[HS, CHUNK_C], dtype=pl.DT_FP32,
                          target_memory=pl.MemorySpace.Vec)
    tt_c64bh = pl.TileType(shape=[HS, CHUNK_C], dtype=pl.DT_BF16,
                           target_memory=pl.MemorySpace.Vec)
    tt_a = pl.TileType(shape=[CHUNK_C, 1], dtype=pl.DT_FP32,
                       target_memory=pl.MemorySpace.Vec,
                       valid_shape=[-1, -1])
    tt_d = pl.TileType(shape=[1, HEAD_D], dtype=pl.DT_FP32,
                       target_memory=pl.MemorySpace.Vec)
    tt_c64 = pl.TileType(shape=[CHUNK_C, CHUNK_C], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Vec)
    tt_c64b = pl.TileType(shape=[CHUNK_C, CHUNK_C], dtype=pl.DT_BF16,
                          target_memory=pl.MemorySpace.Vec)

    gate_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[CHUNK_C, HEAD_D], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Mat, layout=pl.NZ),
        addrs=[L1_G0, L1_G1], mutex_ids=[11, 12],
        fwd_ids=_GATE_FWD_IDS, bwd_ids=_GATE_BWD_IDS,
    )

    cs_g = pl.make_tile_group(
        type=pl.TileType(shape=[CHUNK_C, HEAD_D], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Vec),
        addrs=[VA_CS0], mutex_ids=[16],
        fwd_ids=(4,), bwd_ids=(6,),
    )

    tt_tnd_input = pl.TileType(shape=[HS, HEAD_D], dtype=pl.DT_BF16,
                               target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1])
    x_b_g = pl.make_tile_group(type=tt_tnd_input, addrs=[VA_X_B], mutex_ids=[0])
    qn_h_g = pl.make_tile_group(
        type=tt_b32, addrs=[0x34800, 0x36800], mutex_ids=[10, 13])
    kn_h_g = pl.make_tile_group(
        type=tt_b32, addrs=[0x38800, 0x3A800], mutex_ids=[14, 15])
    xk_g = pl.make_tile_group(type=tt_tnd_input, addrs=[0x21000], mutex_ids=[17])

    xg_g = pl.make_tile_group(type=tt_tnd_input, addrs=[0x2000], mutex_ids=[18])
    gate_f_g = pl.make_tile_group(type=tt_f32, addrs=[VA_GATE_F], mutex_ids=[5])
    gate_nz_g = pl.make_tile_group(
        type=pl.TileType(shape=[HS, HEAD_D], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Vec, layout=pl.NZ,
                         valid_shape=[-1, -1], compact=1),
        addrs=[VA_GATE_NZ], mutex_ids=[6])
    a_g = pl.make_tile_group(type=tt_a, addrs=[VA_A_LOG], mutex_ids=[8])
    d_g = pl.make_tile_group(type=tt_d, addrs=[VA_DT_B], mutex_ids=[9])

    m3_out_b_g = pl.make_tile_group(type=tt_b32, addrs=[VA_QN_B], mutex_ids=[3])
    m3_csl_g = pl.make_tile_group(
        type=pl.TileType(shape=[1, HEAD_D], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Vec),
        addrs=[0x20800], mutex_ids=[21])

    m4_kn_b_g = pl.make_tile_group(type=tt_b, addrs=[VA_KN_B], mutex_ids=[4])
    m4_qn_b_g = pl.make_tile_group(type=tt_b, addrs=[VA_QN_B], mutex_ids=[3])
    m4_opscr_b16_g = pl.make_tile_group(
        type=pl.TileType(shape=[ROW_BLOCK, HEAD_D], dtype=pl.DT_BF16,
                         target_memory=pl.MemorySpace.Vec),
        addrs=[VA_M4_SCR_B16, VA_M4_SCR_B16 + 0x1000], mutex_ids=[6, 19])
    m4_ref_g = pl.make_tile_group(
        type=pl.TileType(shape=[1, HEAD_D], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Vec),
        addrs=[0x20A00], mutex_ids=[30])

    _tt_op16 = pl.TileType(shape=[ROW_BLOCK, HEAD_D], dtype=pl.DT_BF16,
                           target_memory=pl.MemorySpace.Mat, layout=pl.NZ,
                           valid_shape=[-1, -1], compact=1)

    lk0_g = pl.make_tile_group(type=_tt_op16, addrs=[0x14000, 0x15000], mutex_ids=[31, 27])
    lk1_g = pl.make_tile_group(type=_tt_op16, addrs=[0x16000, 0x17000], mutex_ids=[28, 29])
    lk2_g = pl.make_tile_group(type=_tt_op16, addrs=[0x18000, 0x19000], mutex_ids=[30, 31])
    lk3_g = pl.make_tile_group(type=_tt_op16, addrs=[0x1A000, 0x1B000], mutex_ids=[27, 28])
    lq0_g = pl.make_tile_group(type=_tt_op16, addrs=[0x1C000, 0x1D000], mutex_ids=[29, 30])
    lq1_g = pl.make_tile_group(type=_tt_op16, addrs=[0x1E000, 0x1F000], mutex_ids=[31, 27])
    lq2_g = pl.make_tile_group(type=_tt_op16, addrs=[0x20000, 0x21000], mutex_ids=[28, 29])
    lq3_g = pl.make_tile_group(type=_tt_op16, addrs=[0x22000, 0x23000], mutex_ids=[30, 31])
    _tt_rw = pl.TileType(shape=[CHUNK_C, HEAD_D], dtype=pl.DT_BF16,
                         target_memory=pl.MemorySpace.Mat, layout=pl.NZ,
                         valid_shape=[-1, -1], compact=1)
    _tt_rz64 = pl.TileType(shape=[HEAD_D, CHUNK_C], dtype=pl.DT_BF16,
                           target_memory=pl.MemorySpace.Mat, layout=pl.ZN,
                           valid_shape=[-1, -1], compact=1)
    rW0_g = pl.make_tile_group(type=_tt_rw, addrs=[0x34000, 0x56000], mutex_ids=[27, 28])
    rW1_g = pl.make_tile_group(type=_tt_rw, addrs=[0x38000, 0x5A000], mutex_ids=[29, 30])
    rW2_g = pl.make_tile_group(type=_tt_rw, addrs=[0x3C000, 0x5E000], mutex_ids=[31, 27])
    rW3_g = pl.make_tile_group(type=_tt_rw, addrs=[0x40000, 0x62000], mutex_ids=[28, 29])
    rZ0_g = pl.make_tile_group(type=_tt_rz64, addrs=[0x34000, 0x56000], mutex_ids=[27, 28])
    rZ1_g = pl.make_tile_group(type=_tt_rz64, addrs=[0x38000, 0x5A000], mutex_ids=[29, 30])
    rZ2_g = pl.make_tile_group(type=_tt_rz64, addrs=[0x3C000, 0x5E000], mutex_ids=[31, 27])
    rZ3_g = pl.make_tile_group(type=_tt_rz64, addrs=[0x40000, 0x62000], mutex_ids=[28, 29])

    # kn_h_g[1] ends at 0x3C800; the lagged M5 still needs its last rows.
    opnz_g = pl.make_tile_group(
        type=pl.TileType(shape=[ROW_BLOCK, HEAD_D], dtype=pl.DT_BF16,
                         target_memory=pl.MemorySpace.Vec, layout=pl.NZ,
                         valid_shape=[-1, -1], compact=1),
        addrs=[0x3C800],
        mutex_ids=[2])
    m4_la_g = pl.make_tile_group(
        type=pl.TileType(shape=[ROW_BLOCK, HEAD_D], dtype=pl.DT_BF16,
                         target_memory=pl.MemorySpace.Left, layout=pl.NZ),
        addrs=[0x4000, 0xD000], mutex_ids=[13, 23])
    m4_b_g = pl.make_tile_group(
        type=pl.TileType(shape=[HEAD_D, ROW_BLOCK], dtype=pl.DT_BF16,
                         target_memory=pl.MemorySpace.Right, layout=pl.ZN),
        addrs=[0x8000, 0x9000, 0xA000, 0xB000],
        mutex_ids=[14, 16, 17, 18])
    m4_c_g = pl.make_tile_group(
        type=pl.TileType(shape=[ROW_BLOCK, ROW_BLOCK], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Acc, layout=pl.NZ,
                         fractal=1024),
        addrs=[0x8000, 0x8400, 0x8800, 0x8C00],
        mutex_ids=[15, 19, 20, 21])

    l_pre = pl.make_tile(
        pl.TileType(shape=[CHUNK_C, CHUNK_C], dtype=pl.DT_FP32,
                    target_memory=pl.MemorySpace.Mat, layout=pl.NZ),
        addr=L1_L)
    L_l0a_g = pl.make_tile_group(
        type=pl.TileType(shape=[CHUNK_C, CHUNK_C], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Left, layout=pl.NZ),
        addrs=[LA0], mutex_ids=[13])
    g_l0b_g = pl.make_tile_group(
        type=pl.TileType(shape=[CHUNK_C, HEAD_D], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Right, layout=pl.ZN),
        addrs=[RA0], mutex_ids=[14])
    acc_g = pl.make_tile_group(
        type=pl.TileType(shape=[CHUNK_C, HEAD_D], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Acc, layout=pl.NZ,
                         fractal=1024),
        addrs=[CA0], mutex_ids=[15])

    m5_kkt_g = pl.make_tile_group(
        type=tt_c64h, addrs=[VA_M5_KKT], mutex_ids=[1])
    m5_mqk_g = pl.make_tile_group(
        type=tt_c64h, addrs=[VA_M5_MQK], mutex_ids=[2])
    m5_mqk_b_g = pl.make_tile_group(
        type=tt_c64bh, addrs=[0xE100], mutex_ids=[2])

    m5_beta_raw_b_g = pl.make_tile_group(
        type=tt_c64b, addrs=[0x12000], mutex_ids=[2])
    m5_beta_blk_f_g = pl.make_tile_group(
        type=tt_c64, addrs=[VA_M5_BETA_F], mutex_ids=[3])
    m5_v_b_g = pl.make_tile_group(
        type=tt_tnd_input, addrs=[VA_M5_VB], mutex_ids=[4])
    m5_v_operand_g = pl.make_tile_group(
        type=tt_b32, addrs=[VA_M5_VF], mutex_ids=[6])
    m5_kd_operand_g = pl.make_tile_group(
        type=tt_b32, addrs=[VA_M5_KNF], mutex_ids=[5])

    m5_kd_nz_g = pl.make_tile_group(
        type=pl.TileType(shape=[HS, HEAD_D], dtype=pl.DT_BF16,
                         target_memory=pl.MemorySpace.Vec, layout=pl.NZ,
                         valid_shape=[-1, -1], compact=1),
        addrs=[P3_UB_XY], mutex_ids=[16])

    tt_m6_l64 = pl.TileType(shape=[CHUNK_C, CHUNK_C], dtype=pl.DT_FP32,
                            target_memory=pl.MemorySpace.Mat, layout=pl.NZ,
                            valid_shape=[-1, -1])
    tt_m6_lc128 = pl.TileType(shape=[CHUNK_C, HEAD_D], dtype=pl.DT_FP32,
                              target_memory=pl.MemorySpace.Acc, layout=pl.NZ,
                              fractal=1024)
    tt_m6_vhs = pl.TileType(shape=[HS, CHUNK_C], dtype=pl.DT_FP32,
                            target_memory=pl.MemorySpace.Vec,
                            pad=pl.TilePad.zero, valid_shape=[-1, -1])
    tt_m6_vhs33 = pl.TileType(shape=[HS + 1, CHUNK_C], dtype=pl.DT_FP32,
                              target_memory=pl.MemorySpace.Vec,
                              pad=pl.TilePad.zero, valid_shape=[-1, -1])
    tt_m6_nz = pl.TileType(shape=[HS + 1, CHUNK_C], dtype=pl.DT_FP32,
                           target_memory=pl.MemorySpace.Vec, layout=pl.NZ)
    x_ub_g = pl.make_tile_group(type=tt_m6_vhs, addrs=[P3_UB_X], mutex_ids=[15])
    xy_ub_g = pl.make_tile_group(type=tt_m6_vhs33, addrs=[P3_UB_XY], mutex_ids=[16])

    snz_ub_g = pl.make_tile_group(type=tt_m6_nz, addrs=[P3_UB_XY], mutex_ids=[16])
    y_l1_g = pl.make_tile_group(type=tt_m6_l64, addrs=[P3_L1_Y0, P3_L1_Y1],
                                mutex_ids=[0, 1])
    _t71_l1_fp16 = pl.TileType(shape=[64, 64], dtype=pl.DT_FP16,
                               target_memory=pl.MemorySpace.Mat, layout=pl.NZ,
                               fractal=512, compact=0)
    t71_n_db_g = pl.make_tile_group(type=_t71_l1_fp16,
                                    addrs=[0x76000, 0x78000],
                                    mutex_ids=[24, 25])
    t71_eye = pl.make_tile(_t71_l1_fp16, addr=0x7A000)
    t71_zero = pl.make_tile(_t71_l1_fp16, addr=0x7C000)
    t71_scratch_g = pl.make_tile_group(type=_t71_l1_fp16, addrs=172032, mutex_ids=[26])
    t71_p_g = pl.make_tile_group(type=_t71_l1_fp16, addrs=180224, mutex_ids=[27])
    t71_tmp_g = pl.make_tile_group(type=_t71_l1_fp16, addrs=188416, mutex_ids=[28])
    _t71_l1_fp16_d = pl.TileType(shape=[16, 16], dtype=pl.DT_FP16,
                                 target_memory=pl.MemorySpace.Mat, layout=pl.NZ,
                                 fractal=512, compact=0)
    t71_tmpd0_g = pl.make_tile_group(type=_t71_l1_fp16_d, addrs=188416, mutex_ids=[28])
    t71_tmpd1_g = pl.make_tile_group(type=_t71_l1_fp16_d, addrs=188416 + 0xA00, mutex_ids=[29])
    t71_tmpd2_g = pl.make_tile_group(type=_t71_l1_fp16_d, addrs=188416 + 0x1400, mutex_ids=[30])
    t71_tmpd3_g = pl.make_tile_group(type=_t71_l1_fp16_d, addrs=188416 + 0x1E00, mutex_ids=[31])
    _t71_l1_bf16 = pl.TileType(shape=[64, 64], dtype=pl.DT_BF16,
                               target_memory=pl.MemorySpace.Mat, layout=pl.NZ,
                               fractal=512, compact=0)
    t71_inv_bf16_g = pl.make_tile_group(type=_t71_l1_bf16, addrs=147456, mutex_ids=[29])
    t71_eye_bf16 = pl.make_tile(_t71_l1_bf16, addr=0x7E000)

    _t71_a = pl.TileType(shape=[64, 64], dtype=pl.DT_FP16,
                         target_memory=pl.MemorySpace.Left, layout=pl.NZ,
                         fractal=512, compact=0)
    _t71_b = pl.TileType(shape=[64, 64], dtype=pl.DT_FP16,
                         target_memory=pl.MemorySpace.Right, layout=pl.ZN,
                         fractal=512, compact=0)
    t71_a_g = pl.make_tile_group(type=_t71_a, addrs=[0x5000, 0x7000], mutex_ids=[18, 19])
    t71_b_g = pl.make_tile_group(type=_t71_b, addrs=[0xC000, 0xE000], mutex_ids=[20, 21])
    _t71_ad = pl.TileType(shape=[16, 16], dtype=pl.DT_FP16,
                          target_memory=pl.MemorySpace.Left, layout=pl.NZ,
                          fractal=512, compact=0)
    _t71_bd = pl.TileType(shape=[16, 16], dtype=pl.DT_FP16,
                          target_memory=pl.MemorySpace.Right, layout=pl.ZN,
                          fractal=512, compact=0)
    t71_ad0_g = pl.make_tile_group(type=_t71_ad, addrs=[0x5000, 0x7000], mutex_ids=[18, 19])
    t71_ad1_g = pl.make_tile_group(type=_t71_ad, addrs=[0x5A00, 0x7A00], mutex_ids=[0, 1])
    t71_ad2_g = pl.make_tile_group(type=_t71_ad, addrs=[0x6400, 0x8400], mutex_ids=[2, 3])
    t71_ad3_g = pl.make_tile_group(type=_t71_ad, addrs=[0x6E00, 0x8E00], mutex_ids=[4, 5])
    t71_a10_g = pl.make_tile_group(type=_t71_ad, addrs=[0x5200, 0x7200], mutex_ids=[18, 19])
    t71_a32_g = pl.make_tile_group(type=_t71_ad, addrs=[0x6600, 0x8600], mutex_ids=[18, 19])
    t71_bd0_g = pl.make_tile_group(type=_t71_bd, addrs=[0xC000, 0xE000], mutex_ids=[20, 21])
    t71_bd1_g = pl.make_tile_group(type=_t71_bd, addrs=[0xCA00, 0xEA00], mutex_ids=[6, 7])
    t71_bd2_g = pl.make_tile_group(type=_t71_bd, addrs=[0xD400, 0xF400], mutex_ids=[8, 9])
    t71_bd3_g = pl.make_tile_group(type=_t71_bd, addrs=[0xDE00, 0xFE00], mutex_ids=[10, 11])
    t71_b10_g = pl.make_tile_group(type=_t71_bd, addrs=[0xC800, 0xE800], mutex_ids=[20, 21])
    t71_b32_g = pl.make_tile_group(type=_t71_bd, addrs=[0xDC00, 0xFC00], mutex_ids=[20, 21])
    _t71_bt = pl.TileType(shape=[16, 32], dtype=pl.DT_FP16,
                          target_memory=pl.MemorySpace.Right, layout=pl.ZN,
                          fractal=512, compact=0)
    t71_bt0_g = pl.make_tile_group(type=_t71_bt, addrs=[0xC000, 0xE000], mutex_ids=[20, 21])
    t71_bt1_g = pl.make_tile_group(type=_t71_bt, addrs=[0xC800, 0xE800], mutex_ids=[20, 21])
    t71_bl0_g = pl.make_tile_group(type=_t71_bt, addrs=[0xD000, 0xF000], mutex_ids=[20, 21])
    t71_bl1_g = pl.make_tile_group(type=_t71_bt, addrs=[0xD800, 0xF800], mutex_ids=[20, 21])

    _t71_c = pl.TileType(shape=[64, 64], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Acc, layout=pl.NZ,
                         fractal=1024, compact=0)
    t71_c_g = pl.make_tile_group(type=_t71_c, addrs=[0x9000, 0x11000], mutex_ids=[22, 23])
    _t71_c16 = pl.TileType(shape=[16, 16], dtype=pl.DT_FP32,
                           target_memory=pl.MemorySpace.Acc, layout=pl.NZ,
                           fractal=1024, compact=0)

    t71_auxd0_g = pl.make_tile_group(type=_t71_c16, addrs=[0xD000, 0x15000], mutex_ids=[24, 25])
    t71_auxd1_g = pl.make_tile_group(type=_t71_c16, addrs=[0xE400, 0x16400], mutex_ids=[26, 27])
    t71_auxd2_g = pl.make_tile_group(type=_t71_c16, addrs=[0xF800, 0x17800], mutex_ids=[28, 29])
    t71_auxd3_g = pl.make_tile_group(type=_t71_c16, addrs=[0x10C00, 0x18C00], mutex_ids=[30, 31])
    t71_aux_g = pl.make_tile_group(type=_t71_c, addrs=[0xD000, 0x15000], mutex_ids=[24, 25])

    _t71_a_bf16 = pl.TileType(shape=[64, 64], dtype=pl.DT_BF16,
                              target_memory=pl.MemorySpace.Left, layout=pl.NZ,
                              fractal=512, compact=0)
    _t71_b_bf16 = pl.TileType(shape=[64, 64], dtype=pl.DT_BF16,
                              target_memory=pl.MemorySpace.Right, layout=pl.ZN,
                              fractal=512, compact=0)
    t71_a_bf16_g = pl.make_tile_group(type=_t71_a_bf16, addrs=[0x5000, 0x7000], mutex_ids=[18, 19])
    t71_b_bf16_g = pl.make_tile_group(type=_t71_b_bf16, addrs=[0xC000, 0xE000], mutex_ids=[20, 21])

    _t71_st_nd_h = pl.TileType(shape=[32, 64], dtype=pl.DT_FP16,
                               target_memory=pl.MemorySpace.Vec,
                               valid_shape=[-1, -1], compact=0)
    _t71_st_nz_h = pl.TileType(shape=[32, 64], dtype=pl.DT_FP16,
                               target_memory=pl.MemorySpace.Vec, layout=pl.NZ,
                               fractal=512, valid_shape=[-1, -1], compact=1)
    _t71_st_nd_b = pl.TileType(shape=[32, 64], dtype=pl.DT_BF16,
                               target_memory=pl.MemorySpace.Vec,
                               valid_shape=[-1, -1], compact=0)
    _t71_st_nz_b = pl.TileType(shape=[32, 64], dtype=pl.DT_BF16,
                               target_memory=pl.MemorySpace.Vec, layout=pl.NZ,
                               fractal=512, valid_shape=[-1, -1], compact=1)
    t71_nd_h_g = pl.make_tile_group(type=_t71_st_nd_h, addrs=[49152], mutex_ids=[16])
    t71_nz_h_g = pl.make_tile_group(type=_t71_st_nz_h, addrs=[53248], mutex_ids=[16])
    t71_nd_b_g = pl.make_tile_group(type=_t71_st_nd_b, addrs=[49152], mutex_ids=[16])
    t71_nz_b_g = pl.make_tile_group(type=_t71_st_nz_b, addrs=[53248], mutex_ids=[16])
    tt_m6_l64 = pl.TileType(shape=[CHUNK_C, CHUNK_C], dtype=pl.DT_BF16,
                            target_memory=pl.MemorySpace.Mat, layout=pl.NZ,
                            valid_shape=[-1, -1])
    tt_m6_l128 = pl.TileType(shape=[CHUNK_C, HEAD_D], dtype=pl.DT_BF16,
                             target_memory=pl.MemorySpace.Mat)
    tt_m6_la = pl.TileType(shape=[CHUNK_C, CHUNK_C], dtype=pl.DT_BF16,
                           target_memory=pl.MemorySpace.Left, layout=pl.NZ)
    tt_m6_lb128 = pl.TileType(shape=[CHUNK_C, HEAD_D], dtype=pl.DT_BF16,
                              target_memory=pl.MemorySpace.Right, layout=pl.ZN)
    m6_la_g = pl.make_tile_group(type=tt_m6_la, addrs=[LA0], mutex_ids=[18])
    m6_inv_l1_g = pl.make_tile_group(type=tt_m6_l64, addrs=[P3_L1_X], mutex_ids=[31])
    m6_op_l1_g = pl.make_tile_group(type=tt_m6_l128, addrs=[L6_OP], mutex_ids=[19])

    tt_m6_hd64 = pl.TileType(shape=[HEAD_D, CHUNK_C], dtype=pl.DT_BF16,
                             target_memory=pl.MemorySpace.Mat)
    tt_m6_la_hd64 = pl.TileType(shape=[HEAD_D, CHUNK_C], dtype=pl.DT_BF16,
                                target_memory=pl.MemorySpace.Left, layout=pl.NZ)
    tt_m6_lb64 = pl.TileType(shape=[CHUNK_C, CHUNK_C], dtype=pl.DT_BF16,
                             target_memory=pl.MemorySpace.Right, layout=pl.ZN)
    tt_m6_lc_ut = pl.TileType(shape=[HEAD_D, CHUNK_C], dtype=pl.DT_FP32,
                              target_memory=pl.MemorySpace.Acc, layout=pl.NZ,
                              fractal=1024)
    m6_vbt_l1_g = pl.make_tile_group(type=tt_m6_hd64, addrs=[L6_OP2],
                                     mutex_ids=[24])
    m6_la_vbt_g = pl.make_tile_group(type=tt_m6_la_hd64, addrs=[L6_LA_VBT],
                                     mutex_ids=[25])
    tt_m6_l64_zn = pl.TileType(shape=[CHUNK_C, CHUNK_C], dtype=pl.DT_BF16,
                               target_memory=pl.MemorySpace.Mat, layout=pl.ZN,
                               valid_shape=[-1, -1])
    m6_invt_zn_g = pl.make_tile_group(type=tt_m6_l64_zn, addrs=[P3_L1_X],
                                      mutex_ids=[31])
    m6_lb_invt_g = pl.make_tile_group(type=tt_m6_lb64, addrs=[L6_LB_OP2],
                                      mutex_ids=[25])

    m6_lb128_g = pl.make_tile_group(type=tt_m6_lb128, addrs=[L6_LB_OP], mutex_ids=[20])
    m6_w_lc_g = pl.make_tile_group(type=tt_m6_lc128, addrs=[L6_LC_W], mutex_ids=[21])
    m6_u_lc_g = pl.make_tile_group(type=tt_m6_lc_ut, addrs=[L6_LC_U], mutex_ids=[22])

    str1 = HEAD_D * (1 - layout_code) + S * HEAD_D * layout_code
    str_qk = (Nk * HEAD_D) * (1 - layout_code) + HEAD_D * layout_code
    str_vv = (Nv * HEAD_D) * (1 - layout_code) + HEAD_D * layout_code
    q_v = pl.make_tensor(q, [B, Nk, T, HEAD_D], [Nk * T * HEAD_D, HEAD_D, Nk * HEAD_D, 1])
    k_v = pl.make_tensor(k, [B, Nk, T, HEAD_D], [Nk * T * HEAD_D, HEAD_D, Nk * HEAD_D, 1])
    g_v = pl.make_tensor(g, [B, Nv, T, HEAD_D], [Nv * T * HEAD_D, HEAD_D, Nv * HEAD_D, 1])

    ws_nk_bytes = B * Nk * S * HEAD_D * 2
    qn_o = pl.make_tensor(workspace, [B, Nk, S, HEAD_D],
                          [Nk * S * HEAD_D, str1, str_qk, 1], dtype=pl.DT_BF16)
    kn_o = pl.make_tensor(workspace + ws_nk_bytes, [B, Nk, S, HEAD_D],
                          [Nk * S * HEAD_D, str1, str_qk, 1], dtype=pl.DT_BF16)
    qd_o = pl.make_tensor(qd_out, [B, Nv, S, HEAD_D], [Nv * S * HEAD_D, str1, str_vv, 1])
    kr_o = pl.make_tensor(kr_out, [B, Nv, S, HEAD_D], [Nv * S * HEAD_D, str1, str_vv, 1])
    v_v = pl.make_tensor(v, [B, Nv, T, HEAD_D], [Nv * T * HEAD_D, HEAD_D, Nv * HEAD_D, 1])
    w_o = pl.make_tensor(w_out, [B, Nv, S, HEAD_D], [Nv * S * HEAD_D, str1, str_vv, 1])
    upre_o = pl.make_tensor(upre_out, [B, Nv, HEAD_D, S],
                            [Nv * HEAD_D * S, HEAD_D * S, S, 1])
    gam_o = pl.make_tensor(gam_out, [B, Nv, nc, HEAD_D],
                           [Nv * nc * HEAD_D, nc * HEAD_D, HEAD_D, 1])
    mqk_o = pl.make_tensor(mqk_out, [B, Nv, nc, CHUNK_C, CHUNK_C],
                           [Nv * nc * CHUNK_C * CHUNK_C, nc * CHUNK_C * CHUNK_C,
                            CHUNK_C * CHUNK_C, CHUNK_C, 1])
    v_beta_bf16_o = pl.make_tensor(workspace + ws_nk_bytes * 2, [B, Nv, S, HEAD_D],
                                   [Nv * S * HEAD_D, str1, str_vv, 1], dtype=pl.DT_BF16)
    v_beta_lo_o = pl.make_tensor(workspace + ws_nk_bytes * 2 + B * Nv * S * HEAD_D * 2,
                                [B, Nv, S, HEAD_D],
                                [Nv * S * HEAD_D, str1, str_vv, 1], dtype=pl.DT_BF16)
    a_view = pl.make_tensor(a_log, [Nv, 1], [1, 1])

    w_first = core_id
    c_f = w_first % nc
    src_f, valid_f = tnd_locate_chunk(cu_seqlens, c_f, R)
    with pl.section_vector():
        sub_l = pl.get_subblock_idx()
        r0_l = sub_l * HS
        ltril_rows_vf(x_ub_g.current(), r0_l)
        nd2nz_sg(x_ub_g.current(), xy_ub_g.current(), HS)
        pl.set_validshape(snz_ub_g.current(), [HS, CHUNK_C])
        pl.insert(l_pre, snz_ub_g.current(), [r0_l, 0])
        if w_first < nc * total:
            bh_f = w_first // nc
            b_f = bh_f // Nv
            hv_f = bh_f % Nv
            hk_f = hv_f // nq
            tnd_m1_prefetch_v(q_v, k_v, g_v, b_f, hk_f, hv_f,
                          src_f, valid_f, x_b_g, xk_g, xg_g)
    src = src_f
    valid = valid_f

    pw_valid = 0
    pc = 0
    pb = 0
    phv = 0
    ps0 = 0
    p_gsi = 0
    p_src = 0
    p_valid = 64
    for work in pl.range(core_id, nc * total, num_cores):
        c = work % nc
        bh = work // nc
        b = bh // Nv
        hv = bh % Nv
        hk = hv // nq
        s0 = c * CHUNK_C
        w_seq = (work - core_id) // num_cores
        gate_slot_idx = w_seq % 2
        if pw_valid:
            with pl.section_cube():
                m4_kkt_mqk_blocks(m5_kkt_g, m5_mqk_g,
                                  p_gsi,
                                  lk0_g, lk1_g, lk2_g, lk3_g,
                                  lq0_g, lq1_g, lq2_g, lq3_g,
                                  rZ0_g, rZ1_g, rZ2_g, rZ3_g,
                                  m4_la_g, m4_b_g, m4_c_g)
        with pl.section_vector():
            tnd_m1_gate_norm_v(qn_o, kn_o, a_view, lower_bound,
                           dt_bias, gate_l1, gate_slot_idx, b, hv, hk, s0,
                           1 if w_seq >= 2 else 0,
                           x_b_g, xk_g, xg_g, qn_h_g, kn_h_g, gate_f_g,
                           gate_nz_g, a_g, d_g, valid)
        if pw_valid:
            with pl.section_vector():
                tnd_m5_finish_v(beta, v_v, kn_h_g[p_gsi],
                            mqk_o, v_beta_bf16_o, v_beta_lo_o,
                            layout_code, cs_g,
                            1 if w_seq == 1 else 0,
                            pc, pb, phv, ps0, Nv, p_src, p_valid,
                            y_l1_g[0], xy_ub_g, snz_ub_g, m5_kd_nz_g,
                            m6_op_l1_g[0],
                            m5_beta_raw_b_g, m5_beta_blk_f_g, m5_v_b_g,
                            m5_kkt_g, m5_mqk_g, m5_mqk_b_g,
                            m5_v_operand_g, m5_kd_operand_g,
                            t71_n_db_g[p_gsi], t71_eye, t71_zero, t71_eye_bf16,
                            t71_nd_h_g[0], t71_nz_h_g[0], t71_nd_b_g[0], t71_nz_b_g[0],
                            x_ub_g)
        with pl.section_cube():
            m2_cumsum_c(gate_l1, cs_g, gate_slot_idx,
                        pw_valid, l_pre, L_l0a_g, g_l0b_g, acc_g)
        with pl.section_vector():
            m3_qd_kr_gam_v(qn_o, kn_o, qd_o, kr_o, gam_o, cs_g, scale,
                           c, b, hv, hk, s0, gate_slot_idx, Nv, nc,
                           qn_h_g, kn_h_g, m3_out_b_g, m3_csl_g,
                           m4_kn_b_g, m4_qn_b_g,
                           m4_opscr_b16_g, m4_ref_g,
                           lk0_g, lk1_g, lk2_g, lk3_g,
                           lq0_g, lq1_g, lq2_g, lq3_g,
                           rW0_g, rW1_g, rW2_g, rW3_g,
                           opnz_g)

        if pw_valid:
            m6_neumann_cube(
                t71_a_g, t71_b_g, t71_ad0_g, t71_ad1_g,
                t71_ad2_g, t71_ad3_g, t71_a10_g, t71_a32_g,
                t71_bd0_g, t71_bd1_g, t71_bd2_g, t71_bd3_g,
                t71_b10_g, t71_b32_g, t71_bt0_g, t71_bt1_g,
                t71_bl0_g, t71_bl1_g, t71_zero, t71_eye,
                t71_aux_g, t71_auxd0_g, t71_auxd1_g, t71_auxd2_g,
                t71_auxd3_g, t71_a_bf16_g, t71_b_bf16_g, t71_eye_bf16,
                v_beta_bf16_o, v_beta_lo_o, w_o, upre_o, t71_c_g,
                t71_n_db_g, p_gsi, pb, phv,
                ps0, t71_scratch_g, t71_tmpd0_g, t71_tmpd1_g,
                t71_tmpd2_g, t71_tmpd3_g, t71_p_g, t71_tmp_g,
                t71_inv_bf16_g, x_ub_g, m6_inv_l1_g, m6_lb128_g,
                m6_op_l1_g, m6_la_g, m6_w_lc_g, m6_vbt_l1_g,
                m6_la_vbt_g, m6_lb_invt_g, m6_invt_zn_g, m6_u_lc_g)

        nw = pl.min(work + num_cores, nc * total - 1)
        c_n = nw % nc
        bh_n = nw // nc
        src_n, valid_n = tnd_locate_chunk(cu_seqlens, c_n, R)
        with pl.section_vector():
            tnd_m1_prefetch_v(q_v, k_v, g_v, bh_n // Nv, (bh_n % Nv) // nq,
                          bh_n % Nv, src_n, valid_n, x_b_g, xk_g, xg_g)
        pw_valid = 1
        pc = c
        pb = b
        phv = hv
        ps0 = s0
        p_gsi = gate_slot_idx
        p_src = src
        p_valid = valid
        src = src_n
        valid = valid_n

    n_works = (nc * total - core_id + num_cores - 1) // num_cores
    if pw_valid:
        with pl.section_cube():
            m4_kkt_mqk_blocks(m5_kkt_g, m5_mqk_g,
                              p_gsi,
                              lk0_g, lk1_g, lk2_g, lk3_g,
                              lq0_g, lq1_g, lq2_g, lq3_g,
                              rZ0_g, rZ1_g, rZ2_g, rZ3_g,
                              m4_la_g, m4_b_g, m4_c_g)
        with pl.section_vector():
            tnd_m5_finish_v(beta, v_v, kn_h_g[p_gsi],
                        mqk_o, v_beta_bf16_o, v_beta_lo_o,
                        layout_code, cs_g,
                        1 if n_works == 1 else 0,
                        pc, pb, phv, ps0, Nv, p_src, p_valid,
                        y_l1_g[0], xy_ub_g, snz_ub_g, m5_kd_nz_g,
                        m6_op_l1_g[0],
                        m5_beta_raw_b_g, m5_beta_blk_f_g, m5_v_b_g,
                        m5_kkt_g, m5_mqk_g, m5_mqk_b_g,
                        m5_v_operand_g, m5_kd_operand_g,
                        t71_n_db_g[p_gsi], t71_eye, t71_zero, t71_eye_bf16,
                        t71_nd_h_g[0], t71_nz_h_g[0], t71_nd_b_g[0], t71_nz_b_g[0],
                        x_ub_g)
    if n_works >= 1:
        with pl.section_vector():
            pass
            for _ in pl.range(0, pl.min(n_works, 2)):
                pl.system.wait_cross_core(pipe=pl.PipeType.V, event_id=3)
    if pw_valid:
        m6_neumann_cube(
            t71_a_g, t71_b_g, t71_ad0_g, t71_ad1_g,
            t71_ad2_g, t71_ad3_g, t71_a10_g, t71_a32_g,
            t71_bd0_g, t71_bd1_g, t71_bd2_g, t71_bd3_g,
            t71_b10_g, t71_b32_g, t71_bt0_g, t71_bt1_g,
            t71_bl0_g, t71_bl1_g, t71_zero, t71_eye,
            t71_aux_g, t71_auxd0_g, t71_auxd1_g, t71_auxd2_g,
            t71_auxd3_g, t71_a_bf16_g, t71_b_bf16_g, t71_eye_bf16,
            v_beta_bf16_o, v_beta_lo_o, w_o, upre_o, t71_c_g,
            t71_n_db_g, p_gsi, pb, phv,
            ps0, t71_scratch_g, t71_tmpd0_g, t71_tmpd1_g,
            t71_tmpd2_g, t71_tmpd3_g, t71_p_g, t71_tmp_g,
            t71_inv_bf16_g, x_ub_g, m6_inv_l1_g, m6_lb128_g,
            m6_op_l1_g, m6_la_g, m6_w_lc_g, m6_vbt_l1_g,
            m6_la_vbt_g, m6_lb_invt_g, m6_invt_zn_g, m6_u_lc_g)


@pl.jit(auto_mutex=True, tiling_key=_Stage2TndTilingKey)
def flash_kda_stage2_tnd_kernel(
    cu_seqlens: pl.Tensor[[pl.DYNAMIC], pl.DT_INT64],
    # 输入
    layout_code: pl.DT_INT64,
    qd_out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16],
    kr_out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16],
    gam_out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    mqk_out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16],
    w_out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16],
    upre_out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    initial_state: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    # 输出
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16],
    final_state: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
):
    DV = Stage2VRows  # noqa: F821 - compile-time TilingKey field
    DV_SPLIT = HEAD_D // DV
    core_id = pl.get_block_idx() // pl.get_subblock_num()
    num_cores = pl.get_block_num()

    B = qd_out.shape[0]
    S = qd_out.shape[1] * (1 - layout_code) + qd_out.shape[2] * layout_code
    Nv = qd_out.shape[2] * (1 - layout_code) + qd_out.shape[1] * layout_code
    nc = S // CHUNK_C
    R = initial_state.shape[0]
    total = R * Nv
    T = out.shape[1]

    tt_m7_lw = pl.TileType(shape=[CHUNK_C, HEAD_D], dtype=pl.DT_BF16,
                           target_memory=pl.MemorySpace.Left, layout=pl.NZ)
    tt_m7_lw2 = pl.TileType(shape=[CHUNK_C, CHUNK_C], dtype=pl.DT_BF16,
                            target_memory=pl.MemorySpace.Left, layout=pl.NZ)
    tt_m7_rb64 = pl.TileType(shape=[CHUNK_C, HEAD_D], dtype=pl.DT_BF16,
                             target_memory=pl.MemorySpace.Right, layout=pl.ZN)
    tt_m7_la_snapT = pl.TileType(shape=[DV, HEAD_D], dtype=pl.DT_BF16,
                                 target_memory=pl.MemorySpace.Left, layout=pl.NZ)
    tt_m7_rb_wt = pl.TileType(shape=[HEAD_D, CHUNK_C], dtype=pl.DT_BF16,
                              target_memory=pl.MemorySpace.Right, layout=pl.ZN)
    tt_m7_la_ut = pl.TileType(shape=[DV, CHUNK_C], dtype=pl.DT_BF16,
                              target_memory=pl.MemorySpace.Left, layout=pl.NZ)
    tt_m7_lc_udT = pl.TileType(shape=[DV, CHUNK_C], dtype=pl.DT_FP32,
                               target_memory=pl.MemorySpace.Acc, layout=pl.NZ,
                               fractal=1024)
    tt_m7_lc_delT = pl.TileType(shape=[DV, HEAD_D], dtype=pl.DT_FP32,
                                target_memory=pl.MemorySpace.Acc, layout=pl.NZ,
                                fractal=1024)
    tt_m7_st_f = pl.TileType(shape=[DV, CHUNK_C], dtype=pl.DT_FP32,
                             target_memory=pl.MemorySpace.Vec)
    tt_m7_st_b = pl.TileType(shape=[DV, CHUNK_C], dtype=pl.DT_BF16,
                             target_memory=pl.MemorySpace.Vec)
    tt_m7_ut_b = pl.TileType(shape=[DV // 2, CHUNK_C], dtype=pl.DT_BF16,
                             target_memory=pl.MemorySpace.Vec)
    if DV == 16:
        m7_sT_g = pl.make_tile_group(
            type=tt_m7_st_f, addrs=[VA_M7_S, VA_M7_S + 0x1000], mutex_ids=[0, 8])
        m7_delT_g = pl.make_tile_group(
            type=tt_m7_st_f, addrs=[VA_M7_DEL, VA_M7_DEL + 0x2000], mutex_ids=[1, 9])
        tt_m7_udT_half = pl.TileType(shape=[DV // 2, CHUNK_C], dtype=pl.DT_FP32,
                                     target_memory=pl.MemorySpace.Vec)
        m7_udT_g = pl.make_tile_group(
            type=tt_m7_udT_half, addrs=[VA_M7_UD, VA_M7_UD + 0x1000], mutex_ids=[2, 10])

        tt_m7_ut_b_c = pl.TileType(shape=[DV // 2, CHUNK_C], dtype=pl.DT_BF16,
                                   target_memory=pl.MemorySpace.Vec,
                                   valid_shape=[-1, -1], compact=1)
        tt_m7_ut_nz = pl.TileType(shape=[DV // 2, CHUNK_C], dtype=pl.DT_BF16,
                                  target_memory=pl.MemorySpace.Vec, layout=pl.NZ,
                                  valid_shape=[-1, -1], compact=1)
        m7_ut_b_g = pl.make_tile_group(type=tt_m7_ut_b_c,
                                       addrs=[VA_M7_UB, VA_M7_UB + 0x400],
                                       mutex_ids=[4, 11])
        m7_ut_nz_g = pl.make_tile_group(type=tt_m7_ut_nz, addrs=[VA_M7_UTNZ],
                                        mutex_ids=[3])
        tt_m7_snapb_c = pl.TileType(shape=[DV, CHUNK_C], dtype=pl.DT_BF16,
                                    target_memory=pl.MemorySpace.Vec,
                                    valid_shape=[-1, -1], compact=1)
        tt_m7_snapnz = pl.TileType(shape=[DV, CHUNK_C], dtype=pl.DT_BF16,
                                   target_memory=pl.MemorySpace.Vec, layout=pl.NZ,
                                   valid_shape=[-1, -1], compact=1)
        m7_snapb_c_g = pl.make_tile_group(type=tt_m7_snapb_c, addrs=[VA_M7_UB + 0x1000],
                                          mutex_ids=[6])
        m7_snapnz_g = pl.make_tile_group(type=tt_m7_snapnz, addrs=[VA_M7_UB + 0x2000],
                                         mutex_ids=[7])
        m7_snapT_b_g = pl.make_tile_group(type=tt_m7_st_b, addrs=[VA_M7_SNAP],
                                          mutex_ids=[5])

        # Each Upre bank holds (DV // 2) * 64 FP32 values.
        upre_f4_g = pl.make_tile_group(
            type=tt_m7_udT_half, addrs=[P4A_UB_UPRE0, P4A_UB_UPRE1], mutex_ids=[28, 29])
        gam_f4_g = pl.make_tile_group(
            type=pl.TileType(shape=[1, HEAD_D], dtype=pl.DT_FP32,
                             target_memory=pl.MemorySpace.Vec),
            addrs=[P4A_UB_GAM0, P4A_UB_GAM1], mutex_ids=[30, 31])
        wt_l1_g = pl.make_tile_group(
            type=pl.TileType(shape=[HEAD_D, CHUNK_C], dtype=pl.DT_BF16,
                             target_memory=pl.MemorySpace.Mat),
            addrs=[P4A_L1_W0, P4A_L1_W1], mutex_ids=[24, 25])
        kr_l1_g = pl.make_tile_group(
            type=pl.TileType(shape=[CHUNK_C, HEAD_D], dtype=pl.DT_BF16,
                             target_memory=pl.MemorySpace.Mat),
            addrs=[P4A_L1_KR0, P4A_L1_KR1], mutex_ids=[26, 27])
        qd_l1_g = pl.make_tile_group(
            type=pl.TileType(shape=[CHUNK_C, HEAD_D], dtype=pl.DT_BF16,
                             target_memory=pl.MemorySpace.Mat),
            addrs=[P4A_L1_QD0, P4A_L1_QD1], mutex_ids=[22, 23])
        mqk_l1_g = pl.make_tile_group(
            type=pl.TileType(shape=[CHUNK_C, CHUNK_C], dtype=pl.DT_BF16,
                             target_memory=pl.MemorySpace.Mat),
            addrs=[P4A_L1_MQK0, P4A_L1_MQK1], mutex_ids=[20, 21])
        tt_m7_snapin = pl.TileType(shape=[DV, HEAD_D], dtype=pl.DT_BF16,
                                   target_memory=pl.MemorySpace.Mat,
                                   layout=pl.NZ,
                                   valid_shape=[-1, -1], compact=1)
        snapin_l1_g = pl.make_tile_group(
            type=tt_m7_snapin, addrs=[P4A_L1_SNAPIN0, P4A_L1_SNAPIN1],
            mutex_ids=[28, 19])
        snapin_zn_l1_g = pl.make_tile_group(
            type=pl.TileType(shape=[HEAD_D, DV], dtype=pl.DT_BF16,
                             target_memory=pl.MemorySpace.Mat,
                             layout=pl.ZN,
                             valid_shape=[-1, -1], compact=1),
            addrs=[P4A_L1_SNAPIN0, P4A_L1_SNAPIN1],
            mutex_ids=[28, 19])
        tt_m7_unz = pl.TileType(shape=[DV, CHUNK_C], dtype=pl.DT_BF16,
                               target_memory=pl.MemorySpace.Mat,
                               layout=pl.NZ,
                               valid_shape=[-1, -1], compact=1)
        u_nz_l1_g = pl.make_tile_group(
            type=tt_m7_unz, addrs=[P4A_L1_UNZ0, P4A_L1_UNZ1],
            mutex_ids=[22, 23])
        u_zn_l1_g = pl.make_tile_group(
            type=pl.TileType(shape=[CHUNK_C, DV], dtype=pl.DT_BF16,
                             target_memory=pl.MemorySpace.Mat,
                             layout=pl.ZN,
                             valid_shape=[-1, -1], compact=1),
            addrs=[P4A_L1_UNZ0, P4A_L1_UNZ1],
            mutex_ids=[22, 23])
        m7_la_snapT_g = pl.make_tile_group(type=tt_m7_la_snapT, addrs=[LA0],
                                           mutex_ids=[13])
        m7_la_ut_g = pl.make_tile_group(type=tt_m7_la_ut, addrs=[0x2000],
                                        mutex_ids=[13])
        m7_la_qd_g = pl.make_tile_group(type=tt_m7_lw, addrs=[0x4000],
                                        mutex_ids=[13])
        m7_la_mqk_g = pl.make_tile_group(type=tt_m7_lw2, addrs=[0x8000],
                                         mutex_ids=[13])

        m7_rb_wt_g = pl.make_tile_group(type=tt_m7_rb_wt, addrs=[0x0],
                                        mutex_ids=[14])
        m7_rb_kr_g = pl.make_tile_group(type=tt_m7_rb64, addrs=[0x4000],
                                        mutex_ids=[14])
        m7_rb_snapt_g = pl.make_tile_group(
            type=pl.TileType(shape=[HEAD_D, DV], dtype=pl.DT_BF16,
                             target_memory=pl.MemorySpace.Right, layout=pl.ZN),
            addrs=[0x8000], mutex_ids=[14])
        m7_rb_uot_g = pl.make_tile_group(
            type=pl.TileType(shape=[CHUNK_C, DV], dtype=pl.DT_BF16,
                             target_memory=pl.MemorySpace.Right, layout=pl.ZN),
            addrs=[0x8800], mutex_ids=[14])
        m7_lc_udT_g = pl.make_tile_group(type=tt_m7_lc_udT, addrs=[L7_LC_UD],
                                         mutex_ids=[15])
        m7_lc_delT_g = pl.make_tile_group(type=tt_m7_lc_delT, addrs=[L7_LC_DEL],
                                          mutex_ids=[17])
        m7_lc_o_g = pl.make_tile_group(
            type=pl.TileType(shape=[CHUNK_C, DV], dtype=pl.DT_FP32,
                             target_memory=pl.MemorySpace.Acc),
            addrs=[0x20000], mutex_ids=[18])
        tail_o_g = pl.make_tile_group(
            type=pl.TileType(shape=[HS, DV], dtype=pl.DT_FP32,
                             target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1]),
            addrs=[0x36000, 0x36800], mutex_ids=[12, 13])
        tail_b_g = pl.make_tile_group(
            type=pl.TileType(shape=[HS, DV], dtype=pl.DT_BF16,
                             target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1]),
            addrs=[0x37000], mutex_ids=[14])
    else:
        m7_sT_g = pl.make_tile_group(
            type=tt_m7_st_f, addrs=[0, DV * 256], mutex_ids=[0, 8])
        m7_delT_g = pl.make_tile_group(
            type=tt_m7_st_f, addrs=[DV * 512, DV * 768], mutex_ids=[1, 9])
        tt_m7_udT_half = pl.TileType(shape=[DV // 2, CHUNK_C], dtype=pl.DT_FP32,
                                     target_memory=pl.MemorySpace.Vec)
        m7_udT_g = pl.make_tile_group(
            type=tt_m7_udT_half, addrs=[DV * 1024, DV * 1152], mutex_ids=[2, 10])

        tt_m7_ut_b_c = pl.TileType(shape=[DV // 2, CHUNK_C], dtype=pl.DT_BF16,
                                   target_memory=pl.MemorySpace.Vec,
                                   valid_shape=[-1, -1], compact=1)
        tt_m7_ut_nz = pl.TileType(shape=[DV // 2, CHUNK_C], dtype=pl.DT_BF16,
                                  target_memory=pl.MemorySpace.Vec, layout=pl.NZ,
                                  valid_shape=[-1, -1], compact=1)
        m7_ut_b_g = pl.make_tile_group(type=tt_m7_ut_b_c,
                                       addrs=[DV * 1536, DV * 1600],
                                       mutex_ids=[4, 11])
        m7_ut_nz_g = pl.make_tile_group(type=tt_m7_ut_nz, addrs=[DV * 1664],
                                        mutex_ids=[3])
        tt_m7_snapb_c = pl.TileType(shape=[DV, CHUNK_C], dtype=pl.DT_BF16,
                                    target_memory=pl.MemorySpace.Vec,
                                    valid_shape=[-1, -1], compact=1)
        tt_m7_snapnz = pl.TileType(shape=[DV, CHUNK_C], dtype=pl.DT_BF16,
                                   target_memory=pl.MemorySpace.Vec, layout=pl.NZ,
                                   valid_shape=[-1, -1], compact=1)
        m7_snapb_c_g = pl.make_tile_group(type=tt_m7_snapb_c, addrs=[DV * 1728],
                                          mutex_ids=[6])
        m7_snapnz_g = pl.make_tile_group(type=tt_m7_snapnz, addrs=[DV * 1856],
                                         mutex_ids=[7])
        m7_snapT_b_g = pl.make_tile_group(type=tt_m7_st_b, addrs=[DV * 1984],
                                          mutex_ids=[5])

        # Each Upre bank holds (DV // 2) * 64 FP32 values.
        upre_f4_g = pl.make_tile_group(
            type=tt_m7_udT_half, addrs=[DV * 1280, DV * 1408], mutex_ids=[28, 29])
        gam_f4_g = pl.make_tile_group(
            type=pl.TileType(shape=[1, HEAD_D], dtype=pl.DT_FP32,
                             target_memory=pl.MemorySpace.Vec),
            addrs=[DV * 2112, DV * 2112 + 512], mutex_ids=[30, 31])
        wt_l1_g = pl.make_tile_group(
            type=pl.TileType(shape=[HEAD_D, CHUNK_C], dtype=pl.DT_BF16,
                             target_memory=pl.MemorySpace.Mat),
            addrs=[P4A_L1_W0, P4A_L1_W1], mutex_ids=[24, 25])
        kr_l1_g = pl.make_tile_group(
            type=pl.TileType(shape=[CHUNK_C, HEAD_D], dtype=pl.DT_BF16,
                             target_memory=pl.MemorySpace.Mat),
            addrs=[P4A_L1_KR0, P4A_L1_KR1], mutex_ids=[26, 27])
        qd_l1_g = pl.make_tile_group(
            type=pl.TileType(shape=[CHUNK_C, HEAD_D], dtype=pl.DT_BF16,
                             target_memory=pl.MemorySpace.Mat),
            addrs=[P4A_L1_QD0, P4A_L1_QD1], mutex_ids=[22, 23])
        mqk_l1_g = pl.make_tile_group(
            type=pl.TileType(shape=[CHUNK_C, CHUNK_C], dtype=pl.DT_BF16,
                             target_memory=pl.MemorySpace.Mat),
            addrs=[P4A_L1_MQK0, P4A_L1_MQK1], mutex_ids=[20, 21])
        tt_m7_snapin = pl.TileType(shape=[DV, HEAD_D], dtype=pl.DT_BF16,
                                   target_memory=pl.MemorySpace.Mat,
                                   layout=pl.NZ,
                                   valid_shape=[-1, -1], compact=1)
        snapin_l1_g = pl.make_tile_group(
            type=tt_m7_snapin, addrs=[0x50000, 0x50000 + DV * 256],
            mutex_ids=[28, 19])
        snapin_zn_l1_g = pl.make_tile_group(
            type=pl.TileType(shape=[HEAD_D, DV], dtype=pl.DT_BF16,
                             target_memory=pl.MemorySpace.Mat,
                             layout=pl.ZN,
                             valid_shape=[-1, -1], compact=1),
            addrs=[0x50000, 0x50000 + DV * 256],
            mutex_ids=[28, 19])
        tt_m7_unz = pl.TileType(shape=[DV, CHUNK_C], dtype=pl.DT_BF16,
                               target_memory=pl.MemorySpace.Mat,
                               layout=pl.NZ,
                               valid_shape=[-1, -1], compact=1)
        u_nz_l1_g = pl.make_tile_group(
            type=tt_m7_unz, addrs=[0x50000 + DV * 512, 0x50000 + DV * 640],
            mutex_ids=[22, 23])
        u_zn_l1_g = pl.make_tile_group(
            type=pl.TileType(shape=[CHUNK_C, DV], dtype=pl.DT_BF16,
                             target_memory=pl.MemorySpace.Mat,
                             layout=pl.ZN,
                             valid_shape=[-1, -1], compact=1),
            addrs=[0x50000 + DV * 512, 0x50000 + DV * 640],
            mutex_ids=[22, 23])
        m7_la_snapT_g = pl.make_tile_group(type=tt_m7_la_snapT, addrs=[LA0],
                                           mutex_ids=[13])
        m7_la_ut_g = pl.make_tile_group(type=tt_m7_la_ut, addrs=[0x2000],
                                        mutex_ids=[13])
        m7_la_qd_g = pl.make_tile_group(type=tt_m7_lw, addrs=[0x4000],
                                        mutex_ids=[13])
        m7_la_mqk_g = pl.make_tile_group(type=tt_m7_lw2, addrs=[0x8000],
                                         mutex_ids=[13])

        m7_rb_wt_g = pl.make_tile_group(type=tt_m7_rb_wt, addrs=[0x0],
                                        mutex_ids=[14])
        m7_rb_kr_g = pl.make_tile_group(type=tt_m7_rb64, addrs=[0x4000],
                                        mutex_ids=[14])
        m7_rb_snapt_g = pl.make_tile_group(
            type=pl.TileType(shape=[HEAD_D, DV], dtype=pl.DT_BF16,
                             target_memory=pl.MemorySpace.Right, layout=pl.ZN),
            addrs=[0x8000], mutex_ids=[14])
        m7_rb_uot_g = pl.make_tile_group(
            type=pl.TileType(shape=[CHUNK_C, DV], dtype=pl.DT_BF16,
                             target_memory=pl.MemorySpace.Right, layout=pl.ZN),
            addrs=[0x8800], mutex_ids=[14])
        m7_lc_udT_g = pl.make_tile_group(type=tt_m7_lc_udT, addrs=[L7_LC_UD],
                                         mutex_ids=[15])
        m7_lc_delT_g = pl.make_tile_group(type=tt_m7_lc_delT, addrs=[L7_LC_DEL],
                                          mutex_ids=[17])
        m7_lc_o_g = pl.make_tile_group(
            type=pl.TileType(shape=[CHUNK_C, DV], dtype=pl.DT_FP32,
                             target_memory=pl.MemorySpace.Acc),
            addrs=[0x20000], mutex_ids=[18])
        tail_o_g = pl.make_tile_group(
            type=pl.TileType(shape=[HS, DV], dtype=pl.DT_FP32,
                             target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1]),
            addrs=[DV * 2112 + 1024, DV * 2240 + 1024], mutex_ids=[12, 13])
        tail_b_g = pl.make_tile_group(
            type=pl.TileType(shape=[HS, DV], dtype=pl.DT_BF16,
                             target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1]),
            addrs=[DV * 2368 + 1024], mutex_ids=[14])
    str1 = HEAD_D * (1 - layout_code) + S * HEAD_D * layout_code
    str_vv = (Nv * HEAD_D) * (1 - layout_code) + HEAD_D * layout_code

    qd_o = pl.make_tensor(qd_out, [B, Nv, S, HEAD_D], [Nv * S * HEAD_D, str1, str_vv, 1])
    kr_o = pl.make_tensor(kr_out, [B, Nv, S, HEAD_D], [Nv * S * HEAD_D, str1, str_vv, 1])
    w_o = pl.make_tensor(w_out, [B, Nv, S, HEAD_D], [Nv * S * HEAD_D, str1, str_vv, 1])
    upre_o = pl.make_tensor(upre_out, [B, Nv, HEAD_D, S],
                            [Nv * HEAD_D * S, HEAD_D * S, S, 1])
    out_o = pl.make_tensor(out, [1, Nv, T, HEAD_D],
                           [Nv * T * HEAD_D, HEAD_D, Nv * HEAD_D, 1])
    final_state_o = pl.make_tensor(final_state, [R, Nv, HEAD_D, HEAD_D],
                                   [Nv * HEAD_D * HEAD_D, HEAD_D * HEAD_D, HEAD_D, 1])
    gam_o = pl.make_tensor(gam_out, [B, Nv, nc, HEAD_D],
                           [Nv * nc * HEAD_D, nc * HEAD_D, HEAD_D, 1])
    mqk_o = pl.make_tensor(mqk_out, [B, Nv, nc, CHUNK_C, CHUNK_C],
                           [Nv * nc * CHUNK_C * CHUNK_C, nc * CHUNK_C * CHUNK_C,
                            CHUNK_C * CHUNK_C, CHUNK_C, 1])

    sub = pl.get_subblock_idx()

    total_dv = total * DV_SPLIT
    first_m7t = core_id
    fb8 = first_m7t // (Nv * DV_SPLIT)
    frem = first_m7t % (Nv * DV_SPLIT)
    fhv8 = frem // DV_SPLIT
    fco = (frem % DV_SPLIT) * DV
    with pl.section_vector():
        st_init = m7_sT_g.next()
        pl.load(st_init, initial_state, [fb8, fhv8, fco, sub * (HEAD_D // 2)])
        sbt_init = m7_snapT_b_g.next()
        pl.cast(sbt_init, st_init, mode=pl.RoundMode.CAST_ROUND)
        snz_init = m7_snapnz_g.next()
        pl.set_validshape(snz_init, [DV, CHUNK_C])
        pl.set_validshape(sbt_init, [DV, CHUNK_C])
        pl.move(snz_init, sbt_init)
        pl.insert(snapin_l1_g[0], snz_init, [0, sub * (HEAD_D // 2)])
        pl.system.set_cross_core(pipe=pl.PipeType.MTE3,
                                 event_id=_M7_SNAP_EVENT)

    # Per-core sweep count: snapshot L1 bank parity is local to this core's
    # task sequence (prologue hardcodes slot 0), not the global chunk prefix.
    processed_chunks = 0
    for m7t in pl.range(core_id, total_dv, num_cores):
        st_task = m7_sT_g.current()
        b8 = m7t // (Nv * DV_SPLIT)
        first_chunk, count = tnd_seq_span(cu_seqlens, b8, R)
        first_token = cu_seqlens[b8]
        seq_end = cu_seqlens[b8 + 1]
        base_slot = processed_chunks % 2
        mrem = m7t % (Nv * DV_SPLIT)
        hv8 = mrem // DV_SPLIT
        co = (mrem % DV_SPLIT) * DV
        with pl.section_cube():
            pl.system.wait_cross_core(pipe=pl.PipeType.MTE1,
                                      event_id=_M7_SNAP_EVENT)
            pl.load(wt_l1_g[base_slot], w_o, [0, hv8, first_chunk * CHUNK_C, 0], order=[3, 2])
            pl.load(kr_l1_g[base_slot], kr_o, [0, hv8, first_chunk * CHUNK_C, 0])
            pl.load(qd_l1_g[base_slot], qd_o, [0, hv8, first_chunk * CHUNK_C, 0])
            pl.load(mqk_l1_g[base_slot], mqk_o, [0, hv8, first_chunk, 0, 0])
        with pl.section_vector():
            pl.load(upre_f4_g[base_slot], upre_o,
                    [0, hv8, co + sub * (DV // 2), first_chunk * CHUNK_C])
            pl.load(gam_f4_g[base_slot], gam_o, [0, hv8, first_chunk, 0])
        for c8 in pl.range(0, count):
            chunk = first_chunk + c8
            s8 = chunk * CHUNK_C
            out_start = first_token + c8 * CHUNK_C
            valid = pl.min(CHUNK_C, seq_end - out_start)
            is_last = (c8 == count - 1)
            sl = (base_slot + c8) % 2
            ps = (base_slot + c8 + 1) % 2
            with pl.section_cube():
                if c8 + 1 < count:
                    pl.load(wt_l1_g[ps], w_o, [0, hv8, s8 + CHUNK_C, 0],
                            order=[3, 2])
                    pl.load(kr_l1_g[ps], kr_o, [0, hv8, s8 + CHUNK_C, 0])
                    pl.load(qd_l1_g[ps], qd_o, [0, hv8, s8 + CHUNK_C, 0])
                    pl.load(mqk_l1_g[ps], mqk_o, [0, hv8, chunk + 1, 0, 0])
                wt_rb = m7_rb_wt_g.next()
                pl.move(wt_rb, wt_l1_g[sl])
                qd_la = m7_la_qd_g.next()
                pl.move(qd_la, qd_l1_g[sl])
                if c8 > 0:
                    pl.system.wait_cross_core(pipe=pl.PipeType.MTE1,
                                              event_id=_P4A_SNAP_BANK_EVENT)
                    snap_la = m7_la_snapT_g.next()
                    pl.move(snap_la, snapin_l1_g[sl])
                else:
                    snap_la = m7_la_snapT_g.next()
                    pl.move(snap_la, snapin_l1_g[base_slot])
                ud_lc = m7_lc_udT_g.next()
                pl.matmul(ud_lc, snap_la, wt_rb)
                ud_t = m7_udT_g.next()
                pl.move(ud_t, ud_lc,
                        acc_to_vec_mode=pl.AccToVecMode.DualModeSplitM)
                pl.system.set_cross_core(pipe=pl.PipeType.FIX,
                                         event_id=_M7_UD_EVENTS[c8 % 2])
                snapt_rb = m7_rb_snapt_g.next()
                if c8 > 0:
                    pl.move(snapt_rb, snapin_zn_l1_g[sl])
                else:
                    pl.move(snapt_rb, snapin_zn_l1_g[base_slot])
                o_lc = m7_lc_o_g.next()
                pl.matmul(o_lc, qd_la, snapt_rb,
                        phase=pl.AccPhase.Partial)
            with pl.section_vector():
                if c8 + 1 < count:
                    pl.load(upre_f4_g[ps], upre_o,
                            [0, hv8, co + sub * (DV // 2), s8 + CHUNK_C])
                    pl.load(gam_f4_g[ps], gam_o, [0, hv8, chunk + 1, 0])
                pl.system.wait_cross_core(pipe=pl.PipeType.V,
                                               event_id=_M7_UD_EVENTS[c8 % 2])
                ud = m7_udT_g.next()
                u_b = m7_ut_b_g.next()
                m7_ut_add_upre_bf16_vf(ud, upre_f4_g[sl], u_b, 0, DV)
                unz = m7_ut_nz_g.next()
                pl.set_validshape(unz, [DV // 2, CHUNK_C])
                pl.set_validshape(u_b, [DV // 2, CHUNK_C])
                pl.move(unz, u_b)
                pl.insert(u_nz_l1_g[sl], unz, [sub * (DV // 2), 0])
                pl.system.set_cross_core(pipe=pl.PipeType.MTE3, event_id=_M7_U_EVENT)
                st = st_task
                m7_scale_t_vf(st, gam_f4_g[sl], sub * (HEAD_D // 2), DV)
            with pl.section_cube():
                kr_rb = m7_rb_kr_g.next()
                pl.move(kr_rb, kr_l1_g[sl])
                mqk_la = m7_la_mqk_g.next()
                pl.move(mqk_la, mqk_l1_g[sl])
                pl.system.wait_cross_core(pipe=pl.PipeType.MTE1, event_id=_M7_U_EVENT)
                u_la = m7_la_ut_g.next()
                pl.move(u_la, u_nz_l1_g[sl])
                del_lc = m7_lc_delT_g.next()
                pl.matmul(del_lc, u_la, kr_rb)
                de_t = m7_delT_g.next()
                pl.move(de_t, del_lc,
                        acc_to_vec_mode=pl.AccToVecMode.DualModeSplitN)
                pl.system.set_cross_core(pipe=pl.PipeType.FIX,
                                         event_id=_M7_D_EVENTS[c8 % 2])
                uot_rb = m7_rb_uot_g.next()
                pl.move(uot_rb, u_zn_l1_g[sl])
                pl.matmul_acc(o_lc, o_lc, mqk_la, uot_rb,
                             phase=pl.AccPhase.Final)
                if valid == CHUNK_C:
                    pl.store(out_o, o_lc, [0, hv8, out_start, co], order=[2, 3],
                             phase=pl.STPhase.Final)
                else:
                    tail_c = tail_o_g.next()
                    pl.set_validshape(tail_c, [HS, DV])
                    pl.move(tail_c, o_lc,
                            acc_to_vec_mode=pl.AccToVecMode.DualModeSplitM,
                            phase=pl.STPhase.Final)
                    pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=_TND_OUT_EVENT)
            with pl.section_vector():
                pl.system.wait_cross_core(pipe=pl.PipeType.V,
                                               event_id=_M7_D_EVENTS[c8 % 2])
                if valid < CHUNK_C:
                    pl.system.wait_cross_core(pipe=pl.PipeType.V, event_id=_TND_OUT_EVENT)
                    tail_v = tail_o_g.next()
                    rows = pl.max(0, pl.min(HS, valid - sub * HS))
                    if rows > 0:
                        pl.set_validshape(tail_v, [rows, DV])
                        tail_b = tail_b_g.next()
                        pl.set_validshape(tail_b, [rows, DV])
                        pl.cast(tail_b, tail_v, mode=pl.RoundMode.CAST_ROUND)
                        pl.store(out_o, tail_b, [0, hv8, out_start + sub * HS, co], order=[2, 3])
                if is_last:
                    next_m7t = m7t + num_cores
                    if next_m7t < total_dv:
                        nb8 = next_m7t // (Nv * DV_SPLIT)
                        nrem = next_m7t % (Nv * DV_SPLIT)
                        nhv8 = nrem // DV_SPLIT
                        nco = (nrem % DV_SPLIT) * DV
                        next_slot = (processed_chunks + count) % 2
                        st_next = m7_sT_g.next()
                        pl.load(st_next, initial_state, [nb8, nhv8, nco, sub * (HEAD_D // 2)])
                        sbt_next = m7_snapT_b_g.next()
                        pl.cast(sbt_next, st_next, mode=pl.RoundMode.CAST_ROUND)
                        snz_next = m7_snapnz_g.next()
                        pl.set_validshape(snz_next, [DV, CHUNK_C])
                        pl.set_validshape(sbt_next, [DV, CHUNK_C])
                        pl.move(snz_next, sbt_next)
                        pl.insert(snapin_l1_g[next_slot], snz_next,
                                  [0, sub * (HEAD_D // 2)])
                        pl.system.set_cross_core(pipe=pl.PipeType.MTE3,
                                                 event_id=_M7_SNAP_EVENT)
                st = st_task
                de = m7_delT_g.next()
                sbc = m7_snapb_c_g.next()
                m7_state_add_snap_vf(st, de, st, sbc, DV)
                if is_last:
                    pl.store(final_state_o, st, [b8, hv8, co, sub * (HEAD_D // 2)])
                if c8 < count - 1:
                    snz = m7_snapnz_g.next()
                    pl.set_validshape(snz, [DV, CHUNK_C])
                    pl.set_validshape(sbc, [DV, CHUNK_C])
                    pl.move(snz, sbc)
                    pl.insert(snapin_l1_g[ps], snz, [0, sub * (HEAD_D // 2)])
                    pl.system.set_cross_core(pipe=pl.PipeType.MTE3,
                                             event_id=_P4A_SNAP_BANK_EVENT)

        # Empty sequences own no chunks: pass the task's initial state through
        # to final_state and keep the per-task snap prefetch/event chain alive
        # for the next task (the c8 loop above never runs).
        if count == 0:
            with pl.section_vector():
                pl.store(final_state_o, st_task, [b8, hv8, co, sub * (HEAD_D // 2)])
                next_m7t = m7t + num_cores
                if next_m7t < total_dv:
                    nb8 = next_m7t // (Nv * DV_SPLIT)
                    nrem = next_m7t % (Nv * DV_SPLIT)
                    nhv8 = nrem // DV_SPLIT
                    nco = (nrem % DV_SPLIT) * DV
                    next_slot = (processed_chunks + count) % 2
                    st_next = m7_sT_g.next()
                    pl.load(st_next, initial_state, [nb8, nhv8, nco, sub * (HEAD_D // 2)])
                    sbt_next = m7_snapT_b_g.next()
                    pl.cast(sbt_next, st_next, mode=pl.RoundMode.CAST_ROUND)
                    snz_next = m7_snapnz_g.next()
                    pl.set_validshape(snz_next, [DV, CHUNK_C])
                    pl.set_validshape(sbt_next, [DV, CHUNK_C])
                    pl.move(snz_next, sbt_next)
                    pl.insert(snapin_l1_g[next_slot], snz_next,
                              [0, sub * (HEAD_D // 2)])
                    pl.system.set_cross_core(pipe=pl.PipeType.MTE3,
                                             event_id=_M7_SNAP_EVENT)
        processed_chunks += count


def _tnd_cu_device(cu_seqlens, device):
    """Normalise cu_seqlens to int64 on ``device`` with no device-to-host sync.

    CPU input costs one H2D copy; a same-device input passes through (int32 is
    cast in place on device).  Boundary validation is the caller's duty: the
    varlen entry checks CPU tensors for free and trusts device tensors; both
    TND kernels derive the chunk spans on-chip from the passed tensor.
    """
    if not isinstance(cu_seqlens, torch.Tensor):
        raise TypeError("TND requires a 1-D int32/int64 cu_seqlens tensor")
    if cu_seqlens.ndim != 1 or cu_seqlens.numel() < 2:
        raise ValueError("cu_seqlens must have shape [R+1], R >= 1")
    if cu_seqlens.dtype not in (torch.int32, torch.int64):
        raise TypeError("cu_seqlens must be int32 or int64")
    if cu_seqlens.device != device and cu_seqlens.device.type != "cpu":
        raise ValueError("cu_seqlens must be on CPU or on the input device")
    if not 1 <= cu_seqlens.numel() - 1 <= B_MAX:
        raise ValueError(f"TND supports 1 <= R <= {B_MAX}")
    return cu_seqlens.detach().to(device=device, dtype=torch.int64)


def _tnd_tensor(name, tensor, shape, dtype, device):
    if not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != tuple(shape):
        raise ValueError(f"{name} must have shape {tuple(shape)}")
    if tensor.dtype != dtype:
        raise TypeError(f"{name} must have dtype {dtype}")
    if tensor.device != device or device.type != "npu":
        raise ValueError(f"{name} must be on the same NPU as the inputs")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


def _flash_kda_stage1_tnd(q, k, v, g, beta, scale, a_log, dt_bias,
                         lower_bound, cu_seqlens):
    if q.ndim != 3 or v.ndim != 3:
        raise ValueError("TND q/k/v/g must have rank 3")
    T, Nk, D = q.shape
    Nv = v.shape[1]
    if D != HEAD_D or not (1 <= Nk <= NK_MAX and 1 <= Nv <= NV_MAX) or Nv % Nk:
        raise ValueError("TND requires D=128, supported head counts, and Nv % Nk == 0")
    if not isinstance(scale, (int, float)) or not isinstance(lower_bound, (int, float)):
        raise TypeError("scale and lower_bound must be Python scalars")
    if not -5.0 <= float(lower_bound) <= 0.0:
        raise ValueError("lower_bound must be in [-5, 0]")
    for name, x, shape in (("q", q, (T, Nk, D)), ("k", k, (T, Nk, D)),
                           ("v", v, (T, Nv, D)), ("g", g, (T, Nv, D)),
                           ("beta", beta, (T, Nv))):
        _tnd_tensor(name, x, shape, torch.bfloat16, q.device)
    _tnd_tensor("a_log", a_log, (Nv,), torch.float32, q.device)
    _tnd_tensor("dt_bias", dt_bias, (Nv, D), torch.float32, q.device)
    cu = _tnd_cu_device(cu_seqlens, q.device)
    # Worst-case padded geometry from host-known shapes only (no cu sync):
    # J = sum(ceil(len_i/64)) <= floor(T/64) + R.  Chunks past the real total
    # run zero-filled and write into the padded region stage2 never reads.
    R = cu.numel() - 1
    J = T // CHUNK_C + R
    P = J * CHUNK_C
    def alloc(shape, dtype):
        return torch.empty(shape, dtype=dtype, device=q.device)
    qd, kr, w = (alloc((1, P, Nv, D), torch.bfloat16) for _ in range(3))
    # Expose the true physical V-major layout, unlike the legacy dense view.
    upre = alloc((1, Nv, D, P), torch.float32)
    gam = alloc((1, Nv, J, D), torch.float32)
    mqk = alloc((1, Nv, J, CHUNK_C, CHUNK_C), torch.bfloat16)
    workspace = alloc((4 * (Nk + Nv) * P * D,), torch.uint8)
    cores = min(_get_platform_config()[0], Nv * J)
    flash_kda_stage1_tnd_kernel[None, cores](
        cu,
        q.unsqueeze(0), k.unsqueeze(0), g.unsqueeze(0), beta.unsqueeze(0), v.unsqueeze(0),
        a_log, dt_bias, float(lower_bound), float(scale), 0,
        qd, kr, gam, mqk, w, upre, workspace)
    return qd, kr, gam, mqk, w, upre


def _flash_kda_stage2_tnd(qd, kr, gam, mqk, w, upre, initial_state,
                         cu_seqlens, tokens=None):
    if qd.ndim != 4 or qd.shape[0] != 1 or qd.shape[3] != HEAD_D:
        raise ValueError("TND Stage2 qd must be [1, P, Nv, 128]")
    _, P, Nv, D = qd.shape
    if P % CHUNK_C or not 1 <= Nv <= NV_MAX:
        raise ValueError("TND Stage2 requires complete internal chunks and supported Nv")
    cu = _tnd_cu_device(cu_seqlens, qd.device)
    R = cu.numel() - 1
    if tokens is None:
        # Standalone debug entry: T only exists on the host when cu_seqlens
        # does; an NPU-resident cu_seqlens would need a D2H sync to size `out`.
        if cu_seqlens.device.type != "cpu":
            raise ValueError(
                "TND Stage2 requires tokens (or CPU cu_seqlens) when "
                "cu_seqlens lives on the NPU")
        tokens = int(cu_seqlens[-1])
    if P != (tokens // CHUNK_C + R) * CHUNK_C:
        raise ValueError("cu_seqlens disagree with the Stage1 padded-chunk allocation")
    J = tokens // CHUNK_C + R
    for name, x, shape, dtype in (
        ("qd", qd, (1, P, Nv, D), torch.bfloat16),
        ("kr", kr, (1, P, Nv, D), torch.bfloat16),
        ("w", w, (1, P, Nv, D), torch.bfloat16),
        ("upre", upre, (1, Nv, D, P), torch.float32),
        ("gam", gam, (1, Nv, J, D), torch.float32),
        ("mqk", mqk, (1, Nv, J, CHUNK_C, CHUNK_C), torch.bfloat16),
        ("initial_state", initial_state, (R, Nv, D, D), torch.float32),
    ):
        _tnd_tensor(name, x, shape, dtype, qd.device)
    out = torch.empty((tokens, Nv, D), dtype=torch.bfloat16, device=qd.device)
    # Empty sequences pass their state through on-chip, so `final` needs no
    # clone and every R sequence is covered by the task grid.
    final = torch.empty_like(initial_state)
    cube_cores = _get_platform_config()[0]
    v_rows = _stage2_v_rows(Nv, R * Nv, cube_cores, tnd=True)
    cores = min(cube_cores, R * Nv * (HEAD_D // v_rows))
    flash_kda_stage2_tnd_kernel[None, cores, {"Stage2VRows": v_rows}](
        cu, 0, qd, kr, gam, mqk, w, upre, initial_state, out.unsqueeze(0), final)
    return out, final
