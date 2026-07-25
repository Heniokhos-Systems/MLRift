#!/usr/bin/env python3
"""KRNN — the flat on-target model container for std/nn_model.mlr.

===========================================================================
FORMAT SPECIFICATION  (version 1)
---------------------------------------------------------------------------
This block is duplicated VERBATIM in std/nn_model.mlr. If you change one,
change the other; the runner reads every field by fixed byte offset and has
no way to notice that the emitter has drifted.

Everything is little-endian. There is no string data, no varints, no
alignment padding beyond what is written below, and nothing that requires
an allocator or a parser: the runner does load32(base + constant).

  file = header (64 B)
       | layer records (layer_count * 160 B)
       | output records (output_count * 16 B)
       | blob (weights, biases, multipliers, shifts, LUTs)

--- header, 64 bytes ---------------------------------------------------
  0x00 u32  magic          'K','R','N','N' = 0x4E4E524B
  0x04 u32  version        1
  0x08 u32  layer_count
  0x0C u32  arena_bytes    peak activation arena the runner needs
  0x10 u32  layers_off     byte offset of the layer table
  0x14 u32  layer_stride   bytes per layer record (160)
  0x18 u32  outputs_off    byte offset of the output table
  0x1C u32  output_count
  0x20 u32  blob_off       byte offset of the parameter blob
  0x24 u32  blob_bytes
  0x28 u32  input_off      arena offset the caller writes the input to
  0x2C u32  input_h
  0x30 u32  input_w
  0x34 u32  input_c
  0x38 i32  input_zp       zero point of the input quantisation
  0x3C u32  input_scale_q16  round(input_scale * 65536), informational

--- layer record, 160 bytes -------------------------------------------
  0x00 u32  op             see the opcode table below
  0x04 u32  in0_off        arena offset of input A
  0x08 u32  in1_off        arena offset of input B (Add only)
  0x0C u32  out_off        arena offset of the output
  0x10 u32  in_h
  0x14 u32  in_w
  0x18 u32  in_c
  0x1C u32  out_h
  0x20 u32  out_w
  0x24 u32  out_c
  0x28 u32  kh             kernel / pool height   (Resize: unused)
  0x2C u32  kw
  0x30 u32  stride_h       (Resize: integer H scale factor)
  0x34 u32  stride_w       (Resize: integer W scale factor)
  0x38 u32  pad_h          symmetric padding
  0x3C u32  pad_w
  0x40 u32  depth_mult     depthwise depth multiplier
  0x44 u32  q_stride       1 = per-channel requant, 0 = per-tensor
  0x48 u32  w_off          blob offset of the int8 weights   (NONE = 0xFFFFFFFF)
  0x4C u32  bias_off       blob offset of the int32 biases   (NONE)
  0x50 u32  mult_off       blob offset of the int32 mults    (NONE)
  0x54 u32  shift_off      blob offset of the int32 shifts   (NONE)
  0x58 i32  in_zp          zero point of input A
  0x5C i32  out_zp
  0x60 i32  act_min        output clamp, quantised domain
  0x64 i32  act_max
  0x68 u32  n_elem         out_h * out_w * out_c (elementwise ops use it)
  0x6C i32  in1_zp         zero point of input B (Add)
  0x70 i32  a_mult         Add: rescale A onto the common scale
  0x74 i32  a_shift
  0x78 i32  b_mult         Add: rescale B
  0x7C i32  b_shift
  0x80 i32  o_mult         Add: rescale the sum onto the output scale
  0x84 i32  o_shift
  0x88 u32  left_shift     Add: TFLite's intermediate left shift (20)
  0x8C u32  lut_off        blob offset of a 256-byte int8 LUT (NONE)
  0x90 u32  flags          bit 0 = this layer's output is a graph output
  0x94 u32  reserved0
  0x98 u32  reserved1
  0x9C u32  reserved2

  Every `shift` field uses TFLite's SIGNED convention, which is what
  nn_mul_by_qm() consumes: POSITIVE shifts the input left first, NEGATIVE
  divides the product right afterwards. (int8_sim.py internally uses the
  opposite sign; the emitter negates on the way out. Getting this backwards
  is a silent factor of 2^2s.)

--- output record, 16 bytes -------------------------------------------
  0x00 u32  arena_off
  0x04 u32  n_elem
  0x08 i32  zero_point
  0x0C u32  scale_q16      round(scale * 65536); dequantise on the host

--- opcodes ------------------------------------------------------------
  0  NOP          layout-only Transpose/Reshape; output aliases the input
  1  CONV2D       nn_conv2d_int8
  2  DEPTHWISE    nn_depthwise_conv2d_int8
  3  POINTWISE    nn_pointwise_conv_int8 (1x1, stride 1, no pad)
  4  RELU         nn_relu_int8, in place (in0_off == out_off)
  5  MAXPOOL      nn_maxpool_int8
  6  ADD          nn_add_int8
  7  RESIZE_NN    nn_resize_nearest_int8
  8  SIGMOID_LUT  nn_sigmoid_lut_int8
  9  COPY         nn_copy_i8 (a materialised layout change)
 10  AVGPOOL      nn_avgpool_int8
===========================================================================
"""
import struct

MAGIC = 0x4E4E524B
VERSION = 1
HEADER_BYTES = 64
LAYER_STRIDE = 160
OUTPUT_STRIDE = 16
NONE = 0xFFFFFFFF

OP_NOP = 0
OP_CONV2D = 1
OP_DEPTHWISE = 2
OP_POINTWISE = 3
OP_RELU = 4
OP_MAXPOOL = 5
OP_ADD = 6
OP_RESIZE_NN = 7
OP_SIGMOID_LUT = 8
OP_COPY = 9
OP_AVGPOOL = 10

OP_NAMES = {
    OP_NOP: "nop", OP_CONV2D: "conv2d", OP_DEPTHWISE: "depthwise",
    OP_POINTWISE: "pointwise", OP_RELU: "relu", OP_MAXPOOL: "maxpool",
    OP_ADD: "add", OP_RESIZE_NN: "resize_nn", OP_SIGMOID_LUT: "sigmoid_lut",
    OP_COPY: "copy", OP_AVGPOOL: "avgpool",
}

# Field order of a layer record. Kept as a list so the emitter cannot get an
# offset wrong: index i lives at byte 4*i.
LAYER_FIELDS = [
    "op", "in0_off", "in1_off", "out_off",
    "in_h", "in_w", "in_c", "out_h", "out_w", "out_c",
    "kh", "kw", "stride_h", "stride_w", "pad_h", "pad_w",
    "depth_mult", "q_stride",
    "w_off", "bias_off", "mult_off", "shift_off",
    "in_zp", "out_zp", "act_min", "act_max",
    "n_elem", "in1_zp",
    "a_mult", "a_shift", "b_mult", "b_shift", "o_mult", "o_shift",
    "left_shift", "lut_off", "flags", "reserved0", "reserved1", "reserved2",
]
assert len(LAYER_FIELDS) * 4 == LAYER_STRIDE

SIGNED_FIELDS = {"in_zp", "out_zp", "act_min", "act_max", "in1_zp",
                 "a_mult", "a_shift", "b_mult", "b_shift", "o_mult", "o_shift"}


def blank_layer():
    d = {k: 0 for k in LAYER_FIELDS}
    for k in ("w_off", "bias_off", "mult_off", "shift_off", "lut_off"):
        d[k] = NONE
    d["act_min"] = -128
    d["act_max"] = 127
    return d


def pack_layer(d):
    out = bytearray()
    for k in LAYER_FIELDS:
        v = int(d[k])
        if k in SIGNED_FIELDS:
            out += struct.pack("<i", v)
        else:
            out += struct.pack("<I", v & 0xFFFFFFFF)
    assert len(out) == LAYER_STRIDE
    return bytes(out)


def pack_header(layer_count, arena_bytes, layers_off, outputs_off,
                output_count, blob_off, blob_bytes, input_off,
                input_h, input_w, input_c, input_zp, input_scale_q16):
    h = struct.pack("<IIIIIIII", MAGIC, VERSION, layer_count, arena_bytes,
                    layers_off, LAYER_STRIDE, outputs_off, output_count)
    h += struct.pack("<IIIIII", blob_off, blob_bytes, input_off,
                     input_h, input_w, input_c)
    h += struct.pack("<iI", input_zp, input_scale_q16)
    assert len(h) == HEADER_BYTES, len(h)
    return h


def pack_output(arena_off, n_elem, zp, scale_q16):
    return struct.pack("<IIiI", arena_off, n_elem, zp, scale_q16)


# ---------------------------------------------------------------------------
# Blob builder
# ---------------------------------------------------------------------------

class Blob:
    def __init__(self):
        self.buf = bytearray()

    def _align(self, n):
        while len(self.buf) % n:
            self.buf.append(0)

    def add_i8(self, arr):
        off = len(self.buf)
        self.buf += arr.astype("int8").tobytes()
        return off

    # numpy-free variant, so tests/nn/nn_model_ref.py can build a model with
    # nothing but the standard library.
    def add_i8_list(self, values):
        off = len(self.buf)
        for v in values:
            self.buf += struct.pack("<b", int(v))
        return off

    def add_i32(self, values):
        self._align(4)
        off = len(self.buf)
        for v in values:
            self.buf += struct.pack("<i", int(v))
        return off

    def add_bytes(self, b):
        off = len(self.buf)
        self.buf += b
        return off

    def __len__(self):
        return len(self.buf)
