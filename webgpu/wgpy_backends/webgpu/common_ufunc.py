from wgpy_backends.webgpu.ufunc import create_ufunc

binary_number_types = ["BB->B", "ii->i", "ff->f"]
binary_compare_types = ["??->?", "BB->?", "ii->?", "ff->?"]
add = create_ufunc("add", binary_number_types, "out0 = in0 + in1")
sub = create_ufunc("sub", binary_number_types, "out0 = in0 - in1")
mul = create_ufunc("mul", binary_number_types, "out0 = in0 * in1")
truediv = create_ufunc(
    "truediv", ["BB->f", "ii->f", "ff->f"], "out0 = f32(in0) / f32(in1)"
)
# may be used in0 < 0 and y > 0 (-5**2), but in0 < 0 is undefined in GLSL pow.
# it is useful in normalization, so applying abs.
pow = create_ufunc("pow", binary_number_types, "out0 = pow(abs(in0), in1)")
lt = create_ufunc("lt", binary_compare_types, "out0 = in0 < in1")
le = create_ufunc("le", binary_compare_types, "out0 = in0 <= in1")
eq = create_ufunc("eq", binary_compare_types, "out0 = in0 == in1")
ne = create_ufunc("ne", binary_compare_types, "out0 = in0 != in1")
ge = create_ufunc("ge", binary_compare_types, "out0 = in0 >= in1")
gt = create_ufunc("gt", binary_compare_types, "out0 = in0 > in1")

# currently, NaN is not handled
maximum = create_ufunc(
    "maximum",
    [
        ("??->?", "out0 = in0 || in1"),
        "BB->B",
        "ii->i",
        ("ff->f", "out0 = max(in0, in1)"),
    ],
    "out0 = max(in0, in1)",
)
fmax = create_ufunc(
    "fmax",
    [
        ("??->?", "out0 = in0 || in1"),
        "BB->B",
        "ii->i",
        ("ff->f", "out0 = max(in0, in1)"),
    ],
    "out0 = max(in0, in1)",
)
minimum = create_ufunc(
    "minimum",
    [
        ("??->?", "out0 = in0 && in1"),
        "BB->B",
        "ii->i",
        ("ff->f", "out0 = min(in0, in1)"),
    ],
    "out0 = min(in0, in1)",
)
fmin = create_ufunc(
    "fmin",
    [
        ("??->?", "out0 = in0 && in1"),
        "BB->B",
        "ii->i",
        ("ff->f", "out0 = min(in0, in1)"),
    ],
    "out0 = min(in0, in1)",
)
clip = create_ufunc(
    "clip",
    [
        "BBB->B",
        "iii->i",
        "fff->f",
    ],
    "out0 = clamp(in0, in1, in2)",
)

unary_types = ["?->?", "B->B", "i->i", "f->f"]
pos = create_ufunc("pos", unary_types, "out0 = in0")
neg = create_ufunc(
    "neg",
    [
        ("?->?", "out0 = !in0"),  # note: error in numpy
        ("B->B", "out0 = (256u - in0) & 255u"),
        "i->i",
        "f->f",
    ],
    "out0 = -in0",
)
abs = create_ufunc(
    "abs",
    [
        ("?->?", "out0 = in0"),
        ("B->B", "out0 = in0"),
        "i->i",
        "f->f",
    ],
    "out0 = abs(in0)",
)
invert = create_ufunc(
    "invert",
    [
        ("?->?", "out0 = !in0"),
        ("B->B", "out0 = 255u - in0"),
        "i->i",
    ],
    "out0 = ~in0",
)
sign = create_ufunc(
    "sign",
    [
        ("B->B", "if (in0 == 0) { out0 = 0; } else { out0 = 1; }"),
        "i->i",
    ],
    "out0 = sign(in0)",
)

unary_float_types = ["f->f"]
exp = create_ufunc("exp", unary_float_types, "out0 = exp(in0)")
log = create_ufunc("log", unary_float_types, "out0 = log(in0)")
tanh = create_ufunc("tanh", unary_float_types, "out0 = tanh(in0)")
sqrt = create_ufunc("sqrt", unary_float_types, "out0 = sqrt(in0)")
reciprocal = create_ufunc("reciprocal", unary_float_types, "out0 = 1.0 / in0")

# cupy specific
rsqrt = create_ufunc("rsqrt", unary_float_types, "out0 = inverseSqrt(in0)")
