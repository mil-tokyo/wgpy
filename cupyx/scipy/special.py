import wgpy as cp

backend = cp.get_backend_name()
if backend == "webgpu":
    from wgpy_backends.webgpu.elementwise_kernel import ElementwiseKernel
elif backend == "webgl":
    from wgpy_backends.webgl.elementwise_kernel import ElementwiseKernel

_erf_kernel = None


def erf(x):
    global _erf_kernel
    if backend == "webgpu":
        if _erf_kernel is None:
            _erf_kernel = ElementwiseKernel(
                in_params="f32 x",
                out_params="f32 y",
                operation="""
const a1 = 0.254829592;
const a2 = -0.284496736;
const a3 = 1.421413741;
const a4 = -1.453152027;
const a5 = 1.061405429;
const p = 0.3275911;

let absx = abs(x);
let t = 1.0 / (1.0 + p * absx);
let z = 1.0 - ((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t * exp(-absx * absx);

if (x < 0.0) {
    y = -z;
} else {
    y = z;
}
                """,
                name="erf",
            )
        return _erf_kernel(x)
    elif backend == "webgl":
        if _erf_kernel is None:
            _erf_kernel = ElementwiseKernel(
                in_params="float x",
                out_params="float y",
                operation="""
float a1 = 0.254829592;
float a2 = -0.284496736;
float a3 = 1.421413741;
float a4 = -1.453152027;
float a5 = 1.061405429;
float p = 0.3275911;

float sign = x < 0.0 ? -1.0 : 1.0;
float absx = abs(x);
float t = 1.0 / (1.0 + p * absx);
float z = 1.0 - ((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t * exp(-absx * absx);

y = sign * z;
                """,
                name="erf",
            )
        return _erf_kernel(x)
    raise ValueError
