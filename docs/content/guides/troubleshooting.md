# Troubleshooting

First check that you are running from the directory containing `pixi.toml`,
with the pinned compiler and the local source path:

```sh
pixi run --locked mojo --version
pixi run --locked mojo run -I src your_program.mojo
```

Then compare the failing call with a related [example](../examples/index.md)
and its [API declaration](../reference/index.md).

| Symptom | What to check |
|---|---|
| `module 'apn_mojo' not found` | Pass `-I src` from the project directory, or an absolute path to that directory's `src`. |
| A native integer calculation overflows before conversion | Convert the native operands to `Integer` before the arithmetic. Converting an already overflowing native result cannot restore it. Arbitrary-width integer literals have their own exact constructor. |
| A batch comparison cannot be used as a condition | It returns a `Mask`; call `.any()` or `.all()`. |
| `native < value` fails to compile | Put the library value on the left, or convert the native value to the appropriate library type. |
| A one-element vector does not broadcast | Operators on two vectors require equal lengths, as do mapped extents in `vmap`. Share a scalar, or use an elementwise `batch.*` function or a lifted broadcasting call. See the [shape comparison](../reference/batch.md#higher-rank-arithmetic). |
| Updating a slice variable leaves the original unchanged | Assign through the original batch, such as `values[1:3] = 0`. |
| JSON input is rejected | Check the exact [schema](../reference/conversion.md#json). Numeric payload fields are canonical strings; field types and required metadata matter. |
| A conversion limit does not affect later work | Pass `limits=` to each conversion. It does not impose a general arithmetic or process-memory limit. |
| A small slice keeps a large allocation alive | The slice retains its backing storage. Copy its elements into a new batch when you need to release that storage. |
| `vmap(add)` or `vmap[apn_mojo.add]` is rejected | Use brackets and a single family declaration, such as `vmap[apn_mojo.float.add]()`. |
| A mapped function cannot capture a local variable | Pass it as an argument, using `in_axes=None` when it should be shared. Context objects hold only their documented numerical settings. |
| An exact lifted fold cannot represent an intermediate | Use `lift[f](exact=False)` if per-step rounding is appropriate. Exact folds raise for intermediates such as one third that are not finite binary fractions. |
| Ball batch operators or JSON fail to compile | Use `batch.add`, `batch.exp`, or other supported functions, or map scalar ball functions through `vmap`. Those batch operators and JSON are not implemented. |
| An operation returns an indeterminate ball | Check whether part of the input interval is outside the function's domain, such as a divisor containing zero. |
| A function exceeds `max_precision` | Check the function and argument in the error. Increase the context's working budget if that work is justified; this does not change the requested output precision. See [precision and accuracy](precision.md). |
| Interval bounds stay wide at higher precision | Check input uncertainty, repeated use of correlated inputs, proximity to a singularity, and the function's working budget. Recompute from original inputs; changing a result's format cannot restore lost information. |
| `to_float_if_certain` returns `None` | The enclosure does not establish one rounded result under that context. Recompute more accurately, improve uncertain inputs, or choose a less demanding target precision. |
| A larger thread count does not speed up a batch | The work may be too short or use a serial mapping signature. Inspect [parallel mapping](../reference/batch.md#parallel-mapping) and [thread controls](../reference/batch.md#threads). |
| `to_list()` still returns APN numbers | Use `to_native[dtype]()` for native values and keep `shape()` separately. A Ball midpoint export discards its uncertainty. |
| Native integer export rejects a fraction | Choose `batch.floor`, `batch.ceil`, `batch.trunc`, or `batch.round` first if that rounding is intended; integer export itself is exact. |

## Understanding error messages

Read the operation name and remedy in the error message. An element failure
also includes its logical index; a configuration error can occur before any
element is visited. The messages are written for people and may change, so
do not parse them as stable error codes.

For an unexplained failure, prepare a small reproduction as described in
[Try and review the API](review.md).
