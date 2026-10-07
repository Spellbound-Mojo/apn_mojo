# Try and review the API

Start with a small calculation you know well. As you try the library, note
what you expected, what happened, and where the API helped or got in the way.
Those details make feedback easier to act on.

## A short evaluation route

1. Run the [first program](../index.md#first-program). Change the inputs and
   check that saved copies keep their values.
2. Try [integer division](../tutorial/integers.md#arithmetic-and-division) with
   negative operands, then compare it with exact [fractions](../tutorial/rationals.md).
3. Choose a [Float precision](../tutorial/floats.md) and compare decimal text
   with input from a native float. Try [complex arithmetic](../tutorial/complex.md)
   or [ball bounds](../reference/ball.md) if your calculation needs them.
4. Work through [batches and selections](../tutorial/batches.md), including
   a slice, a mask, and an overlapping assignment.
5. Trigger an [error](../tutorial/errors.md) and check the destination afterward.
6. Save and restore a value through [JSON](../tutorial/conversion.md), then
   try a conversion limit that is too small.
7. Adapt [counters and totals](totals.md) to your own inputs.

## Key review criteria

| Area | Questions to consider |
|---|---|
| Construction | Is it clear which inputs are accepted and when they round? |
| Arithmetic | Are division rules, result types, and precision choices predictable? |
| Batches | Can you tell which shapes broadcast and which variable an update changes? |
| Errors | Does the message help you understand and fix the failed call? |
| Interchange | Do the text and JSON formats preserve the information you need? |
| Daily use | Can you express the calculation without depending on private storage details? |

## Reporting feedback and edge cases

For a bug report, include:

- A small, self-contained Mojo program that shows the problem.
- The expected result and the actual output or error.
- The compiler version from `pixi run --locked mojo --version`.
- Any relevant precision, rounding mode, shape, or conversion limits.

For an API suggestion, show the call you would like to write and how it would
help your calculation. Before reporting an issue, check
[support and limitations](status.md) for known boundaries.
