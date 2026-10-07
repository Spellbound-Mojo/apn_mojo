# Design principles

These rules define the promises made by the API and the choices that support
them. Changing a rule requires an explicit design decision and updated
documentation.

## Exactness is a promise

1. **Exact families stay exact.** Integer and Rational arithmetic never silently
   wraps, saturates, or rounds. Storage grows within checked limits. Integer
   literals keep their full value; argument adapters must not narrow them to a
   native `Int` before the operation.
2. **Floats round once.** A Float result is the mathematical result rounded
   once to its destination format. A Complex result follows that rule per
   component. Exact `sum`, `dot`, and default lifted folds retain exact working
   values; ordered modes round at each prescribed step. A longer expression
   can accumulate error across operations. Transcendental functions meet the
   same correct-rounding contract.
3. **Representation does not change the answer.** Inline and heap storage,
   thread scheduling, and dispatch paths must agree on values, formats, and
   checked failures. Result types follow the API, not the size of a value.
4. **Numerical choices are explicit.** Contexts supply formats, rounding, traps,
   and supported work budgets. Calls without a context follow documented
   operand-format rules. There is no global or thread-local rounding context.
5. **Balls enclose their results.** A ball operation accounts for every point
   in its input intervals and for rounding. Bounds may be wider than necessary.
   An operation undefined on part of an interval returns an indeterminate ball.

## One scalar function per family

6. **Define the mathematics once.** Each family has one declaration for a named
   mathematical function. Root functions dispatch by argument type to those
   family declarations.
7. **Reuse scalar functions for batches.** `vmap` maps supported scalar
   signatures; `lift` builds folds and outer products for its supported families;
   the NumPy-style functions of `apn_mojo.batch` map each root function's family
   declarations with `vmap`. Batch layout must not require a separate
   mathematical algorithm.
8. **Keep functions pure.** Numerical results do not depend on hidden mutable
   state. A checked update failure leaves the destination unchanged. Mapping
   may evaluate a failing element again to produce its diagnostic.
9. **Use free functions for mathematical operations.** Write `sqrt(x)`, with
   `abs` and `norm_sqr` for complex magnitudes. Value inspection and conversion
   methods remain methods.

### Why one declaration per name and family

`vmap` and `lift` take a compile-time function value. Mojo 1.1 cannot treat an
overloaded root name as that one value, so callers use family names such as
`apn_mojo.float.add`.

Argument adapters let one declaration accept several numeric types while
preserving exact inputs. This avoids a separate overload for every type pair.
Batch functions live in `apn_mojo.batch` because adding them to the root
overloads would conflict with scalar-to-batch construction. Each takes its
batch element type as a parameter, which also determines its context type.

Rounding status stays internal and controls traps. Do not add a second public
copy of each function just to return flags: exact comparisons, directed
rounding, and ball bounds address the corresponding numerical questions.

## Storage serves the scalars

10. **Keep scalar handles compact.** The Integer handle is 16 bytes on the
    qualified target. View layout stays in the batch container so it does not
    enlarge every scalar argument and result.
11. **Borrow elements when mapping.** Batches store scalar handles so functions
    can read them directly. Outputs still need storage, and copying a handle
    can involve reference counting. Avoid claims that an entire mapping is
    allocation-free. Keeping scalar handles in batch storage avoids
    reconstructing values from columns for each function call.

## Performance discipline

12. **Compare with the scalar loop.** Batch operators and mapping should approach
    a loop doing the same work on the same values. This is a target, not an
    unconditional performance guarantee.
13. **Keep gains that justify their cost.** Estimate the possible benefit,
    measure warmed and pinned runs, alternate builds, and report geometric
    means with drift. Preserve the measurement conditions.
14. **Use supported memory mechanisms.** Use standard allocator calls, atomics,
    and Mojo facilities. Avoid private allocator internals or thread-local keys.
15. **Budget compilation too.** Compile sequentially and use the test runner's
    12 GB guard.

## Native Mojo

16. **Use ordinary Mojo APIs.** Expose imports, structs, traits, functions, and
    compile-time parameters. Numerical algorithms are written in Mojo. The
    implementation uses FFI for system allocation and threading, not an
    external numerical backend.
17. **Work within the compiler's supported model.** Capturing closures cannot
    serve as mapped compile-time functions; pass runtime data explicitly.
18. **Do not fork the compiler or standard library.** Keep the implementation
    within the released toolchain's capabilities.

## Non-goals

The current project does not provide a general tensor framework, new hardware
DTypes, automatic vectorization of arbitrary code, decimal floating-point
arithmetic, or symbolic algebra. It also makes no constant-time or cryptographic
guarantees.

Production arithmetic does not call GMP, MPFR, MPC, or Python. Use standard
Mojo components where they meet the contract; keep custom machinery only when
semantics, safety, or measurements justify it.
