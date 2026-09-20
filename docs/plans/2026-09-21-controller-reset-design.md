# Selective PID history reset

Evidence: runs/singlejuggle-reset-01.json shows physical selective reset passes, but controller output differs from a fresh controller by 0.01747644. Reset observations have done=false and is_init=true. Upstream uses done to clear its PID state.

Keep the original CTBR conversion, gains, mass, saturation and controller implementation. Add a project-owned subclass of its transform, overriding TorchRL's reset callback to clear only selected rows of integ and last_body_rate. Do not alter termination flags or reset other environments. No upstream edits are required.

Alternatives: changing done would conflate episode termination and initialization; clearing all controller rows would couple unrelated environments. Explicit selective reset uses the existing lifecycle and preserves both invariants.

Verification: rerun the real 16-environment selective reset case, require selected physical/history state cleared, untouched physics and controller history unchanged, and first post-reset motor outputs equal to a fresh controller within 1e-6. Preserve the failing native report. Record use of the project adapter in subsequent training configuration and provenance.
