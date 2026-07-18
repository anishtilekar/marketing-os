# ADR 0001: `tools` may import `services.cost_guard`

**Status:** Accepted — 2026-07-16

## Decision

`marketingos.tools.base` imports `cost_guarded` from
`marketingos.services.cost_guard`. This is a deliberate, single exception to the
usual dependency direction (services depend on tools, not the reverse).

## Context

Budget enforcement must be **structural**, not conventional: `Tool.__init_subclass__`
wraps every subclass's `invoke` with the cost guard at class-creation time, so a tool
that reaches its provider unpriced cannot be defined. That requires the ABC itself to
reference the guard.

The alternative — each tool applying `@cost_guarded` by hand — keeps the layering clean
but makes the ₹100 ceiling opt-in, and a tool author who forgets the decorator silently
bypasses it. A ceiling that depends on remembering is not a ceiling.

## Consequences

- No import cycle: `cost_guard` imports `Tool` only under `TYPE_CHECKING`, and depends
  at runtime on `models.cost` and `exceptions.budget` only.
- The exception is confined to `tools/base.py`. Other `tools` modules must not import
  from `services`; concrete tools (e.g. `tools/llm/gemini_client.py`) import
  `CostGuard` for typing only and never apply the decorator themselves.
- If enforcement later moves behind a protocol owned by `tools`, this ADR should be
  superseded rather than quietly reversed.
- Guarded by `tests/unit/test_cost_guard.py::test_guard_fires_on_a_subclass_that_never_applied_the_decorator`.
