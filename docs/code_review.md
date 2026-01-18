# Codebase Foundation Review

## Summary
- Core module boundaries are sensible, but `cli.py` currently blends UI, config assembly, and artifact writing, which will slow Phase 2 integration.
- Configuration models exist, yet the CLI bypasses them and builds dicts inline, increasing drift risk as schemas evolve.
- Hashing and JSON serialization are deterministic, but semantic vs run-specific config separation is not yet enforced.
- Error handling is generally user-friendly, with one critical runtime failure if git is unavailable.
- The UI layer is cohesive and reusable, but progress styling is not fully theme-driven.

## Must-Fix Before Phase 2
- Missing git fallback can crash `arbiter run` when git is not installed or not in PATH.  
  Reference: `src/arbiter/manifest.py:49-76`.  
  Recommendation: catch `OSError` in `get_git_info` and return `"unknown"` with `git_dirty=False` so runs remain reproducible even outside git.

## Should-Fix Soon
- Schema drift risk: `ResolvedConfig` exists but the CLI constructs a raw dict and bypasses the dataclass.  
  Reference: `src/arbiter/config.py:107-128`, `src/arbiter/cli.py:143-152`.  
  Recommendation: route CLI assembly through a `ResolvedConfig` builder or factory to keep schema changes centralized.
- Budget semantics should remain a total-cap `trial_budget.k_max` per instance and must not be multiplied by Q(c) size.  
  Reference: `src/arbiter/cli.py:63-173`.  
  Recommendation: move budget semantics into config with explicit `K_max` and keep planning logic separate from Q(c) size.
- Semantic vs run-specific metadata is mixed under `run` in the resolved config, making semantic hashing and reproducibility harder.  
  Reference: `src/arbiter/config.py:53-72`, `src/arbiter/cli.py:133-154`.  
  Recommendation: split run metadata (run_id, output_dir, timestamps) from semantic config and add a semantic hash computed on the latter.
- Run folder writes are not atomic and can leave partial artifacts if a write fails mid-way.  
  Reference: `src/arbiter/storage.py:15-33`, `src/arbiter/cli.py:126-177`.  
  Recommendation: write into a temporary directory and rename on success, or clean up on failure.
- Persona selection mode can stay `"sample_uniform"` even if the persona bank fails to load.  
  Reference: `src/arbiter/cli.py:81-103`.  
  Recommendation: reset selection to `"none"` on load failure or treat it as a hard validation error.
- Prompt defaults are embedded directly in the CLI flow, making reuse and future test coverage harder.  
  Reference: `src/arbiter/cli.py:53-121`.  
  Recommendation: move defaults into a dedicated module or config constants.

## Nice-to-Have
- Progress bars use hardcoded colors that bypass the theme palette.  
  Reference: `src/arbiter/ui/progress.py:18-31`.  
  Recommendation: use theme style names or constants to keep UI palette cohesive.
- `build_q_distribution` eagerly materializes the cartesian product; this may be memory heavy when the ladder expands.  
  Reference: `src/arbiter/config.py:223-255`.  
  Recommendation: consider streaming/generator-based construction or compute lengths without allocating full lists.
