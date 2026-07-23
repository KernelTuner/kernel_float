# Cache Modifiers

On Nvidia GPUs, each load or store instruction can be annotated with a cache modifier hint that tells the hardware how to treat the accessed cache lines.
For example, data that is only read once benefits from bypassing the cache entirely.
Kernel Float exposes these hints through the `kf::cache_modifier` enumeration.

## The `cache_modifier` enum

`kf::cache_modifier` mirrors CUDA's `__ldca`/`__ldcg`/`__ldcs`/`__ldlu`/`__ldcv` load intrinsics and `__stwb`/`__stcg`/`__stcs`/`__stwt` store intrinsics:

| Value                       | Meaning                                                     | Load intrinsic | Store intrinsic |
|-----------------------------|-------------------------------------------------------------|-----------------|------------------|
| `normal`                    | No hint, caching behavior is left to the compiler (default) | -               | -                |
| `cache_all` (alias `ca`/`wb`) | Cache at all levels, data is likely to be reused            | `__ldca`        | `__stwb`         |
| `cache_global` (alias `cg`) | Cache at the global (L2) level only, bypassing L1           | `__ldcg`        | `__stcg`         |
| `streaming` (alias `cs`)    | Streaming access, data is likely accessed only once         | `__ldcs`        | `__stcs`         |
| `uncached` (alias `cv`/`wt`) | Bypass caching entirely and always go to memory             | `__ldcv`        | `__stwt`         |
| `last_use` (alias `lu`)     | Last use; the cache line will not be reused (loads only)    | `__ldlu`        | -                |

Note that you can use both the short PTX-derived name (`ca`, `cg`, `cs`, `cv`, `wt`, `lu`)
or a more descriptive alias (`cache_all`, `cache_global`, `streaming`, `uncached`, `last_use`).

`last_use` has no store equivalent (there is no `__stlu` intrinsic), so using it with a store operation silently falls back to a plain, unmodified store.

## Using `read_aligned`/`write_aligned`

The modifier is an extra (optional, defaulted) template argument on `read_aligned` and `write_aligned`:

```cpp
float* pointer = ...;

// Read 4 elements as a streaming access. This avoids polluting the L1 cache.
kf::vec<float, 4> a = kf::read_aligned<4, kf::cache_modifier::streaming>(pointer);

// Write 4 elements, explicitly bypassing L1 (cache at the L2/global level only).
kf::write_aligned<4, kf::cache_modifier::cache_global>(pointer, a);
```

Both `read_aligned` and `write_aligned` place the modifier right after the alignment, so `N` (the number of elements) does not need to be repeated: it defaults to the given alignment, as shown above.

Calling `read_aligned`/`write_aligned` without explicitly specifying a modifier (as shown in the [memory operations guide](memory.md)) is equivalent to explicitly passing `cache_modifier::normal`.

## Using `vector_ptr`/`access_policy`

The `cache_modifier` can also be combined with `vector_ptr` by using the alias `cache_vec_ptr`:

```cpp
float* pointer = ...;

// Create the vector pointer with streaming access
kf::cache_vec_ptr<kf::cache_modifier::streaming, float, 4> a = kf::make_vec_ptr<4>(pointer);

// Read data using the stream access policy
kf::vec<float, 4> v = a[0];

// Write data using the stream access policy
a[0] = 2 * v;
```


## Fallback behavior

Cache modifiers are a *hint*, not a guarantee. Kernel Float will attempt to emit the correct instruction if possible, 
but will fall back to regular load/store if not possible. There are several cases:

* The code is not being compiled for a CUDA device (e.g., host or HIP). The cache intrinsics are CUDA-specific.
* The requested modifier is `normal` (or `last_use` for a store, since it has no store equivalent).
* The data type being accessed has a size that does not match the supported cache intrinsics (1, 2, 4, 8, or 16 bytes). 

In these cases, the cache hint is ignored and a regular load/store instruction is emitted.

