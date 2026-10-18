# CAT4D-Style Dynamic 3DGS PLY For Brush

This folder documents the dynamic Gaussian splat PLY format used by the
CAT4D website's embedded Brush viewer. Use this as a handoff for agents that
need to export our own dynamic 3DGS into a browser-loadable Brush asset.

Brush officially supports normal `.ply`, `.compressed.ply`, zipped splat
animations, and "a special ply that includes delta frames." CAT4D uses that
special delta-frame PLY. The CAT4D site loads one URL per scene and calls
`window.app.load_url(url)` on a Brush WASM `EmbeddedApp`.

## Goal

Write one binary little-endian PLY file containing:

1. A base/static Gaussian splat set for frame 0.
2. Per-frame quantized deltas for geometry at frames 1..F-1.

The file must preserve one stable Gaussian identity/order across all frames.
This format is not a general topology-changing animation format: splats should
not be born, removed, or reordered after frame 0.

## Observed CAT4D Example

Example file:

```text
https://storage.googleapis.com/realtime-nerf-360/cat4d/kling-forest-elf-fire_deform3dgs_pruned.ply
```

Observed header summary:

```text
ply
format binary_little_endian 1.0
comment Vertical Axis: z
element vertex 80411
...
property float opacity
element meta_delta_min_1 1
...
element meta_delta_max_1 1
...
element delta_vertex_1 80411
...
element meta_delta_min_63 1
element meta_delta_max_63 1
element delta_vertex_63 80411
end_header
```

This sample has 80411 splats, 64 total frames, one base frame, and 63 delta
frames. Its deployed Brush WASM contains the expected loader strings:
`meta_delta_min_`, `meta_delta_max_`, and `delta_vertex_`.

## Header Layout

Use ASCII PLY header text terminated by `end_header\n`, followed immediately by
binary payload.

Base splat element:

```text
ply
format binary_little_endian 1.0
comment Vertical Axis: z
element vertex {N}
property float x
property float y
property float z
property float rot_0
property float rot_1
property float rot_2
property float rot_3
property float scale_0
property float scale_1
property float scale_2
property float f_dc_0
property float f_dc_1
property float f_dc_2
property float f_rest_0
...
property float f_rest_44
property float opacity
```

Then, for every animation frame `k = 1..F-1`, append exactly this trio of PLY
elements:

```text
element meta_delta_min_{k} 1
property float x
property float y
property float z
property float rot_0
property float rot_1
property float rot_2
property float rot_3
property float scale_0
property float scale_1
property float scale_2
element meta_delta_max_{k} 1
property float x
property float y
property float z
property float rot_0
property float rot_1
property float rot_2
property float rot_3
property float scale_0
property float scale_1
property float scale_2
element delta_vertex_{k} {N}
property ushort x
property ushort y
property ushort z
property ushort rot_0
property ushort rot_1
property ushort rot_2
property ushort rot_3
property ushort scale_0
property ushort scale_1
property ushort scale_2
```

Finish with:

```text
end_header
```

Notes:

- The CAT4D example uses 45 `f_rest_*` values, matching standard degree-3 3DGS
  spherical harmonics: 15 non-DC coefficients times 3 color channels.
- If your model has only RGB/DC color, still write `f_rest_0..f_rest_44` as
  zeros for maximum compatibility with the CAT4D-style header.
- CAT4D's delta blocks only animate position, rotation, and scale. Color,
  opacity, and SH are static in this observed format. If you need dynamic
  color/opacity, use a `.zip` of full per-frame PLYs or extend Brush and the
  writer together.

## Binary Payload Order

After the header, write binary rows in the same order as the header elements:

1. `vertex`: `N` base rows.
2. For each frame `k = 1..F-1`:
   - `meta_delta_min_{k}`: one row of 10 float32 values.
   - `meta_delta_max_{k}`: one row of 10 float32 values.
   - `delta_vertex_{k}`: `N` rows of 10 uint16 values.

All numeric values are little-endian:

- `float` is IEEE `float32`, little-endian.
- `ushort` is unsigned 16-bit integer, little-endian.

Base row property order:

```text
x y z
rot_0 rot_1 rot_2 rot_3
scale_0 scale_1 scale_2
f_dc_0 f_dc_1 f_dc_2
f_rest_0 ... f_rest_44
opacity
```

Delta row property order:

```text
x y z
rot_0 rot_1 rot_2 rot_3
scale_0 scale_1 scale_2
```

## What To Store In The Delta Blocks

Use frame 0 as the base. For each later frame `k`, compute component-wise
deltas relative to frame 0:

```text
delta[k, splat, attr] = frame[k, splat, attr] - frame[0, splat, attr]
```

The 10 animated attributes are:

```text
x, y, z, rot_0, rot_1, rot_2, rot_3, scale_0, scale_1, scale_2
```

For each frame `k` and each attribute independently:

```text
min_attr = min(delta[:, attr])
max_attr = max(delta[:, attr])
q = round((delta - min_attr) / (max_attr - min_attr) * 65535)
q = clamp(q, 0, 65535)
```

Write the 10 `min_attr` values into `meta_delta_min_{k}`, the 10 `max_attr`
values into `meta_delta_max_{k}`, and each quantized `q` row into
`delta_vertex_{k}`.

Brush should reconstruct with:

```text
decoded_delta = min_attr + (q / 65535.0) * (max_attr - min_attr)
frame_value = base_value + decoded_delta
```

If `max_attr == min_attr`, write equal min/max values and write `q = 0` for all
splats for that attribute.

Important: the additive-delta interpretation is based on CAT4D's observed field
names and file organization. After downloading examples, validate by parsing
one example or by writing a tiny two-frame file with all-zero deltas. It should
render as a static splat in Brush.

## Quaternion And Scale Conventions

- Keep the same quaternion component order as your static 3DGS exporter uses
  for `rot_0..rot_3`. Do not reorder between base and deltas.
- Normalize quaternions before computing deltas when possible.
- If Brush rendering looks distorted, first check quaternion ordering and sign
  continuity. Quaternion `q` and `-q` represent the same rotation but produce
  huge component deltas; enforce temporal sign continuity before export.
- `scale_0..scale_2` should be the standard 3DGS log-scale values, not
  exponentiated radii, unless your static Brush PLY exporter already uses a
  different convention that you have verified.
- `opacity` should be the raw 3DGS opacity/logit value used by common 3DGS PLY
  exporters, not post-sigmoid alpha.

## Size Sanity Check

For the CAT4D-compatible degree-3 SH layout, each base vertex has:

```text
3 position + 4 rotation + 3 scale + 3 SH DC + 45 SH rest + 1 opacity
= 59 float32 values
= 236 bytes per splat
```

Each delta frame has:

```text
2 metadata rows * 10 float32 values = 80 bytes
N delta rows * 10 uint16 values = N * 20 bytes
```

Expected file size:

```text
header_bytes + N * 236 + (F - 1) * (80 + N * 20)
```

The observed CAT4D sample is about 120 MB for `N = 80411` and `F = 64`.

## Minimal Writer Pseudocode

```python
import numpy as np
import struct

ATTRS = ["x", "y", "z", "rot_0", "rot_1", "rot_2", "rot_3",
         "scale_0", "scale_1", "scale_2"]

def write_f32_row(f, values):
    f.write(struct.pack("<" + "f" * len(values), *values))

def write_u16_row(f, values):
    f.write(struct.pack("<" + "H" * len(values), *values))

def quantize_delta(delta_2d):
    # delta_2d shape: [N, 10]
    mn = delta_2d.min(axis=0).astype(np.float32)
    mx = delta_2d.max(axis=0).astype(np.float32)
    span = mx - mn
    q = np.zeros(delta_2d.shape, dtype=np.uint16)

    good = span > 1e-12
    q_float = np.zeros(delta_2d.shape, dtype=np.float32)
    q_float[:, good] = (delta_2d[:, good] - mn[good]) / span[good] * 65535.0
    q[:, good] = np.clip(np.rint(q_float[:, good]), 0, 65535).astype(np.uint16)
    return mn, mx, q

# frames_geom shape: [F, N, 10], ordered like ATTRS.
# base_full shape: [N, 59], ordered like the base vertex properties.
with open("scene_deform3dgs_pruned.ply", "wb") as f:
    f.write(make_header(N=base_full.shape[0], frames=frames_geom.shape[0]).encode("ascii"))

    for row in base_full.astype(np.float32):
        write_f32_row(f, row)

    base_geom = frames_geom[0]
    for k in range(1, frames_geom.shape[0]):
        delta = (frames_geom[k] - base_geom).astype(np.float32)
        mn, mx, q = quantize_delta(delta)
        write_f32_row(f, mn)
        write_f32_row(f, mx)
        for row in q:
            write_u16_row(f, row)
```

## Validation Checklist

Before calling the file done:

1. Header opens with `ply` and `format binary_little_endian 1.0`.
2. Header has `element vertex {N}` and one `delta_vertex_k {N}` for every
   non-base frame.
3. Every delta frame has exactly matching `meta_delta_min_k`,
   `meta_delta_max_k`, and `delta_vertex_k` names.
4. Binary payload size matches the size formula above.
5. A zero-delta two-frame test renders exactly like the base frame.
6. A small animated test with one translated splat moves in the expected
   direction.
7. Random decoded deltas round-trip close to the source values:
   `abs(decoded - original) <= span / 65535 + epsilon`.
8. The exported file loads in Brush locally or in a hosted web viewer.

## Browser Hosting Notes

For a CAT4D-like website:

```js
import init, * as bindings from "/brush-demo/brush_app.js";
await init({ module_or_path: "/brush-demo/brush_app_bg.wasm" });

const app = new bindings.EmbeddedApp("brush_canvas", "?zen=true");
app.load_url("https://your-storage.example/scene_deform3dgs_pruned.ply");
```

Host the `.ply` over HTTPS with CORS enabled for the website origin. Google
Cloud Storage, S3, or a static file server is fine. Use `Content-Type:
application/octet-stream`. Range support is useful for large files, but the
critical requirement is that the browser can fetch the file from the viewer
origin.

## Files In This Scratch Folder

- `cat4d_index.html`: local copy of the CAT4D page wiring the Brush viewer.
- `app.js`: CAT4D's `loadViewer(url)` helper.
- `brush_app.js` and `brush_app_bg.wasm`: deployed CAT4D Brush WASM bundle.
- `kling_forest_head_256k.ply`: first 256 KiB of one CAT4D example, enough to
  inspect the full header.
- `brush_README.md`: current Brush README at the time this note was written.
