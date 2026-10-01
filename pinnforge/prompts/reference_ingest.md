# Reference ingest

The operator has supplied a reference solution for a new PINNForge task, and
you turn it into the task's **scoring truth**: every physics-informed network
submitted to the task will be scored by its relative L2 distance to what you
produce. Get the numbers across exactly; do not smooth, resample or "fix" them
unless the task description below asks for it.

## What the task is

{intake}

## The operator's data

It has been copied to `{user_data}` (a file or a directory). It may be `.npy`,
`.npz`, `.csv`, `.mat`, `.h5`/`.hdf5`, `.vtu`/`.vtk`, or something else —
inspect it and read it with whatever it needs. It may also be the exact
solution as a formula (a `.py` or `.txt`): then choose the points yourself, as
a task author would — a tensor-product grid over the whole domain, boundaries
included, on the order of 100–300 points per dimension and fewer along time —
and evaluate the formula there in float64. Python for this work:
`{python}` — an environment of your own; install readers into it with

    {uv} pip install --python {python} <package>

(`scipy` reads `.mat`, `h5py` reads HDF5, `meshio` reads VTK). Never install
into the workspace's `.venv`, and never read another task's `private/`.

Work in `{workdir}`. Your time limit is **{budget_min} minutes**.

## What you must leave behind

You have seen the answers, so nothing you write may reach the task package or
its author except the grid's coordinates: the framework checks that
`eval_grid.csv` is the only file you wrote outside `{private_dir}`, and hands
the author nothing of yours but the grid's shape and column names — which it
reads from the files themselves, not from your account of them. Name the
grid's columns plainly (`x`, `y`, `t`, `phi`): short identifiers — a letter or
`_`, then letters, digits or `_`, at most 32 characters, no runs of three or
more digits. A header is public too, and one that is not plain is rejected;
so is a grid any of whose columns holds the reference.

1. **`{task_dir}/eval_grid.csv`** — public: one row per point that will be
   scored, a header line naming the coordinate columns in the order a
   network takes them (e.g. `x,y` or `t,x` or `t,x,y`), then the numbers as
   plain decimals — `np.savetxt(path, grid, delimiter=",", header="x,y",
   comments="", fmt="%.17g")` — never Python reprs such as `np.float64(0.5)`;
   the framework rejects a grid `np.loadtxt` cannot read.
   Keep every point of the data unless it is plainly not part of the domain
   (NaN rows, ghost cells). If a mask is needed later (say, points inside a
   body), add it as an extra column *after* the coordinates (e.g. `phi`, or a
   0/1 flag) — it must be computable from the public grid alone.
2. **`{private_dir}/ref_values.npy`** — float64, shape `(rows, C)`, one row per
   grid row in the same order, one column per solution field. Finite
   everywhere.
3. **`{workdir}/ingest.json`** —

       {"grid_columns": ["x", "y"], "input_columns": 2,
        "ref_columns": ["u", "v", "p"], "rows": 65536,
        "source": "what the data is, as far as you can tell",
        "notes": "anything the task author must know"}

   `input_columns` is how many leading grid columns are coordinates (a
   whole number); `ref_columns` names each column of `ref_values.npy` in
   order, with identifiers as plain as the grid's. `source` and `notes` are
   for the operator only; they stay in the private directory.
4. **`{workdir}/report.md`** — what the data holds, what you kept and dropped,
   and the checks you made: the boundary and initial conditions of the task
   description hold on the data (to what tolerance), the values are finite
   and of a plausible magnitude, nothing is duplicated.

If the data cannot be the reference for the task as described — the wrong
equation, the wrong domain, missing fields — stop, say exactly why in
`report.md` and in your final reply, and leave the outputs unwritten.

End with a two-line reply: rows and columns written, and the largest boundary
condition violation you measured.
