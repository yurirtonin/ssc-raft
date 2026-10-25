# SSC-RAFT: Reconstruction Algorithms for Tomography

---

# LOCAL FORK NOTES — systematic half-pixel shift fix (2026-08-12)

**This checkout diverges from upstream `cnpem/ssc-raft`.** Three CUDA kernels were
patched to fix the systematic shift of
[issue #3](https://github.com/cnpem/ssc-raft/issues/3). Everything in this section
describes local changes only; the rest of the README is upstream.

## The bug

Every index lookup in the parallel-beam CUDA kernels truncated towards zero
(`(int)v`) instead of rounding (`(int)(v + 0.5f)`). Truncation biases each lookup by
half a sample. In object space that is a rigid `(0.5, 0.5)` voxel translation of the
volume, and a rigid translation seen through a projection is exactly
`a·cos θ + b·sin θ` — a first harmonic of amplitude `0.5·√2 = 0.707 px`.

`raft_fbp` carried a **second, independent** half-pixel error: its object grid put
the rotation axis at `n/2` while the forward projector uses `(n-1)/2`. The two
errors added, giving the ≈1.4 px wave reported in the issue.

Because the bias lives in the operators and not in the data, it appears even for a
perfect unshifted sinogram, which is why an iterative reprojection alignment
re-measured and re-applied it on every pass and never converged — `Proj(Reproj(A)) ≠ A`.

## What was changed

All under `cuda/src/geometries/parallel/`:

| file | change |
| --- | --- |
| `forward_methods/radon_rt.cu` | `Radon_RT_version_sscRadon`: round instead of truncate; `(x+1)` → `(x+ax)`, `(y+1)` → `(y+ay)` (the hard-coded `1` was a latent bug, masked only because the Python wrapper hard-codes `a = 1.0`) |
| `reconstruction_methods/fbp.cu` | `FBPBackProjectionRT`: `xmin`/`ymin` centred on `(n-1)/2` instead of `n/2`, matching the projector. The detector term needed no change — `(t-tmin)/dt` with `tmin = -pixel*nrays/2` already equals `(n-1)/2` centring with rounding folded in |
| `reconstruction_methods/kernels/em_RT_kernel.cu` | same rounding fix at all six sites (two forward kernels, three backprojections); `raft_em` has the bug in its own kernels |

Each site carries a comment referencing the issue so it does not get "tidied" away.

## Measured effect

Perfect sinogram, no shifts, one alignment iteration; wave amplitude is the fitted
first harmonic `c₀ + a·cos θ + b·sin θ` of the recovered horizontal shift:

| build | wave amplitude | dx RMS | round trip `fbp(radon)` vs truth |
| --- | --- | --- | --- |
| upstream | 1.372 px | 0.971 px | 0.9659 |
| + `radon_rt.cu` fix | 0.683 px | 0.484 px | 0.9825 |
| **+ `fbp.cu` fix (current)** | **0.019 px** | 0.019 px | **0.9904** |

A 73× reduction. `raft_em` went from 0.324 px to 0.076 px. Cross-checking with an
independent projector (ASTRA `parallel3d`) gives 0.010 px, confirming projector and
reconstructor now agree.

Diagnosis method, if it needs repeating: reimplement `Radon_RT_version_sscRadon` in
numpy (it reproduces the GPU output to 0.4 % RMS), measure the per-angle detector
offset against ASTRA, and fit the first harmonic. The truncating version returns
`a = -0.488, b = +0.488` — ∓½ voxel, the truncation itself.

## ⚠️ Things to be aware of

1. **Reconstructions moved by half a pixel.** The `fbp.cu` centring change shifts
   every `raft_fbp` output by `(0.5, 0.5)` px relative to the old behaviour. This is
   the correct position — it is what makes the operator pair consistent — but **any
   rotation-axis offset calibrated against the old build must be re-checked**, and
   results are not bit-comparable with anything reconstructed before this date.

2. **Not all code paths are covered.** Only the parallel-beam RT paths were audited
   and fixed. The following may carry the same truncation and were **not** measured:
   - `reconstruction_methods/bst.cu` (the `dic['method'] = 'BST'` branch)
   - `forward_methods/radon_fst.cu`
   - the whole `geometries/conebeam/` tree
   - `Radon_RT_version_Pyraft_Giov`, an unused kernel in `radon_rt.cu`, still truncates

3. **There is still a convergence floor.** The residual creeps up slowly across
   iterations (0.019 → 0.074 px RMS over 5 passes), so `threshold = 1e-2` in an
   iterative reprojection alignment will still never be reached. Use ≈`5e-2` as a
   realistic target. Going below that needs bilinear interpolation in the projector
   instead of nearest-neighbour sampling — a larger change, not done here.

4. **`raft_em` is improved but not clean** (0.076 px vs 0.019 px for `raft_fbp`). It
   is iterative with its own convergence behaviour; prefer `raft_fbp` when the
   sub-pixel budget matters.

5. **This is not upstream.** Re-cloning, `git checkout` of these files, or a
   `pip install sscRaft` will silently restore the bug.

## Rebuilding after further CUDA edits

`make` and `cmake --build` **do not work** in this tree: `_skbuild`'s cached
`CMAKE_MAKE_PROGRAM` points at a `ninja` inside a deleted pip build environment.
Until the tree is regenerated, compile and link by hand:

```bash
cd /home/yrossitonin/code/ssc-raft/_skbuild/linux-x86_64-3.14/cmake-build
R=/net/roentgen/home/yrossitonin/code/ssc-raft

# 1. recompile each changed .cu (flags copied from build.ninja)
nvcc -forward-unknown-to-host-compiler -Draft_EXPORTS \
  -I$R/cuda/inc -I$R/cuda/inc/common -I$R/cuda/inc/common10 \
  -isystem /usr/local/cuda-12.3/targets/x86_64-linux/include \
  --expt-extended-lambda -O3 -DNDEBUG -std=c++17 \
  "--generate-code=arch=compute_70,code=[compute_70,sm_70]" \
  -Xcompiler=-fPIC -use_fast_math \
  -x cu -c $R/cuda/src/geometries/parallel/forward_methods/radon_rt.cu \
  -o CMakeFiles/raft.dir/cuda/src/geometries/parallel/forward_methods/radon_rt.cu.o

# 2. relink all 32 objects
grep '^build libraft.so:' build.ninja | tr ' ' '\n' | grep '\.cu\.o$' > /tmp/objs.txt
nvcc -forward-unknown-to-host-compiler --expt-extended-lambda -O3 -DNDEBUG \
  "--generate-code=arch=compute_70,code=[compute_70,sm_70]" \
  -shared -Xcompiler=-fPIC -Xlinker -soname=libraft.so -o libraft.so \
  $(tr '\n' ' ' < /tmp/objs.txt) \
  /usr/local/cuda-12.3/targets/x86_64-linux/lib/libcufft.so \
  /usr/local/cuda-12.3/targets/x86_64-linux/lib/libcublas.so \
  /usr/local/cuda-12.3/targets/x86_64-linux/lib/libcublasLt.so \
  /usr/local/cuda-12.3/targets/x86_64-linux/lib/libculibos.a \
  -lcudadevrt -lcudart_static -lrt -lpthread -ldl

# 3. install where Python actually loads it from
cp libraft.so /home/yrossitonin/envs/raft/lib/python3.14/site-packages/sscRaft/lib/libraft.so
cp libraft.so ../cmake-install/sscRaft/lib/libraft.so
```

`sscRaft/rafttypes.py` loads `lib/libraft.so` **relative to its own directory**, so
step 3 is what actually makes a rebuild take effect. The copy into `cmake-install/`
is bookkeeping only; the live library is the one in
`envs/raft/lib/python3.14/site-packages/sscRaft/lib/`.

⚠️ **Do not run Python with this repo as the working directory.** The checkout has
no `sscRaft/lib/` of its own, so the repo's `sscRaft/` package shadows the installed
one on `sys.path` and the import dies with

```
OSError: .../ssc-raft/sscRaft/lib/libraft.so: cannot open shared object file
```

Run from anywhere else, or copy `libraft.so` into `sscRaft/lib/` if you want the
checkout to be importable in place — in which case that copy also has to be kept
current on every rebuild.

## Reverting

The pre-fix binary is kept at
`_skbuild/linux-x86_64-3.14/cmake-build/libraft.so.orig`. To A/B:

```bash
cp _skbuild/linux-x86_64-3.14/cmake-build/libraft.so.orig \
   /home/yrossitonin/envs/raft/lib/python3.14/site-packages/sscRaft/lib/libraft.so
```

Source-side: `git checkout cuda/src/geometries/parallel/` then rebuild as above.

## Related

The iterative reprojection alignment that exposed this lives in
`phdtoolbox/phdtoolbox/alignment.py`, which also gained an ASTRA forward-projector
backend (`radon_method='astra'`) used as the independent reference for the numbers
above. ASTRA rotates the opposite way to `radon_RT`, so that backend negates the
angles internally — do not "fix" that sign.

---

## Authors

* Eduardo X. Miqueles, LNLS/CNPEM
* Paola Ferraz, LNLS/CNPEM

## Contributors

* Larissa M. Moreno, LNLS/CNPEM
* João F. G. de Albuquerque Oliveira, LNLS/CNPEM
* Alan Zanoni Peixinho, LNLS/CNPEM
* Yuri Rossi Tonin, Georg-August-Universität Göttingen

## Past contributors

* Otávio M. Paiano
* Gilberto Martinez Jr.
* Giovanni Baraldi
* Janito Vaqueiro Ferreira Filho
* Fernando S. Furusato
* Matheus F. Sarmento, LNLS/CNPEM
* Nikoley Koshev
* Elias Helou

## Acknowledgements

We would like to acknowledge the Brazilian Ministry of Science, Technology, and Innovation MCTI for supporting this work through the Brazilian Center for Research in Energy and Materials (CNPEM).
We want to thank Petrobras (Manhattan Project) - ID 2021/00018-3 for funding part of this project - and Mogno beamline from Sirius, for many fruitful discussion about the reconstruction strategies.
We also thank FAPESP/CEPID 2013/07375-0 for funding Nikolay Koshev, and FAPESP 2016/16238-4 for funding Gilberto Martinez Jr.

### Contact

Sirius Scientific Computing Team: [gcc@lnls.br](malito:gcc@lnls.br)

## Documentation

The package documentation can be found on the GCC website [https://gcc.lnls.br/ssc/ssc-raft/index.html](https://gcc.lnls.br/ssc/ssc-raft/index.html) inside the CNPEM network.
Also, the `HTML` documentation can be found in the source directory `./docs/build/index.html` and can be opened with your preferred brownser.

## Install

This package uses `C`, `C++`, `CUDA` and `Python3`.
See bellow for full requirements.

The library `sscRaft` can be installed with form the source code or by `pip`/`git` if inside the CNPEM network.

### GITHUB

One can clone our public [github](https://github.com/cnpem/ssc-raft/) repository and install the latest version by:

```bash
git clone https://github.com/cnpem/ssc-raft.git 
cd ssc-raft
make clean && make
```

For a specific version, one can use:

```bash
    git clone https://github.com/cnpem/ssc-raft.git --branch v<version> --single-branch
    cd ssc-raft 
    make clean && make
```

The `<version>` is the version of the `sscRaft` to be installed. Example, to install version 3.1.1

```bash
    git clone https://github.com/cnpem/ssc-raft.git --branch v3.1.1 --single-branch
    cd ssc-raft 
    make clean && make
```

### Source code from Zenodo

The source code can be downloaded from [zenodo website](https://zenodo.org/) under the DOI:[10.5281/zenodo.10988342](https://doi.org/10.5281/zenodo.10988342).
On the left panel, one can find
the available versions. Select the version want and download the ``ssc-raft.tar.gz`` with the source files, one can decompress by

```bash
    tar -xvf ssc-raft.tar.gz
```

To compile the source files, enter the following command inside the folder

```bash
    make clean && make
```

### PIP

---
> **Warning:** This installation option is available only inside the CNPEM network.
---

One can install the latest version of sscRaft directly from the `pip server`

```bash
    pip install sscRaft==<version> --index-url https://gitlab.cnpem.br/api/v4/projects/1978/packages/pypi/simple

```

Where `<version>` is the version number of the `sscRaft`

```bash
    pip install sscRaft==3.0.3 --index-url https://gitlab.cnpem.br/api/v4/projects/1978/packages/pypi/simple
```

### GITLAB

---
> **Warning:** For this installation option is available only inside the CNPEM network.
---

One can clone our [gitlab](https://gitlab.cnpem.br/) repository and install the latest version by:

```bash
git clone https://gitlab.cnpem.br/GCC/ssc-raft.git 
cd ssc-raft
make clean && make
```

For a specific version, one can use:

```bash
    git clone https://gitlab.cnpem.br/GCC/ssc-raft.git --branch v<version> --single-branch
    cd ssc-raft 
    make clean && make
```

The `<version>` is the version of the `sscRaft` to be installed. Example, to install version 3.1.1

```bash
    git clone https://gitlab.cnpem.br/GCC/ssc-raft.git --branch v3.1.1 --single-branch
    cd ssc-raft 
    make clean && make
```

## Memory

Be careful using GPU functions due to memory allocation.

## Requirements

Before installation, you will need the following packages installed:

* `CUDA >= 10.0.0`
* `C`
* `C++`
* `Python >= 3.8.0`
* `PIP`
* `libcurl4-openssl-dev`

This package supports nvidia ``GPUs`` with capabilities ``7.0`` or superior and a compiler with support to ``c++17``.

The following modules are used:

* `CUBLAS`
* `CUFFT`
* `PTHREADS`
* `CMAKE>=3.10`

The following `Python3` modules are used:

* `scikit-build>=0.17.0`
* `setuptools>=64.0.0`
* `numpy`
* `scikit-image >=0.19.3`
* `scipy`
* `matplotlib`
* `SharedArray`
* `h5py`

## Uninstall

To uninstall `sscRaft` use the command, independent of the instalation method,

```bash
    pip uninstall sscRaft 
```
