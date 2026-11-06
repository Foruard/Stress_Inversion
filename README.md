# singleFaultInversion

A fault stress inversion and stress-map toolkit driven by focal mechanisms (fault slip vectors).
Several faults can be supplied at once; each fault is inverted independently.

> Chinese version: [README.md](README.md)

The project contains exactly two Python files:

| File | Purpose |
| --- | --- |
| `singleFaultInversion.py` | Core library: stress tensor algebra, 5-D stress space, inversion grid, stress maps, fault-activation risk map |
| `main.py` | Example driver: Bohai focal mechanisms → World Stress Map comparison → figures |

All comments and docstrings in the source code are in English.

---

## Table of contents

1. [Overview](#1-overview)
2. [Repository layout](#2-repository-layout)
3. [Requirements](#3-requirements)
4. [Quick start](#4-quick-start)
5. [Data files](#5-data-files)
6. [Conventions](#6-conventions)
7. [Method](#7-method)
8. [API reference](#8-api-reference)
9. [main.py workflow](#9-mainpy-workflow)
10. [Recipes](#10-recipes)
11. [Performance and known limitations](#11-performance-and-known-limitations)
12. [FAQ](#12-faq)
13. [References](#13-references)

---

## 1. Overview

- **Stress tensor algebra** – build a normalised reduced stress tensor from four parameters (α, β, γ, R) and evaluate the normal traction, shear traction, Mohr–Coulomb criterion and slip vector on a fault plane.
- **5-D stress space** – map stress tensors onto the 5-D unit sphere of Sato & Yamaji (2006), with conversion both ways between a 5-vector and a 3×3 tensor.
- **Inversion** – for each fault, search a 4-D grid (α, β, γ, R) for stress states whose predicted slip direction lies within a tolerance of the observed slip, then average them with a distance-weighted iteration (attractor).
- **Interpretation** – eigendecomposition gives σ1/σ2/σ3 and the stress ratio Φ; the principal axis closest to vertical determines the mechanism type (Normal / Strike_Slip / Reverse) and the azimuth.
- **Figures** – stress direction maps with mechanism symbols, World Stress Map comparison, σ1/σ3 polar plots, beach balls and a fault-activation risk map.
- **Database matching** – compare two candidate nodal-plane solutions with the nearest World Stress Map station and pick the more plausible one from its REGIME and azimuth.

## 2. Repository layout

```
E:/code
├─ singleFaultInversion.py      # core library
├─ main.py                      # example driver
├─ README.md                    # documentation (Chinese)
├─ README_EN.md                 # this file (English)
├─ data/
│  ├─ focal_data.csv            # focal mechanism catalogue (124 events)
│  ├─ WSM_Database_2025.csv     # World Stress Map database (~100k stations)
│  ├─ mesh_360_180_10.csv       # pre-computed 4-D grid (~716k rows, 51 MB)
│  ├─ normal_stress.csv         # stress state table [α, β, γ, R], sample data
│  └─ reverse_stress.csv        # stress state table [α, β, γ, R], sample data
├─ background/
│  └─ Satellite_map.png         # satellite basemap (6931 × 6935, RGBA)
└─ plate/
   └─ boundary.csv              # discrete plate / tectonic boundary points
```

All relative paths (`./data/...`, `./background/...`, `./plate/...`) are resolved against the
**current working directory**, so run the scripts from inside `E:/code`.

## 3. Requirements

- Python 3.8 or newer (developed with Python 3.11)
- Third-party packages:

| Package | Used for |
| --- | --- |
| `numpy` | matrices, eigendecomposition, vector algebra |
| `pandas` | reading focal mechanisms / WSM database / boundary points |
| `matplotlib` | all plotting |
| `scipy` | `scipy.linalg.eigh`, `RBFInterpolator` spatial interpolation |
| `tqdm` | progress bars for inversion and drawing |

```bash
pip install numpy pandas matplotlib scipy tqdm
```

Fonts: the code sets `plt.rcParams['font.family'] = 'Times New Roman'`. If that font is not
installed, matplotlib falls back to its default and emits a warning; the computations are unaffected.

## 4. Quick start

```bash
cd E:/code
python main.py
```

Running `main.py` executes, in order: a synthetic test → the Bohai focal mechanism inversion →
the World Stress Map comparison → figures Fig4, Fig8–Fig12. The script **does not save any image**
by itself; it only keeps the `fig` objects (see [section 10](#10-recipes)).

Minimal example with a single fault:

```python
import numpy as np
import singleFaultInversion as sfi

run = sfi.solution()                        # default 4-D grid (~60k stress states)

# Fault parameters [strike, dip, slip] in radians; use a 2-D array even for one fault
faults = np.deg2rad([[30.0, 45.0, -90.0]])

stress = run.peakFindInversion(faults, limitInDeg=2)[0]   # states with misfit <= 2 deg
Phi, eigvecs, attractor = run.calAttractor(run.vector5D(stress))
azi, aziType = run.fromEigvecsCalAzi(eigvecs.reshape(1, 3, 3))

print("Phi =", Phi, "azimuth =", azi, "type =", aziType)
```

> Note: the default grid is built **when the module is imported** (about 60k states), so the first
> `import singleFaultInversion` takes a few seconds up to about ten seconds. This is expected.

## 5. Data files

### 5.1 `data/focal_data.csv` — focal mechanisms

| Column | Meaning |
| --- | --- |
| (first, unnamed) | row index |
| `name` | event identifier |
| `strike1, dip1, slip1` | primary nodal plane: strike / dip / rake (degrees) |
| `strike2, dip2, slip2` | conjugate nodal plane: strike / dip / rake (degrees) |
| `lat, lon` | epicentre latitude and longitude (degrees) |
| `depth` | depth (km); not used by the current workflow |

`main.py` reads it with `pd.read_csv(..., index_col=1).values[:, 1:10]`: the `name` column becomes the
index and the unnamed row-index column is skipped, leaving 9 values per event
`[strike1, dip1, slip1, strike2, dip2, slip2, lat, lon, depth]`, which the script then slices.

### 5.2 `data/WSM_Database_2025.csv` — World Stress Map

Required columns (names must be upper case):

| Column | Meaning |
| --- | --- |
| `LAT`, `LON` | station latitude / longitude (degrees) |
| `AZI` | maximum horizontal stress azimuth (degrees; axial data) |
| `QUALITY` | quality rank; only `A/B/C` are used |
| `REGIME` | stress regime: `NF` normal, `TF` reverse, `SS` strike-slip, `U/NS/TS` no usable type |

The file contains many other columns (`SITE`, `DEPTH`, `METHOD`, ...); only the five above are read.

### 5.3 `plate/boundary.csv` — boundary points

The first column is an index, followed by **longitude and latitude** in degrees. Fully empty rows and
columns are dropped on load, leaving an `N × 2` array.

### 5.4 `background/Satellite_map.png` — basemap

- The image must match `area = [minLon, maxLon, minLat, maxLat]` exactly: **top left = (minLon, maxLat),
  bottom right = (maxLon, minLat)**.
- `locTrans` applies a spherical Mercator conversion in the y direction, so the basemap should be in
  Mercator projection; with a plate carrée image the latitude axis is stretched slightly.
- The current basemap is 6931 × 6935 pixels. The risk map converts it to floating point and interpolates
  every pixel, which needs a lot of memory — see [section 11](#11-performance-and-known-limitations).

### 5.5 `mesh_360_180_10.csv` / `normal_stress.csv` / `reverse_stress.csv`

All three are `N × 4` tables of 4-D grid points / stress states with columns `[α, β, γ, R]`, angles in
radians and `R` a ratio in 0–1. They can be passed directly to `run.setMeshgrid(...)` or kept as
archived inversion results. `main.py` mentions `mesh_360_180_10.csv` only in a comment.

## 6. Conventions

| Item | Convention |
| --- | --- |
| Axes | direction vectors: east `[1, 0, 0]`, north `[0, -1, 0]`, up `[0, 0, 1]` |
| Azimuth `azi` | clockwise from north, in degrees; treated as axial data (0° ≡ 180°), differences limited to 0–90° |
| `strike` | clockwise from north to the strike direction, `0 ≤ strike ≤ 2π` |
| `dip` | downward from the horizontal plane to the fault plane, `0 ≤ dip ≤ π/2` |
| `slip` (rake) | angle between the strike direction and the hanging-wall slip direction, **counter-clockwise positive**, `-π ≤ slip ≤ π` |
| α, β, γ | intrinsic z-y-z Euler angles, `0 ≤ α ≤ 2π`, `0 ≤ β ≤ π/2`, `0 ≤ γ ≤ π` |
| R (Φ) | stress shape ratio `R = (σ1 - σ2) / (σ1 - σ3)`, `0 ≤ R ≤ 1` |
| Principal stresses | sorted descending: `σ1 ≥ σ2 ≥ σ3` |
| Sign | compression positive; the tensor is deviatoric, so adding an isotropic term does not change the results |

## 7. Method

```
focal mechanisms (strike, dip, slip)      WSM database (LAT, LON, AZI, QUALITY, REGIME)
        │                                          │
        │  4-D grid (α, β, γ, R)                    │
        ├─ fitness: angle between predicted shear and observed slip
        │       └─ keep grid points with angle <= limitInDeg → candidate stress states
        │
        ├─ 5-D vectors + iterative weighted mean (attractor) → mean stress tensor
        ├─ eigendecomposition → σ1/σ2/σ3, Φ
        ├─ mechanism type from the principal axis closest to vertical; azimuth azi / aziType
        │
        └──────────────► matchAziWithDatabase: nearest station, choose one of the two candidates
                                   │
                                   └─ figures: stress map / comparison / risk map
```

## 8. API reference

`import singleFaultInversion as sfi`. Inheritance chain:
`solution → picture → geology → stressParameter`, so a single `run = sfi.solution()` exposes every
method of the four classes.

### 8.1 Helper functions

| Function | Description |
| --- | --- |
| `is_2d_array(arr)` | True for a 2-D array or a list of lists |
| `CalAngle(v1, v2)` | rotation angle that takes `v1` onto `v2` (counter-clockwise positive, in `[0, 2π)`) |
| `add_right_cax(ax, pad, width)` | append a colorbar axes of equal height on the right of `ax` |

### 8.2 `stressParameter`

```python
stressParameter(grid=[α, β, γ, R], fault=[strike, dip, slip], friction=0.6)
```

| Method | Description |
| --- | --- |
| `setGrid(newGrid)` / `setGridInDeg(newGrid)` | set the stress parameters (radians / degrees) |
| `setFault(newFault)` / `setFaultInDeg(newFault)` | set the fault geometry (radians / degrees) |
| `setFriction(mu)` / `getStress()` / `getFault()` | friction coefficient and getters |
| `Rx/Ry/Rz(rad)`, `Rzyz(a,b,g)`, `Rccc(order,a,b,c)` | elementary and composite rotation matrices |
| `invariantValues(S)` | the three stress invariants |
| `normalizationIn5D()` | normalised stress tensor used for the 5-D mapping (with axes rotation) |
| `vector5D(stress=None)` | one state → `(5,)`; an `(n,4)` array → `(n,5)` stress vectors |
| `distance5D(v1, v2)` | Euclidean distance between two 5-vectors |
| `s5_to_stress_tensor(v5, normalize=False)` | 5-vector → 3×3 tensor |
| `s5Array_to_stress_tensor(v5, normalize=False)` | `(n,5)` → `(n,3,3)` |
| `strikeVector()` / `surfaceVector()` / `faultNormalVector()` | strike / fault-plane / hanging-wall normal unit vectors |
| `directionCosine(vec, isDefaultRotation=True, order=(), rotation=())` | direction cosines in the rotated frame |
| `shearStressVector(..., R=0.5, isUnit=True)` | shear traction direction in the principal axes frame |
| `shearStressVectorTransform(...)` | the same direction rotated back to the original frame |
| `shearStress()` / `normalStress()` | shear / normal traction (with the `k1`, `k2` factors) |
| `MC()` | Mohr–Coulomb value: `|τ| - μ|σn|` |
| `slipVector()` | hanging-wall slip vector from the fault geometry |

### 8.3 `meshgrid`

```python
meshgrid(N1=100, N2=100, N3=5, isRad=True)
```

| Method | Description |
| --- | --- |
| `setN1/setN2/setN3` | number of directions / rotations per direction / R values per rotation |
| `FibonacciGrid(isSimplify=False)` | Fibonacci points on the upper hemisphere |
| `normalFibonacciGrid(isSimplify=False)` | Fibonacci points on the lower hemisphere |
| `rotationGrid(point)` / `CartesiusToEuler(point)` / `RGrid(euler)` | expand one point into α/β/γ/R |
| `fourDimensionalGrid(isSimplify=False)` | build the complete 4-D grid, returns `(n,4)` |

The defaults `N1=100, N2=100, N3=5` give **60,600** stress states.

### 8.4 `geology`

```python
geology(platesBoundary='./plate/boundary.csv')
```

| Method | Description |
| --- | --- |
| `getBoundary()` / `getREarth()` | boundary array / Earth radius (6371 km) |
| `Mercator([lon, lat])` | lon/lat → spherical Mercator plane coordinates (km) |
| `locTrans(LonAndLat, pictureX, pictureY, area)` | lon/lat → pixel coordinates of the basemap |
| `matchAziWithDatabase(filePath, inputData, area=(110,131,29,46))` | choose one of two candidates from the nearest WSM station |

`matchAziWithDatabase` expects `inputData` of shape `(n, 6)` with rows
`[azi1, aziType1, azi2, aziType2, lat, lon]`. It returns an object array of shape `(n, 3)`:
`[selectedAzi, selectedAziType, chosen]`, where `chosen = 0` means the first candidate and `1` the second.

### 8.5 `picture`

```python
picture(path='./', lineLength=0.5, color='jet', figureName="", lineWidth=1,
        pointSize=20, dpi=600, rlabel_fontsize=25, thetalabel_fontsize=25,
        cbar_labelsize=25)
```

| Method | Description |
| --- | --- |
| `setPath/setLineLength/setColor/setFigureName/setLineWidth/setPointSize/setDpi` | plotting defaults |
| `setRlabel_fontsize/setThetalabel_fontsize/setCbar_labelsize` | font sizes |
| `save(fig, pictureFormat='.png', pictureName='picture')` | save a figure into `path` |
| `polar_to_cartesian` / `cartesian_to_polar` / `vectorAddition` | polar ↔ Cartesian conversion and vector addition |
| `calculate_circle_from_three_points` / `line_circle_intersection` | geometry helpers for beach balls |
| `stressMapPlus(...)` | stress direction map (basemap + segments + mechanism symbols + boundary) |
| `stressMapWithDatabase(...)` | WSM (white) versus inversion (black) comparison |
| `FaultsActivationRiskMap(...)` | fault activation risk map |
| `apmMap(lon, lat, apm, background, area)` | absolute plate motion arrows |
| `visualization(stress, ...)` | σ1 / σ3 polar plots coloured by Φ |
| `beachball(faults)` | beach balls |
| `strikeAndDip(stress)` | strike/dip of the principal axes and stress ratio |

Signatures of the main plotting routines:

```python
stressMapPlus(lon, lat, azi, aziType, background, area,
              isTitle=False, boundaryPoints=None, boundaryColor='yellow',
              boundarySize=8, boundaryLine=False, boundaryLineWidth=0.8)

stressMapWithDatabase(lon, lat, azi, aziType, background, area, csvFile,
                      isTitle=False, numTicks=5, boundaryPoints=None,
                      boundaryColor='yellow', boundarySize=15,
                      boundaryLine=False, boundaryLineWidth=0.4)

FaultsActivationRiskMap(eigvecs, Phi, focalData, background, area,
                        isTitle=False, smoothing=0.5, numTicks=5,
                        colorbarPad=0.15, terrainStrength=0.45,
                        saturationScale=1.2, boundaryPoints=None,
                        boundaryColor='yellow', boundarySize=15,
                        boundaryLine=False, boundaryLineWidth=0.8)
```

- `azi` is in degrees and `aziType ∈ {'Normal', 'Reverse', 'Strike_Slip'}`; `stressMapWithDatabase`
  additionally accepts REGIME aliases such as `U`, `N`, `R`, `SS` (unknown types are passed as `'U'`).
- Symbols: Normal = open red circle, Reverse = open blue square, Strike_Slip = open green triangle;
  squares and triangles are rotated with the segment direction.
- `boundaryPoints` is an `(N, 2)` array of `[longitude, latitude]`; `[nan, nan]` entries break the
  boundary line into segments.
- `FaultsActivationRiskMap` expects `focalData` columns `[strike, dip, slip, lat, lon]` in degrees and
  returns `(fig, risk)`.
- `visualization` takes a sequence of `[α, β, γ, Φ]` states in radians.

### 8.6 `solution`

```python
solution(meshgrid=meshgrid(100, 100, 5).fourDimensionalGrid())
```

| Method | Description |
| --- | --- |
| `setMeshgrid(grid)` / `getMeshGrid()` / `getMeshgridLength()` | replace / read the 4-D grid |
| `fitness(gridNumber, faultsParameter)` | fitness of one fault at one grid point: `(1 - cosθ)/2`, 0 = perfect fit |
| `peakFindInversion(faults, limitInDeg=2, isMC=False)` | list of `(n,4)` candidate stress states, one entry per fault |
| `calAttractor(vectorArray)` | iterate the attractor, returns `(Phi, eigvecs, attractor)` |
| `attractor(tensors, oldAttractor, is5DVector=False)` | distance-weighted mean |
| `stressDifference(A, B)` / `combinedVectorAverageLength(vectors)` | stress tensor distance / mean resultant length |
| `stressResolution(tensor, isArray=True)` | eigendecomposition, returns `(Phi, eigvecs)` |
| `cal5DVectorDifference(v1, v2)` / `calDeviation(eigvecs, target='z')` | difference of 5-D solutions / axis–target angle |
| `fromEigvecsCalAzi(eigvecs)` | `(n,3,3)` → `(azi_deg, [aziType, ...])` |
| `randomTrialStress(amount)` | random stress states |
| `faultsCreateSimulation(...)` / `customFaultsSlipCalculation(faults, stress)` | synthetic faults / slip angle from a stress state |

Note on `peakFindInversion`: `faults` must be a 2-D array `(n, 3)` in radians. Even for a single fault
write `np.deg2rad([[strike, dip, slip]])`, otherwise the function iterates over individual numbers.

## 9. main.py workflow

| Step | Content | Results |
| --- | --- | --- |
| 0 | Create `solution()` (default grid); optionally `setMeshgrid` a custom grid | `run` |
| — | **Synthetic test (unnumbered, currently executed)**: two artificial faults and their σ1/σ3 polar plots | `Fig4a`, `Fig4b` |
| 1 | Read `focal_data.csv`, extract primary/conjugate planes and epicentres (only the first 2 events are used) | `faults`, `cfaults`, `BohaiFaultsLocation` |
| 2 | Peak-search inversion for both plane sets (`limitInDeg=2`) | `stresses`, `cstresses` |
| 3 | Primary plane: 5-D vectors → attractor → eigendecomposition → azimuth and type | `azi/aziType`, `BohaiEigvecs/BohaiPhi` |
| 4 | Conjugate plane: same treatment | `cazi/caziType`, `cBohaiEigvecs/cBohaiPhi` |
| 5 | Draw both candidate stress maps | `Fig8`, `Fig9` |
| 6 | Compare with the WSM database and choose the better nodal plane per event | `res`, `finalEigvecs/finalPhi/finalFocal` |
| 7 | Draw the final stress map, the comparison map and the risk map | `Fig10`, `Fig11`, `Fig12`, `risk` |

> ⚠️ The synthetic test block (`fault1 = ...` to `Fig4b = ...`) is currently **not commented out** and
> therefore runs with `main.py`. It sets `color` to `coolwarm` and `dpi` to `600`, which affects all
> later figures. To run the Bohai data only, comment the block out or wrap it in `''' ... '''`.

## 10. Recipes

### 10.1 Saving figures

```python
run.setPath('./figures')                  # the directory must exist; savefig does not create it
run.save(Fig8, '.png', 'Fig8_primary')
run.save(Fig12, '.png', 'Fig12_risk')
run.save(Fig12, '.pdf', 'Fig12_risk')     # vector output
```

### 10.2 Colours, line width, DPI

```python
run.setColor('coolwarm')     # any matplotlib colormap
run.setDpi(600)
run.setLineLength(2)         # segment length in degrees
run.setLineWidth(1.2)
run.setPointSize(25)
```

### 10.3 Custom / pre-computed grid

```python
import pandas as pd
grid = pd.read_csv('./data/mesh_360_180_10.csv', index_col=0).values   # (n, 4)
run.setMeshgrid(grid)
```

### 10.4 A small grid for fast experiments

```python
import singleFaultInversion as sfi
fast = sfi.solution(sfi.meshgrid(40, 40, 3).fourDimensionalGrid())   # about 6,560 states
```

### 10.5 Synthetic faults from a stress state (with noise)

```python
import numpy as np

# with isDeg=True the stress state is given in degrees: [α, β, γ, R]
stress = [[0.0, 90.0, 0.0, 0.5]]
faults = run.faultsCreateSimulation(
    faultsNumberForPerStress=50,
    stress=stress,
    noise=3.0,                # noise amplitude in degrees
    isGausssian=True,
    isDeg=True,
    isSeed=True, seed=2025,
)                              # returns (n, 3) [strike, dip, slip] in radians
```

### 10.6 Risk value without plotting

```python
run.setFaultInDeg([strike, dip, slip])
risk = np.linalg.norm(run.shearStress()) / np.linalg.norm(run.normalStress())
```

## 11. Performance and known limitations

- **Large default grid**: `N1=100, N2=100, N3=5` gives 60,600 stress states and `peakFindInversion`
  visits every one of them, once per fault. Use a smaller grid such as `sfi.meshgrid(40, 40, 3)` while
  debugging.
- **Import cost**: the default grid is a default argument of `solution.__init__`, so it is built when
  the module is imported; `import singleFaultInversion` therefore takes a few seconds.
- **Risk map memory**: `FaultsActivationRiskMap` converts the whole basemap (6931 × 6935) to floating
  point RGB/HSV and interpolates every pixel, which can peak at several GB of RAM and take a long time.
  If memory is short, down-sample the basemap first (e.g. 2000 px wide); `area` does not need to change.
- **`fitness` handles a single fault only**: `fitness(gridNumber, faultsParameter)` implements the
  continuous single-fault fitness `(1 - cosθ)/2`; invert each fault separately with
  `peakFindInversion`. A joint multi-fault objective would have to be implemented (and weighted)
  separately.
- **WSM data quality**: `stressMapWithDatabase` requires every `REGIME` value to be recognised and
  non-empty, otherwise it raises; clean the table first (`dropna`, consistent upper case).
  `matchAziWithDatabase` matches column names `LAT/LON/AZI/QUALITY/REGIME` exactly (case-sensitive).
- **Basemap projection**: `locTrans` uses a spherical Mercator conversion in y, so a plate carrée image
  is stretched slightly; the image extent must still match `area` exactly.
- **Non-Latin labels**: all figure text is English with Times New Roman. To render CJK labels, also set
  `plt.rcParams['font.sans-serif']` and `axes.unicode_minus`.

## 12. FAQ

**Q: `ValueError: No valid data with QUALITY A, B or C exists in the given area.`**
A: There is no high-quality station inside the selection window. `matchAziWithDatabase` defaults to
`(110, 131, 29, 46)`; check that the data lie inside it or pass an explicit `area`.

**Q: `ValueError: The database is missing the following columns: ...`**
A: `WSM_Database_2025.csv` must contain the column names `LAT, LON, AZI, QUALITY, REGIME` in upper case.

**Q: `ValueError: Each row of inputData must contain 6 values ...`**
A: `inputData` must be `(n, 6)` with rows `[azi1, aziType1, azi2, aziType2, lat, lon]`.

**Q: Can several faults be inverted jointly?**
A: Not directly. `fitness` evaluates a single fault and `peakFindInversion` inverts every fault in
`faults` separately. For a joint inversion, build your own weighted multi-fault objective function.

**Q: The run is very slow or runs out of memory.**
A: Debug the inversion with a small grid first, and use a lower-resolution basemap (about 2000 px) for
the risk map.

**Q: `FileNotFoundError: ./background/Satellite_map.png`**
A: Relative paths depend on the working directory. Run from `E:/code`, or use an absolute path.

## 13. References

- Sato, K. & Yamaji, A. (2006): parameterisation and normalisation of the 5-D stress vector (basis of
  `vector5D` and `s5_to_stress_tensor`).
- World Stress Map Project: <https://www.world-stress-map.org/> (source of `WSM_Database_2025.csv`).
- Michael, A.J. (1984): least-squares framework for inverting stress from slip vectors (physical
  background of `fitness`).
- Author: 77891. Research code — please contact the author before reusing it in a paper or project.
