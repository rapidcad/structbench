# StructBench

StructBench is a benchmark for evaluating the capability of any LLM to generate
structurally valid CAD components from functional requirements. Each instance
specifies applied loads, boundary conditions, and a design space; the desired
output is a CAD object that performs best under load. Intermediate steps, such
as requirement representation and CAD modeling method, are purposely left open
to allow for diverse approaches. StructBench provides a novel evaluation
protocol grounded in physical metrics rather than geometric similarity.

![Representative StructBench instances](samples.png)

## Task

Each benchmark instance defines a triplet `(D, B, F)` consisting of the
admissible design domain, boundary conditions (regions where the component is
physically fixed), and applied load configuration, and tasks a model with
generating geometry `G ⊆ D` that satisfies structural performance criteria
under FEA. A model receives this specification as natural language and must
produce a CadQuery program whose executed geometry satisfies the structural
constraints.

## Metrics

StructBench evaluates generated designs on four metrics that reflect the
sequential nature of engineering validation: a design must respect its
geometric envelope before it can transfer load, transfer load before it can
be simulated, and survive simulation before its mass efficiency is
meaningful. The metrics are therefore evaluated as a cascade, where each acts
as a precondition for the next.

- **Design Space Violation (DSV)** — the fraction of generated material that
  falls outside the specified design space volume `V_D`:

  `DSV = V_outside / V_D ∈ [0, ∞)`

  An ideal design achieves `DSV = 0`.

- **Load Path Validity (LPV)** — a binary precondition: does a continuous
  path of material connect the loaded region to the constrained region?
  Without this connectivity, the structure is mechanically degenerate and FEA
  cannot be executed. If `LPV = 0`, all subsequent metrics are undefined and
  FEA is not run.

- **Safety Factor Compliance (SFC)** — given a design that passes the LPV
  gate, FEA is executed with material properties fixed to aluminum alloy
  Al6061. Let `σ_vM,95` denote the 95th percentile of the element-wise von
  Mises stress under the prescribed loading, and `σ_y` the yield strength.
  The safety factor is `η = σ_y / σ_vM,95`, and `SFC = 1` iff `η ≥ 1`. The
  p95 statistic is used consistently for both benchmark evaluation and
  training — compared with the raw maximum, it is less sensitive to
  mesh-dependent single-element stress spikes and localized numerical
  singularities while remaining focused on the high-stress tail. Fixing the
  material across all benchmark cases removes a trivial degree of freedom (a
  model cannot satisfy structural requirements by selecting a stronger
  material) and ensures that performance differences reflect genuine
  geometric reasoning.

- **Mass Efficiency (ME)** — among designs that satisfy structural validity,
  engineering optimization demands minimizing material usage as a proxy for
  cost, weight, and sustainability. Generated volume is normalized by the
  design space volume to ensure comparability across load cases of different
  scales:

  `ME = V_generated / V_D ∈ (0, 1]`

  Lower values indicate more efficient use of material. ME is a measurement
  of material usage; the incentive towards lightweight design arises from the
  StructScore composite below, which rewards lower ME among structurally
  valid designs.

### Composite StructScore

For leaderboard comparison, a model-level composite is defined from the
benchmark-wide aggregate of each component:

`StructScore = (1 − DSV) · LPV · SFC · (1 − ME)`

where the aggregate is taken over all evaluated benchmark generations; binary
LPV and SFC aggregates are reported as rates. This multiplicative form
jointly rewards design-space compliance, load-path validity, structural
safety, and material efficiency.

## Usage

Run the benchmark harness against a generated STEP file for a given load
case:

```bash
python vendor/structbench/benchmark.py \
    path/to/model.step \
    vendor/structbench/data/json/l_bracket.json \
    --output-dir results/l_bracket \
    --output-json results/l_bracket/result.json
```

Load cases live in [`data/json/`](data/json/) (75 files, one per instance);
`data/step/`, `data/inp/`, and `data/img/` hold the corresponding reference
geometry, FEA input decks, and preview images. See `python
vendor/structbench/benchmark.py --help` for the full CLI, and the module
docstring in [`benchmark.py`](benchmark.py) for the output JSON schema and
relevant environment variables.

For details on the training corpus StructBench is held out from, see
[`datacard.md`](datacard.md).

