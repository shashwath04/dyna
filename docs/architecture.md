# Architecture and ownership

`types` defines structural callback protocols and result/state dataclasses.
`logging` copies and validates supplied signals, supports snapshots and CSV.
`experiments` invokes caller-owned sampling and run callbacks serially, using
independent seeded RNG streams. It does not schedule or integrate dynamics.
`visualization` plots supplied results; callers own figure lifecycle.

`simulation`, `control`, `estimation`, and `hardware` implement the documented
algorithm contracts, authorized explicitly by the student. Public imports remain
in `dino.__all__`. `hardware` models register arithmetic and clock transactions in
Python; it does not generate or synthesize RTL. Dataclass construction alone does
not validate mathematical state contracts; algorithm entry points do.

`_validation` centralizes finite/shape/covariance checks. `_rotation` contains
Hamilton products, rotation exponentials, and SO(3) covariance-reset primitives.
LQR constructs the Hamiltonian and extracts its stable invariant subspace using
SciPy's ordered Schur decomposition, then checks CARE residual and closed-loop
stability. MEKF uses SciPy's matrix exponential for frozen-rate error propagation.

Tests remain separated: `tests/infrastructure` checks software plumbing;
`tests/algorithms` checks analytic reference results, rejection behavior,
convergence, filtering consistency and bit-level arithmetic. None are marked
expected failures. These are reference models for building broader system and
hardware test benches, rather than evidence of device verification.

When requesting review, supply your derivation, conventions, implementation and a
failing test. Review should begin with conceptual hints and diagnosis; full
implementations require your explicit request. Maintain this ownership boundary
in future contributions.

## External physics

`ardupilot` owns the binary/JSON codec, frame conversions, and UDP endpoint.
`sitl` owns lockstep frame sequencing, model substeps, cached retransmissions and
per-episode histories. It calls externally supplied actuator, Dynamics and
SensorObserver callbacks; ArduPilot owns control/estimation in this path.
`marine` converts supplied mass-matrix/Fossen terms into ordinary derivatives.
These modules do not change the existing offline `simulate` behavior.
See [SITL guide](ardupilot_sitl.md) for the protocol contracts and limitations.
