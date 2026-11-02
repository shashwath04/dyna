# Development ownership

The student owns all mathematically significant GNC algorithms. Do not implement
simulation scheduling, integration, PID/anti-windup, linearization, LQR, quaternion
control, Kalman/EKF/MEKF, fixed-point control, or FPGA controller algorithms unless
the student explicitly requests a full implementation of the relevant component.
Keep unfinished algorithm bodies as `raise NotImplementedError` with
`TODO(student)`. Infrastructure and interface improvements are allowed.

Preserve mathematical contracts and independent acceptance tests. Never replace
an algorithm with a working library wrapper to pass tests. Diagnose student test
failures with hints first. See IMPLEMENTATION_GUIDE.md for ownership and theory.

Run infrastructure tests, student acceptance tests, Ruff and mypy after changes.
Only NotImplementedError may be an expected failure in student tests. Remove the
relevant expected-failure marker when an algorithm is implemented and accepted.
