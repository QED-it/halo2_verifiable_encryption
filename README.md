# Verifiable Encryption
This repository contains the source code and necessary files to implement
verifiable encryption using [Halo2](https://github.com/zcash/halo2).
It lets a prover convince a verifier of some property of a message `m`
while the message itself is kept private inside an ElGamal ciphertext `C`.

The prover commits to `m` by publishing `C = ElGamal.Enc(pk, p_m)`, where
`p_m` is the curve-point encoding of `m`. A Halo2 zk-SNARK then proves
that `C` is a well-formed encryption of `p_m = Encode(m; r_encode)`, and
(in the second circuit) that `m` is additionally the private key of a
DSA public key `pk_dsa = [m]G`. Fiat-Shamir is instantiated with Blake2b.

## Documentation
[The white paper](https://github.com/QED-it/halo2_verifiable_encryption/blob/main/src/Verifiable_Encryption_using_Halo2.pdf)

## Curves and parameters
- Proving system: Halo2 (PLONK-ish) from the `QED-it/halo2` fork,
  branch `verifiable-encryption`.
- Curves: the Pasta cycle (`pallas` / `vesta`). Proofs are over `vesta`;
  the circuit field is `pallas::Base` (= `Fp`); ElGamal and point
  encoding live on the `pallas` curve.
- Circuit size: `K = 11` (i.e. `2^11` rows) for both circuits.
- Transcript: `Blake2bWrite` / `Blake2bRead` with `Challenge255`.

## Crate layout
- `src/lib.rs` — crate root; re-exports `add_sub_mul`, `encode`,
  `elgamal`, `constants`; keeps `circuits` module-private.
- `src/circuits/verifiable_encryption.rs` — Task 1 circuit
  (Encode + ElGamal, no relation).
- `src/circuits/verifiable_encryption_with_relation.rs` — Task 2
  circuit (Task 1 + DSA key relation).
- `src/add_sub_mul/chip.rs` — minimal add / sub / mul custom chip
  used to enforce the affine encoding equation `p_m.x = r_encode + m`.
- `src/elgamal/elgamal.rs` — raw EC-ElGamal keygen, encrypt, decrypt.
- `src/elgamal/extended_elgamal.rs` — encode-then-encrypt / decrypt-then-decode
  wrapper; defines the public `DataInTransmit { ct, r_encode }`.
- `src/encode/encode.rs` — scalar → curve-point encoder (rejection sampling
  on `y^2 = x^3 + 5`).
- `src/encode/utf8.rs` — helpers to pack UTF-8 strings into `pallas::Base`.
- `src/constants/fixed_bases.rs` — fixed bases for the Halo2 `EccChip`.

## Projects

### Task 1 — Verifiable Encryption without relation
[`src/circuits/verifiable_encryption.rs`](https://github.com/QED-it/halo2_verifiable_encryption/blob/main/src/circuits/verifiable_encryption.rs)
implements a system for the prover to prove to the verifier that she
knows a message `m` that encrypts to a ciphertext `C`.
[[doc, Section 3.2](https://github.com/QED-it/halo2_verifiable_encryption/blob/main/src/Verifiable_Encryption_using_Halo2.pdf)]

The circuit proves:
1. `Encode(m; r_encode) = p_m`, checked as `p_m.x = r_encode + m`
   (the curve equation `p_m.x^3 + 5 = p_m.y^2` is already enforced
   implicitly by `NonIdentityPoint::new`).
2. `ct_1 = [r_enc]G`, with `G` the `pallas` generator.
3. `ct_2 = p_m + [r_enc] · pk_elgamal`.

Private inputs: `m`, `p_m`, `r_enc`.
Public instance (7 rows): `[0, ct_1.x, ct_1.y, ct_2.x, ct_2.y,
pk_elgamal.x, pk_elgamal.y]`.

### Task 2 — Verifiable Encryption with relation
[`src/circuits/verifiable_encryption_with_relation.rs`](https://github.com/QED-it/halo2_verifiable_encryption/blob/main/src/circuits/verifiable_encryption_with_relation.rs)
implements a system for the prover to prove to the verifier that she
knows a message `m` that encrypts to a ciphertext `C`, **and** that
`m` is the private key of a digital-signature keypair.
[[doc, Section 3.3](https://github.com/QED-it/halo2_verifiable_encryption/blob/main/src/Verifiable_Encryption_using_Halo2.pdf)]

The circuit reuses Task 1's configuration and `check_encryption`, and
adds the extra constraint:

4. `pk_dsa = [m]G`, where `m` is reinterpreted as a scalar.

Public instance (9 rows): the 7 rows from Task 1, followed by
`pk_dsa.x` (row 7) and `pk_dsa.y` (row 8).

## Proof-flow summary
Both circuits follow the same pipeline in their tests:
1. `Params::new(K)` — SRS / commitment parameters for `K = 11`.
2. Build circuit: call `create_circuit(m, elgamal_keypair)`, which runs
   `extended_elgamal_encrypt` off-circuit to obtain `p_m`, `r_encode`,
   `r_enc`, asserts round-trip decryption, and wraps everything in the
   circuit struct.
3. Build instance: `to_halo2_instance()` lays out the public-input vector.
4. `plonk::keygen_vk` → `plonk::keygen_pk`.
5. `plonk::create_proof` with `Blake2bWrite` transcript.
6. `plonk::verify_proof` with `SingleVerifier` and a matching
   `Blake2bRead` transcript.
7. Assert `proof.len() == CircuitCost::measure(K, circuit).proof_size(instance.len())`.

## Test instructions

### Task 1 — Verifiable encryption
Round-trip test (splits a sample string into 31-byte blocks and
proves/verifies each one):
```bash
cargo test --package halo2_verifiable_encryption --lib circuits::verifiable_encryption::tests::round_trip
```

### Task 2 — Verifiable encryption with relation
Round-trip (positive) test:
```bash
cargo test --package halo2_verifiable_encryption --lib circuits::verifiable_encryption_with_relation::tests::round_trip
```
Negative test — tamper with the public `dsa_public_key` after proving
and expect verification to fail:
```bash
cargo test --package halo2_verifiable_encryption --lib circuits::verifiable_encryption_with_relation::tests::negative_test
```
Negative test — generate the proof from a witness whose DSA private key
differs from the one implied by the public instance and expect
verification to fail:
```bash
cargo test --package halo2_verifiable_encryption --lib circuits::verifiable_encryption_with_relation::tests::negative_witness_test
```

### Supporting unit tests
The `encode` and `elgamal` modules ship their own 1000-iteration
round-trip tests:
```bash
cargo test --package halo2_verifiable_encryption --lib encode::encode::tests
cargo test --package halo2_verifiable_encryption --lib elgamal::elgamal::tests
cargo test --package halo2_verifiable_encryption --lib elgamal::extended_elgamal::tests
```

### Running everything
```bash
cargo test
```
