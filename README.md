# Verifiable Encryption
This repository contains the source code and necessary files to implement 
verifiable encryption using Halo2.
This work solves a task that allows a prover to prove some property of a message m, 
while the message is given in an encrypted form.

## Documentation
[The white paper](https://github.com/QED-it/halo2_verifiable_encryption/blob/main/src/Verifiable_Encryption_using_Halo2.pdf)


## Projects 

[[verifiable_encryption.rs](https://github.com/QED-it/halo2_verifiable_encryption/blob/main/src/tasks/verifiable_encryption.rs
)] implements a system for the prover to prove to the verifier that she knows the knowledge of a
message m that encrypts to a cipehrtext C. [[doc, Section 3.2](https://github.com/QED-it/halo2_verifiable_encryption/blob/main/src/Verifiable_Encryption_using_Halo2.pdf)]


[[verifiable_encryption_with_relation.rs](https://github.com/QED-it/halo2_verifiable_encryption/blob/main/src/tasks/verifiable_encryption_with_relation.rs
)] implements a system for the prover to prove to the verifier that she knows the knowledge of a message m that
encrypts to a cipehrtext C. Additionally, the message is a private key of a digital signature scheme.
 [[doc, Section 3.3](https://github.com/QED-it/halo2_verifiable_encryption/blob/main/src/Verifiable_Encryption_using_Halo2.pdf)]

## Test Instructions

### Verifiable encryption
To run the round trip test:
```bash
cargo test --package halo2_verifiable_encryption --lib tasks::verifiable_encryption::tests::round_trip
```
### Verifiable encryption with relation
To run the round trip test:
```bash
cargo test --package halo2_verifiable_encryption --lib tasks::verifiable_encryption_with_relation::tests::round_trip
```
To run the negative test with tampered public input:
```bash
cargo test --package halo2_verifiable_encryption --lib tasks::verifiable_encryption_with_relation::tests::negative_test
```
To run the negative test with tampered witness:
```bash
cargo test --package halo2_verifiable_encryption --lib tasks::verifiable_encryption_with_relation::tests::negative_witness_test
```
