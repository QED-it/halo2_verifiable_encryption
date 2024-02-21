# Verifiable Encryption
This repository contains the source code and necessary files to implement 
verifiable encryption using Halo2.
This work solves a task that allows a prover to prove some property of a message m, 
while the message is given in an encrypted form.

## Documentation
[The white paper](https://github.com/QED-it/halo2_verifiable_encryption/blob/main/src/Verifiable_Encryption_using_Halo2.pdf)


## Projects 

Task 1 is a warm up task for the prover to prove to the verifier that she knows the knowledge of a
message m that encrypts to a cipehrtext C. [[Source Code](https://github.com/QED-it/halo2_verifiable_encryption/blob/main/src/tasks/task1.rs
)] [[Section 3.2](https://github.com/QED-it/halo2_verifiable_encryption/blob/main/src/Verifiable_Encryption_using_Halo2.pdf)]


Task 2 is for the prover to prove to the verifier that she knows the knowledge of a message m that
encrypts to a cipehrtext C. Additionally, the message is a private key of a digital signature scheme.
[[Source Code](https://github.com/QED-it/halo2_verifiable_encryption/blob/main/src/tasks/task2.rs
)] [[Section 3.3](https://github.com/QED-it/halo2_verifiable_encryption/blob/main/src/Verifiable_Encryption_using_Halo2.pdf)]

## Test Instructions

### Task 1
To run the round trip test:
```bash
cargo test --package halo2_verifiable_encryption --lib tasks::task1::tests::round_trip
```
### Task 2
To run the round trip test:
```bash
cargo test --package halo2_verifiable_encryption --lib tasks::task2::tests::round_trip
```
To run the negative test:
```bash
cargo test --package halo2_verifiable_encryption --lib tasks::task2::tests::negative_test
```
