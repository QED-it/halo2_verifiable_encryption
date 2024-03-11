/// Defined in [Verifiable Encryption using Halo2][Section 3.3. Task 2 - Verifiable Encryption with relation R].
/// Format a circuit and an instance for Encode, Elgamal decryption and DSA keypair
/// A round trip test to prove ciphertext is an encryption of an encoded scalar message
/// and the scalar message is the private key of a DSA keypair
///
/// This file shows how to build a proof of knowledge of message in a ciphertext
/// Prove:
/// (1) Encode(m; r_encode) = p_m, that is,
/// (1.1) p_m.x = r_encode + m
/// (1.2) p_m.x^3 + 5 = p_m.y^2 (redundant check, if p_m is not on the curve, the point operations will fail)
/// (2) C = ElGamal.Enc(pk, p_m)
/// (2.1) ct_1 = [r_enc]G, G is the generator of E
/// (2.2) ct_2 = p_m +[r_enc]pk_elgamal
/// (3) pk_dsa = [m]G (new constraint compared to task1)
///
/// - secret input `m`;
/// - secret input `p_m`;
/// - secret input `r_enc`;
/// - public group element `ct_1 := [r_enc]G`
/// - public group element `ct_2 := p_m + [r]elgamal_public_key`
/// - public random element `r_encode`
/// - public group element `elgamal_public_key`
/// - public group element `dsa_public_key`
/// - public generator `G`;


use ff::{Field, PrimeField};
use group::prime::PrimeCurveAffine;
use group::{Curve, Group};
use halo2_gadgets::ecc::chip::{EccChip};
use halo2_gadgets::ecc::{NonIdentityPoint, ScalarVar};
use halo2_gadgets::utilities::UtilitiesInstructions;
use halo2_proofs::{
    circuit::{Layouter, SimpleFloorPlanner, Value},
    plonk::{Circuit, ConstraintSystem, Error},
};
use halo2_proofs::circuit::{AssignedCell, Chip};
use pasta_curves::arithmetic::CurveAffine;
use pasta_curves::{Fp, pallas, vesta};
use pasta_curves::pallas::{Affine, Base};
use rand;
use rand::rngs::OsRng;
use crate::add_sub_mul::chip::{ AddSubMulChip, AddSubMulInstructions};
use crate::circuits::verifiable_encryption::{VeEncCircuit, VeConfig, VeEncInstance};
use crate::elgamal::elgamal::ElGamalKeypair;
use crate::circuits::verifiable_encryption;
use crate::constants::fixed_bases::VerifiableEncryptionFixedBases;

const K: u32 = 11;
const DSA_PK_X: usize = 7;
const DSA_PK_Y: usize = 8;


#[derive(Default, Clone)]
struct VeCircuit {
    ve_enc_circuit: VeEncCircuit,
    dsa_public_key: pallas::Point,
}

impl Circuit<pallas::Base> for VeCircuit {
    type Config = VeConfig;

    type FloorPlanner = SimpleFloorPlanner;

    fn without_witnesses(&self) -> Self {
        Self::default()
    }

    fn configure(meta: &mut ConstraintSystem<pallas::Base>) -> Self::Config {
        VeEncCircuit::configure(meta)
    }

    fn synthesize(
        &self,
        config: Self::Config,
        mut layouter: impl Layouter<pallas::Base>,
    ) -> Result<(), Error> {
        let ecc_chip = EccChip::construct(config.ecc_config.clone());
        let add_sub_mul_chip = AddSubMulChip::new(config.add_sub_mul_config.clone());

        // Load 10-bit lookup table.
        config.ecc_config.lookup_config.load(&mut layouter)?;

        let column = ecc_chip.config().advices[0];

        // witness message point p_m
        let p_m = NonIdentityPoint::new(
            ecc_chip.clone(),
            layouter.namespace(|| "load p_m"),
            self.ve_enc_circuit.p_m.as_ref().map(|p_m| p_m.to_affine()),
        )?;
        // load randomness r_encode
        let r_encode = add_sub_mul_chip.load_private(
            layouter.namespace(|| "load r_encode"),
            Value::known(self.ve_enc_circuit.data_in_transmit.r_encode),
        )?;

        // load dsa_private_key = message
        let message =
            add_sub_mul_chip.load_private(layouter.namespace(|| "load message"), self.ve_enc_circuit.m)?;

        // load r_enc
        let assigned_r_enc =
            ecc_chip.load_private(layouter.namespace(|| "load r_enc"), column, self.ve_enc_circuit.r_enc)?;

        // elgamal_public_key
        let elgamal_public_key = NonIdentityPoint::new(
            ecc_chip.clone(),
            layouter.namespace(|| "load elgamal_public_key"),
            Value::known(self.ve_enc_circuit.elgamal_public_key.to_affine()),
        )?;

        check_encryption_and_relation(
            config,
            layouter,
            ecc_chip,
            add_sub_mul_chip,
            p_m,
            r_encode,
            message,
            assigned_r_enc,
            elgamal_public_key,
        )
    }
}

pub(crate) fn check_encryption_and_relation(
    config: VeConfig,
    mut layouter: impl Layouter<pallas::Base>,
    ecc_chip:  EccChip<VerifiableEncryptionFixedBases>,
    add_sub_mul_chip: AddSubMulChip,
    p_m: NonIdentityPoint<Affine, EccChip<VerifiableEncryptionFixedBases>>,
    r_encode: AssignedCell<Fp, Fp>,
    message: AssignedCell<Fp, Fp>,
    assigned_r_enc:  AssignedCell<Base, Base>,
    elgamal_public_key: NonIdentityPoint<Affine, EccChip<VerifiableEncryptionFixedBases>>,
) -> Result<(), Error>
{
    // check relation
    // generator
    let generator = NonIdentityPoint::new(
        ecc_chip.clone(),
        layouter.namespace(|| "load generator"),
        Value::known(pallas::Affine::generator()),
    )?;
    // (3) dsa_public_key = [m]generator
    // convert message to scalar, it is dsa_private_key
    let m_scalar = ScalarVar::from_base(
        ecc_chip.clone(),
        layouter.namespace(|| "m to scalar"),
        &message,
    )?;

    // compute dsa_pk_expected = [message_scalar]generator
    let (dsa_pk_expected, _) = {
        generator.mul(
            layouter.namespace(|| "[m_scalar]generator"),
            m_scalar,
        )?
    };

    // Constrain dsa_pk_expected to equal public input dsa_pk
    layouter.constrain_instance(
        dsa_pk_expected.inner().x().cell(),
        config.instance,
        DSA_PK_X,
    )?;
    layouter.constrain_instance(
        dsa_pk_expected.inner().y().cell(),
        config.instance,
        DSA_PK_Y,
    )?;

    // check encryption
    crate::circuits::verifiable_encryption::check_encryption(
        config,
        layouter,
        ecc_chip,
        add_sub_mul_chip,
        p_m,
        r_encode,
        message,
        assigned_r_enc,
        elgamal_public_key,
    )
}


/// Public inputs
#[derive(Clone, Debug)]
pub struct VeInstance {
    ve_enc_instance: VeEncInstance,
    dsa_public_key: pallas::Point,
}

impl VeInstance {
    fn to_halo2_instance(&self) -> [[vesta::Scalar; 9]; 1] {
        let mut instance = [vesta::Scalar::random(OsRng); 9];

        let ve_enc_instance = self.ve_enc_instance.clone();
        let ve_enc_instance = ve_enc_instance.to_halo2_instance();
        for i in 0..7 {
            instance[i] = ve_enc_instance[0][i];
        }

        instance[DSA_PK_X] = *self.dsa_public_key.to_affine().coordinates().unwrap().x();
        instance[DSA_PK_Y] = *self.dsa_public_key.to_affine().coordinates().unwrap().y();

        [instance]
    }
}
fn create_circuit(message: pallas::Base, elgamal_keypair: ElGamalKeypair) -> VeCircuit {
    let ve_enc_circuit = verifiable_encryption::create_circuit(message,elgamal_keypair);

    // map base to scalar
    let dsa_private_key = pallas::Scalar::from_repr(message.to_repr()).unwrap();

    // Calculate the dsa public key [dsa_private_key]G
    let dsa_public_key = pallas::Point::generator() * dsa_private_key;

    VeCircuit {
        ve_enc_circuit: ve_enc_circuit,
        dsa_public_key: dsa_public_key,
    }
}
#[cfg(test)]
mod tests {
    use super::{create_circuit, VeInstance, K};
    use crate::elgamal::elgamal::ElGamalKeypair;
    use ff::{Field};
    use group::Group;
    use halo2_proofs::plonk::SingleVerifier;
    use halo2_proofs::poly::commitment::Params;
    use halo2_proofs::transcript::{Blake2bRead, Blake2bWrite, Challenge255};
    use halo2_proofs::{plonk};
    use pasta_curves::{pallas, vesta};
    use rand::rngs::OsRng;
    use crate::circuits::verifiable_encryption::{VeEncInstance};

    #[test]
    fn round_trip() {
        let mut rng = OsRng;

        // Elgamal keygen
        let elgamal_keypair = ElGamalKeypair::new();

        // Setup phase: generate parameters for the circuit.
        let params = Params::new(K);

        // generate a random dsa private key
        // it will be encoded to a point on ECC, and be an input to elgamal encryption
        let dsa_private_key = pallas::Base::random(&mut rng);

        // Step 1. create a circuit
        let circuit = vec![create_circuit(dsa_private_key, elgamal_keypair.clone())];

        // Step 2. arrange the public instance.
        let ve_enc_instance = VeEncInstance{
            data_in_transmit: circuit[0].ve_enc_circuit.data_in_transmit.clone(),
            elgamal_public_key: circuit[0].ve_enc_circuit.elgamal_public_key.clone(),
        };
        let instance = vec![VeInstance {
            ve_enc_instance: ve_enc_instance,
            dsa_public_key: circuit[0].dsa_public_key.clone(),
        }];

        // Instance transformation
        let instance: Vec<_> = instance.iter().map(|i| i.to_halo2_instance()).collect();
        let instance: Vec<Vec<_>> = instance
            .iter()
            .map(|i| i.iter().map(|c| &c[..]).collect())
            .collect();
        let instance: Vec<_> = instance.iter().map(|i| &i[..]).collect();

        // Step 3. generate the verification key vk and proving key pk from the params and circuit.
        let vk = plonk::keygen_vk(&params, &circuit[0]).unwrap();
        let pk = plonk::keygen_pk(&params, vk.clone(), &circuit[0]).unwrap();

        // Step 4. Proving phase: create a proof with public instance and witness.
        // The proof generation will need an internal transcript for Fiat-Shamir transformation.
        let mut transcript = Blake2bWrite::<_, vesta::Affine, _>::init(vec![]);
        plonk::create_proof(
            &params,
            &pk.clone(),
            &circuit,
            &instance,
            &mut rng,
            &mut transcript,
        )
            .unwrap();
        let proof = transcript.finalize();

        // Step 5. Verification phase: verify the proof against the public instance.
        let strategy = SingleVerifier::new(&params);
        let mut transcript: Blake2bRead<&[u8], vesta::Affine, Challenge255<vesta::Affine>> =
            Blake2bRead::init(&proof[..]);
        let verify = plonk::verify_proof(&params, &vk, strategy, &instance, &mut transcript);
        // Round-trip assertion: check the proof is valid and matches expected values.
        assert!(verify.is_ok());

        // Calculate the circuit cost
        let circuit_cost = halo2_proofs::dev::CircuitCost::<pasta_curves::vesta::Point, _>::measure(
            K,
            &circuit[0],
        );
        let expected_proof_size = usize::from(circuit_cost.proof_size(instance.len()));
        println!("Proof length: {}", proof.len());
        assert_eq!(proof.len(), expected_proof_size);
    }

    #[test]
    fn negative_test() {
        let mut rng = OsRng;

        // Elgamal keygen
        let elgamal_keypair = ElGamalKeypair::new();

        // Setup phase: generate parameters for the circuit.
        let params = Params::new(K);

        // generate a random dsa private key
        // it will be encoded to a point on ECC, and be an input to elgamal encryption
        let dsa_private_key = pallas::Base::random(&mut rng);

        // Step 1. create a circuit
        let circuit = vec![create_circuit(dsa_private_key, elgamal_keypair.clone())];

        // Step 2. arrange the public instance.
        let ve_enc_instance = VeEncInstance{
            data_in_transmit: circuit[0].ve_enc_circuit.data_in_transmit.clone(),
            elgamal_public_key: circuit[0].ve_enc_circuit.elgamal_public_key.clone(),
        };
        let instance = vec![VeInstance {
            ve_enc_instance: ve_enc_instance.clone(),
            dsa_public_key: circuit[0].dsa_public_key.clone(),
        }];

        // Instance transformation
        let instance: Vec<_> = instance.iter().map(|i| i.to_halo2_instance()).collect();
        let instance: Vec<Vec<_>> = instance
            .iter()
            .map(|i| i.iter().map(|c| &c[..]).collect())
            .collect();
        let instance: Vec<_> = instance.iter().map(|i| &i[..]).collect();

        // Step 3. generate the verification key vk and proving key pk from the params and circuit.
        let vk = plonk::keygen_vk(&params, &circuit[0]).unwrap();
        let pk = plonk::keygen_pk(&params, vk.clone(), &circuit[0]).unwrap();

        // Step 4. Proving phase: create a proof with public instance and witness.
        // The proof generation will need an internal transcript for Fiat-Shamir transformation.
        let mut transcript = Blake2bWrite::<_, vesta::Affine, _>::init(vec![]);
        plonk::create_proof(
            &params,
            &pk.clone(),
            &circuit,
            &instance,
            &mut rng,
            &mut transcript,
        )
            .unwrap();
        let proof = transcript.finalize();

        // Step 5, the negative part: Introduce a modification in the public instance to simulate inconsistency
        // For example, tamper with the DSA public key or any other critical part that will invalidate the proof
        // Let's assume `tampered_dsa_public_key` is a public key from a different DSA private key
        let tampered_dsa_public_key = pallas::Point::random(&mut rng); // This should ideally be a different key
        let tampered_instance = vec![VeInstance {
            ve_enc_instance: ve_enc_instance,
            dsa_public_key: tampered_dsa_public_key,
        }];
        let tampered_instance: Vec<_> = tampered_instance.iter().map(|i| i.to_halo2_instance()).collect();
        let tampered_instance: Vec<Vec<_>> = tampered_instance.iter().map(|i| i.iter().map(|c| &c[..]).collect()).collect();
        let tampered_instance: Vec<_> = tampered_instance.iter().map(|i| &i[..]).collect();


        // Step 6. Verification phase: verify the proof against the public instance.
        let strategy = SingleVerifier::new(&params);
        let mut tampered_transcript: Blake2bRead<&[u8], vesta::Affine, Challenge255<vesta::Affine>> = Blake2bRead::init(&proof[..]);
        let verify = plonk::verify_proof(&params, &vk, strategy, &tampered_instance, &mut tampered_transcript);
        // The verification should fail, demonstrating the negative test case
        assert!(verify.is_err(), "Expected verification to fail due to tampered input, but it succeeded.");
    }

    #[test]
    fn negative_witness_test() {
        let mut rng = OsRng;

        // Elgamal keygen
        let elgamal_keypair = ElGamalKeypair::new();

        // Setup phase: generate parameters for the circuit.
        let params = Params::new(K);

        // generate a random dsa private key
        // it will be encoded to a point on ECC, and be an input to elgamal encryption
        let dsa_private_key = pallas::Base::random(&mut rng);

        // Step 1. create a circuit
        let circuit = vec![create_circuit(dsa_private_key, elgamal_keypair.clone())];

        // Step 2. arrange the public instance.
        let ve_enc_instance = VeEncInstance{
            data_in_transmit: circuit[0].ve_enc_circuit.data_in_transmit.clone(),
            elgamal_public_key: circuit[0].ve_enc_circuit.elgamal_public_key.clone(),
        };
        let instance = vec![VeInstance {
            ve_enc_instance: ve_enc_instance,
            dsa_public_key: circuit[0].dsa_public_key.clone(),
        }];

        // Instance transformation
        let instance: Vec<_> = instance.iter().map(|i| i.to_halo2_instance()).collect();
        let instance: Vec<Vec<_>> = instance
            .iter()
            .map(|i| i.iter().map(|c| &c[..]).collect())
            .collect();
        let instance: Vec<_> = instance.iter().map(|i| &i[..]).collect();

        // Step 3, the negative part: Introduce a modification in the witness to simulate inconsistency
        // For example, tamper with the DSA private key or any other critical part that will invalidate the proof
        // Let's assume `tampered_dsa_private_key` is a different DSA private key
        let tampered_dsa_private_key = pallas::Base::random(&mut rng);
        let tampered_circuit = vec![create_circuit(tampered_dsa_private_key, elgamal_keypair.clone())];

        // Step 4. generate the verification key vk and proving key pk from the params and tampered_circuit.
        let vk = plonk::keygen_vk(&params, &tampered_circuit[0]).unwrap();
        let pk = plonk::keygen_pk(&params, vk.clone(), &tampered_circuit[0]).unwrap();

        // Step 5. Proving phase: create a proof with public instance and tampered_witness.
        // The proof generation will need an internal transcript for Fiat-Shamir transformation.
        let mut transcript = Blake2bWrite::<_, vesta::Affine, _>::init(vec![]);
        plonk::create_proof(
            &params,
            &pk.clone(),
            &tampered_circuit,
            &instance,
            &mut rng,
            &mut transcript,
        )
            .unwrap();
        let proof = transcript.finalize();

        // Step 6. Verification phase: verify the proof against the public instance.
        let strategy = SingleVerifier::new(&params);
        let mut tampered_transcript: Blake2bRead<&[u8], vesta::Affine, Challenge255<vesta::Affine>> = Blake2bRead::init(&proof[..]);
        let verify = plonk::verify_proof(&params, &vk, strategy, &instance, &mut tampered_transcript);
        // The verification should fail, demonstrating the negative test case
        assert!(verify.is_err(), "Expected verification to fail due to tampered input, but it succeeded.");
    }
}