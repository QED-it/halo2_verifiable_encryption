/// Defined in [Verifiable Encryption using Halo2][Section 3.2. Task 1 - Verifiable Encryption without relation R].
/// Format a circuit and an instance for Encode and Elgamal encryption
/// A round trip test to prove ciphertext(s) are encryption(s) of message block(s)
///
/// This file shows how to build a proof of knowledge of message in a ciphertext
/// Prove:
/// (1) Encode(m; r_encode) = p_m, that is,
/// (1.1) p_m.x = r_encode + m
/// (1.2) p_m.x^3 + 5 = p_m.y^2 (redundant check, if p_m is not on the curve, the point operations will fail)
/// (2) C = ElGamal.Enc(pk, p_m)
/// (2.1) ct_1 = [r_enc]G, G is the generator of E
/// (2.2) ct_2 = p_m +[r_enc]pk_elgamal
///
/// - secret input `m`;
/// - secret input `p_m`;
/// - secret input `r_enc`;
/// - public group element `ct_1 := [r_enc]G`
/// - public group element `ct_2 := p_m + [r]elgamal_public_key`
/// - public random element `r_encode`
/// - public group element `elgamal_public_key`
/// - public generator `G`;


use crate::add_sub_mul::chip::{
    AddInstructions, AddSubMulChip, AddSubMulConfig, AddSubMulInstructions,
    SubInstructions,
};
use crate::constants::fixed_bases::VerifiableEncryptionFixedBases;
use crate::elgamal::extended_elgamal::{DataInTransmit, extended_elgamal_decrypt, extended_elgamal_encrypt};
use ff::{Field, PrimeField};
use group::prime::PrimeCurveAffine;
use group::Curve;
use halo2_gadgets::ecc::chip::{EccChip, EccConfig};
use halo2_gadgets::ecc::{NonIdentityPoint, ScalarVar};
use halo2_gadgets::utilities::UtilitiesInstructions;
use halo2_proofs::{
    circuit::{Chip, Layouter, SimpleFloorPlanner, Value},
    plonk::{Circuit, Column, ConstraintSystem, Error, Instance as InstanceColumn},
};
use pasta_curves::arithmetic::CurveAffine;
use pasta_curves::{Fp, pallas, vesta};
use rand::rngs::OsRng;
use crate::elgamal::elgamal::ElGamalKeypair;
use halo2_gadgets::utilities::lookup_range_check::LookupRangeCheckConfig;
use halo2_proofs::circuit::AssignedCell;
use pasta_curves::pallas::{Affine, Base};

const K: u32 = 11;

const ZERO: usize = 0;
const ELGAMAL_CT1_X: usize = 1;
const ELGAMAL_CT1_Y: usize = 2;

const ELGAMAL_CT2_X: usize = 3;
const ELGAMAL_CT2_Y: usize = 4;
const ELGAMAL_PK_X: usize = 5;
const ELGAMAL_PK_Y: usize = 6;

#[derive(Clone, Debug)]
pub struct VeConfig {
    pub(crate) instance: Column<InstanceColumn>,
    pub(crate) ecc_config: EccConfig<VerifiableEncryptionFixedBases>,
    pub(crate) add_sub_mul_config: AddSubMulConfig,
}

#[derive(Default,Clone)]
pub struct VeEncCircuit {
    pub(crate) data_in_transmit: DataInTransmit,
    pub(crate) elgamal_public_key: pallas::Point,
    pub(crate) m: Value<pallas::Base>,
    pub(crate) p_m: Value<pallas::Point>,
    pub(crate) r_enc: Value<pallas::Base>,
}
impl Circuit<pallas::Base> for VeEncCircuit {
    type Config = VeConfig;
    type FloorPlanner = SimpleFloorPlanner;

    fn without_witnesses(&self) -> Self {
        Self::default()
    }

    fn configure(meta: &mut ConstraintSystem<pallas::Base>) -> Self::Config {

        let advices = [
            meta.advice_column(),
            meta.advice_column(),
            meta.advice_column(),
            meta.advice_column(),
            meta.advice_column(),
            meta.advice_column(),
            meta.advice_column(),
            meta.advice_column(),
            meta.advice_column(),
            meta.advice_column(),
        ];
        let table_idx = meta.lookup_table_column();
        let table_range_check_tag = meta.lookup_table_column();

        // Instance column used for public inputs
        let instance = meta.instance_column();
        meta.enable_equality(instance);

        // Permutation over all advice columns.
        for advice in advices.iter() {
            meta.enable_equality(*advice);
        }

        let lagrange_coeffs = [
            meta.fixed_column(),
            meta.fixed_column(),
            meta.fixed_column(),
            meta.fixed_column(),
            meta.fixed_column(),
            meta.fixed_column(),
            meta.fixed_column(),
            meta.fixed_column(),
        ];

        // Shared fixed column for loading constants
        let constant = lagrange_coeffs[0];
        meta.enable_constant(constant);

        // Shared advice column for loading advice
        let advice = [advices[8], advices[9]];

        let add_sub_mul_config = AddSubMulChip::configure(meta, advice, instance, constant);

        let range_check =
            LookupRangeCheckConfig::configure(meta, advices[9], table_idx, table_range_check_tag);


        // Configuration for curve point operations.
        // This uses 10 advice columns and spans the whole circuit.
        let ecc_config = EccChip::<VerifiableEncryptionFixedBases>::configure(
            meta,
            advices,
            lagrange_coeffs,
            range_check,
        );

        VeConfig {
            instance,
            ecc_config,
            add_sub_mul_config,
        }
    }

    fn synthesize(
        &self,
        config: Self::Config,
        mut layouter: impl Layouter<pallas::Base>,
    ) -> Result<(), Error> {
        // Construct the add, sub, mul chip.
        let add_sub_mul_chip = AddSubMulChip::new(config.add_sub_mul_config.clone());
        // Construct the ECC chip.
        let ecc_chip = EccChip::construct(config.ecc_config.clone());

        // Load 10-bit lookup table.
        config.ecc_config.lookup_config.load(&mut layouter)?;

        let column = ecc_chip.config().advices[0];

        // witness message point p_m
        let p_m = NonIdentityPoint::new(
            ecc_chip.clone(),
            layouter.namespace(|| "load p_m"),
            self.p_m.as_ref().map(|p_m| p_m.to_affine()),
        )?;
        // load randomness r_encode
        let r_encode = add_sub_mul_chip.load_private(
            layouter.namespace(|| "load r_encode"),
            Value::known(self.data_in_transmit.r_encode),
        )?;

        // load dsa_private_key = message
        let message =
            add_sub_mul_chip.load_private(layouter.namespace(|| "load message"), self.m)?;

        // load r_enc
        let assigned_r_enc =
            ecc_chip.load_private(layouter.namespace(|| "load r_enc"), column, self.r_enc)?;

        // elgamal_public_key
        let elgamal_public_key = NonIdentityPoint::new(
            ecc_chip.clone(),
            layouter.namespace(|| "load elgamal_public_key"),
            Value::known(self.elgamal_public_key.to_affine()),
        )?;


        check_encryption(
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

pub(crate) fn check_encryption(
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
    // (1) Encode(m; r_encode) = p_m, that is,
    // (1.1) p_m.x = r_encode + m

    // compute res = m + r_encode - p_m.x
    let exp_m = add_sub_mul_chip.add(
        layouter.namespace(|| "m + r_encode"),
        message.clone(),
        r_encode,
    )?;
    let res = add_sub_mul_chip.sub(
        layouter.namespace(|| "m + r_encode - p_m.x"),
        exp_m,
        p_m.inner().x(),
    )?;

    // check if res = 0
    add_sub_mul_chip.check_result(layouter.namespace(|| "check res"), res, 0)?;

    // (2) C = ElGamal.Enc(pk, p_m)
    // (2.1) ct_1 = [r_enc]generator
    // r_enc
    let r_enc = ScalarVar::from_base(
        ecc_chip.clone(),
        layouter.namespace(|| "r_enc"),
        &assigned_r_enc,
    )?;

    // generator
    let generator = NonIdentityPoint::new(
        ecc_chip.clone(),
        layouter.namespace(|| "load generator"),
        Value::known(pallas::Affine::generator()),
    )?;

    // compute [r_enc]generator
    let (ct1_expected, _) =
        { generator.mul(layouter.namespace(|| "[r_enc]generator"), r_enc)? };

    // Constrain ct1_expected to equal public input ct1
    layouter.constrain_instance(
        ct1_expected.inner().x().cell(),
        config.instance,
        ELGAMAL_CT1_X,
    )?;
    layouter.constrain_instance(
        ct1_expected.inner().y().cell(),
        config.instance,
        ELGAMAL_CT1_Y,
    )?;

    // (2.2) ct_2 = p_m +[r_enc]pk
    // r_enc
    let r_enc = ScalarVar::from_base(
        ecc_chip.clone(),
        layouter.namespace(|| "r_enc"),
        &assigned_r_enc,
    )?;

    // Constrain elgamal_public_key to equal public input pk
    layouter.constrain_instance(
        elgamal_public_key.inner().x().cell(),
        config.instance,
        ELGAMAL_PK_X,
    )?;
    layouter.constrain_instance(
        elgamal_public_key.inner().y().cell(),
        config.instance,
        ELGAMAL_PK_Y,
    )?;

    // Compute [r_enc]elgamal_public_key
    let (r_mul_pk, _) =
        { elgamal_public_key.mul(layouter.namespace(|| "[r_enc]elgamal_public_key"), r_enc)? };

    // Compute ct_2_expected = [r_enc]elgamal_public_key + p_m
    let ct_2_expected =
        r_mul_pk.add(layouter.namespace(|| "[r_enc]elgamal_public_key+p_m"), &p_m)?;

    // Constrain ct_2_expected to equal public input ct_2
    layouter.constrain_instance(
        ct_2_expected.inner().x().cell(),
        config.instance,
        ELGAMAL_CT2_X,
    )?;
    layouter.constrain_instance(
        ct_2_expected.inner().y().cell(),
        config.instance,
        ELGAMAL_CT2_Y,
    )?;
    Ok(())
}

/// Public inputs
#[derive(Clone, Debug)]
pub struct VeEncInstance {
    pub(crate) data_in_transmit: DataInTransmit,
    pub(crate) elgamal_public_key: pallas::Point,
}

impl VeEncInstance {
    pub(crate) fn to_halo2_instance(&self) -> [[vesta::Scalar; 7]; 1] {
        let mut instance = [vesta::Scalar::random(OsRng); 7];
        instance[ZERO] = vesta::Scalar::zero();

        instance[ELGAMAL_CT1_X] = *self.data_in_transmit.ct.c1.to_affine().coordinates().unwrap().x();
        instance[ELGAMAL_CT1_Y] = *self.data_in_transmit.ct.c1.to_affine().coordinates().unwrap().y();

        instance[ELGAMAL_CT2_X] = *self.data_in_transmit.ct.c2.to_affine().coordinates().unwrap().x();
        instance[ELGAMAL_CT2_Y] = *self.data_in_transmit.ct.c2.to_affine().coordinates().unwrap().y();

        instance[ELGAMAL_PK_X] = *self
            .elgamal_public_key
            .to_affine()
            .coordinates()
            .unwrap()
            .x();
        instance[ELGAMAL_PK_Y] = *self
            .elgamal_public_key
            .to_affine()
            .coordinates()
            .unwrap()
            .y();

        [instance]
    }
}

pub(crate) fn create_circuit(message: pallas::Base, keypair: ElGamalKeypair) -> VeEncCircuit {
    // Elgamal encryption
    let (data_in_transmit, elgamal_secret) =
        extended_elgamal_encrypt(&keypair.public_key, message);
    let decrypted_message =
        extended_elgamal_decrypt(&keypair.private_key, data_in_transmit.clone())
            .expect("Decryption failed");
    // Verify decryption
    assert_eq!(message, decrypted_message);

    // convert r_enc to base value
    let r_enc = pallas::Base::from_repr(elgamal_secret.r_enc.to_repr()).unwrap();

    VeEncCircuit {
        data_in_transmit: data_in_transmit,
        elgamal_public_key: keypair.public_key,
        m: Value::known(message),
        p_m: Value::known(elgamal_secret.p_m),
        r_enc: Value::known(r_enc),
    }
}

#[cfg(test)]
mod tests {
    use std::time::Instant;
    use super::{create_circuit, VeEncInstance, K};
    use crate::elgamal::elgamal::ElGamalKeypair;
    use crate::encode::utf8::{
        convert_string_to_u8_array, convert_u8_array_to_u64_array, split_message_into_blocks,
    };
    use halo2_proofs::poly::commitment::Params;
    use halo2_proofs::transcript::{Blake2bRead, Blake2bWrite, Challenge255};
    use halo2_proofs::{plonk};
    use halo2_proofs::plonk::SingleVerifier;
    use pasta_curves::{pallas, vesta};
    use rand::rngs::OsRng;


    #[test]
    fn round_trip() {
        let mut rng = OsRng;

        // Split the message into blocks
        let test_message = "This is a short message.";
        // let test_message = "This is a long message for test!";

        // Specify the block size as 31 bytes
        let block_size = 31;
        let blocks = split_message_into_blocks(test_message, block_size);

        // Elgamal keygen
        let keypair = ElGamalKeypair::new();

        let start = Instant::now();
        // Setup phase: generate parameters for the circuit.
        let params = Params::new(K);

        let duration = start.elapsed();

        // Print the duration to see how long the function took.
        println!("Time elapsed in my_function() is: {:?}", duration);


        // Create a circuit for each block
        for (_, block) in blocks.iter().enumerate() {
            // convert message block to a Fp element
            let bytes = convert_string_to_u8_array(block);
            let m = pallas::Base::from_raw(convert_u8_array_to_u64_array(bytes));

            // Step 1. create a circuit
            let circuit = vec![create_circuit(m, keypair.clone())];

            // Step 2. arrange the public instance.
            let instance = vec![VeEncInstance {
                data_in_transmit: circuit[0].data_in_transmit.clone(),
                elgamal_public_key: circuit[0].elgamal_public_key.clone(),
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
            let circuit_cost =
                halo2_proofs::dev::CircuitCost::<vesta::Point, _>::measure(K, &circuit[0]);
            let expected_proof_size = usize::from(circuit_cost.proof_size(instance.len()));
            println!("Proof length: {}B", expected_proof_size);

            assert_eq!(proof.len(), expected_proof_size);


        }

    }
}