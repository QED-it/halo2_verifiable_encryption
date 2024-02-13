use crate::constants::fixed_bases::VerifiableEncryptionFixedBases;
use crate::constants::sinsemilla::{
    VerifiableEncryptionCommitDomain, VerifiableEncryptionHashDomain,
};
use ff::Field;
use group::prime::PrimeCurveAffine;
use group::Curve;
use halo2_gadgets::ecc::chip::{EccChip, EccConfig};
/// Defined in [Verifiable Encryption using Halo2][Section 3.3. Task 2 - Verifiable Encryption with relation R].
/// Format a circuit and an instance for Encode, Elgamal decryption and DSA keypair
/// A round trip test to prove ciphertext is an encryption of an encoded scalar message
/// and the scalar message is the private key of a DSA keypair
///
/// This file shows how to build a proof of knowledge of message in a ciphertext
/// Prove:
/// (1) Encode(m; r_encode) = p_m, that is,
/// (1.1) p_m.x = r_encode + m
/// (1.2) p_m.x^3 + 5 = p_m.y^2
/// (2) C = ElGamal.Enc(pk, p_m)
/// (2.1) ct_1 = [r_enc]G, G is the generator of E
/// (2.2) ct_2 = p_m +[r_enc]pk_elgamal
/// (4) pk_dsa = [m]G
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
use halo2_gadgets::ecc::{NonIdentityPoint, ScalarVar};
use halo2_gadgets::sinsemilla::chip::{SinsemillaChip, SinsemillaConfig};
use halo2_proofs::{
    circuit::{Chip, Layouter, SimpleFloorPlanner, Value},
    plonk::{Circuit, Column, ConstraintSystem, Error, Instance as InstanceColumn},
};
use pasta_curves::arithmetic::CurveAffine;
use pasta_curves::{pallas, vesta};
use rand;
use rand::rngs::OsRng;

use crate::add_sub_mul::add_sub_mul::{
    AddInstructions, AddSubMulChip, AddSubMulConfig, AddSubMulInstructions, MulInstructions,
    SubInstructions,
};

use crate::elgamal::extended_elgamal::DataInTransmit;
use halo2_gadgets::utilities::lookup_range_check::LookupRangeCheckConfig;
use halo2_gadgets::utilities::UtilitiesInstructions;
const K: u32 = 11;
const ZERO: usize = 0;
const ELGAMAL_CT1_X: usize = 1;
const ELGAMAL_CT1_Y: usize = 2;

const ELGAMAL_CT2_X: usize = 3;
const ELGAMAL_CT2_Y: usize = 4;
const ELGAMAL_PK_X: usize = 5;
const ELGAMAL_PK_Y: usize = 6;
const DSA_PK_X: usize = 7;
const DSA_PK_Y: usize = 8;

#[derive(Clone, Debug)]
pub struct Config {
    instance: Column<InstanceColumn>,
    ecc_config: EccConfig<VerifiableEncryptionFixedBases>,
    add_sub_mul_config: AddSubMulConfig,
    sinsemilla_config: SinsemillaConfig<
        VerifiableEncryptionHashDomain,
        VerifiableEncryptionCommitDomain,
        VerifiableEncryptionFixedBases,
    >,
}

#[derive(Default, Clone)]
struct MyCircuit {
    ct: DataInTransmit,
    elgamal_public_key: pallas::Point,
    dsa_public_key: pallas::Point,
    m: Value<pallas::Base>,
    p_m: Value<pallas::Point>,
    r_enc: Value<pallas::Base>,
}
impl Circuit<pallas::Base> for MyCircuit {
    type Config = Config;

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

        let lookup = (
            table_idx,
            meta.lookup_table_column(),
            meta.lookup_table_column(),
            table_range_check_tag,
        );

        // Shared advice column for loading advice
        let advice = [advices[8], advices[9]];

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

        let add_sub_mul_config = AddSubMulChip::configure(meta, advice, instance, constant);

        let range_check =
            LookupRangeCheckConfig::configure(meta, advices[9], table_idx, table_range_check_tag);

        let sinsemilla_config = SinsemillaChip::configure(
            meta,
            advices[..5].try_into().unwrap(),
            advices[6],
            lagrange_coeffs[0],
            lookup,
            range_check,
        );

        // Configuration for curve point operations.
        // This uses 10 advice columns and spans the whole circuit.
        let ecc_config = EccChip::<VerifiableEncryptionFixedBases>::configure(
            meta,
            advices,
            lagrange_coeffs,
            range_check,
        );

        Config {
            instance,
            ecc_config,
            add_sub_mul_config,
            sinsemilla_config,
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

        // Load the Sinsemilla generator lookup table used by the whole circuit.
        SinsemillaChip::load(config.sinsemilla_config.clone(), &mut layouter)?;

        let column = ecc_chip.config().advices[0];

        // (1) ct_1 = [r_enc]G, G is the generator of E
        // r_enc
        let assigned_r_enc =
            ecc_chip.load_private(layouter.namespace(|| "load r_enc"), column, self.r_enc)?;
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

        // (2) ct_2 = p_m +[r_enc]pk_elgamal
        // witness message point p_m
        let p_m = NonIdentityPoint::new(
            ecc_chip.clone(),
            layouter.namespace(|| "load p_m"),
            self.p_m.as_ref().map(|p_m| p_m.to_affine()),
        )?;

        // r_enc
        let r_enc = ScalarVar::from_base(
            ecc_chip.clone(),
            layouter.namespace(|| "r_enc"),
            &assigned_r_enc,
        )?;

        // elgamal_public_key
        let elgamal_public_key = NonIdentityPoint::new(
            ecc_chip.clone(),
            layouter.namespace(|| "load elgamal_public_key"),
            Value::known(self.elgamal_public_key.to_affine()),
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

        // (3) Encode(m; r_encode) = p_m, that is,
        // (3.1) pm.x = m + r_encode
        // load randomness r_encode
        let r_encode = add_sub_mul_chip.load_private(
            layouter.namespace(|| "load r_encode"),
            Value::known(self.ct.r_encode),
        )?;

        // load dsa_private_key = message
        let message =
            add_sub_mul_chip.load_private(layouter.namespace(|| "load message"), self.m)?;

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

        // (3.2) p_m.x^3 + 5 = p_m.y^2
        let x2 = add_sub_mul_chip.mul(
            layouter.namespace(|| "x*x"),
            p_m.inner().x().clone(),
            p_m.inner().x().clone(),
        )?;
        let x3 =
            add_sub_mul_chip.mul(layouter.namespace(|| "x*x*x"), p_m.inner().x().clone(), x2)?;

        let five = add_sub_mul_chip
            .load_constant(layouter.namespace(|| "load 5"), pallas::Base::from(5))?;

        let left = add_sub_mul_chip.add(layouter.namespace(|| "x*x*x + 5"), x3, five)?;

        let right = add_sub_mul_chip.mul(
            layouter.namespace(|| "y*y"),
            p_m.inner().y().clone(),
            p_m.inner().y().clone(),
        )?;

        let res = add_sub_mul_chip.sub(layouter.namespace(|| "x*x*x + 5 - y*y"), left, right)?;

        // check if x*x*x + 5 - y*y = 0
        add_sub_mul_chip.check_result(layouter.namespace(|| "check res"), res, 0)?;

        // (4) dsa_public_key = [m]generator
        // convert message to scalar, it is dsa_private_key
        let message_scalar = ScalarVar::from_base(
            ecc_chip.clone(),
            layouter.namespace(|| "message to scalar"),
            &message,
        )?;

        // compute dsa_pk_expected = [message_scalar]generator
        let (dsa_pk_expected, _) = {
            generator.mul(
                layouter.namespace(|| "[message_scalar]generator"),
                message_scalar,
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

        Ok(())
    }
}

/// Public inputs
#[derive(Clone, Debug)]
pub struct MyInstance {
    ct: DataInTransmit,
    elgamal_public_key: pallas::Point,
    dsa_public_key: pallas::Point,
}

impl MyInstance {
    fn to_halo2_instance(&self) -> [[vesta::Scalar; 9]; 1] {
        let mut instance = [vesta::Scalar::random(OsRng); 9];
        instance[ZERO] = vesta::Scalar::zero();

        instance[ELGAMAL_CT1_X] = *self.ct.ct.c1.to_affine().coordinates().unwrap().x();
        instance[ELGAMAL_CT1_Y] = *self.ct.ct.c1.to_affine().coordinates().unwrap().y();

        instance[ELGAMAL_CT2_X] = *self.ct.ct.c2.to_affine().coordinates().unwrap().x();
        instance[ELGAMAL_CT2_Y] = *self.ct.ct.c2.to_affine().coordinates().unwrap().y();

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

        instance[DSA_PK_X] = *self.dsa_public_key.to_affine().coordinates().unwrap().x();
        instance[DSA_PK_Y] = *self.dsa_public_key.to_affine().coordinates().unwrap().y();

        [instance]
    }
}

#[cfg(test)]
mod tests {
    use super::{MyCircuit, MyInstance, K};
    use crate::elgamal::elgamal::ElGamalKeypair;
    use crate::elgamal::extended_elgamal::{extended_elgamal_decrypt, extended_elgamal_encrypt};
    use ff::{Field, PrimeField};
    use group::Group;
    use halo2_proofs::plonk::SingleVerifier;
    use halo2_proofs::poly::commitment::Params;
    use halo2_proofs::transcript::{Blake2bRead, Blake2bWrite, Challenge255};
    use halo2_proofs::{circuit::Value, plonk};
    use pasta_curves::{pallas, vesta};
    use rand::rngs::OsRng;

    fn create_circuit(message: pallas::Base, elgamal_keypair: ElGamalKeypair) -> MyCircuit {
        // map base to scalar
        let dsa_private_key = pallas::Scalar::from_repr(message.to_repr()).unwrap();

        // Calculate the dsa public key [dsa_private_key]G
        let dsa_public_key = pallas::Point::generator() * dsa_private_key;

        // Elgamal encryption, encrypt message to ciphertext, the underlying message point is p_m
        // p_m.x = m + ciphertext.r
        let (data_in_transmit, elgamal_secret) =
            extended_elgamal_encrypt(&elgamal_keypair.public_key, message);
        let decrypted_message =
            extended_elgamal_decrypt(&elgamal_keypair.private_key, data_in_transmit.clone())
                .expect("Decryption failed");

        // convert r_enc to base value
        let r_enc = pallas::Base::from_repr(elgamal_secret.r_enc.to_repr()).unwrap();

        // Verify decryption
        assert_eq!(message, decrypted_message);

        MyCircuit {
            ct: data_in_transmit,
            elgamal_public_key: elgamal_keypair.public_key,
            dsa_public_key: dsa_public_key,
            m: Value::known(message),
            p_m: Value::known(elgamal_secret.p_m),
            r_enc: Value::known(r_enc),
        }
    }
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
        let instance = vec![MyInstance {
            ct: circuit[0].ct.clone(),
            elgamal_public_key: circuit[0].elgamal_public_key.clone(),
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
}
