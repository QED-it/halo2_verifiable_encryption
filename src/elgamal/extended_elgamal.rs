/// Defined in [Verifiable Encryption using Halo2][Section 2.4. Real Application Process].
use crate::elgamal::elgamal::{
    elgamal_decrypt, elgamal_encrypt, ElGamalCiphertext, ElGamalKeypair,
};
use crate::encode::encode::{decode, encode};
use ff::Field;
use pasta_curves::pallas;

// Define the DataInTransmit tuple
#[derive(Clone, Debug, Default)]
pub struct DataInTransmit {
    pub ct: ElGamalCiphertext,  // ElGamal Ciphertext
    pub r_encode: pallas::Base, // randomness for encoding and decoding
}

// Define the encryptor's witness values
#[derive(Clone, Debug, Default)]
pub struct Witness {
    pub m: pallas::Base,       // message m
    pub p_m: pallas::Point,    // message point p_m
    pub r_enc: pallas::Scalar, // randomness for encryption
}

// Encode + ElGamal encryption
pub fn extended_elgamal_encrypt(
    public_key: &pallas::Point,
    message: pallas::Base,
) -> (DataInTransmit, Witness) {
    // encode m to point p_m
    let (p_m, r_encode) = encode(message);

    // encrypting p_m
    let (ct, witness) = elgamal_encrypt(public_key, p_m);
    assert_eq!(witness.p_m, p_m);
    (
        DataInTransmit { ct, r_encode },
        Witness {
            m: message,
            p_m: p_m,
            r_enc: witness.r_enc,
        },
    )
}

// ElGamal decryption + Decode
pub fn extended_elgamal_decrypt(
    private_key: &pallas::Scalar,
    data_in_transmit: DataInTransmit,
) -> Option<pallas::Base> {
    // Decrypt ct to obtain the message point p_m
    let p_m = elgamal_decrypt(private_key, &data_in_transmit.ct);

    // decode p_m to m
    let m = decode(p_m.unwrap(), data_in_transmit.r_encode);
    Some(m)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_elgamal_encryption_and_decryption() {
        for _ in 0..1000 {
            let keypair = ElGamalKeypair::new();
            use rand::rngs::OsRng;
            let rng = OsRng;
            // encode and encrypt a random message m
            let m = pallas::Base::random(rng);
            let (data_in_transmit, _) = extended_elgamal_encrypt(&keypair.public_key, m);
            let decrypted_plaintext =
                extended_elgamal_decrypt(&keypair.private_key, data_in_transmit)
                    .expect("Decryption failed");

            // Verify decryption
            assert_eq!(m, decrypted_plaintext);
        }
    }
}
