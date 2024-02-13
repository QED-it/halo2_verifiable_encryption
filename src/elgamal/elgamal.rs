/// Defined in [Verifiable Encryption using Halo2][Section 2.2. ECElgamal].
use ff::Field;
use group::Group;
use pasta_curves::pallas;
use rand::rngs::OsRng;

// Define the keypair for the ElGamal cryptosystem
#[derive(Clone, Debug)]
pub struct ElGamalKeypair {
    pub public_key: pallas::Point, // [private_key]G, where G is the generator point
    pub private_key: pallas::Scalar, // A secret scalar
}

// Define the ElGamal ciphertext tuple
#[derive(Clone, Debug, Default)]
pub struct ElGamalCiphertext {
    pub c1: pallas::Point, // [r_enc] G where G is the generator and r_enc is a nonce
    pub c2: pallas::Point, // p_m + [r_enc] public_key
}

// Define the ElGamal encryptor's witness values
#[derive(Clone, Debug, Default)]
pub struct Witness {
    pub p_m: pallas::Point,    // message point p_m
    pub r_enc: pallas::Scalar, // randomness for encryption
}
impl ElGamalKeypair {
    // Generate a new keypair for use with ElGamal encryption
    pub fn new() -> Self {
        // Secure random number generator
        let mut rng = OsRng;

        // Generate the secret scalar
        let private_key = pallas::Scalar::random(&mut rng);

        // Calculate the public key G^private_key
        let public_key = pallas::Point::generator() * private_key;

        Self {
            public_key,
            private_key,
        }
    }
}

// ElGamal encryption
pub fn elgamal_encrypt(
    public_key: &pallas::Point,
    p_m: pallas::Point,
) -> (ElGamalCiphertext, Witness) {
    let mut rng = OsRng;

    // Generate a random nonce r_enc
    let r_enc = pallas::Scalar::random(&mut rng);
    // c1 = [r_enc]G
    let c1 = pallas::Point::generator() * r_enc;
    // c2 = p_m + [r_enc]public_key
    let c2 = p_m + public_key * r_enc;
    (
        ElGamalCiphertext { c1, c2 },
        Witness {
            p_m: p_m,
            r_enc: r_enc,
        },
    )
}

// ElGamal decryption
pub fn elgamal_decrypt(
    private_key: &pallas::Scalar,
    ciphertext: &ElGamalCiphertext,
) -> Option<pallas::Point> {
    // Decrypt the message point p_m using the private key: c2 - [private_key]c1
    // Decode p_m to the message

    // compute p_m = c2 - [private_key]c1
    let p_m = ciphertext.c2 - ciphertext.c1 * private_key;

    Some(p_m)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_elgamal_encryption_and_decryption() {
        use rand::rngs::OsRng;
        let rng = OsRng;
        for _ in 0..1000 {
            let keypair = ElGamalKeypair::new();
            // Your message
            let p_m = pallas::Point::random(rng);

            // Elgamal encryption
            let (ciphertext, _) = elgamal_encrypt(&keypair.public_key, p_m);

            // Elgamal decryption
            let decrypted_plaintext =
                elgamal_decrypt(&keypair.private_key, &ciphertext).expect("Decryption failed");

            // Verify decryption
            assert_eq!(p_m, decrypted_plaintext);
        }
    }
}
