use crate::encode::utf8::convert_u8_array_to_u64_array;
/// Defined in [Verifiable Encryption using Halo2][Section 2.3. Encode a Message into a Point].
/// encode allows to encode a Fp message to an ECC point
/// decode to decode an ECC point to a Fp message
use ff::Field;
use group::prime::PrimeCurveAffine;
use group::Curve;
use pasta_curves::arithmetic::CurveAffine;
use pasta_curves::{pallas, Fp};
use rand::rngs::OsRng;
use subtle::CtOption;

/// Defined in [Verifiable Encryption using Halo2][Section 2.3. Encode a Message into a Point][Encode, step 3 - 4].
pub fn find_point_from_scalar(x_m: pallas::Base) -> Option<pallas::Point> {
    // compute a point on curve by the x value
    let y_square = x_m * x_m * x_m + pallas::Base::from(5);
    let y: CtOption<Fp> = y_square.sqrt();
    // if a square root y exists, return p_m =(x_m,y), otherwise, return None
    if y.is_some().unwrap_u8() == 1u8 {
        let m_point = pallas::Affine::from_xy(x_m, y.unwrap()).unwrap();
        let p_m = m_point.to_curve();
        return Some(p_m);
    }
    return None;
}

/// Encode function
pub fn encode(m: pallas::Base) -> (pallas::Point, pallas::Base) {
    let mut x_m;
    let mut r;
    let p_m;

    loop {
        // Defined in [Verifiable Encryption using Halo2][Section 2.3. Encode a Message into a Point][Encode, step 1-2].
        // add a random element r to the message to ensure there exists a point with x-coordinates x_m on curve
        // repeat until a point is found
        let rng = OsRng;
        r = pallas::Base::random(rng);
        x_m = m + r;

        // find a point by x_m
        match find_point_from_scalar(x_m) {
            Some(point) => {
                p_m = point;
                break;
            }
            None => continue,
        }
    }
    return (p_m, r);
}

/// Decode function
pub fn decode(pt: pallas::Point, r: pallas::Base) -> pallas::Base {
    // get the x-coordinate x_m of the affine point (x_m, y)
    // compute m = x_m -r
    pt.to_affine().coordinates().unwrap().x() - r
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encode::utf8::{convert_string_to_u8_array, convert_u8_array_to_string};
    use ff::PrimeField;

    #[test]
    fn test_encode_decode_scalar() {
        use rand::rngs::OsRng;
        let rng = OsRng;
        for _ in 0..1000 {
            let scalar = pallas::Base::random(rng);

            let (encoded, r) = encode(scalar);

            let decoded = decode(encoded, r);

            // Check if decoding(encoding(scalar)) == scalar
            assert_eq!(scalar, decoded);
        }
    }

    #[test]
    fn test_encode_decode_string() {
        use rand::{distributions::Alphanumeric, Rng};
        let mut rng = OsRng;

        for _ in 0..1000 {
            // check the encode and decode for strings shorter than 32
            // generate a random length for string
            let num = rng.gen_range(0..32);

            // generate a random string of length num
            let random_string: String = OsRng
                .sample_iter(&Alphanumeric)
                .take(num)
                .map(char::from)
                .collect();

            let m = pallas::Base::from_raw(convert_u8_array_to_u64_array(
                convert_string_to_u8_array(&random_string),
            ));

            let (encoded, r) = encode(m);
            let decoded = decode(encoded, r);

            let str = convert_u8_array_to_string(decoded.to_repr());
            // Check if decoding(encoding(random_string)) == random_string
            assert_eq!(random_string, str);
        }
    }
}
