//! Sinsemilla generators
use crate::constants::fixed_bases::VerifiableEncryptionFixedBases;
use group::Curve;

use halo2_gadgets::sinsemilla::{CommitDomains, HashDomains};

use crate::constants::fixed_bases::FullWidth;
use halo2_gadgets::ecc::chip::find_zs_and_us;
use pasta_curves::pallas;

use halo2_gadgets::{
    ecc::chip::{H, NUM_WINDOWS},
    sinsemilla::primitives::{self as sinsemilla},
};
use lazy_static::lazy_static;

pub(crate) const PERSONALIZATION: &str = "MerkleCRH";

lazy_static! {
    static ref COMMIT_DOMAIN: sinsemilla::CommitDomain =
        sinsemilla::CommitDomain::new(PERSONALIZATION);
    static ref Q: pallas::Affine = COMMIT_DOMAIN.Q().to_affine();
    static ref R: pallas::Affine = COMMIT_DOMAIN.R().to_affine();
    static ref R_ZS_AND_US: Vec<(u64, [pallas::Base; H])> =
        find_zs_and_us(*R, NUM_WINDOWS).unwrap();
}
#[derive(Debug, Clone, Eq, PartialEq)]
pub(crate) struct VerifiableEncryptionHashDomain;
impl HashDomains<pallas::Affine> for VerifiableEncryptionHashDomain {
    fn Q(&self) -> pallas::Affine {
        *Q
    }
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub(crate) struct VerifiableEncryptionCommitDomain;
impl CommitDomains<pallas::Affine, VerifiableEncryptionFixedBases, VerifiableEncryptionHashDomain>
    for VerifiableEncryptionCommitDomain
{
    fn r(&self) -> FullWidth {
        FullWidth::from_parts(*R, &R_ZS_AND_US)
    }

    fn hash_domain(&self) -> VerifiableEncryptionHashDomain {
        VerifiableEncryptionHashDomain
    }
}
