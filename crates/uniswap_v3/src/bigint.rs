//! Minimal big-integer support for the Uniswap-V3 reference and hint encoding.
//!
//! This is the *trusted reference* side: a small, obviously-correct `U256`
//! (and a 512-bit product type) used to (a) compute the expected `mulDiv`
//! quotient/remainder against which the guest is differentially tested, and
//! (b) encode operands and the prover's `q`/`r` as little-endian base-2^8 limb
//! buffers (`Vec<F>`), exactly as the guest consumes them.
//!
//! Representation: `U256` is four little-endian `u64` words (`limbs[0]` least
//! significant). Limb-vector form (for the guest) is little-endian base-256:
//! `to_le_base256_limbs(32)` yields 32 `F` cells, cell `i` = byte `i`.

// Carry/borrow propagation loops read most clearly with explicit word indices
// (the value at index `i` depends on the carry produced at `i-1`); the iterator
// rewrites clippy suggests obscure that data flow, so we keep indexed loops.
#![allow(clippy::needless_range_loop)]

use lean_vm::F;

use backend::PrimeCharacteristicRing;

/// Unsigned 256-bit integer, little-endian `u64` words.
///
/// NOTE: `Ord`/`PartialOrd` are implemented by hand (most-significant word
/// first). The derived impls would compare `words[0]` (the LEAST significant
/// word) first, which is numerically wrong — a subtle bug we deliberately avoid.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct U256 {
    /// Little-endian 64-bit limbs: `words[0]` is least significant.
    pub words: [u64; 4],
}

/// Unsigned 512-bit integer (the full product of two `U256`), little-endian words.
/// Same hand-written ordering caveat as `U256`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct U512 {
    pub words: [u64; 8],
}

impl Ord for U256 {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Compare from most significant word down.
        for i in (0..4).rev() {
            match self.words[i].cmp(&other.words[i]) {
                std::cmp::Ordering::Equal => {}
                ord => return ord,
            }
        }
        std::cmp::Ordering::Equal
    }
}
impl PartialOrd for U256 {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for U512 {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        for i in (0..8).rev() {
            match self.words[i].cmp(&other.words[i]) {
                std::cmp::Ordering::Equal => {}
                ord => return ord,
            }
        }
        std::cmp::Ordering::Equal
    }
}
impl PartialOrd for U512 {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl U256 {
    pub const ZERO: U256 = U256 { words: [0; 4] };
    pub const ONE: U256 = U256 { words: [1, 0, 0, 0] };
    /// `2^256 - 1`.
    pub const MAX: U256 = U256 {
        words: [u64::MAX; 4],
    };

    pub fn from_u64(x: u64) -> U256 {
        U256 {
            words: [x, 0, 0, 0],
        }
    }

    /// Parse a base-10 decimal string into a `U256` (wrapping mod 2^256; callers
    /// pass in-range values). Used by tests to write large oracle constants
    /// readably. Each step: acc = acc*10 + digit, via the existing carrying ops.
    pub fn from_dec_str(s: &str) -> U256 {
        let mut acc = U256::ZERO;
        let ten = U256::from_u64(10);
        for ch in s.bytes() {
            assert!(ch.is_ascii_digit(), "from_dec_str: non-digit '{}'", ch as char);
            // acc *= 10  (low 256 bits of the product; values stay < 2^256).
            acc = acc.full_mul(ten).low_u256();
            let (s2, _carry) = acc.overflowing_add(U256::from_u64((ch - b'0') as u64));
            acc = s2;
        }
        acc
    }

    /// `Q96 = 2^96`.
    pub const Q96: U256 = U256 {
        words: [0, 0x0000_0001_0000_0000, 0, 0],
    };

    /// `type(uint160).max = 2^160 - 1`.
    pub const UINT160_MAX: U256 = U256 {
        words: [u64::MAX, u64::MAX, 0x0000_0000_FFFF_FFFF, 0],
    };

    /// `type(uint128).max = 2^128 - 1`.
    pub const UINT128_MAX: U256 = U256 {
        words: [u64::MAX, u64::MAX, 0, 0],
    };

    /// True iff `self < 2^160` (fits uint160).
    pub fn fits_uint160(self) -> bool {
        self.words[2] >> 32 == 0 && self.words[3] == 0
    }

    /// True iff `self < 2^128` (fits uint128).
    pub fn fits_uint128(self) -> bool {
        self.words[2] == 0 && self.words[3] == 0
    }

    /// Build a `U256` from 32 little-endian bytes (byte 0 = least significant).
    pub fn from_le_bytes(bytes: &[u8; 32]) -> U256 {
        let mut words = [0u64; 4];
        for (w, chunk) in words.iter_mut().zip(bytes.chunks_exact(8)) {
            *w = u64::from_le_bytes(chunk.try_into().unwrap());
        }
        U256 { words }
    }

    /// 32 little-endian bytes (byte 0 = least significant).
    pub fn to_le_bytes(self) -> [u8; 32] {
        let mut out = [0u8; 32];
        for (i, w) in self.words.iter().enumerate() {
            out[i * 8..i * 8 + 8].copy_from_slice(&w.to_le_bytes());
        }
        out
    }

    pub fn is_zero(self) -> bool {
        self.words == [0; 4]
    }

    /// Zero-extend to a `U512`.
    pub fn to_u512(self) -> U512 {
        let mut words = [0u64; 8];
        words[..4].copy_from_slice(&self.words);
        U512 { words }
    }

    /// Carrying addition modulo `2^256`; returns `(sum, carry_out)`.
    pub fn overflowing_add(self, rhs: U256) -> (U256, bool) {
        let mut words = [0u64; 4];
        let mut carry = 0u128;
        for i in 0..4 {
            let s = self.words[i] as u128 + rhs.words[i] as u128 + carry;
            words[i] = s as u64;
            carry = s >> 64;
        }
        (U256 { words }, carry != 0)
    }

    /// Wrapping subtraction modulo `2^256` (caller ensures `self >= rhs` when an
    /// exact difference is required).
    pub fn wrapping_sub(self, rhs: U256) -> U256 {
        let mut words = [0u64; 4];
        let mut borrow = 0i128;
        for i in 0..4 {
            let d = self.words[i] as i128 - rhs.words[i] as i128 - borrow;
            if d < 0 {
                words[i] = (d + (1i128 << 64)) as u64;
                borrow = 1;
            } else {
                words[i] = d as u64;
                borrow = 0;
            }
        }
        U256 { words }
    }

    /// Low 256 bits of `self * rhs`, asserting the high 256 bits are zero (so the
    /// product genuinely fits in `U256`). Test convenience for building exact
    /// dividends like `a*denom`.
    pub fn full_mul_into_u256_unchecked(self, rhs: U256) -> U256 {
        let p = self.full_mul(rhs);
        assert_eq!(p.words[4..], [0, 0, 0, 0], "full_mul_into_u256: product exceeds uint256");
        U256 {
            words: [p.words[0], p.words[1], p.words[2], p.words[3]],
        }
    }

    /// Full 256x256 -> 512-bit product (no truncation).
    pub fn full_mul(self, rhs: U256) -> U512 {
        let mut words = [0u64; 8];
        for i in 0..4 {
            let mut carry = 0u128;
            for j in 0..4 {
                let cur = words[i + j] as u128
                    + self.words[i] as u128 * rhs.words[j] as u128
                    + carry;
                words[i + j] = cur as u64;
                carry = cur >> 64;
            }
            words[i + 4] = words[i + 4].wrapping_add(carry as u64);
        }
        U512 { words }
    }

    /// Logical left shift by `bits` (modulo `2^256`).
    #[allow(clippy::should_implement_trait)] // intentional inherent method, not std::ops::Shl
    pub fn shl(self, bits: u32) -> U256 {
        if bits >= 256 {
            return U256::ZERO;
        }
        let word_shift = (bits / 64) as usize;
        let bit_shift = bits % 64;
        let mut words = [0u64; 4];
        for i in (0..4).rev() {
            if i < word_shift {
                continue;
            }
            let lo = self.words[i - word_shift];
            let mut v = (lo as u128) << bit_shift;
            if bit_shift > 0 && i > word_shift {
                v |= (self.words[i - word_shift - 1] as u128) >> (64 - bit_shift);
            }
            words[i] = v as u64;
        }
        U256 { words }
    }

    /// Logical right shift by `bits` (floor).
    #[allow(clippy::should_implement_trait)] // intentional inherent method, not std::ops::Shr
    pub fn shr(self, bits: u32) -> U256 {
        if bits >= 256 {
            return U256::ZERO;
        }
        let word_shift = (bits / 64) as usize;
        let bit_shift = bits % 64;
        let mut words = [0u64; 4];
        for i in 0..4 {
            if i + word_shift >= 4 {
                break;
            }
            let mut v = (self.words[i + word_shift] as u128) >> bit_shift;
            if bit_shift > 0 && i + word_shift + 1 < 4 {
                v |= (self.words[i + word_shift + 1] as u128) << (64 - bit_shift);
            }
            words[i] = v as u64;
        }
        U256 { words }
    }

    /// Little-endian base-256 limbs as `F` cells; `width` cells (one byte each).
    /// `width` must be at least the byte length of `self` (we assert no nonzero
    /// byte is dropped, so a too-small width is caught rather than silently
    /// truncating — which would desync the hint contract).
    pub fn to_le_base256_limbs(self, width: usize) -> Vec<F> {
        let bytes = self.to_le_bytes();
        if width < 32 {
            for &byte in bytes.iter().skip(width) {
                assert_eq!(byte, 0, "to_le_base256_limbs: value does not fit in {width} limbs");
            }
        }
        let mut out = Vec::with_capacity(width);
        for i in 0..width {
            let byte = if i < 32 { bytes[i] } else { 0 };
            out.push(F::from_u32(byte as u32));
        }
        out
    }
}

impl U512 {
    pub const ZERO: U512 = U512 { words: [0; 8] };

    pub fn is_zero(self) -> bool {
        self.words == [0; 8]
    }

    /// Low 256 bits (drops the high four words). Used by `U256::from_dec_str`.
    pub fn low_u256(self) -> U256 {
        U256 {
            words: [self.words[0], self.words[1], self.words[2], self.words[3]],
        }
    }

    /// In-place left shift by 1 bit (modulo `2^512`).
    pub fn shl1_in_place(&mut self) {
        let mut carry = 0u64;
        for w in &mut self.words {
            let new_carry = *w >> 63;
            *w = (*w << 1) | carry;
            carry = new_carry;
        }
    }

    /// Wrapping subtraction of a `U256` (caller ensures `self >= rhs`).
    pub fn wrapping_sub(self, rhs: U512) -> U512 {
        let mut words = [0u64; 8];
        let mut borrow = 0i128;
        for i in 0..8 {
            let d = self.words[i] as i128 - rhs.words[i] as i128 - borrow;
            if d < 0 {
                words[i] = (d + (1i128 << 64)) as u64;
                borrow = 1;
            } else {
                words[i] = d as u64;
                borrow = 0;
            }
        }
        U512 { words }
    }

    /// `floor(self / 2^bits)` truncated to a `U256` (asserts the result fits).
    /// Used for the exact `(ratio * magic) >> 128` shift in TickMath.
    pub fn shr_to_u256(self, bits: u32) -> U256 {
        assert!(bits < 512, "shr_to_u256: shift too large");
        let word_shift = (bits / 64) as usize;
        let bit_shift = bits % 64;
        let mut words = [0u64; 8];
        for i in 0..8 {
            if i + word_shift >= 8 {
                break;
            }
            let mut v = (self.words[i + word_shift] as u128) >> bit_shift;
            if bit_shift > 0 && i + word_shift + 1 < 8 {
                v |= (self.words[i + word_shift + 1] as u128) << (64 - bit_shift);
            }
            words[i] = v as u64;
        }
        assert_eq!(words[4..], [0, 0, 0, 0], "shr_to_u256: result exceeds uint256");
        U256 {
            words: [words[0], words[1], words[2], words[3]],
        }
    }

    /// 64 little-endian bytes (byte 0 = least significant).
    pub fn to_le_bytes(self) -> [u8; 64] {
        let mut out = [0u8; 64];
        for (i, w) in self.words.iter().enumerate() {
            out[i * 8..i * 8 + 8].copy_from_slice(&w.to_le_bytes());
        }
        out
    }

    /// Little-endian base-256 limbs as `F` cells (one byte each), padded/truncated
    /// to `width`. Truncation asserts the dropped bytes are zero.
    pub fn to_le_base256_limbs(self, width: usize) -> Vec<F> {
        let bytes = self.to_le_bytes();
        if width < 64 {
            for &byte in bytes.iter().skip(width) {
                assert_eq!(byte, 0, "U512::to_le_base256_limbs: value does not fit in {width} limbs");
            }
        }
        let mut out = Vec::with_capacity(width);
        for i in 0..width {
            let byte = if i < 64 { bytes[i] } else { 0 };
            out.push(F::from_u32(byte as u32));
        }
        out
    }

    /// `floor(self / denom)` and `self mod denom`, as `(q, r)` with both fitting
    /// in `U256` *for the inputs this crate uses* (caller guarantees the quotient
    /// of a 512-bit dividend by a uint256 denominator fits the relevant width).
    /// Schoolbook long division over the 512 bits, bit by bit.
    ///
    /// The running remainder is held in 512 bits so that `(r << 1) | bit` never
    /// overflows even when `denom` is close to `2^256` (a `U256` remainder would
    /// lose the top bit on the shift — that was a real correctness bug). Since
    /// `r < denom < 2^256` throughout, `r` stays well inside 512 bits.
    pub fn divmod_u256(self, denom: U256) -> (U512, U256) {
        assert!(!denom.is_zero(), "division by zero");
        let denom512 = denom.to_u512();
        let mut q = U512::ZERO;
        let mut r = U512::ZERO;
        // Process dividend bits from most significant (bit 511) down to 0.
        for bit in (0..512).rev() {
            r.shl1_in_place();
            let word = bit / 64;
            let off = bit % 64;
            let dividend_bit = (self.words[word] >> off) & 1;
            if dividend_bit == 1 {
                r.words[0] |= 1;
            }
            if r >= denom512 {
                r = r.wrapping_sub(denom512);
                let qw = bit / 64;
                let qoff = bit % 64;
                q.words[qw] |= 1u64 << qoff;
            }
        }
        // r < denom < 2^256, so its high words are zero: narrow back to U256.
        debug_assert_eq!(r.words[4..], [0, 0, 0, 0]);
        let r256 = U256 {
            words: [r.words[0], r.words[1], r.words[2], r.words[3]],
        };
        (q, r256)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn u(x: u64) -> U256 {
        U256::from_u64(x)
    }

    #[test]
    fn add_sub_roundtrip() {
        let a = U256::from_le_bytes(&[0xAB; 32]);
        let b = U256::from_le_bytes(&[0x12; 32]);
        let (s, _) = a.overflowing_add(b);
        assert_eq!(s.wrapping_sub(b), a);
    }

    #[test]
    fn full_mul_small() {
        let a = u(1_000_000_007);
        let b = u(998_244_353);
        let p = a.full_mul(b);
        let expect = 1_000_000_007u128 * 998_244_353u128;
        let lo = p.words[0] as u128 | ((p.words[1] as u128) << 64);
        assert_eq!(lo, expect);
        assert_eq!(p.words[2], 0);
    }

    #[test]
    fn divmod_basic() {
        let prod = u(1234567).full_mul(u(7654321));
        let denom = u(7654321);
        let (q, r) = prod.divmod_u256(denom);
        assert_eq!(q.words[0], 1234567);
        assert!(r.is_zero());
        assert_eq!(q.words[2..], [0, 0, 0, 0, 0, 0]);
    }

    /// Reconstruct `q*d + r` into 8 little-endian words and verify it equals
    /// `prod` exactly (asserting nothing flows past word 8). `q` is up to 512
    /// bits, so this does NOT assume the quotient fits in 256 bits.
    fn check_euclid(prod: U512, q: U512, d: U256, r: U256) {
        let mut w = [0u128; 9]; // 8 words + overflow lane
        for i in 0..8 {
            for j in 0..4 {
                if i + j < 8 {
                    w[i + j] += q.words[i] as u128 * d.words[j] as u128;
                } else {
                    // any product landing past word 8 must be zero for a 512-bit identity
                    assert_eq!(q.words[i] as u128 * d.words[j] as u128, 0, "q*d exceeds 512 bits");
                }
            }
        }
        // add remainder
        for j in 0..4 {
            w[j] += r.words[j] as u128;
        }
        // propagate carries
        let mut out = [0u64; 8];
        let mut carry = 0u128;
        for i in 0..8 {
            let s = w[i] + carry;
            out[i] = s as u64;
            carry = s >> 64;
        }
        carry += w[8];
        assert_eq!(carry, 0, "q*d + r overflows 512 bits");
        assert_eq!(U512 { words: out }, prod);
    }

    #[test]
    fn divmod_with_remainder() {
        // (a*b) / d where the quotient EXCEEDS 256 bits (d is small): verify the
        // full Euclidean identity q*d + r == a*b and r < d, with q at 512 bits.
        let a = U256::from_le_bytes(&[0x9F; 32]);
        let b = U256::from_le_bytes(&[0x37; 32]);
        let d = U256::from_le_bytes(&{
            let mut z = [0u8; 32];
            z[0] = 0xFD;
            z[1] = 0xAC;
            z[5] = 0x11;
            z
        });
        let prod = a.full_mul(b);
        let (q, r) = prod.divmod_u256(d);
        check_euclid(prod, q, d, r);
        assert!(r < d);
        // sanity: this case really does need >256-bit quotient
        assert_ne!(q.words[4..], [0, 0, 0, 0]);
    }

    #[test]
    fn limbs_le_base256_roundtrip() {
        let a = U256::from_le_bytes(&[0x01, 0x02, 0x03, 0x04, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        let limbs = a.to_le_base256_limbs(32);
        assert_eq!(limbs.len(), 32);
        assert_eq!(limbs[0], F::from_u32(1));
        assert_eq!(limbs[3], F::from_u32(4));
        assert_eq!(limbs[4], F::from_u32(0));
    }
}
