//! The carry-less products of the vector kernels, at the width of the multiplier: a register of
//! 128-bit lanes, one with PCLMULQDQ alone, two with VPCLMULQDQ and four with AVX-512 as well,
//! and one with NEON's PMULL. Every operation is lane by lane, so a kernel written once serves
//! every width.
//!
//! The intrinsics are safe to call: the module is compiled only where its features are.

#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

cfg_select! {
    target_arch = "aarch64" => {
        pub type V = uint64x2_t;
    }
    all(target_feature = "avx512f", target_feature = "vpclmulqdq") => {
        pub type V = __m512i;
    }
    target_feature = "vpclmulqdq" => {
        pub type V = __m256i;
    }
    _ => {
        pub type V = __m128i;
    }
}

/// NEON has what the Skip kernel takes; `store`, `load_packs` and `splat` serve x86's other
/// kernels only.
pub trait Vector: Copy {
    /// 128-bit lanes.
    const N: usize;
    /// # Safety
    /// `p` is valid for reading `2 * N` words.
    unsafe fn load(p: *const u64) -> Self;
    /// # Safety
    /// `p` is valid for writing `2 * N` words.
    #[cfg(target_arch = "x86_64")]
    unsafe fn store(p: *mut u64, v: Self);
    /// `2 * N` packs, a word each.
    ///
    /// # Safety
    /// `p` is valid for reading `2 * N` packs.
    #[cfg(target_arch = "x86_64")]
    unsafe fn load_packs(p: *const u32) -> Self;
    fn zero() -> Self;
    #[cfg(target_arch = "x86_64")]
    fn splat(x: u64) -> Self;
    fn xor(self, other: Self) -> Self;
    /// In each lane, the product of the words `IMM` picks, as `_mm_clmulepi64_si128` does.
    fn clmul<const IMM: i32>(self, other: Self) -> Self;
    /// In each lane, the high word moved to the low one.
    fn high(self) -> Self;
    /// In each lane, the low word of each.
    fn unpacklo(self, other: Self) -> Self;
    /// The sum of the lanes.
    fn sum(self) -> u128;
}

#[cfg(target_arch = "x86_64")]
impl Vector for __m128i {
    const N: usize = 1;
    #[inline(always)]
    unsafe fn load(p: *const u64) -> Self {
        unsafe { _mm_loadu_si128(p.cast()) }
    }
    #[inline(always)]
    unsafe fn store(p: *mut u64, v: Self) {
        unsafe { _mm_storeu_si128(p.cast(), v) }
    }
    #[inline(always)]
    unsafe fn load_packs(p: *const u32) -> Self {
        unsafe { _mm_cvtepu32_epi64(_mm_loadl_epi64(p.cast())) }
    }
    #[inline(always)]
    fn zero() -> Self {
        unsafe { _mm_setzero_si128() }
    }
    #[inline(always)]
    fn splat(x: u64) -> Self {
        unsafe { _mm_set1_epi64x(x as i64) }
    }
    #[inline(always)]
    fn xor(self, other: Self) -> Self {
        unsafe { _mm_xor_si128(self, other) }
    }
    #[inline(always)]
    fn clmul<const IMM: i32>(self, other: Self) -> Self {
        unsafe { _mm_clmulepi64_si128::<IMM>(self, other) }
    }
    #[inline(always)]
    fn high(self) -> Self {
        unsafe { _mm_srli_si128::<8>(self) }
    }
    #[inline(always)]
    fn unpacklo(self, other: Self) -> Self {
        unsafe { _mm_unpacklo_epi64(self, other) }
    }
    #[inline(always)]
    fn sum(self) -> u128 {
        // SAFETY: both are 128 bits with no invalid patterns.
        unsafe { core::mem::transmute::<Self, u128>(self) }
    }
}

#[cfg(target_feature = "vpclmulqdq")]
impl Vector for __m256i {
    const N: usize = 2;
    #[inline(always)]
    unsafe fn load(p: *const u64) -> Self {
        unsafe { _mm256_loadu_si256(p.cast()) }
    }
    #[inline(always)]
    unsafe fn store(p: *mut u64, v: Self) {
        unsafe { _mm256_storeu_si256(p.cast(), v) }
    }
    #[inline(always)]
    unsafe fn load_packs(p: *const u32) -> Self {
        unsafe { _mm256_cvtepu32_epi64(_mm_loadu_si128(p.cast())) }
    }
    #[inline(always)]
    fn zero() -> Self {
        unsafe { _mm256_setzero_si256() }
    }
    #[inline(always)]
    fn splat(x: u64) -> Self {
        unsafe { _mm256_set1_epi64x(x as i64) }
    }
    #[inline(always)]
    fn xor(self, other: Self) -> Self {
        unsafe { _mm256_xor_si256(self, other) }
    }
    #[inline(always)]
    fn clmul<const IMM: i32>(self, other: Self) -> Self {
        unsafe { _mm256_clmulepi64_epi128::<IMM>(self, other) }
    }
    #[inline(always)]
    fn high(self) -> Self {
        unsafe { _mm256_bsrli_epi128::<8>(self) }
    }
    #[inline(always)]
    fn unpacklo(self, other: Self) -> Self {
        unsafe { _mm256_unpacklo_epi64(self, other) }
    }
    #[inline(always)]
    fn sum(self) -> u128 {
        unsafe { _mm256_castsi256_si128(self).sum() ^ _mm256_extracti128_si256::<1>(self).sum() }
    }
}

#[cfg(all(target_feature = "avx512f", target_feature = "vpclmulqdq"))]
impl Vector for __m512i {
    const N: usize = 4;
    #[inline(always)]
    unsafe fn load(p: *const u64) -> Self {
        unsafe { _mm512_loadu_si512(p.cast()) }
    }
    #[inline(always)]
    unsafe fn store(p: *mut u64, v: Self) {
        unsafe { _mm512_storeu_si512(p.cast(), v) }
    }
    #[inline(always)]
    unsafe fn load_packs(p: *const u32) -> Self {
        unsafe { _mm512_cvtepu32_epi64(_mm256_loadu_si256(p.cast())) }
    }
    #[inline(always)]
    fn zero() -> Self {
        unsafe { _mm512_setzero_si512() }
    }
    #[inline(always)]
    fn splat(x: u64) -> Self {
        unsafe { _mm512_set1_epi64(x as i64) }
    }
    #[inline(always)]
    fn xor(self, other: Self) -> Self {
        unsafe { _mm512_xor_si512(self, other) }
    }
    #[inline(always)]
    fn clmul<const IMM: i32>(self, other: Self) -> Self {
        unsafe { _mm512_clmulepi64_epi128::<IMM>(self, other) }
    }
    #[inline(always)]
    fn high(self) -> Self {
        unsafe { _mm512_bsrli_epi128::<8>(self) }
    }
    #[inline(always)]
    fn unpacklo(self, other: Self) -> Self {
        unsafe { _mm512_unpacklo_epi64(self, other) }
    }
    #[inline(always)]
    fn sum(self) -> u128 {
        unsafe { _mm512_castsi512_si256(self).sum() ^ _mm512_extracti64x4_epi64::<1>(self).sum() }
    }
}

#[cfg(target_arch = "aarch64")]
impl Vector for uint64x2_t {
    const N: usize = 1;
    #[inline(always)]
    unsafe fn load(p: *const u64) -> Self {
        // SAFETY: the caller's, two words at `p`.
        unsafe { vld1q_u64(p) }
    }
    #[inline(always)]
    fn zero() -> Self {
        // SAFETY: NEON is in every aarch64 target, and the intrinsic takes no pointer.
        unsafe { vdupq_n_u64(0) }
    }
    #[inline(always)]
    fn xor(self, other: Self) -> Self {
        // SAFETY: as `zero`.
        unsafe { veorq_u64(self, other) }
    }
    /// PMULL and PMULL2 on the registers, written out: the intrinsic takes low words as scalars,
    /// which the compiler loads or moves apart from their registers.
    #[inline(always)]
    fn clmul<const IMM: i32>(self, other: Self) -> Self {
        use primitives::field::gf2_64::aarch64::{pmull_hi_asm, pmull_lo_asm};
        // SAFETY: PMULL needs `aes`, which the module is compiled only with; no pointer is taken.
        unsafe {
            match IMM {
                0x00 => pmull_lo_asm(self, other),
                0x01 => pmull_lo_asm(self.high(), other),
                0x10 => pmull_lo_asm(self, other.high()),
                _ => pmull_hi_asm(self, other),
            }
        }
    }
    #[inline(always)]
    fn high(self) -> Self {
        // SAFETY: as `zero`.
        unsafe { vextq_u64::<1>(self, vdupq_n_u64(0)) }
    }
    #[inline(always)]
    fn unpacklo(self, other: Self) -> Self {
        // SAFETY: as `zero`.
        unsafe { vzip1q_u64(self, other) }
    }
    #[inline(always)]
    fn sum(self) -> u128 {
        // SAFETY: both are 128 bits with no invalid patterns.
        unsafe { core::mem::transmute::<Self, u128>(self) }
    }
}

#[cfg(test)]
mod tests {
    use primitives::field::gf2_64::mul_wide;
    use primitives::test_rng::Rng;

    use super::*;

    /// Every `IMM` multiplies the words it picks in each lane, and `high` and `unpacklo` move theirs.
    #[test]
    fn products_are_of_the_picked_words() {
        let mut rng = Rng::new(34);
        let (a, b): ([u64; 8], [u64; 8]) = (
            std::array::from_fn(|_| rng.next_u64()),
            std::array::from_fn(|_| rng.next_u64()),
        );
        // SAFETY: eight words each, and `2 * N` is at most eight.
        let (va, vb) = unsafe { (V::load(a.as_ptr()), V::load(b.as_ptr())) };
        // Over the lanes, the sum of the products of word `x` of `a` and word `y` of `b`.
        let sum = |x: usize, y: usize| (0..V::N).fold(0, |acc, l| acc ^ mul_wide(a[2 * l + x], b[2 * l + y]));
        assert_eq!(va.clmul::<0x00>(vb).sum(), sum(0, 0));
        assert_eq!(va.clmul::<0x01>(vb).sum(), sum(1, 0));
        assert_eq!(va.clmul::<0x10>(vb).sum(), sum(0, 1));
        assert_eq!(va.clmul::<0x11>(vb).sum(), sum(1, 1));
        assert_eq!(va.high().clmul::<0x00>(vb).sum(), sum(1, 0));
        assert_eq!(va.high().clmul::<0x11>(vb).sum(), 0);
        assert_eq!(va.unpacklo(vb).clmul::<0x10>(vb).sum(), sum(0, 1));
        assert_eq!(va.unpacklo(vb).clmul::<0x01>(va).sum(), sum(0, 0));
    }
}
