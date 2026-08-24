pub fn xorshift64(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

/// Fill `buf` with deterministic pseudo-random bytes derived from `seed`.
pub fn fill_xorshift(buf: &mut [u8], seed: u64) {
    let mut state = seed.max(1);
    for chunk in buf.chunks_mut(8) {
        let v = xorshift64(&mut state).to_le_bytes();
        for (dst, src) in chunk.iter_mut().zip(v.iter()) {
            *dst = *src;
        }
    }
}
