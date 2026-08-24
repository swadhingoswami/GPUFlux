//! Minimal binary codec for the value types stored in redb.
//!
//! redb 2.6 does not ship derive macros, so `redb::Value` is implemented by
//! hand using a tiny fixed-order encoding. Keeping this dependency-free avoids
//! pulling in serde/bincode for a handful of structs.

pub(crate) fn push_u64(out: &mut Vec<u8>, x: u64) {
    out.extend_from_slice(&x.to_le_bytes());
}

pub(crate) fn push_u32(out: &mut Vec<u8>, x: u32) {
    out.extend_from_slice(&x.to_le_bytes());
}

pub(crate) fn push_f64(out: &mut Vec<u8>, x: f64) {
    out.extend_from_slice(&x.to_le_bytes());
}

pub(crate) fn push_bool(out: &mut Vec<u8>, b: bool) {
    out.push(if b { 1 } else { 0 });
}

pub(crate) fn push_str(out: &mut Vec<u8>, s: &str) {
    push_u64(out, s.len() as u64);
    out.extend_from_slice(s.as_bytes());
}

pub(crate) fn push_vec_f64(out: &mut Vec<u8>, v: &[f64]) {
    push_u64(out, v.len() as u64);
    for x in v {
        push_f64(out, *x);
    }
}

pub(crate) fn push_opt_f64(out: &mut Vec<u8>, x: Option<f64>) {
    match x {
        Some(v) => {
            push_bool(out, true);
            push_f64(out, v);
        }
        None => push_bool(out, false),
    }
}

pub(crate) fn push_opt_bool(out: &mut Vec<u8>, x: Option<bool>) {
    match x {
        Some(v) => {
            push_bool(out, true);
            push_bool(out, v);
        }
        None => push_bool(out, false),
    }
}

pub(crate) fn take_u64(data: &mut &[u8]) -> u64 {
    let (head, tail) = data.split_at(8);
    *data = tail;
    u64::from_le_bytes(head.try_into().expect("u64"))
}

pub(crate) fn take_u32(data: &mut &[u8]) -> u32 {
    let (head, tail) = data.split_at(4);
    *data = tail;
    u32::from_le_bytes(head.try_into().expect("u32"))
}

pub(crate) fn take_f64(data: &mut &[u8]) -> f64 {
    let (head, tail) = data.split_at(8);
    *data = tail;
    f64::from_le_bytes(head.try_into().expect("f64"))
}

pub(crate) fn take_bool(data: &mut &[u8]) -> bool {
    let b = data[0];
    *data = &data[1..];
    b != 0
}

pub(crate) fn take_str(data: &mut &[u8]) -> String {
    let len = take_u64(data) as usize;
    let s = std::str::from_utf8(&data[..len]).expect("utf8");
    *data = &data[len..];
    s.to_string()
}

pub(crate) fn take_vec_f64(data: &mut &[u8]) -> Vec<f64> {
    let len = take_u64(data) as usize;
    let mut v = Vec::with_capacity(len);
    for _ in 0..len {
        v.push(take_f64(data));
    }
    v
}

pub(crate) fn take_opt_f64(data: &mut &[u8]) -> Option<f64> {
    if take_bool(data) {
        Some(take_f64(data))
    } else {
        None
    }
}

pub(crate) fn take_opt_bool(data: &mut &[u8]) -> Option<bool> {
    if take_bool(data) {
        Some(take_bool(data))
    } else {
        None
    }
}
