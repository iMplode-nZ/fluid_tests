use std::f32::consts::PI;

use luisa::lang::types::core::*;
use luisa::prelude::*;
use luisa_compute as luisa;

fn normalizer(h: f32) -> f32 {
    (40.0 / 7.0) / (PI * h * h)
}

// Taken from https://docs.rs/salva2d/latest/src/salva2d/kernel/cubic_spline_kernel.rs.html.
pub fn apply(r: F32, h: f32) -> F32 {
    track!({
        let q = r / h;

        let rhs = if q < 0.5 {
            let q2 = q * q;
            1.0 + 6.0 * (q2 * q - q2)
        } else if q.cmple(1.0) {
            let nq = 1.0 - q;
            2.0 * nq * nq * nq
        } else {
            0.0_f32.expr()
        };
        normalizer(h) * rhs
    })
}

pub fn apply_diff(r: F32, h: f32) -> F32 {
    track!({
        let q = r / h;

        let rhs = if q > 1.0 || q < 1.0e-5 {
            0.0_f32.expr()
        } else if q < 0.5 {
            6.0 * (q * 3.0 - 2.0) * q
        } else {
            let nq = 1.0 - q;
            -6.0 * nq * nq
        };

        normalizer(h) * rhs / h
    })
}
