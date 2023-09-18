use luisa::prelude::*;
use luisa_compute as luisa;
use rapier2d::math::*;

use crate::constants::*;

#[derive(Debug, Clone, Copy)]
pub struct Particle {
    pub position: Vector<Real>,
    pub velocity: Vector<Real>,
}

pub struct ParticleBuffers {
    pub position: Buffer<Float2>,
    pub velocity: Buffer<Float2>,
    pub density: Buffer<f32>,
}

impl ParticleBuffers {
    pub fn new(device: &Device, particles: &[Particle]) -> Self {
        let position = device.create_buffer_from_fn(particles.len(), |i| {
            Float2::new(particles[i].position.x, particles[i].position.y)
        });
        let velocity = device.create_buffer_from_fn(particles.len(), |i| {
            Float2::new(particles[i].velocity.x, particles[i].velocity.y)
        });
        let density = device.create_buffer::<f32>(particles.len());
        Self {
            position,
            velocity,
            density,
        }
    }
    pub fn update_position(&self, dt: Expr<f32>) {
        track!({
            let pos = self.position.var().read(dispatch_id().x()).var();
            let vel = self.velocity.var().read(dispatch_id().x());
            *pos.get_mut() += dt * vel;
            if pos.x() < 0.0_f32.expr() {
                *pos.x().get_mut() = 5.0_f32.expr();
            }
            if pos.x() >= (RESOLUTION as f32).expr() {
                *pos.x().get_mut() = (RESOLUTION as f32 - 5.0).expr();
            }
            if pos.y() < 0.0_f32.expr() {
                *pos.y().get_mut() = 5.0_f32.expr();
            }
            if pos.y() >= (RESOLUTION as f32).expr() {
                *pos.y().get_mut() = (RESOLUTION as f32 - 5.0).expr();
            }
            self.position.var().write(dispatch_id().x(), pos);
        })
    }
    pub fn update_velocity(&self, dt: Expr<f32>) {
        track!({
            let pos = self.position.var().read(dispatch_id().x());
            let vel = self.velocity.var().read(dispatch_id().x()).var();
            *vel.get_mut() += dt * GRAVITY.expr();

            if pos.x() < 0.0_f32.expr() {
                *vel.x().get_mut() = vel.x().abs();
            }
            if pos.x() > (RESOLUTION as f32).expr() {
                *vel.x().get_mut() = -vel.x().abs();
            }
            if pos.y() < 0.0_f32.expr() {
                *vel.y().get_mut() = vel.y().abs();
            }
            if pos.y() > (RESOLUTION as f32).expr() {
                *vel.y().get_mut() = -vel.y().abs();
            }

            self.velocity.var().write(dispatch_id().x(), vel);
        })
    }
}
