use luisa::prelude::*;
use luisa::printer::*;
use luisa_compute as luisa;

use crate::grid::GridBuffers;
use crate::particle::ParticleBuffers;
use crate::{constants::*, kernel};

pub struct DfsphBuffers {
    alpha: Buffer<f32>,
    // In correctDensityError, correctDivergenceError: κ_i / ρ_i * dt.
    stiffness_div_density: Buffer<f32>,
    predicted_density: Buffer<f32>,
}
impl DfsphBuffers {
    pub fn new(device: &Device, particle_count: usize) -> Self {
        let alpha = device.create_buffer::<f32>(particle_count);

        let stiffness_div_density = device.create_buffer::<f32>(particle_count);
        let divergence = device.create_buffer::<f32>(particle_count);
        Self {
            alpha,
            stiffness_div_density,
            predicted_density: divergence,
        }
    }
}

pub struct DfsphExecutor {
    pub buffers: DfsphBuffers,
    pub particles: ParticleBuffers,
    pub grid: GridBuffers,
}
impl DfsphExecutor {
    pub fn new(particles: ParticleBuffers, grid: GridBuffers, buffers: DfsphBuffers) -> Self {
        Self {
            buffers,
            particles,
            grid,
        }
    }
    pub fn compute_factors(&self) {
        let Self {
            buffers,
            particles,
            grid,
        } = self;
        track!({
            let pos = particles.position.var().read(dispatch_id().x());
            let density = 0.0_f32.var();
            let sum = Var::<Float2>::zeroed();
            let sum_sq = 0.0_f32.var();
            grid.on_neighbors(&pos, |i| {
                let neighbor_pos = particles.position.var().read(i);
                let norm = (pos - neighbor_pos).length();
                *density.get_mut() += PARTICLE_MASS * kernel::apply(norm, KERNEL_SIZE);
                if dispatch_id().x() != i {
                    let dir = (pos - neighbor_pos) / norm;
                    let x = PARTICLE_MASS * kernel::apply_diff(norm, KERNEL_SIZE);
                    *sum.get_mut() += x * dir;
                    *sum_sq.get_mut() += x * x;
                }
            });
            let alpha = sum.length_squared() + *sum_sq;
            particles.density.var().write(dispatch_id().x(), density);
            buffers.alpha.var().write(dispatch_id().x(), alpha);
        })
    }
    fn calculate_divergence(&self, pos: Expr<Float2>, vel: Expr<Float2>) -> Expr<f32> {
        track!({
            let divergence = 0.0_f32.var();

            self.grid.on_neighbors(&pos, |i| {
                if dispatch_id().x() != i {
                    let neighbor_pos = self.particles.position.var().read(i);
                    let neighbor_vel = self.particles.velocity.var().read(i);
                    let norm = (pos - neighbor_pos).length();
                    let dir = (pos - neighbor_pos) / norm;
                    let dw = kernel::apply_diff(norm, KERNEL_SIZE) * dir;
                    *divergence.get_mut() += PARTICLE_MASS * (vel - neighbor_vel).dot(dw);
                }
            });
            *divergence
        })
    }
    pub fn correct_density_error_predict_density(&self, dt: Expr<f32>) {
        track!({
            let pos = self.particles.position.var().read(dispatch_id().x());
            let vel = self.particles.velocity.var().read(dispatch_id().x());
            let divergence = self.calculate_divergence(pos, vel);
            let density = self.particles.density.var().read(dispatch_id().x());
            let predicted_density = density + dt * divergence;
            self.buffers
                .predicted_density
                .var()
                .write(dispatch_id().x(), predicted_density);
        })
    }
    pub fn correct_density_error_compute_stiffness(&self, dt: Expr<f32>) {
        track!({
            let density = self.particles.density.var().read(dispatch_id().x());
            let predicted_density = self.buffers.predicted_density.var().read(dispatch_id().x());
            let alpha = self.buffers.alpha.var().read(dispatch_id().x());
            let stiffness_div_density = (predicted_density - REST_DENSITY) * alpha / (dt * density);
            self.buffers
                .stiffness_div_density
                .var()
                .write(dispatch_id().x(), stiffness_div_density);
        })
    }
    pub fn correct_divergence_error_compute_stiffness(&self, _dt: Expr<f32>) {
        track!({
            let pos = self.particles.position.var().read(dispatch_id().x());
            let vel = self.particles.velocity.var().read(dispatch_id().x());
            let divergence = self.calculate_divergence(pos, vel);
            let alpha = self.buffers.alpha.var().read(dispatch_id().x());
            let stiffness_div_density = divergence / (alpha.max(1e-6));
            self.buffers
                .stiffness_div_density
                .var()
                .write(dispatch_id().x(), stiffness_div_density);
        })
    }
    pub fn adapt_velocity(&self, _dt: Expr<f32>) {
        track!({
            let pos = self.particles.position.var().read(dispatch_id().x());
            let vel = self.particles.velocity.var().read(dispatch_id().x()).var();
            let st = self
                .buffers
                .stiffness_div_density
                .var()
                .read(dispatch_id().x());

            self.grid.on_neighbors(&pos, |i| {
                if dispatch_id().x() != i {
                    let neighbor_pos = self.particles.position.var().read(i);
                    let neighbor_st = self.buffers.stiffness_div_density.var().read(i);
                    let norm = (pos - neighbor_pos).length();
                    let dir = (pos - neighbor_pos) / norm;
                    let dw = kernel::apply_diff(norm, KERNEL_SIZE) * dir;
                    *vel.get_mut() -= PARTICLE_MASS * (st + neighbor_st) * dw;
                }
            });
            self.particles.velocity.var().write(dispatch_id().x(), vel);
        })
    }
}
