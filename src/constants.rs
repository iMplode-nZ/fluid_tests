use luisa_compute::prelude::*;

pub const RESOLUTION: u32 = 512;
pub const KERNEL_SIZE: f32 = 3.0;
pub const GRID_SIZE: usize = (RESOLUTION as f32 / KERNEL_SIZE) as usize;
pub const PAD_OFFSET: usize = 1;
pub const PADDED_SIZE: usize = GRID_SIZE + 2 * PAD_OFFSET;
pub const MAX_PARTICLES_PER_CELL: usize = 64;
pub const PARTICLE_MASS: f32 = 1.0;
pub const GRAVITY: Float2 = Float2::new(0.0, 0.0);
pub const REST_DENSITY: f32 = 1.0;
