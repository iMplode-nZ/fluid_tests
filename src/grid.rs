use luisa::prelude::*;
use luisa_compute as luisa;

use crate::{constants::*, particle::ParticleBuffers};

pub struct GridBuffers {
    pub grid: Buffer<u32>,
    pub cell_sizes: Buffer<u32>,
}

impl GridBuffers {
    pub fn new(device: &Device) -> Self {
        let grid = device.create_buffer::<u32>(PADDED_SIZE * PADDED_SIZE * MAX_PARTICLES_PER_CELL);
        let cell_sizes = device.create_buffer::<u32>(PADDED_SIZE * PADDED_SIZE);
        Self { grid, cell_sizes }
    }
    pub fn on_neighbors(&self, pos: &Expr<Float2>, f: impl Fn(Expr<u32>)) {
        track! {
            let cell_x = (pos.x() / KERNEL_SIZE).floor().int() + PAD_OFFSET as i32;
            let cell_y = (pos.y() / KERNEL_SIZE).floor().int() + PAD_OFFSET as i32;
            for i in -1..=1 {
                for j in -1..=1 {
                    let cell_index: Expr<i32> = (cell_x + i) + (cell_y + j) * PADDED_SIZE as i32;
                    let cell_index = cell_index.uint();
                    let cell_size = self.cell_sizes.var().read(cell_index);
                    for k in 0_u32.expr()..cell_size {
                        let particle_index = self
                            .grid
                            .var()
                            .read(cell_index * MAX_PARTICLES_PER_CELL as u32 + k);
                        f(particle_index);
                    }
                }
            }
        }
    }
    pub fn clear_grid(&self) {
        track! {
            self.cell_sizes.var().write(dispatch_id().x(), 0);
        }
    }
    pub fn fill_grid(&self, particles: &ParticleBuffers) {
        track! {
            let pos = particles.position.var().read(dispatch_id().x());
            let cell_x = (pos.x() / KERNEL_SIZE).floor().int() + PAD_OFFSET as i32;
            let cell_y = (pos.y() / KERNEL_SIZE).floor().int() + PAD_OFFSET as i32;
            let cell_index = cell_x + cell_y * PADDED_SIZE as i32;
            let cell_index = cell_index.uint();
            let cell_size = self.cell_sizes.var().atomic_fetch_add(cell_index, 1);

            self.grid.var().write(
                cell_index * MAX_PARTICLES_PER_CELL as u32 + cell_size,
                dispatch_id().x(),
            );
        }
    }
}
