use std::env::current_exe;

use bluenoise::BlueNoise;
use rand::Rng;
use rand_pcg::Pcg64Mcg;

use rapier2d::math::*;

use winit::event::{Event, WindowEvent};

use luisa::prelude::*;
use luisa_compute as luisa;

mod constants;
mod dfsph;
mod grid;
mod kernel;
mod particle;

use constants::*;
use dfsph::{DfsphBuffers, DfsphExecutor};
use grid::GridBuffers;
use particle::{Particle, ParticleBuffers};

fn main() {
    let ctx = luisa::Context::new(current_exe().unwrap());
    let device = ctx.create_device("cuda");

    let event_loop = winit::event_loop::EventLoop::new();
    let window = winit::window::WindowBuilder::new()
        .with_inner_size(winit::dpi::LogicalSize::new(RESOLUTION, RESOLUTION))
        .with_resizable(false)
        .build(&event_loop)
        .unwrap();
    let swapchain = device.create_swapchain(
        &window,
        &device.default_stream(),
        RESOLUTION,
        RESOLUTION,
        false,
        false,
        3,
    );

    let mut particles = Vec::new();
    let mut rng = Pcg64Mcg::new(0);
    let mut noise = BlueNoise::<Pcg64Mcg>::new(100.0, 100.0, 0.7 * KERNEL_SIZE);
    noise.with_samples(10);
    for sample in noise {
        let particle = Particle {
            position: Vector::new(sample.x, sample.y)
                + Vector::repeat(RESOLUTION as f32 / 2.0 - 50.0),
            velocity: Vector::zeros(), // Vector::new(rng.gen_range(-1.0..1.0), rng.gen_range(-1.0..1.0)) * 100.0,
        };
        particles.push(particle);
    }
    let particle_count = particles.len();
    let particles = ParticleBuffers::new(&device, &particles);
    let grid = GridBuffers::new(&device);
    let dfsph = DfsphBuffers::new(&device, particle_count);
    let executor = DfsphExecutor::new(particles, grid, dfsph);

    let display =
        device.create_tex2d::<Float4>(swapchain.pixel_storage(), RESOLUTION, RESOLUTION, 1);

    let clear_display = device.create_kernel_async::<fn()>(&|| {
        display
            .var()
            .write(dispatch_id().xy(), Float4::expr(0.0, 0.0, 0.0, 1.0));
    });

    let draw_particles = device.create_kernel_async::<fn()>(&|| {
        let pos = executor.particles.position.var().read(dispatch_id().x());
        display
            .var()
            .write(pos.uint(), Float4::expr(1.0, 0.0, 0.0, 1.0));
    });

    let clear_grid = device.create_kernel_async::<fn()>(&|| executor.grid.clear_grid());
    let fill_grid =
        device.create_kernel_async::<fn()>(&|| executor.grid.fill_grid(&executor.particles));

    let update_position =
        device.create_kernel_async::<fn(f32)>(&|dt| executor.particles.update_position(dt));
    let update_velocity =
        device.create_kernel_async::<fn(f32)>(&|dt| executor.particles.update_velocity(dt));

    let compute_factors = device.create_kernel_async::<fn()>(&|| executor.compute_factors());
    let correct_density_error_predict_density = device
        .create_kernel_async::<fn(f32)>(&|dt| executor.correct_density_error_predict_density(dt));
    let correct_density_error_compute_stiffness = device
        .create_kernel_async::<fn(f32)>(&|dt| executor.correct_density_error_compute_stiffness(dt));
    let adapt_velocity = device.create_kernel_async::<fn(f32)>(&|dt| executor.adapt_velocity(dt));
    let correct_divergence_error_compute_stiffness = device.create_kernel_async::<fn(f32)>(&|dt| {
        executor.correct_divergence_error_compute_stiffness(dt)
    });

    let ds = [particle_count as u32, 1, 1];

    fill_grid.dispatch(ds);
    compute_factors.dispatch(ds);

    event_loop.run(move |event, _, control_flow| {
        control_flow.set_poll();
        match event {
            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                window_id,
            } if window_id == window.id() => {
                let mut img_buffer = vec![[0u8; 4]; (RESOLUTION * RESOLUTION) as usize];
                {
                    let scope = device.default_stream().scope();
                    scope.submit([display.view(0).copy_to_async(&mut img_buffer)]);
                }
                {
                    let img = image::RgbImage::from_fn(RESOLUTION, RESOLUTION, |x, y| {
                        let i = x + y * RESOLUTION;
                        let px = img_buffer[i as usize];
                        image::Rgb([px[0], px[1], px[2]])
                    });
                    img.save("particles.png").unwrap();
                }
                control_flow.set_exit();
            }
            Event::MainEventsCleared => {
                window.request_redraw();
            }
            Event::RedrawRequested(_) => {
                let scope = device.default_stream().scope();
                scope.present(&swapchain, &display);
                let dt = 0.01;
                let mut commands = vec![
                    clear_display.dispatch_async([RESOLUTION, RESOLUTION, 1]),
                    draw_particles.dispatch_async(ds),
                    update_velocity.dispatch_async(ds, &dt),
                ];
                for _ in 0..2 {
                    commands.extend(
                        [
                            correct_density_error_predict_density.dispatch_async(ds, &dt),
                            correct_density_error_compute_stiffness.dispatch_async(ds, &dt),
                            adapt_velocity.dispatch_async(ds, &dt),
                        ]
                        .into_iter(),
                    );
                }
                commands.extend(
                    [
                        update_position.dispatch_async(ds, &dt),
                        clear_grid.dispatch_async([(PADDED_SIZE * PADDED_SIZE) as u32, 1, 1]),
                        fill_grid.dispatch_async(ds),
                        compute_factors.dispatch_async(ds),
                    ]
                    .into_iter(),
                );
                for _ in 0..1 {
                    commands.extend(
                        [
                            correct_divergence_error_compute_stiffness.dispatch_async(ds, &dt),
                            adapt_velocity.dispatch_async(ds, &dt),
                        ]
                        .into_iter(),
                    );
                }

                scope.submit(commands);
                window.request_redraw();
            }
            _ => {}
        }
    });
}
