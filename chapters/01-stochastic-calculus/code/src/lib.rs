//! # Stochastic Calculus for Algorithmic Trading
//!
//! This crate provides implementations of fundamental stochastic processes
//! and numerical methods used in quantitative finance.
//!
//! ## Modules
//!
//! - [`brownian`] - Brownian motion (Wiener process) generation
//! - [`sde`] - Stochastic Differential Equations framework
//! - [`gbm`] - Geometric Brownian Motion
//! - [`heston`] - Heston stochastic volatility model
//! - [`jump_diffusion`] - Merton jump-diffusion model
//! - [`monte_carlo`] - Monte Carlo methods with variance reduction
//!
//! ## Example
//!
//! ```rust
//! use stochastic_calculus::gbm::GeometricBrownianMotion;
//!
//! let gbm = GeometricBrownianMotion::new(100.0, 0.1, 0.2);
//! let mut rng = rand::thread_rng();
//!
//! // Generate a single path
//! let path = gbm.generate_path(&mut rng, 252, 1.0 / 252.0);
//! println!("Final price: {:.2}", path.last().unwrap());
//! ```

pub mod brownian;
pub mod gbm;
pub mod heston;
pub mod jump_diffusion;
pub mod monte_carlo;
pub mod sde;

pub use brownian::BrownianMotion;
pub use gbm::GeometricBrownianMotion;
pub use heston::HestonModel;
pub use jump_diffusion::MertonJumpDiffusion;
pub use sde::{EulerMaruyama, SDE};
