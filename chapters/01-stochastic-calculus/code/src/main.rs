//! Example usage of the stochastic calculus library
//!
//! Run with: cargo run --release

use stochastic_calculus::{
    gbm::GeometricBrownianMotion,
    heston::HestonModel,
    jump_diffusion::MertonJumpDiffusion,
    monte_carlo::{
        black_scholes_call, call_payoff, monte_carlo_antithetic, monte_carlo_basic,
        monte_carlo_control_variate, monte_carlo_parallel,
    },
    BrownianMotion,
};

fn main() {
    println!("=== Stochastic Calculus for Trading ===\n");

    example_brownian_motion();
    example_gbm();
    example_heston();
    example_jump_diffusion();
    example_monte_carlo_option_pricing();
}

fn example_brownian_motion() {
    println!("--- Brownian Motion ---");

    let bm = BrownianMotion::new(0.0);
    let mut rng = rand::thread_rng();

    // Generate a path
    let n_steps = 1000;
    let dt = 0.01;
    let path = bm.generate_path(&mut rng, n_steps, dt);

    let t_end = n_steps as f64 * dt;
    let final_value = path.last().unwrap();

    println!("Path length: {} steps over T = {:.1}", n_steps, t_end);
    println!("Final value W_T = {:.4}", final_value);

    // Verify quadratic variation
    let qv = stochastic_calculus::brownian::quadratic_variation(&path);
    println!("Quadratic variation [W,W]_T = {:.4} (expected ≈ {:.1})", qv, t_end);
    println!();
}

fn example_gbm() {
    println!("--- Geometric Brownian Motion ---");

    let gbm = GeometricBrownianMotion::new(
        100.0,  // Initial price
        0.10,   // 10% expected return
        0.20,   // 20% volatility
    );

    // Generate sample paths
    let n_paths = 5;
    let n_steps = 252; // 1 year of daily data
    let dt = 1.0 / 252.0;

    println!("Initial price: $100.00");
    println!("Expected return: 10%/year, Volatility: 20%/year");
    println!("\nSample paths (1 year):");

    for i in 1..=n_paths {
        let mut rng = rand::thread_rng();
        let path = gbm.generate_path(&mut rng, n_steps, dt);
        let final_price = path.last().unwrap();
        let return_pct = (final_price / 100.0 - 1.0) * 100.0;
        println!("  Path {}: ${:.2} ({:+.1}%)", i, final_price, return_pct);
    }

    // Expected value
    let t = 1.0;
    let expected = gbm.expected_value(t);
    println!("\nTheoretical E[S_T] = ${:.2}", expected);
    println!();
}

fn example_heston() {
    println!("--- Heston Stochastic Volatility Model ---");

    let model = HestonModel::new(
        100.0,  // Initial price
        0.04,   // Initial variance (20% vol squared)
        0.05,   // 5% drift
        2.0,    // Mean reversion speed
        0.04,   // Long-term variance
        0.3,    // Vol of vol
        -0.7,   // Negative correlation (leverage effect)
    );

    println!("Parameters:");
    println!("  S0 = $100, V0 = 0.04 (vol ≈ 20%)");
    println!("  κ = 2.0, θ = 0.04, ξ = 0.3, ρ = -0.7");
    println!(
        "  Feller condition: {}",
        if model.feller_condition_satisfied() {
            "satisfied ✓"
        } else {
            "NOT satisfied ✗"
        }
    );

    let mut rng = rand::thread_rng();
    let n_steps = 252;
    let dt = 1.0 / 252.0;

    let path = model.simulate_euler(&mut rng, n_steps, dt);

    let final_price = path.prices.last().unwrap();
    let final_vol = path.variances.last().unwrap().sqrt() * 100.0;

    println!("\nSimulated 1-year path:");
    println!("  Final price: ${:.2}", final_price);
    println!("  Final volatility: {:.1}%", final_vol);
    println!();
}

fn example_jump_diffusion() {
    println!("--- Merton Jump-Diffusion Model ---");

    let model = MertonJumpDiffusion::new(
        100.0,  // Initial price
        0.10,   // 10% drift
        0.15,   // 15% diffusion vol
        3.0,    // 3 jumps per year on average
        -0.02,  // Mean jump of -2%
        0.08,   // Jump vol of 8%
    );

    println!("Parameters:");
    println!("  S0 = $100, μ = 10%, σ = 15%");
    println!("  λ = 3 jumps/year, μ_J = -2%, σ_J = 8%");
    println!("  Expected jump size κ = {:.4}", model.kappa());

    let mut rng = rand::thread_rng();
    let n_steps = 252;
    let dt = 1.0 / 252.0;

    println!("\nSample paths (1 year):");
    for i in 1..=5 {
        let path = model.simulate_log(&mut rng, n_steps, dt);
        let final_price = path.last().unwrap();
        let return_pct = (final_price / 100.0 - 1.0) * 100.0;

        // Count large moves (potential jumps)
        let large_moves: usize = path
            .windows(2)
            .filter(|w| (w[1] / w[0]).ln().abs() > 0.03)
            .count();

        println!(
            "  Path {}: ${:.2} ({:+.1}%), large moves: {}",
            i, final_price, return_pct, large_moves
        );
    }
    println!();
}

fn example_monte_carlo_option_pricing() {
    println!("--- Monte Carlo Option Pricing ---");

    let s0 = 100.0;
    let k = 100.0; // ATM strike
    let r = 0.05;  // Risk-free rate
    let sigma = 0.2;
    let t = 1.0;   // 1 year

    // Use risk-neutral measure
    let gbm = GeometricBrownianMotion::new(s0, r, sigma);

    println!("European Call Option:");
    println!("  S0 = $100, K = $100, r = 5%, σ = 20%, T = 1 year");

    // Black-Scholes (analytical)
    let bs_price = black_scholes_call(s0, k, r, sigma, t);
    println!("\nBlack-Scholes price: ${:.4}", bs_price);

    // Monte Carlo methods
    let n_paths = 100_000;
    let discount = (-r * t).exp();

    println!("\nMonte Carlo with {} paths:", n_paths);

    let mut rng = rand::thread_rng();

    // Basic MC
    let basic = monte_carlo_basic(&gbm, call_payoff(k), t, n_paths, &mut rng);
    println!(
        "  Basic:      ${:.4} ± {:.4} (discounted: ${:.4})",
        basic.estimate,
        basic.std_error,
        basic.estimate * discount
    );

    // Antithetic MC
    let antithetic = monte_carlo_antithetic(&gbm, call_payoff(k), t, n_paths, &mut rng);
    println!(
        "  Antithetic: ${:.4} ± {:.4} (discounted: ${:.4})",
        antithetic.estimate,
        antithetic.std_error,
        antithetic.estimate * discount
    );

    // Control Variate MC
    let cv = monte_carlo_control_variate(&gbm, call_payoff(k), t, n_paths, &mut rng);
    println!(
        "  Control V:  ${:.4} ± {:.4} (discounted: ${:.4})",
        cv.estimate,
        cv.std_error,
        cv.estimate * discount
    );

    // Parallel MC (for comparison)
    let parallel = monte_carlo_parallel(&gbm, call_payoff(k), t, n_paths);
    println!(
        "  Parallel:   ${:.4} ± {:.4} (discounted: ${:.4})",
        parallel.estimate,
        parallel.std_error,
        parallel.estimate * discount
    );

    println!("\nVariance reduction effectiveness:");
    println!(
        "  Antithetic reduces SE by {:.1}%",
        (1.0 - antithetic.std_error / basic.std_error) * 100.0
    );
    println!(
        "  Control variate reduces SE by {:.1}%",
        (1.0 - cv.std_error / basic.std_error) * 100.0
    );
}
