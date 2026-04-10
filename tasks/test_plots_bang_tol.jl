"""
Focused plot smoke-test for the optim_LFI_alloc branch.

Checks that:
  * `plot_irf` / `plot_irf!` work with default settings, custom `tol`, and `tol` + algorithm
  * `plot_solution` / `plot_solution!` work with default, second-order, and custom `tol`
  * `plot_conditional_forecast` / `plot_conditional_forecast!` run without error
  * `plot_model_estimates` / `plot_model_estimates!` run with a synthetic dataset
  * all bang (!) variants successfully overlay on a previous plot

All plots are produced with `show_plots = true` (headless / CI safe).

Run with:
    julia --project=. tasks/test_plots_bang_tol.jl
"""

using Test
using MacroModelling
using Random
using AxisKeys
import StatsPlots
import MacroModelling: Tolerances, FirstOrderTolerances, HigherOrderTolerances,
                       SolverTolerances, NsssTolerances, clear_solution_caches!

Random.seed!(42)
# StatsPlots.gr()   # force GR backend — no display / X server needed

# ── models ───────────────────────────────────────────────────────────────────
@info "Loading models…"
include(joinpath(@__DIR__, "../models/FS2000.jl"))
include(joinpath(@__DIR__, "../models/Gali_2015_chapter_3_nonlinear.jl"))

# ── custom tolerances ─────────────────────────────────────────────────────────
tight_tol = Tolerances(
    first_order  = FirstOrderTolerances(qme = SolverTolerances(acceptance_tol = 1e-10)),
    second_order = HigherOrderTolerances(sylvester = SolverTolerances(acceptance_tol = 1e-10)),
)

loose_tol = Tolerances(
    nsss = NsssTolerances(xtol = 1e-10, ftol = 1e-10),
)

# ── synthetic data for model-estimates / conditional-forecast tests ───────────
sim_fs   = simulate(FS2000)
obs_fs   = FS2000.constants.post_model_macro.var[1:2]
data_fs  = sim_fs(obs_fs, :, :simulate) .-
           FS2000.caches.non_stochastic_steady_state[1:2]

sim_nl   = simulate(Gali_2015_chapter_3_nonlinear)
obs_nl   = Gali_2015_chapter_3_nonlinear.constants.post_model_macro.var[1:2]
data_nl  = sim_nl(obs_nl, :, :simulate) .-
           Gali_2015_chapter_3_nonlinear.caches.non_stochastic_steady_state[1:2]

# ── KeyedArray conditions helper (used by plot_conditional_forecast) ──────────
function make_conditions(var::Symbol, n_periods::Int, target_val::Float64)
    m = Matrix{Union{Nothing,Float64}}(nothing, 1, n_periods)
    m[1, end] = target_val
    KeyedArray(m; Variables = [var], Periods = 1:n_periods)
end

# ════════════════════════════════════════════════════════════════════════════
@testset verbose = true "plot_irf and plot_irf!" begin

    @testset "FS2000 — default" begin
        p = plot_irf(FS2000; shocks = :e_a, show_plots = true)
        @test p isa Array
    end

    @testset "FS2000 — bang compare overlay (parameters)" begin
        p  = plot_irf(FS2000; show_plots = true)
        p! = plot_irf!(FS2000; parameters = :alp => 0.36, show_plots = true)
        @test p! isa Array
    end

    @testset "FS2000 — bang stack overlay" begin
        p  = plot_irf(FS2000; show_plots = true)
        p! = plot_irf!(FS2000; shock_size = 2, plot_type = :stack,
                       show_plots = true)
        @test p! isa Array
    end

    @testset "FS2000 — bang negative_shock" begin
        p  = plot_irf(FS2000; shocks = :e_a, show_plots = true)
        p! = plot_irf!(FS2000; shocks = :e_a, negative_shock = true,
                       show_plots = true)
        @test p! isa Array
    end

    @testset "FS2000 — bang second_order overlay" begin
        p  = plot_irf(FS2000; shocks = :e_a, show_plots = true)
        p! = plot_irf!(FS2000; shocks = :e_a,
                       algorithm = :pruned_second_order, show_plots = true)
        @test p! isa Array
    end

    @testset "Gali — tol option (first_order)" begin
        clear_solution_caches!(Gali_2015_chapter_3_nonlinear, :first_order)
        p = plot_irf(Gali_2015_chapter_3_nonlinear;
                     shocks = :eps_a,
                     tol = tight_tol, parameters = :β => 0.985,
                     show_plots = true)
        @test p isa Array
    end

    @testset "Gali — tol option (second_order)" begin
        clear_solution_caches!(Gali_2015_chapter_3_nonlinear, :second_order)
        p = plot_irf(Gali_2015_chapter_3_nonlinear;
                     shocks = :eps_a,
                     algorithm = :pruned_second_order,
                     tol = tight_tol, parameters = :β => 0.984,
                     show_plots = true)
        @test p isa Array
    end

    @testset "Gali — bang tol overlay" begin
        clear_solution_caches!(Gali_2015_chapter_3_nonlinear, :first_order)
        p  = plot_irf(Gali_2015_chapter_3_nonlinear;
                      shocks = :eps_a, parameters = :β => 0.983,
                      show_plots = true)
        p! = plot_irf!(Gali_2015_chapter_3_nonlinear;
                       shocks = :eps_a,
                       tol = loose_tol, parameters = :β => 0.982,
                       show_plots = true)
        @test p! isa Array
    end

    @testset "Gali — generalised_irf bang" begin
        p  = plot_irf(Gali_2015_chapter_3_nonlinear;
                      shocks = :eps_a,
                      algorithm = :pruned_second_order, show_plots = true)
        p! = plot_irf!(Gali_2015_chapter_3_nonlinear;
                       shocks = :eps_a,
                       algorithm = :pruned_second_order,
                       generalised_irf = true, show_plots = true)
        @test p! isa Array
    end

    @testset "Gali — qme_algorithm doubling" begin
        clear_solution_caches!(Gali_2015_chapter_3_nonlinear, :first_order)
        p = plot_irf(Gali_2015_chapter_3_nonlinear;
                     shocks = :eps_a,
                     quadratic_matrix_equation_algorithm = :doubling,
                     parameters = :β => 0.981,
                     show_plots = true)
        @test p isa Array
    end

    @testset "Gali — sylvester_algorithm bartels_stewart" begin
        clear_solution_caches!(Gali_2015_chapter_3_nonlinear, :second_order)
        p = plot_irf(Gali_2015_chapter_3_nonlinear;
                     shocks = :eps_a,
                     algorithm = :second_order,
                     sylvester_algorithm = :bartels_stewart,
                     show_plots = true)
        @test p isa Array
    end

end  # plot_irf testset

# ════════════════════════════════════════════════════════════════════════════
@testset verbose = true "plot_solution and plot_solution!" begin

    @testset "FS2000 — default first_order" begin
        p = plot_solution(FS2000, :k; show_plots = true)
        @test p isa Array
    end

    @testset "FS2000 — bang second_order overlay" begin
        p  = plot_solution(FS2000, :k; show_plots = true)
        p! = plot_solution!(FS2000, :k;
                            algorithm = :pruned_second_order, show_plots = true)
        @test p! isa Array
    end

    @testset "Gali — default first_order" begin
        p = plot_solution(Gali_2015_chapter_3_nonlinear, :A; show_plots = true)
        @test p isa Array
    end

    @testset "Gali — bang labels + parameters" begin
        p  = plot_solution(Gali_2015_chapter_3_nonlinear, :A;
                           parameters = :β => 0.99, label = "β=0.99",
                           show_plots = true)
        p! = plot_solution!(Gali_2015_chapter_3_nonlinear, :A;
                            parameters = :β => 0.97, label = "β=0.97",
                            show_plots = true)
        @test p! isa Array
    end

    @testset "Gali — tol option" begin
        clear_solution_caches!(Gali_2015_chapter_3_nonlinear, :first_order)
        p = plot_solution(Gali_2015_chapter_3_nonlinear, :A;
                          tol = tight_tol, parameters = :β => 0.986,
                          show_plots = true)
        @test p isa Array
    end

    @testset "Gali — bang tol overlay" begin
        clear_solution_caches!(Gali_2015_chapter_3_nonlinear, :first_order)
        p  = plot_solution(Gali_2015_chapter_3_nonlinear, :A;
                           parameters = :β => 0.985, show_plots = true)
        p! = plot_solution!(Gali_2015_chapter_3_nonlinear, :A;
                            tol = tight_tol, parameters = :β => 0.984,
                            show_plots = true)
        @test p! isa Array
    end

    @testset "Gali — qme_algorithm" begin
        clear_solution_caches!(Gali_2015_chapter_3_nonlinear, :first_order)
        p = plot_solution(Gali_2015_chapter_3_nonlinear, :A;
                          quadratic_matrix_equation_algorithm = :doubling,
                          parameters = :β => 0.983,
                          show_plots = true)
        @test p isa Array
    end

end  # plot_solution testset

# ════════════════════════════════════════════════════════════════════════════
@testset verbose = true "plot_conditional_forecast and plot_conditional_forecast!" begin

    cndtns1 = make_conditions(:y, 8, 1.4)
    cndtns2 = make_conditions(:y, 4, 2.01)

    @testset "FS2000 — default" begin
        p = plot_conditional_forecast(FS2000, cndtns1; show_plots = true)
        @test p isa Array
    end

    @testset "FS2000 — bang compare overlay" begin
        p  = plot_conditional_forecast(FS2000, cndtns1; show_plots = true)
        p! = plot_conditional_forecast!(FS2000, cndtns2;
                                        label = "alt target",
                                        show_plots = true)
        @test p! isa Array
    end

    @testset "FS2000 — bang stack overlay" begin
        p  = plot_conditional_forecast(FS2000, cndtns1; show_plots = true)
        p! = plot_conditional_forecast!(FS2000, cndtns2;
                                        plot_type = :stack,
                                        show_plots = true)
        @test p! isa Array
    end

end  # plot_conditional_forecast testset

# ════════════════════════════════════════════════════════════════════════════
@testset verbose = true "plot_model_estimates and plot_model_estimates!" begin

    @testset "FS2000 — default kalman" begin
        p = plot_model_estimates(FS2000, data_fs; show_plots = true)
        @test p isa Array
    end

    @testset "FS2000 — bang different parameters" begin
        p  = plot_model_estimates(FS2000, data_fs; show_plots = true)
        p! = plot_model_estimates!(FS2000, data_fs;
                                   parameters = :alp => 0.36,
                                   show_plots = true)
        @test p! isa Array
    end

    @testset "FS2000 — bang inversion filter" begin
        p  = plot_model_estimates(FS2000, data_fs; show_plots = true)
        p! = plot_model_estimates!(FS2000, data_fs;
                                   filter = :inversion,
                                   show_plots = true)
        @test p! isa Array
    end

    @testset "FS2000 — tol option" begin
        clear_solution_caches!(FS2000, :first_order)
        p = plot_model_estimates(FS2000, data_fs;
                                 tol = tight_tol,
                                 parameters = :alp => 0.357,
                                 show_plots = true)
        @test p isa Array
    end

    @testset "FS2000 — bang tol overlay" begin
        clear_solution_caches!(FS2000, :first_order)
        p  = plot_model_estimates(FS2000, data_fs;
                                  parameters = :alp => 0.358, show_plots = true)
        p! = plot_model_estimates!(FS2000, data_fs;
                                   tol = loose_tol,
                                   parameters = :alp => 0.355,
                                   show_plots = true)
        @test p! isa Array
    end

    @testset "FS2000 — bang smooth=false" begin
        p  = plot_model_estimates(FS2000, data_fs; show_plots = true)
        p! = plot_model_estimates!(FS2000, data_fs;
                                   smooth = false, show_plots = true)
        @test p! isa Array
    end

    @testset "Gali — inversion filter + bang smooth=false" begin
        p  = plot_model_estimates(Gali_2015_chapter_3_nonlinear, data_nl;
                                  filter = :inversion, show_plots = true)
        p! = plot_model_estimates!(Gali_2015_chapter_3_nonlinear, data_nl;
                                   filter = :inversion, smooth = false,
                                   show_plots = true)
        @test p! isa Array
    end

end  # plot_model_estimates testset

@info "All plot smoke-tests finished."
