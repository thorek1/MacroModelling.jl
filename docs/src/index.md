# `MacroModelling.jl`

**Author: Thore Kockerols (@thorek1)**

The goal of `MacroModelling.jl` is to reduce coding time and speed up model development.

The package supports dicsrete-time DSGE models and the timing of a variable reflects when the variable is decided (end of period for stock variables).

As of now the package can:

- parse a model written with user friendly syntax (variables are followed by time indices `...[2], [1], [0], [-1], [-2]...`, or `[x]` for shocks)
- (tries to) solve the model only knowing the model equations and parameter values (no steady state file needed)
- calculate first, second, and third order perturbation solutions using (forward or reverse-mode) automatic differentiation (AD)
- calculate (generalised) impulse response functions, simulate the model, or do conditional forecasts
- calibrate parameters using (non stochastic) steady state relationships
- match model moments
- estimate the model on data (Kalman filter using first order perturbation)
- **differentiate** (forward AD) the model solution (first order perturbation), Kalman filter loglikelihood (reverse-mode AD), model moments, steady state, **with respect to the parameters**

The package is not:

- guaranteed to find the non stochastic steady state
- the fastest package around if you already have a fast way to find the NSSS

The former has to do with the fact that solving systems of nonlinear equations is hard (an active area of research). Especially in cases where the values of the solution are far apart (have a high standard deviation - e.g. `sol = [-46.324, .993457, 23523.3856]`), the algorithms have a hard time finding a solution. The recommended way to tackle this is to set bounds in the [`@parameters`](@ref) part (e.g. `r < 0.2`), so that the initial points are closer to the final solution (think of steady state interest rates not being higher than 20% - meaning not being higher than 0.2 or 1.2 depending on the definition).

The latter has to do with the fact that julia code is fast once compiled, and that the package can spend more time finding the non stochastic steady state. This means that it takes more time from executing the code to define the model and parameters for the first time to seeing the first plots than with most other packages. But, once the functions are compiled and the non stochastic steady state has been found the user can benefit from the object oriented nature of the package and generate outputs or change parameters very fast.

The package contains the following models in the `models` folder:

- [Aguiar and Gopinath (2007)](https://www.journals.uchicago.edu/doi/10.1086/511283) `Aguiar_Gopinath_2007.jl`
- [Ascari and Sbordone (2014)](https://www.aeaweb.org/articles?id=10.1257/jel.52.3.679) `Ascari_sbordone_2014.jl`
- [Caldara et al. (2012)](https://www.sciencedirect.com/science/article/abs/pii/S1094202511000433) `Caldara_et_al_2012.jl`
- [Gali (2015)](https://press.princeton.edu/books/hardcover/9780691164786/monetary-policy-inflation-and-the-business-cycle) - Chapter 3 `Gali_2015_chapter_3_nonlinear.jl`
- [Gali and Monacelli (2005)](https://crei.cat/wp-content/uploads/users/pages/roes8739.pdf) - CPI inflation-based Taylor rule `Gali_Monacelli_2005_CITR.jl`
- [Gerali, Neri, Sessa, and Signoretti (2010)](https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1538-4616.2010.00331.x) `GNSS_2010.jl`
- [Ghironi and Melitz (2005)](https://faculty.washington.edu/ghiro/GhiroMeliQJE0805.pdf) `Ghironi_Melitz_2005.jl`
- [Ireland (2004)](http://irelandp.com/pubs/tshocksnk.pdf) `Ireland_2004.jl`
- [Jermann and Quadrini (2012)](https://www.aeaweb.org/articles?id=10.1257/aer.102.1.238) - RBC `JQ_2012_RBC.jl`
- [New Area-Wide Model (2008)](https://www.ecb.europa.eu/pub/pdf/scpwps/ecbwp944.pdf) - Euro Area - US `NAWM_EAUS_2008.jl`
- [Schmitt-Grohé and Uribe (2003)](https://www.sciencedirect.com/science/article/abs/pii/S0022199602000569) - debt premium `SGU_2003_debt_premium.jl`
- [Schorfheide (2000)](https://onlinelibrary.wiley.com/doi/abs/10.1002/jae.582) `FS2000.jl`
- [Smets and Wouters (2003)](https://onlinelibrary.wiley.com/doi/10.1162/154247603770383415) `SW03.jl`
- [Smets and Wouters (2007)](https://www.aeaweb.org/articles?id=10.1257/aer.97.3.586) `SW07.jl`
