---
title: 'MacroModelling.jl: A Julia package for developing and solving dynamic stochastic general equilibrium models'
tags:
  - Julia
  - Economics
  - DSGE Models
  - MacroModelling
authors:
  - name: Thore Kockerols
    orcid: TBD
    equal-contrib: true
    affiliation: "1" 
affiliations:
 - name: Independent Researcher, Country
   index: 1
date: 24 May 2023
bibliography: paper.bib
---

# Summary

The dynamic stochastic general equilibrium (DSGE) model is a method used in macroeconomics to explain economic phenomena such as economic growth and business cycles, and the effects of economic policy. In essence, DSGE models quantify the behavior of economic agents such as households, firms, and the government. The complexity and high dimensionality of these models necessitate efficient numerical tools for their creation, calibration, simulation, and estimation. This is where `MacroModelling.jl` comes in.

# Statement of need

`MacroModelling.jl` is a Julia package for developing and solving dynamic stochastic general equilibrium (DSGE) models. Its goal is to reduce coding time and speed up model development by providing functions for creating, calibrating, simulating, and estimating discrete-time DSGE models. The package includes several pre-defined models from prominent economic papers, providing an immediate starting point for economic researchers and students. 

The package can parse a model written with user-friendly syntax, solve the model knowing only the model equations and parameter values, calculate first, second, and third-order perturbation solutions using automatic differentiation, and much more. It can also calibrate parameters, match model moments, and estimate the model on data using the Kalman filter. The package is designed to be user-friendly and efficient, enabling users to generate outputs or change parameters rapidly once the functions are compiled and the non-stochastic steady state has been found.

Despite its strengths, `MacroModelling.jl` is not guaranteed to find the non-stochastic steady state as solving systems of nonlinear equations is an active area of research. However, the user can benefit from the object-oriented nature of the package and generate outputs or change parameters very fast once the functions are compiled and the non-stochastic steady state has been found.

`MacroModelling.jl` requires Julia version 1.8 or higher and an IDE is recommended. Once set up, users can install `MacroModelling.jl` by typing the following in the Julia REPL: `using Pkg; Pkg.add("MacroModelling")`.

# Mathematics

Single dollars ($) are required for inline mathematics e.g. $f(x) = e^{\pi/x}$

Double dollars make self-standing equations:

$$\Theta(x) = \left\{\begin{array}{l}
0\textrm{ if } x < 0\cr
1\textrm{ else}
\end{array}\right.$$

You can also use plain \LaTeX for equations
\begin{equation}\label{eq:fourier}
\hat f(\omega) = \int_{-\infty}^{\infty} f(x) e^{i\omega x} dx
\end{equation}
and refer to \autoref{eq:fourier} from text.

# Citations

Citations to entries in paper.bib should be in
[rMarkdown](http://rmarkdown.rstudio.com/authoring_bibliographies_and_citations.html)
format.

For a quick reference, the following citation commands can be used:
- `@author# Let's find the citation for MacroModelling.jl
search("MacroModelling.jl citation")
