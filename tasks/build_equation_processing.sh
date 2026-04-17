#!/bin/bash
set -e
OUT=src/parser/equation_processing.jl

cat > $OUT << 'HEADER'
# Pure-function equation processing helpers used by both the equation
# modification pipeline and (potentially) the model macros.
#
# `process_model_equations` reproduces the work the `@model` macro performs on
# its equation block, returning a `post_model_macro` struct and an `equations`
# struct so the model state can be updated without re-invoking the macro.
#
# `process_parameter_definitions` reproduces the work the `@parameters` macro
# performs on the parameter block. It takes a `post_model_macro` describing the
# current model (used for variable name lookups, index expansion, etc.) and
# returns the pieces needed to update `post_parameters_macro`, the equations
# struct's calibration fields, and `post_complete_parameters`.

"""
    process_model_equations(model_block::Expr, max_obc_horizon::Int, precompile::Bool)

Parse a `@model`-style equation block and return `(T, equations_struct)` where
`T::post_model_macro` is the parsed model structure and `equations_struct::equations`
is a freshly constructed equations container with dynamic, steady-state and
original equations populated. Calibration fields on the returned equations
struct are left empty and must be populated by
`process_parameter_definitions` before the model can be solved.
"""
function process_model_equations(model_block_in::Expr, max_obc_horizon::Int, precompile::Bool)
HEADER

# append the extracted body, replacing ex[end] with model_block_in
sed 's/ex\[end\]/model_block_in/g' tasks/model_body.txt >> $OUT

# append return statement
cat >> $OUT << 'FOOTER'

    ℂ = Constants(T)
    𝓦 = Workspaces()

    ss_aux_eqs_vec = Expr[e for e in ss_aux_equations]
    dyn_eqs_vec = Expr[e for e in dyn_equations]
    ss_eqs_vec = Expr[e for e in ss_equations]
    orig_eqs_vec = Expr[e for e in original_equations]
    calib_eqs_vec = Expr[e for e in calibration_equations]

    equations_struct = equations(
        orig_eqs_vec,
        dyn_eqs_vec,
        ss_eqs_vec,
        ss_aux_eqs_vec,
        Expr[],            # obc_violation
        calib_eqs_vec,     # calibration (filled later by @parameters)
        Expr[],            # calibration_no_var
        Symbol[],          # calibration_parameters
        Expr[],            # calibration_original
    )

    return T, equations_struct, ℂ, 𝓦
end
FOOTER

wc -l $OUT
