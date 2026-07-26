"""
Time the incremental cost of loading MacroModelling's direct dependencies.

Each process should be started with `--startup-file=no --history-file=no
--project=.`.  The reported times are incremental costs in this import order:
dependencies shared by multiple packages are charged to the first import that
loads them, so they should not be summed as independent package totals.
"""

function timed_import(package_name::Symbol)
    started = time_ns()
    Base.require(Main, package_name)
    elapsed = (time_ns() - started) / 1.0e9
    println(package_name, " ", round(elapsed; digits=3), " s")
    return elapsed
end

package_order = (
    :DocStringExtensions,
    :ThreadedSparseArrays,
    :PrecompileTools,
    :SpecialFunctions,
    :SymPyPythonCall,
    :PythonCall,
    :Symbolics,
    :Accessors,
    :Dates,
    :LoopVectorization,
    :NLopt,
    :SparseArrays,
    :LinearSolve,
    :FastLapackInterface,
    :Combinatorics,
    :BlockTriangularForm,
    :Subscripts,
    :Krylov,
    :LinearOperators,
    :DataStructures,
    :MacroTools,
    :Suppressor,
    :REPL,
    :Unicode,
    :MatrixEquations,
    :AxisKeys,
    :ChainRulesCore,
    :RuntimeGeneratedFunctions,
    :Reexport,
    :DispatchDoctor,
)

total_started = time_ns()
for package_name in package_order
    timed_import(package_name)
end
total_elapsed = (time_ns() - total_started) / 1.0e9
println("direct imports total ", round(total_elapsed; digits=3), " s")

println("MacroModelling itself")
timed_import(:MacroModelling)
