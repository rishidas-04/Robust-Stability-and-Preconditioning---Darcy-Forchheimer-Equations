module GridapPertubedSaddlePointPreconditioners

    using Gridap
    using Gridap.ReferenceFEs
    using Gridap.FESpaces
    using LinearAlgebra
    using LinearOperators

    include("DarcyTools.jl")
    include("Preconditioners.jl")
    #include("ConstantFESpaces.jl")

    export assemble_perturbed_darcy, assemble_darcy, compute_errors_darcy, compute_errors_perturbed_darcy
    export generate_model2d
    export GridapLinearSolverPreconditioner
end
