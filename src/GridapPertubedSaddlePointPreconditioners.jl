module GridapPertubedSaddlePointPreconditioners

    using Gridap
    using Gridap.ReferenceFEs
    using Gridap.FESpaces
    using LinearAlgebra
    using LinearOperators
    using Printf
    using Krylov

    include("DarcyTools.jl")
    include("Preconditioners.jl")
    #include("ConstantFESpaces.jl")

    export assemble_perturbed_darcy, assemble_darcy, compute_errors_darcy, compute_errors_perturbed_darcy, compute_operator_norm, compute_errors_robust_darcy, assemble_darcy_riesz_mapping_preconditioner_blocks

    export generate_model2d
    export GridapLinearSolverPreconditioner
end
