module GridapPertubedSaddlePointPreconditioners

    using Gridap
    using Gridap.ReferenceFEs
    using Gridap.FESpaces
    using LinearOperators
    using LinearAlgebra
    using Krylov
    using Printf

    include("Preconditioners.jl")
    include("DarcyTools.jl")
    
    #include("ConstantFESpaces.jl")

    export assemble_perturbed_darcy, assemble_darcy, compute_errors_darcy, compute_errors_perturbed_darcy, compute_operator_norm, compute_errors_robust_darcy, assemble_darcy_riesz_mapping_preconditioner_blocks, assemble_linear_elasticity, compute_errors_linear_elasticity

    export generate_model2d
    export GridapLinearSolverPreconditioner
end
