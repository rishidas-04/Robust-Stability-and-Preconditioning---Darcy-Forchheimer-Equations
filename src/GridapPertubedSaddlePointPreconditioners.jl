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

     
    export assemble_perturbed_darcy, assemble_darcy, assemble_darcy_noExact, compute_errors_darcy, compute_errors_perturbed_darcy, compute_operator_norm, compute_errors_robust_darcy,  assemble_linear_elasticity, compute_errors_linear_elasticity

    export generate_model2d, setup_model_labels_unit_square!
    export GridapLinearSolverPreconditioner
end
