module GridapPertubedSaddlePointPreconditioners

    using Gridap
    using Gridap.ReferenceFEs
    using Gridap.FESpaces
    using LinearAlgebra
    using LinearOperators

    include("DarcyTools.jl")
    include("Preconditioners.jl")
    include("ConstantFESpaces.jl")

    export assemble_darcy, compute_errors_darcy
    export generate_model2d
    export GridapLinearSolverPreconditioner
end
