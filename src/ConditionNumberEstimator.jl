using KrylovKit

function estimate_condition_number_operator(A, P)
    # Define the action of the preconditioned operator
    #C = cholesky(Hermitian(P))
    function preconditioned_op(x)
        y = A * x
        z = P \ y   # apply the inverse of the linear operator P
        return z
    end
    x0 = randn(size(A,1))
    λ_max, _,_ = eigsolve(preconditioned_op, x0, 1, :LM; issymmetric=true)
    λ_min, _,_ = eigsolve(preconditioned_op, x0, 1, :SR; issymmetric=true)
    return abs(λ_max[1]) / abs(λ_min[1]) # Returns cond number 
end