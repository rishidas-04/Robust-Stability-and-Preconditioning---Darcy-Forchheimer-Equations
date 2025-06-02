using KrylovKit

function estimate_condition_number_operator(A, P; nev=1)
    # Define the action of the preconditioned operator
    function preconditioned_op(x)
        Px = P \ x
        return A * Px
    end
    x0 = rand(size(A,1))
    λ_max, _,_ = eigsolve(preconditioned_op, x0, nev, :LM)
    λ_min, _,_ = eigsolve(preconditioned_op, x0, nev, :SM)
    return abs(λ_max[1]) / abs(λ_min[1]) # Returns cond number 
end