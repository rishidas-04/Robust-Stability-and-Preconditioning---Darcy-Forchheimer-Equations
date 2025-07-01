module BaerlandDarcyPreconditionerTests
  using PertubedSaddlePointPreconditioners
  using Gridap
  import Gridap: ∇
  using Printf
  using Test
  using Krylov
  using LinearAlgebra
  using LinearOperators


  # We want to reproduce the table from Baerland et al.
  # "A robust preconditioner for the Darcy problem with mixed boundary conditions"
  # https://doi.org/10.1016/j.camwa.2017.01.001

  function assemble_baerland_preconditioner(op, dΩ, dΓ, dΛ, h_e, h_e_Γ, κ; prec_variant=:prec_variant)
    X1,X2,X3 = op.trial
    Y1,Y2,Y3 = op.test 

    N  = num_free_dofs(op.trial)
    Ns = [num_free_dofs(U) for U in op.trial] 
    Np = zeros(Int,length(Ns)+1)
    Np[1]=1
      for i=2:length(Ns)+1
          Np[i]=Np[i-1]+Ns[i-1]
      end
    Pinv=zeros(N,N); 
    Adense = Array(op.op.matrix)
    range1=Np[1]:Np[2]-1
    range2=Np[2]:Np[3]-1
    range3=Np[3]:Np[4]-1


    @assert prec_variant in (:B1,:B2)
    if (prec_variant==:B2)
      a11_B2(u,v) =∫((1.0/κ)*(u⋅v)+(∇⋅u)*(∇⋅v))dΩ
      a22a_B2(p,q)= ∫(p*q)dΩ
      a22b_B2(p,q)= ∫(κ*(∇(p))⋅∇(q))dΩ + ∫((κ/h_e)*jump(p)*jump(q))dΛ + ∫((κ/h_e_Γ)*p*q)dΓ
      a33_B2(ϕ,ψ) = ∫(ϕ*ψ)dΩ
  
      A11_,A22a_,A22b_,A33_ = assemble_matrix(a11_B2,X1,Y1),assemble_matrix(a22a_B2,X2,Y2),assemble_matrix(a22b_B2,X2,Y2),assemble_matrix(a33_B2,X3,Y3)

      A22=LinearOperator(GridapLinearSolverPreconditioner(A22a_))+
             LinearOperator(GridapLinearSolverPreconditioner(A22b_))

      A22ainv=inv(Array(A22a_));
      A22binv=inv(Array(A22b_));  

      Pinv[range2,range2]=A22ainv+A22binv;         
      
    else #THIS IS FOR B1:
      a11_B1(u,v)=∫((1.0/κ)*(u⋅v) + (1.0/κ)*(∇⋅u)*(∇⋅v))dΩ
      a22_B1(p,q)= ∫(κ*p*q)dΩ
      a33_B1(ϕ,ψ) = ∫(κ*ϕ*ψ)dΩ
  
      A11_,A22_,A33_ = assemble_matrix(a11_B1,X1,Y1),assemble_matrix(a22_B1,X2,Y2),assemble_matrix(a33_B1,X3,Y3)
      A22=LinearOperator(GridapLinearSolverPreconditioner(A22_)) 

      A22inv=inv(Array(A22_));  
      Pinv[range2,range2]=A22inv;  

    end

    A11=LinearOperator(GridapLinearSolverPreconditioner(A11_))
    A33=LinearOperator(GridapLinearSolverPreconditioner(A33_))  

    A11inv=inv(Array(A11_));  
    A33inv=inv(Array(A33_));
    Pinv[range1,range1]=A11inv;   
    Pinv[range3,range3]=A33inv;
    evals=eigvals(Pinv*Adense)
    @printf("condition number of the preconditioned system: %1.3e\n", maximum(broadcast(abs,evals))/minimum(broadcast(abs,evals)))

    return BlockDiagonalOperator(A11,A22,A33)
  end

  
  function solve_using_baerland_preconditioner(op,P;
                                                       rtol=1.0e-6,
                                                       itmax=500, 
                                                       verbose=false)
    A=op.op.matrix
    b=op.op.vector
    r=copy(b)
    conv_history = Float64[]
    function custom_stopping_condition(solver::KrylovSolver, A, b, r, tol)
      mul!(r, A, solver.x)
      r .-= b                       # r := b - Ax
      bool = norm(r) ≤ tol*norm(b)  # tolerance based on the 2-norm of the residual
      if (verbose)
         @printf("||b-Ax||/||b||: %16.7e\n",norm(r)/norm(b))
      end
      push!(conv_history,norm(r)/norm(b))
      return bool
    end
    minres_callback(solver) = custom_stopping_condition(solver, A, b, r, rtol)
    (xdofs, stats) = minres(A, b, M=P; itmax=itmax, verbose=1,
                         callback=minres_callback,
                         atol=0.0,
                         rtol=0.0, 
                         etol=0.0)
    xh=FEFunction(op.trial,xdofs)

    if (verbose)
      println("#iters: $(stats.niter)")
    end

    xh, stats, conv_history
  end



  k=0; κ = 1.0E-4; nk = 3

  # RHS for Darcy
  f_rhs  = VectorValue(0.1,0)
  g_rhs  = 0.0

  model=generate_model2d(nk)
  Ω = Triangulation(model)
  dΩ = Measure(Ω,2*(k+2))
  Λ = SkeletonTriangulation(model)
  dΛ = Measure(Λ,2*(k+2)-1)
  h_e = CellField(get_array(∫(1)dΛ),Λ)
  Γ = BoundaryTriangulation(model)
  dΓ = Measure(Γ,2*(k+2)-1)
  h_e_Γ = CellField(get_array(∫(1)dΓ),Γ)
  
  oper  = assemble_darcy_noExact(model,k,κ,f_rhs,g_rhs)
  riesz = assemble_baerland_preconditioner(oper, dΩ, dΓ, dΛ, h_e, h_e_Γ, κ; prec_variant=:B1)
  xh,state = solve_using_baerland_preconditioner(oper,riesz)

end
