module DarcyPreconditionerTests
  using BiotBrinkmanWithVorticity
  using Gridap
  import Gridap: ∇
  using Printf
  using Test
  using Krylov 
  using LinearAlgebra
  using LinearOperators

  # Options
  const write_solution_to_vtk = false

  # manufactured solutions
  u_darcy_ex(x)  = VectorValue((sin(π*x[1]))^2*sin(π*x[2]),
                          sin(π*x[1])*(sin(π*x[2]))^2)
  p_darcy_ex(x)  = sin(π*x[1])*cos(π*x[2])

  function assemble_darcy_riesz_mapping_preconditioner_blocks(op, dΩ, dΛ, dΓ, h_e, h_e_Γ,  κ, ν, c0 ;
                                                              prec_variant=:B1)

    @assert prec_variant in (:B1,:B2)
    if prec_variant == :B1 
      #a11b1(u,v)=∫((ν/κ)*u⋅v + ν/κ*(∇⋅u)*(∇⋅v))dΩ
      a11b1(u,v)=∫((ν/κ)*u⋅v + (ν^2/κ)*(∇⋅u)*(∇⋅v))dΩ
      #a22b1(p,q)=∫((c0+κ/ν)*(p*q))dΩ
      a22b1(p,q)=∫((c0+κ/ν+κ/ν^2)*(p*q))dΩ
    else
      #a11b2(u,v)=∫((ν/κ)*u⋅v + (∇⋅u)*(∇⋅v))dΩ
      a11b2(u,v)=∫((ν/κ)*u⋅v + (1.0+ν^2/κ)*(∇⋅u)*(∇⋅v))dΩ
      a22ab2(p,q)=∫((1.0+c0)*(p*q))dΩ
      a22bb2(p,q)=∫(c0*(p*q))dΩ + ∫(κ/ν*(∇(p))⋅∇(q))dΩ + ∫(κ/ν*1.0/h_e*jump(p)*jump(q))dΛ #+ ∫(κ/ν*1.0/h_e_Γ*p*q)dΓ
    end    
    if (prec_variant==:B1)
      Y1,Y2=op.test
      X1,X2=op.trial
      A11,A22=assemble_matrix(a11b1,X1,Y1),
              assemble_matrix(a22b1,X2,Y2)
      return A11,A22
    else
      Y1,Y2=op.test
      X1,X2=op.trial
      A11=assemble_matrix(a11b2,X1,Y1)
      A22a=assemble_matrix(a22ab2,X2,Y2)
      A22b=assemble_matrix(a22bb2,X2,Y2)
      return A11,A22a,A22b
    end
  end

  function solve_darcy_riesz_mapping_preconditioner(op, dΩ, dΛ, dΓ, h_e, h_e_Γ,
                                                       κ, ν, c0;
                                                       rtol=1.0e-6,
                                                       itmax=500,
                                                       prec_variant=:B1,
                                                       verbose=false)

    if (verbose)
      println("###############")
      println("κ=$(κ)")
      println("prec_variant=$(prec_variant)")
      println("###############")
    end

    blocks=assemble_darcy_riesz_mapping_preconditioner_blocks(op, dΩ, dΛ, dΓ, h_e, h_e_Γ, κ, ν, c0;
                                                              prec_variant=prec_variant)
    
    if (prec_variant==:B1)
      P=BlockDiagonalOperator(LinearOperator.(GridapLinearSolverPreconditioner.(blocks))...) 
    else 
      A11,A22a,A22b=blocks
       A11=LinearOperator(GridapLinearSolverPreconditioner(A11))
       A22=LinearOperator(GridapLinearSolverPreconditioner(A22a))+
             LinearOperator(GridapLinearSolverPreconditioner(A22b))
       P=BlockDiagonalOperator(A11,A22)
      
    end  

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
                         ratol=0.0,
                         rrtol=0.0,
                         etol=0.0)
    xh=FEFunction(op.trial,xdofs)

    if (verbose)
      println("#iters: $(stats.niter)")
    end

    xh, stats, conv_history
  end

  export solve_darcy_riesz_mapping_preconditioner

  # k=0
  # model=generate_model(2)
  # Ω  = Triangulation(model)
  # dΩ = Measure(Ω,2*(k+2))

  # op=assemble_biotbrinkman(model, k, μ, λ, ν, κ, α, c_0, u_ex, p_ex, v_ex)
  # xh,state=
  #    solve_biotbrinkman_riesz_mapping_preconditioner(op, dΩ, μ, λ, ν, κ, α, c_0;
  #                                                    option=:B3)
  # uh, vh, ωh, φh, ph = xh
  # ndofs=num_free_dofs(Gridap.FESpaces.get_fe_space(xh))


  # error_u,error_v,error_ω,error_φ,error_p,lossh=
  #     compute_errors_biotbrinkman(xh, dΩ, μ, λ, ν, κ, α, c_0, u_ex, p_ex, v_ex)

  # println(error_u, " ", error_v, " ", error_ω, " ", error_φ, " ", error_p, " ", lossh)

end
