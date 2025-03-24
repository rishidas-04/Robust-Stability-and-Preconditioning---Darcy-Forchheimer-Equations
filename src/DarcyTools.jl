
function generate_model2d(nk)
  domain =(0,1,0,1)
  n      = 2^nk + 1
  # Discrete model
  partition = (n,n)
  CartesianDiscreteModel(domain, partition) |> simplexify
end


function assemble_perturbed_darcy(model, k, κ, ν, c0, u_ex, p_ex)
   reffe_u = ReferenceFE(raviart_thomas,Float64,k)
   reffe_p = ReferenceFE(lagrangian,Float64,k)

   # FESpaces
   Uh_ = TestFESpace(model,reffe_u,dirichlet_tags="boundary",conformity=:HDiv)
   Qh_ = TestFESpace(model,reffe_p,conformity=:L2)
   Lh_ = ConstantFESpace(model)

   Uh = TrialFESpace(Uh_,u_ex)
   Qh = TrialFESpace(Qh_)
   Lh = TrialFESpace(Lh_)

   Yh = MultiFieldFESpace([Uh_,Qh_,Lh_])
   Xh = MultiFieldFESpace([Uh,Qh,Lh])

   Ω = Triangulation(model)
   dΩ = Measure(Ω,2*(k+2))
   Γ = BoundaryTriangulation(model)
   n_Γ = get_normal_vector(Γ)
   dΓ = Measure(Γ,2*(k+2)-1)

    a((u, p),(v, q)) = ∫( (ν/κ)*(u⋅v) )*dΩ +∫((ν^2/κ)*(∇⋅u)*(∇⋅v) )*dΩ - ∫(q*(∇⋅u))*dΩ - ∫((∇⋅v)*p)*dΩ - ∫(c0*p*q)*dΩ
   b((v, q)) = ∫((ν/κ)*(u_ex⋅v))*dΩ - ∫((ν^2/κ)*((∇(∇⋅u_ex))⋅v))*dΩ + ∫(∇(p_ex)⋅v)*dΩ - ∫(c0*(p_ex*q))*dΩ - ∫(q*(∇⋅u_ex))*dΩ

  # a((u, p),(v, q)) = ∫( (1.0/κ)*(u⋅v) )*dΩ - ∫(q*(∇⋅u))*dΩ - ∫((∇⋅v)*p)*dΩ - ∫(c0*p*q)*dΩ
  #  b((v, q)) = ∫((1.0/κ)*(u_ex⋅v))*dΩ + ∫(∇(p_ex)⋅v)*dΩ - ∫(c0*(p_ex*q))*dΩ - ∫(q*(∇⋅u_ex))*dΩ


   # Build affine FE operator
   op = AffineFEOperator(a,b,Xh,Yh)
   op, dΩ, dΓ
end


function assemble_darcy(model, k, κ, u_ex, p_ex)
  reffe_u = ReferenceFE(raviart_thomas,Float64,k)
  reffe_p = ReferenceFE(lagrangian,Float64,k)

  # FESpaces
  Uh_ = TestFESpace(model,reffe_u,dirichlet_tags="boundary",conformity=:HDiv)
  Qh_ = TestFESpace(model,reffe_p,conformity=:L2)
  Lh_ = ConstantFESpace(model)

  Uh = TrialFESpace(Uh_,u_ex)
  Qh = TrialFESpace(Qh_)
  Lh = TrialFESpace(Lh_)

  Yh = MultiFieldFESpace([Uh_,Qh_,Lh_])
  Xh = MultiFieldFESpace([Uh,Qh,Lh])

  Ω = Triangulation(model)
  dΩ = Measure(Ω,2*(k+2))
  Γ = BoundaryTriangulation(model)
  n_Γ = get_normal_vector(Γ)
  dΓ = Measure(Γ,2*(k+2)-1)

  a((u, p, ϕ),(v, q, ψ)) = ∫( (1/κ)*(u⋅v) )*dΩ - ∫(q*(∇⋅u))*dΩ - ∫((∇⋅v)*p)*dΩ + ∫(p*ψ)dΩ + ∫(q*ϕ)dΩ 
  b((v, q, ψ)) = ∫((1/κ)*(u_ex⋅v))*dΩ + ∫(∇(p_ex)⋅v)*dΩ - ∫(q*(∇⋅u_ex))*dΩ + ∫(p_ex*ψ)dΩ


  # Build affine FE operator
  op = AffineFEOperator(a,b,Xh,Yh)
  op, dΩ, dΓ
end

function compute_errors_perturbed_darcy(xh, dΩ, u_ex, p_ex)
  uh, ph = xh
  error_u = sqrt(sum(∫((u_ex-uh)⋅(u_ex-uh))*dΩ + ∫((∇⋅(u_ex-uh))*(∇⋅(u_ex-uh)))*dΩ))
  error_p = sqrt(sum(∫((p_ex-ph)*(p_ex-ph))*dΩ))
  error_u,error_p
end

function compute_errors_darcy(xh, dΩ, κ, u_ex, p_ex)
  uh, ph, _ = xh
  error_u = sqrt(sum(∫(1.0/κ*(u_ex-uh)⋅(u_ex-uh))*dΩ + ∫((∇⋅(u_ex-uh))*(∇⋅(u_ex-uh)))*dΩ))
  error_p = sqrt(sum(∫((p_ex-ph)*(p_ex-ph))*dΩ))
  error_u,error_p
end

function compute_operator_norm(B,xh)
  x=get_free_dof_values(xh)
  sqrt(dot(x,B*x))
end

function assemble_darcy_riesz_mapping_preconditioner_blocks(op, dΩ, dΓ, dΛ,
  h_e, h_e_Γ, κ; prec_variant=:B1)
  X1,X2,X3 = op.trial
  Y1,Y2,Y3 = op.test 

  @assert prec_variant in (:B1,:B2)
  if (prec_variant==:B1)
    # a11_B1(u,v)=∫((1.0/κ)*(u⋅v)+(1.0/κ)*(∇⋅u)*(∇⋅v))dΩ
    a11_B1(u,v)=∫((1.0/κ)*u⋅v + (1.0^2/κ)*(∇⋅u)*(∇⋅v))dΩ
    # a22_B1(p,q)= ∫(κ*p*q)dΩ
    a22_B1(p,q)=∫((1.0+κ/1.0+κ/1.0^2)*(p*q))dΩ
    a33_B1(ϕ,ψ) = ∫(ϕ*ψ)dΩ

    A11,A22,A33 = assemble_matrix(a11_B1,X1,Y1),assemble_matrix(a22_B1,X2,Y2),assemble_matrix(a33_B1,X3,Y3)
    return A11,A22,A33
  else
    # a11_B2(u,v)=∫((1.0/κ)*(u⋅v)+(∇⋅u)*(∇⋅v))dΩ
    a11_B2(u,v)=∫((1.0/κ)*u⋅v + (1.0+1.0^2/κ)*(∇⋅u)*(∇⋅v))dΩ
    # a22a_B2(p,q)= ∫(p*q)dΩ
    a22a_B2(p,q)=∫((1.0+1.0)*(p*q))dΩ
    # a22b_B2(p,q)=  ∫(κ*(∇(p))⋅∇(q))dΩ + ∫(κ/h_e*jump(p)*jump(q))dΛ + ∫((κ/h_e_Γ)*p*q)dΓ
    a22b_B2(p,q)=∫(1.0*(p*q))dΩ + ∫(κ/1.0*(∇(p))⋅∇(q))dΩ + ∫(κ/1.0*1.0/h_e*jump(p)*jump(q))dΛ #+ ∫(κ/ν*1.0/h_e_Γ*p*q)dΓ
    a33_B2(ϕ,ψ) = ∫(ϕ*ψ)dΩ

    A11,A22a,A22b,A33 = assemble_matrix(a11_B2,X1,Y1),assemble_matrix(a22a_B2,X2,Y2),assemble_matrix(a22b_B2,X2,Y2),assemble_matrix(a33_B2,X3,Y3)
    return A11,A22a,A22b,A33
  end
end



function compute_errors_robust_darcy(xh, op, dΩ, dΓ, dΛ, h_e, h_e_Γ, κ, u_ex, p_ex)

  blocks=assemble_darcy_riesz_mapping_preconditioner_blocks(op, dΩ, dΓ, dΛ,
                                                         h_e, h_e_Γ,
                                                         κ;
                                                         prec_variant=:B2)

  A11,A22a,A22b,A33=blocks
  A11 = LinearOperator(A11)
  A22 = LinearOperator(A22a)+LinearOperator(A22b)

  A22_inv = LinearOperator(GridapLinearSolverPreconditioner(A22a))+
                LinearOperator(GridapLinearSolverPreconditioner(A22b))

  A33 = LinearOperator(A33)   

  uh, ph, _ = xh
  eu = u_ex-uh
  ep = p_ex-ph
  X1,X2,_ = op.trial
  euh=interpolate(eu,X1)
  eph=interpolate(ep,X2)


  freedofs = get_free_dof_values(eph)
  newfreedofs = zeros(length(freedofs))
  auxM = A22_inv
  auxr=copy(freedofs)
  rtol=1.0e-14

  function custom_stopping_condition(solver::KrylovSolver, A, b, r, tol)
    mul!(r, A, solver.x)
    r .-= b                       # r := b - Ax
    bool = norm(r) ≤ tol*norm(b)  # tolerance based on the 2-norm of the residual
    #@printf("ERR NORM: ||b-Ax||/||b||: %16.7e\n",norm(r)/norm(b))
    return bool
  end
  
  minres_callback(solver) = custom_stopping_condition(solver, auxM, freedofs, auxr, rtol)
  # Using minres here for simplicity, but one could 
  # use pcg as well, as A is SPD
  (newfreedofs, stats) = minres(auxM, freedofs, M=A22; itmax=2000, verbose=1,
                        callback=minres_callback,
                        atol=0.0,
                        rtol=0.0,
                        etol=0.0) 

  weighted_norm = sqrt(dot(newfreedofs,freedofs))

  return compute_operator_norm(A11,euh), weighted_norm #compute_operator_norm(A22,eph)

end

function assemble_linear_elasticity(model, k, μ, λ, u_ex, p_ex)
  reffe_u = ReferenceFE(lagrangian,VectorValue{2,Float64},k+1)
  reffe_p = ReferenceFE(lagrangian,Float64,k)
  
  # FESpaces (P2-P0)
   Uh_ = TestFESpace(model,reffe_u,dirichlet_tags="boundary",conformity=:H1)
   Qh_ = TestFESpace(model,reffe_p,conformity=:L2)
   Lh_ = ConstantFESpace(model)

  # FESpaces (Hood-Taylor)
  #  Uh_ = TestFESpace(model,reffe_u,dirichlet_tags="boundary",conformity=:H1)
  #  Qh_ = TestFESpace(model,reffe_p,conformity=:C0)

  # FESpaces (Crouzeix-Raviart)
   #Uh_ = TestFESpace(model,reffe_u,dirichlet_tags="boundary",conformity=:H1)
   #Qh_ = TestFESpace(model,reffe_p,conformity=:L2, constraint=:zeromean)

  # Define non-conforming velocity space
  # Uh_ = TestFESpace(model, reffe_u, dirichlet_tags="boundary", conformity=:H1)
  
  # Define pressure space with mean-zero constraint
  # Qh_ = TestFESpace(model, reffe_p, conformity=:L2, constraint=:zeromean)  

  Uh = TrialFESpace(Uh_,u_ex)
  Qh = TrialFESpace(Qh_)
  Lh = TrialFESpace(Lh_)

  Yh = MultiFieldFESpace([Uh_,Qh_,Lh_])
  Xh = MultiFieldFESpace([Uh,Qh,Lh])

  Ω = Triangulation(model)
  dΩ = Measure(Ω,2*(k+2))
  Γ = BoundaryTriangulation(model)
  n_Γ = get_normal_vector(Γ)
  dΓ = Measure(Γ,2*(k+2))

  Id = TensorValue(1.0,0.0,0.0,1.0)
  
    σ_ex(x) = 2*μ*ε(u_ex)(x)- p_ex(x)*Id
    b_ex(x) = -(∇⋅σ_ex)(x)
  
  lhs((u,p),(v,q)) =  ∫(2*μ*ε(v)⊙ε(u))dΩ -∫((∇⋅v)*p)dΩ -
  ∫((∇⋅u)*q)dΩ       -∫(1/λ*p*q)dΩ   
  
  rhs((v,q)) = ∫(b_ex⋅v)dΩ
  
  op = AffineFEOperator(lhs,rhs,Xh,Yh)
  op, dΩ, dΓ
  
end





function assemble_linear_elasticity_riesz_mapping_preconditioner_blocks(op, dΩ, dΓ, dΛ,
  h_e, h_e_Γ, μ, λ; prec_variant=:B1)
  X1,X2,X3 = op.trial
  Y1,Y2,Y3 = op.test 

  @assert prec_variant in (:B1,:B2)
  if (prec_variant==:B1)
    a11_B1(u,v)=∫(2*μ*ε(v)⊙ε(u))dΩ
    a22_B1(p,q)= ∫((1.0/μ)*p*q)dΩ
    a33_B1(ϕ,ψ) = ∫(ϕ*ψ)dΩ

    A11,A22,A33 = assemble_matrix(a11_B1,X1,Y1),assemble_matrix(a22_B1,X2,Y2),assemble_matrix(a33_B1,X3,Y3)
    return A11,A22,A33
  else
    a11_B2(u,v)=∫(2*μ*ε(v)⊙ε(u))dΩ
    a22a_B2(p,q)= ∫((1.0/λ)*p*q)dΩ
    a22b_B2(p,q)=  ∫((1.0/μ)*p*q)dΩ + ∫(μ/h_e*jump(p)*jump(q))dΛ + ∫((μ/h_e_Γ)*p*q)dΓ
    a33_B2(ϕ,ψ) = ∫(ϕ*ψ)dΩ

    A11,A22a,A22b,A33 = assemble_matrix(a11_B2,X1,Y1),assemble_matrix(a22a_B2,X2,Y2),assemble_matrix(a22b_B2,X2,Y2),assemble_matrix(a33_B2,X3,Y3)
    return A11,A22a,A22b,A33
  end
end



function compute_errors_robust_linear_elasticity(xh, op, dΩ, dΓ, dΛ, h_e, h_e_Γ, μ, λ, u_ex, p_ex)

  blocks=assemble_linear_elasticity_riesz_mapping_preconditioner_blocks(op, dΩ, dΓ, dΛ,
                                                         h_e, h_e_Γ,
                                                         μ, λ;
                                                         prec_variant=:B2)

  A11,A22a,A22b,A33=blocks
  A11 = LinearOperator(A11)
  A22 = LinearOperator(A22a)+LinearOperator(A22b)

  A22_inv = LinearOperator(GridapLinearSolverPreconditioner(A22a))+
                LinearOperator(GridapLinearSolverPreconditioner(A22b))

  A33 = LinearOperator(A33)   

  uh, ph, _ = xh
  eu = u_ex-uh
  ep = p_ex-ph
  X1,X2,_ = op.trial
  euh=interpolate(eu,X1)
  eph=interpolate(ep,X2)


  freedofs = get_free_dof_values(eph)
  newfreedofs = zeros(length(freedofs))
  auxM = A22_inv
  auxr=copy(freedofs)
  rtol=1.0e-14

  function custom_stopping_condition(solver::KrylovSolver, A, b, r, tol)
    mul!(r, A, solver.x)
    r .-= b                       # r := b - Ax
    bool = norm(r) ≤ tol*norm(b)  # tolerance based on the 2-norm of the residual
    #@printf("ERR NORM: ||b-Ax||/||b||: %16.7e\n",norm(r)/norm(b))
    return bool
  end
  
  minres_callback(solver) = custom_stopping_condition(solver, auxM, freedofs, auxr, rtol)
  # Using minres here for simplicity, but one could 
  # use pcg as well, as A is SPD
  (newfreedofs, stats) = minres(auxM, freedofs, M=A22; itmax=2000, verbose=1,
                        callback=minres_callback,
                        atol=0.0,
                        rtol=0.0,
                        etol=0.0) 

  weighted_norm = sqrt(dot(newfreedofs,freedofs))

  return compute_operator_norm(A11,euh), weighted_norm #compute_operator_norm(A22,eph)

end





function compute_errors_linear_elasticity(xh, dΩ, μ, λ, u_ex, p_ex)
  uh, ph = xh
  error_u = sqrt(2*μ)*sqrt(sum(∫((u_ex-uh)⋅(u_ex-uh))*dΩ +∫(∇(u_ex-uh)⊙∇(u_ex - uh))*dΩ))
  #error_p = 1/sqrt(2*μ)*sqrt(sum(∫((p_ex-ph)*(p_ex-ph))*dΩ))
  error_p = sqrt(sum(∫((1.0/√μ)*(p_ex-ph)*(p_ex-ph))*dΩ + ∫((1.0/√λ)*(p_ex-ph)*(p_ex-ph))*dΩ))
  error_u,error_p
end
