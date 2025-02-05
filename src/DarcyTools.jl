
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

   Uh = TrialFESpace(Uh_,u_ex)
   Qh = TrialFESpace(Qh_)

   Yh = MultiFieldFESpace([Uh_,Qh_])
   Xh = MultiFieldFESpace([Uh,Qh])

   Ω = Triangulation(model)
   dΩ = Measure(Ω,2*(k+2))
   Γ = BoundaryTriangulation(model)
   n_Γ = get_normal_vector(Γ)
   dΓ = Measure(Γ,2*(k+2))

   a((u, p),(v, q)) = ∫( (ν/κ)*(u⋅v) )*dΩ +∫((ν^2/κ)*(∇⋅u)*(∇⋅v) )*dΩ - ∫(q*(∇⋅u))*dΩ - ∫((∇⋅v)*p)*dΩ - ∫(c0*p*q)*dΩ
   b((v, q)) = ∫((ν/κ)*(u_ex⋅v))*dΩ - ∫((ν^2/κ)*((∇(∇⋅u_ex))⋅v))*dΩ + ∫(∇(p_ex)⋅v)*dΩ - ∫(c0*(p_ex*q))*dΩ - ∫(q*(∇⋅u_ex))*dΩ


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
  dΓ = Measure(Γ,2*(k+2))

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

function compute_errors_darcy(xh, dΩ, u_ex, p_ex)
  uh, ph, _ = xh
  error_u = sqrt(sum(∫((u_ex-uh)⋅(u_ex-uh))*dΩ + ∫((∇⋅(u_ex-uh))*(∇⋅(u_ex-uh)))*dΩ))
  error_p = sqrt(sum(∫((p_ex-ph)*(p_ex-ph))*dΩ))
  error_u,error_p
end
