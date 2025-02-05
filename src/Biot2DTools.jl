function generate_model_unit_square(nk)
  domain =(0,1,0,1)
  n      = 2^nk
  # Discrete model
  partition = (n,n)
  model = CartesianDiscreteModel(domain, partition) |> simplexify
  model
end


function define_analytical_functions_biot(μ, λ, ν, κ, α, c_0, u_ex, p_ex)
  Id = TensorValue(1,0,0,1)
  φ_ex(x) = α*p_ex(x)-λ*(∇⋅u_ex)(x)
  σ_ex(x) = 2*μ*ε(u_ex)(x)- φ_ex(x)*Id
  b_ex(x) = -(∇⋅σ_ex)(x)
  z_ex(x) = -κ/ν*∇(p_ex)(x)
  g_ex(x) = (c_0 + α^2/λ)*p_ex(x) - α/λ*φ_ex(x) + (∇⋅z_ex)(x)
  return φ_ex, σ_ex, b_ex, g_ex, z_ex
end

function assemble_biot2d(model, k, μ, λ, ν, κ, α, c_0, u_ex, p_ex)
  φ_ex, σ_ex, b_ex, g_ex, z_ex =
     define_analytical_functions_biot(μ, λ, ν, κ, α, c_0, u_ex, p_ex)

   # Define reference FE (P2-P0/RT0-P0)
   reffe_u = ReferenceFE(lagrangian,VectorValue{2,Float64},k+2)
   reffe_φ = ReferenceFE(lagrangian,Float64,k)
   reffe_z = ReferenceFE(raviart_thomas,Float64,k)
   reffe_p = ReferenceFE(lagrangian,Float64,k)
   #reffe_aux = ConstantFESpace(lagrangian,Float64,0)

   # FESpaces
   Uh_ = TestFESpace(model,reffe_u,dirichlet_tags="boundary",conformity=:H1)
   Φh_ = TestFESpace(model,reffe_φ,conformity=:L2)
   Zh_ = TestFESpace(model,reffe_z,dirichlet_tags="boundary",conformity=:HDiv)
   Ph_ = TestFESpace(model,reffe_p,conformity=:L2)
   #Lh_ = ConstantFESpace(model,reffe_aux,Conformity=:L2)
 
   Uh = TrialFESpace(Uh_,u_ex)
   Φh = TrialFESpace(Φh_)
   Zh = TrialFESpace(Zh_,z_ex)
   Ph = TrialFESpace(Ph_)

   Yh = MultiFieldFESpace([Uh_,Φh_,Zh_,Ph_])
   Xh = MultiFieldFESpace([Uh,Φh,Zh,Ph])

   Ω = Triangulation(model)
   dΩ = Measure(Ω,2*(k+3))

   # bilinear form
   lhs((u,φ,z,p),(v,ψ,w,q)) =  ∫(2*μ*ε(v)⊙ε(u))dΩ -∫((∇⋅v)*φ)dΩ -
                               ∫((∇⋅u)*ψ)dΩ       -∫(1/λ*φ*ψ)dΩ                 +∫(α/λ*p*ψ)dΩ +
                                                                 ∫(ν/κ*(z⋅w))dΩ -∫(p*(∇⋅w))dΩ +
                                                   ∫(α/λ*φ*q)dΩ -∫((∇⋅z)*q)dΩ   -∫((c_0+α^2/λ)*p*q)dΩ 

   # linear functional
   rhs((v,ψ,w,q)) = ∫(b_ex⋅v - g_ex*q)dΩ

   # Build affine FE operator
   op = AffineFEOperator(lhs,rhs,Xh,Yh)
end

function compute_errors_biot2d(xh, dΩ, μ, λ, ν, κ, α, c_0, u_ex, p_ex)
  uh, φh, zh, ph = xh
  Xh = get_fe_space(xh)
  _, _, _, Qh = Xh
  φ_ex, σ_ex, b_ex, g_ex, z_ex =
    define_analytical_functions_biot(μ, λ, ν, κ, α, c_0, u_ex, p_ex)

  a(u,v) = ∫( u*v )*dΩ
  l(v) = ∫(v*((c_0+α^2/λ)*ph-α/λ*φh+(∇⋅zh)-g_ex))*dΩ
  projection = AffineFEOperator(a,l,Qh,Qh)
  loss = solve(projection)
  lossh=norm(get_free_dof_values(loss),Inf)

  # errors in the non-weighted norms
  error_u = sqrt(sum(∫((u_ex-uh)⋅(u_ex-uh))*dΩ +∫(∇(u_ex-uh)⊙∇(u_ex - uh))*dΩ))
  error_φ = sqrt(sum(∫((φ_ex-φh)*(φ_ex-φh))*dΩ))
  error_z = sqrt(sum(∫((z_ex-zh)⋅(z_ex-zh))*dΩ + ∫((∇⋅(z_ex-zh))*(∇⋅(z_ex-zh)))*dΩ))
  error_p = sqrt(sum(∫((p_ex-ph)*(p_ex-ph))*dΩ))

  error_u,error_φ,error_z,error_p, lossh
end