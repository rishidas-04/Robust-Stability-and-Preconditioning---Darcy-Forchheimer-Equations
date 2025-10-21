module DarcyForchheimerConvergenceTests_mixedBC_precond

using Gridap
import Gridap: ∇
using Printf
using Test
using LineSearches: BackTracking
using LinearAlgebra
using Krylov          
using PertubedSaddlePointPreconditioners
using LinearOperators
using Gridap.CellData

const κ0 = 1.0e-10
const r = 4.0
const FF = 1.0e05

 κ(x) = κ0*(1+exp(-0.5*(10*x[2]-5.0-sin(10.0*x[1]))^2))
  κinv(x)=  1.0/κ(x)

 function normu(v)
    (v⋅v>0)*(sqrt(v⋅v))^(r-2)
  end

  function deri(v) #derivative of |v| wrt v
    (v⋅v>0)*(r-2)*(sqrt(v⋅v))^(r-4)*v
  end

  f_ex = VectorValue(0.0,0.0)
  g_ex = 0.0
  zerov = VectorValue(0.0,0.0)
  uin(x) = VectorValue(2.5*x[2]*(1-x[2]),0.0)
  
  pout  = 0.0

function solve_ForchheimerNoExact(model; k=k, generate_output=false)
  reffe_u = ReferenceFE(raviart_thomas, Float64, k)
  reffe_p = ReferenceFE(lagrangian, Float64, k)

  # FESpaces
  Uh_ = TestFESpace(model, reffe_u, dirichlet_tags=["Gamma_in","Gamma_topbot"], conformity=:HDiv)
  Qh_ = TestFESpace(model, reffe_p, conformity=:L2)

  Uh = TrialFESpace(Uh_,[uin,zerov])   
  Qh = TrialFESpace(Qh_)

  Yh = MultiFieldFESpace([Uh_, Qh_])
  Xh = MultiFieldFESpace([Uh, Qh])

  Ω = Triangulation(model)
  dΩ = Measure(Ω, 2*(k+2))
  Γp = BoundaryTriangulation(model, tags="Gamma_out")
  n_Γp = get_normal_vector(Γp)
  dΓp = Measure(Γp, 2*(k+2)-1)
  Λ = SkeletonTriangulation(model)
  dΛ = Measure(Λ, 3)
  h_e = CellField(get_array(∫(1)dΛ), Λ)
  h_e_Γ = CellField(get_array(∫(1)dΓp), Γp)

  # nonlinear residual and Jacobian forms
  a(u, v)   = ∫( κinv*(u⋅v))*dΩ
  c(u, w, v) = ∫(FF * v⋅((normu∘(u)) * w))*dΩ
  d1(u, w, v) = ∫(FF * v⋅((normu∘(u)) * w))*dΩ
  d2(u, w, v) = ∫(FF * (u⋅v) * ((deri∘(u)) ⋅ w))*dΩ
  b(v, q)   = ∫(-(∇⋅v) * q) * dΩ
  F(v)      = ∫(f_ex ⋅ v) * dΩ - ∫((pout) * (v⋅n_Γp))*dΓp
  G(q)     = ∫((-1.0)*(g_ex*q))dΩ

  resid((u, p), (v, q)) = a(u, v) + c(u, u, v) + b(v, p) + b(u, q) - F(v) - G(q)
  jacob((u, p), (du, dp), (v, q)) = a(du, v) + d1(u, du, v) + d2(u, du, v) + b(v, dp) + b(du, q)

    op = FEOperator(resid,jacob,Xh,Yh)
    nls = NLSolver(show_trace=true, method=:newton, linesearch=BackTracking(), iterations=25, ftol=1.0E-7)
    solver = FESolver(nls)
    x0=0.01*ones(num_free_dofs(Xh)) 
    xh=FEFunction(Xh,x0)
    (uh,ph), _ = solve!(xh,solver,op)

    
    if generate_output 
      writevtk(Ω,"DFNoExactExtreme_ncells=$(num_cells(model))",order=1,cellfields=["uh"=>uh,"ph"=>ph, "κ"=>κ])
    end

  pa(p, q) = ∫(p*q)*dΩ
  pdivl(q) = ∫(q*(∇⋅uh - g_ex)) * dΩ
  projection = AffineFEOperator(pa, pdivl, Qh, Qh)
  loss_div = solve(projection)
  loss_divh = norm(get_free_dof_values(loss_div), Inf)

  ndofs = Gridap.FESpaces.num_free_dofs(Xh)

  return uh, ph, loss_divh, ndofs, op, dΩ, dΓp, dΛ, h_e, h_e_Γ, Uh, Qh
end

function assemble_darcy_forchheimer_riesz_mapping_preconditioner_blocks(op, dΩ, dΓp, dΛ, h_e, h_e_Γ, κinv, FF,uref; prec_variant=:B1)
  X1, X2 = op.trial
  Y1, Y2 = op.test

  @assert prec_variant in (:B1, :B2)
  if prec_variant == :B1
    a11_B1(u, v) = ∫((x->κinv(x) + FF)*(u⋅v) + ((x->κinv(x) + FF))*(∇⋅u)*(∇⋅v))*dΩ
    a22_B1(p, q) = ∫((x->1.0/ (κinv(x) + FF)) * p * q)*dΩ
    A11 = assemble_matrix(a11_B1, X1, Y1)
    A22 = assemble_matrix(a22_B1, X2, Y2)
    return A11, A22
  else
    # B2 variant (simplified)
    a11_B2(u, v) = ∫((x->κinv(x) + FF)*(u⋅v) + (x->κinv(x) + FF)*(∇⋅u)*(∇⋅v))dΩ + (FF)*∫((x -> (normu(uref(x)))^(r-2))*(v⋅u))dΩ + (FF)*(r-2)*∫((x -> (normu(uref(x)))^(r-4)) * (uref ⋅ u)  * (uref ⋅ v))dΩ
    a22a_B2(p, q) = ∫(p*q)*dΩ
    a22b_B2(p, q) =∫((x ->(κinv(x) + FF))*(∇(p))⋅∇(q))dΩ + ∫(((x->(κinv(x) + FF*( (normu(uref(x)))^(r-2))))/h_e)*jump(p)*jump(q))dΛ + ∫(((x->(κinv(x) + FF*( (normu(uref(x)))^(r-2))))/h_e_Γ)*p*q)*dΓ

    A11 = assemble_matrix(a11_B2, X1, Y1)
    A22a = assemble_matrix(a22a_B2, X2, Y2)
    A22b = assemble_matrix(a22b_B2, X2, Y2)

    return A11, A22a, A22b
  end
end

function compute_errors_robust_darcy_forchheimer(op, dΩ, dΓp, dΛ, h_e, h_e_Γ, κinv, FF, uref,pref,Uh,Qh,uh,ph)
  
  blocks = assemble_darcy_forchheimer_riesz_mapping_preconditioner_blocks(op, dΩ, dΓp, dΛ, h_e, h_e_Γ, κinv, FF,uref; prec_variant=:B1)
  A11, A22 = blocks
  A11_lo = LinearOperator(A11)
  A22_lo = LinearOperator(A22)
  
    ur_ = Interpolable(uref)
    pr_ = Interpolable(pref)
    ur = interpolate_everywhere(ur_, Uh)
    pr = interpolate_everywhere(pr_, Qh)

    # uh, ph = xh
    X1, X2 = op.trial
    # euh = ur - uh
    # eph = pr - ph
    euh = FEFunction(X1, get_free_dof_values(ur) .- get_free_dof_values(uh))
    eph = FEFunction(X2, get_free_dof_values(pr) .- get_free_dof_values(ph))

println("types:")
println("  uref  -> ", typeof(uref))
println("  pref  -> ", typeof(pref))
println("  ur_   -> ", typeof(ur_))
println("  pr_   -> ", typeof(pr_))
println("  ur    -> ", typeof(ur))
println("  pr    -> ", typeof(pr))
println("  uh    -> ", typeof(uh))
println("  ph    -> ", typeof(ph))
println("  euh   -> ", typeof(euh))
println("  eph   -> ", typeof(eph))

  
freedofs = get_free_dof_values(eph)
  newfreedofs = zeros(length(freedofs))
  auxM = A22#_inv
  auxr = copy(freedofs)
  rtol = 1.0e-12

  function custom_stopping_condition(solver, A, b, r, tol)
    mul!(r, A, solver.x)
    r .-= b
    return norm(r) ≤ tol * norm(b)
  end

  minres_callback(solver) = custom_stopping_condition(solver, auxM, freedofs, auxr, rtol)

  (newfreedofs, stats) = minres(auxM, freedofs, M=A22_lo; itmax=2000, verbose=0,
                               callback=minres_callback,
                               atol=0.0,
                               rtol=0.0,
                               etol=0.0)

  weighted_norm = sqrt(dot(newfreedofs, freedofs))

  return compute_operator_norm(A11,euh), compute_operator_norm(A22,eph)# weighted_norm
end

# ---------------------------
# Convergence test : runs on a sequence of refinements
function convergence_test(; nkmax, k=k, generate_output=false)
  eu = Float64[]
  ep = Float64[]
  ru = Float64[]
  rp = Float64[]
  hh = Float64[]
  losses_divh = Float64[]
  nn = Int[]

  push!(ru, 0.0)
  push!(rp, 0.0)

   modelfine = generate_rectangle(9)
    setup_model_labels_rectangle_channel!(modelfine)
    uref,pref,_,_,_ = solve_ForchheimerNoExact(modelfine; k=k, generate_output=true)

  for nk in 1:nkmax
    println("******** Refinement step: $nk")
    model=generate_rectangle(nk-1)
    setup_model_labels_rectangle_channel!(model)
    uh, ph, loss_divh, ndofs, op, dΩ, dΓp, dΛ, h_e, h_e_Γ, Uh, Qh = 
      solve_ForchheimerNoExact(model; k=k, generate_output=false)

    error_u,error_p = compute_errors_robust_darcy_forchheimer(op, dΩ, dΓp, dΛ, h_e, h_e_Γ, κinv, FF, uref,pref,Uh,Qh,uh,ph)

    push!(nn, ndofs)
    push!(hh, sqrt(2)/(2^nk))
    println("******** Total DoFs: ", nn[nk])

    push!(eu, error_u)
    push!(ep, error_p)
    push!(losses_divh, loss_divh)

    if nk > 1
      push!(ru, log(eu[nk]/eu[nk-1]) / log(hh[nk]/hh[nk-1]))
      push!(rp, log(ep[nk]/ep[nk-1]) / log(hh[nk]/hh[nk-1]))
    end
  end

  println("=======================================================")
  println("   DoF  &    h   &   e(u)   &  r(u) &   e(p)   &  r(p) & loss_divh ")
  println("=======================================================")
  for nk in 1:nkmax
    @printf("%7d & %.4f & %.2e & %.3f & %.2e & %.3f & %.2e\n",
            nn[nk], hh[nk], eu[nk], ru[nk], ep[nk], rp[nk], losses_divh[nk])
  end
  println("=======================================================")
end

convergence_test(; nkmax=8, k=0, generate_output=false)

end 
 