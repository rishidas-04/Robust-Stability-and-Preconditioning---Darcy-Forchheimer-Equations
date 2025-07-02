module rishiConvergenceTests
  using PertubedSaddlePointPreconditioners
  using Gridap
  import Gridap: ∇
  using Printf
  using Test

  const write_solution_to_vtk = false

  # Arbitrary model parameters
  const κ = 1.0E-12 # permeability

  u_ex(x)  = VectorValue((κ*π*cos(π*x[1]))*cos(π*x[2]),
                         -κ*π*sin(π*x[1])*(sin(π*x[2])))
  p_ex(x)  = sin(π*x[1])*cos(π*x[2]) 
  p_min_ex(x)  = sin(π*x[1])*cos(π*x[2])
  
  function solve_darcy(op; kwargs...)
    solve(op)
  end

  function solve_minimization_problem(opmin; kwargs...)
    solve(opmin)
  end

  for k in 0:0
    eu = Float64[]
    ep = Float64[]
    ru = Float64[]
    rp = Float64[]
    epmin = Float64[]
    rpmin = Float64[]
    hh = Float64[]
    nn = Int[]

    push!(ru,0.)
    push!(rp,0.)
    push!(rpmin,0.)

    nkmax=6
    for nk in 1:nkmax
        println("******** Refinement step: $nk")
        model=generate_model2d(nk)
        op,dΩ,_=assemble_darcy(model, k, κ, u_ex, p_ex)
        xh=solve_darcy(op)
        uh, ph, _ = xh
        ndofs=num_free_dofs(Gridap.FESpaces.get_fe_space(xh))

        
# Assembling the sub-minimization problem
function assemble_minimization_problem(model, k, κ, p_min_ex)
  
  reffe_p_min = ReferenceFE(lagrangian,Float64,0)

  # FESpaces
  Qminh_ = TestFESpace(model,reffe_p_min,conformity=:L2)
  Lminh_ = ConstantFESpace(model)

  
  Qminh = TrialFESpace(Qminh_,p_min_ex)
  Lminh = TrialFESpace(Lminh_)

  Yminh = MultiFieldFESpace([Qminh_,Lminh_])
  Xminh = MultiFieldFESpace([Qminh,Lminh])

  Ω1 = Triangulation(model)
  # dΩ = Measure(Ω,2*(k+2))
  dΩ1 = Measure(Ω1,4)
  Γ = BoundaryTriangulation(model)
  n_Γ = get_normal_vector(Γ)
  # dΓ = Measure(Γ,2*(k+2)-1)
  dΓ = Measure(Γ,3)


  amin((p, ϕ),(q, ψ)) = ∫( p*q ) * dΩ1 + ∫(κ*∇(p)⋅∇(q))*dΩ1 + ∫(p*ψ) * dΩ1 + ∫(q*ϕ) * dΩ1 
  bmin((q, ψ)) = ∫( (p_ex - ph)*q ) * dΩ1 + ∫((p_ex - ph)*ψ) * dΩ1 

  # Build affine FE operator
  opmin = AffineFEOperator(amin,bmin,Xminh,Yminh)
  opmin, dΩ1, dΓ
end

        opmin,dΩ1,_=assemble_minimization_problem(model, k, κ, p_min_ex)
        yh=solve_minimization_problem(opmin)
        pminh, _ = yh

        if (write_solution_to_vtk)
          Ω = Triangulation(model)
          writevtk(Ω, "convergence_results_$nk",
                  cellfields=["uh"=>uh, "ph"=>ph])
        end


        error_u,error_p,error_pmin =
        computation_error_darcy_minimization(xh, op, yh, opmin, dΩ, dΩ1, κ, u_ex, p_ex, p_min_ex)

        push!(nn,ndofs)
        push!(hh,sqrt(2)/(2^nk))
        println("******** Total DoFs: ", nn[nk])
        
        push!(eu,error_u)
        push!(ep,error_p)
        push!(epmin,error_pmin)
        if nk>1
            push!(ru, log(eu[nk]/eu[nk-1])/log(hh[nk]/hh[nk-1]))
            push!(rp, log(ep[nk]/ep[nk-1])/log(hh[nk]/hh[nk-1]))
            push!(rpmin, log(epmin[nk]/epmin[nk-1])/log(hh[nk]/hh[nk-1]))
        end
    end
    println("=======================================================")
    println("   DoF  &    h   &   e(u)   &  r(u) &   e(p)   &  r(p) ")
    println("=======================================================")
    for nk in 1:nkmax
        @printf("%7d & %.4f & %.2e & %.3f & %.2e & %.3f \n", 
                nn[nk], hh[nk], eu[nk], ru[nk], ep[nk], rp[nk]);
    end
    println("=======================================================")
  end
end