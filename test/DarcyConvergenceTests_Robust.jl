module RobustDarcyConvergenceTests
  using GridapPertubedSaddlePointPreconditioners
  using Gridap
  import Gridap: ∇
  using Printf
  using Test
  using Krylov

  const write_solution_to_vtk = false

  # Arbitrary model parameters
  const κ = 0.000001
  const ν = 1
  const c0 = 1

  u_ex(x)  = VectorValue((sin(π*x[1]))^2*sin(π*x[2]),
                          sin(π*x[1])*(sin(π*x[2]))^2)
  p_ex(x)  = sin(π*x[1])*cos(π*x[2])

  function solve_darcy(op; kwargs...)
    solve(op)
  end

  for k in 0:0
    eu = Float64[]
    ep = Float64[]
    ru = Float64[]
    rp = Float64[]
    hh = Float64[]
    nn = Int[]

    push!(ru,0.)
    push!(rp,0.)

    nkmax=6
    for nk in 1:nkmax
        println("******** Refinement step: $nk")
        model=generate_model2d(nk)
        op,dΩ,dΓ=assemble_darcy(model, k, κ, u_ex, p_ex)
        xh=solve_darcy(op)
        uh, ph = xh
        ndofs=num_free_dofs(Gridap.FESpaces.get_fe_space(xh))

        if (write_solution_to_vtk)
          Ω = Triangulation(model)
          writevtk(Ω, "convergence_results_$nk",
                  cellfields=["uh"=>uh, "ph"=>ph])
        end

        Λ  = SkeletonTriangulation(model)
        dΛ = Measure(Λ,3)
        Γ = BoundaryTriangulation(model)
        dΓ = Measure(Γ,3)
        h_e = CellField(get_array(∫(1)dΛ),Λ)
        h_e_Γ = CellField(get_array(∫(1)dΓ),Γ)


        #error_u,error_p =
        #  compute_errors_darcy(xh, dΩ, κ, u_ex, p_ex)

        error_u,error_p =
          compute_errors_robust_darcy(xh, op, dΩ, dΓ, dΛ, h_e, h_e_Γ, κ, u_ex, p_ex)  

        push!(nn,ndofs)
        push!(hh,sqrt(2)/(2^nk))
        println("******** Total DoFs: ", nn[nk])
        
        push!(eu,error_u)
        push!(ep,error_p)
        if nk>1
            push!(ru, log(eu[nk]/eu[nk-1])/log(hh[nk]/hh[nk-1]))
            push!(rp, log(ep[nk]/ep[nk-1])/log(hh[nk]/hh[nk-1]))
        end
    end
    println("=======================================================")
    println("   DoF  &    h   &   e(u)   &  r(u) &   e(p)   &  r(p) ")
    println("=======================================================")
    for nk in 1:nkmax
        @printf("%7d & %.4f & %.2e & %.3f & %.2e & %.3f\n", 
                nn[nk], hh[nk], eu[nk], ru[nk], ep[nk], rp[nk]);
    end
    println("=======================================================")
  end
end