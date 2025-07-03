module DarcyForchheimerConvergenceTests_NoExactSolution
  using PertubedSaddlePointPreconditioners
  using Gridap
  import Gridap: ∇
  using Printf
  using Test
  using LineSearches: BackTracking
  using LinearAlgebra
  using Gridap.CellData

  # Arbitrary model parameters
  const κ0 = 1.0E-10 # permeability
  const r  = 4.0 # Forchheimer exponent
  const FF = 1.0E5 # Forchheimer coefficient

  function normu(v)
    (v⋅v>0)*(sqrt(v⋅v))^(r-2)
  end

  function deri(v) #derivative of |v| wrt v
    (v⋅v>0)*(r-2)*(sqrt(v⋅v))^(r-4)*v
  end

  κ(x) = κ0*(1+exp(-0.5*(10*x[2]-5.0-sin(10.0*x[1]))^2))
  κinv(x)=  1.0/κ(x)
  #κinv = 1.0/κ0

  f_ex = VectorValue(0.0,0.0)
  g_ex = 0.0
  zerov = VectorValue(0.0,0.0)
  uin(x) = VectorValue(2.5*x[2]*(1-x[2]),0.0)
  pout  = 0.0

  function solve_ForchheimerNoExact(model; k=k, generate_output=false)
    reffe_u = ReferenceFE(raviart_thomas,Float64,k)
    reffe_p = ReferenceFE(lagrangian,Float64,k)

    # FESpaces
    Uh_ = TestFESpace(model,reffe_u,dirichlet_tags=["Gamma_in","Gamma_topbot"],conformity=:HDiv)
    Qh_ = TestFESpace(model,reffe_p,conformity=:L2)

    Uh = TrialFESpace(Uh_,[uin,zerov])
    Qh = TrialFESpace(Qh_)

    Yh = MultiFieldFESpace([Uh_,Qh_])
    Xh = MultiFieldFESpace([Uh,Qh])

    Ω = Triangulation(model)
    dΩ = Measure(Ω,2*(k+2))
    Γout = BoundaryTriangulation(model, tags = "Gamma_out")
    n_Γout = get_normal_vector(Γout)
    dΓout = Measure(Γout,2*(k+2)-1)

    a(u,v)   = ∫( κinv*(u⋅v))*dΩ
    c(u,w,v) = ∫(FF*v⋅((normu∘(u))*w))dΩ
    d1(u,w,v) = ∫(FF*v⋅((normu∘(u))*w))dΩ
    d2(u,w,v) = ∫(FF*(u⋅v)*((deri∘(u))⋅w))dΩ 
    b(v,q)   = ∫(-(∇⋅v)*q)*dΩ
    F(v)     = ∫(f_ex⋅v)*dΩ -∫(pout*(v⋅n_Γout))dΓout
    G(q)     = ∫((-1.0)*(g_ex*q))dΩ

    resid((u,p),(v,q)) = a(u,v) + c(u,u,v) + b(v,p) + b(u,q) - F(v) - G(q) 

    jacob((u,p),(du,dp),(v,q)) = a(du,v) + d1(u,du,v) + d2(u,du,v) + b(v,dp) + b(du,q)  

    oper = FEOperator(resid,jacob,Xh,Yh)
    nls = NLSolver(show_trace=true, method=:newton, linesearch=BackTracking(), iterations=25, ftol=1.0E-6)
    solver = FESolver(nls)
    x0=0.01*ones(num_free_dofs(Xh)) 
    xh=FEFunction(Xh,x0)
    (uh,ph), _ = solve!(xh,solver,oper)

    if generate_output 
      writevtk(Ω,"DFNoExactExtreme_ncells=$(num_cells(model))",order=1,cellfields=["uh"=>uh,"ph"=>ph, "κ"=>κ])
    end

    pa(p,q) = ∫(p*q)*dΩ
    pdivl(q) = ∫(q*(∇⋅uh - g_ex))*dΩ
    projection = AffineFEOperator(pa,pdivl,Qh,Qh)
    loss_div = solve(projection)
    loss_divh=norm(get_free_dof_values(loss_div),Inf)

    uh, ph, loss_divh, dΩ, Uh, Qh, Gridap.FESpaces.num_free_dofs(Xh)
  end


  function compute_errorNoExact(uref,pref,uh,ph,dΩ, Uh,Qh)
    ur_ = Interpolable(uref)
    pr_ = Interpolable(pref)
    ur = interpolate_everywhere(ur_, Uh)
    pr = interpolate_everywhere(pr_, Qh)

    euh = ur - uh
    eph = pr - ph
    eu = (sum(∫((euh⋅euh).^(0.5*3.0))dΩ))^(1.0/3.0) + sqrt(sum(∫((∇⋅euh)*(∇⋅euh))*dΩ))
    ep = sqrt(sum(∫(eph*eph)*dΩ))
    eu, ep
  end


  function  convergence_test(; nkmax, k=k, generate_output=false)
    eu = Float64[]
    ep = Float64[]
    ru = Float64[]
    rp = Float64[]
    hh = Float64[]
    losses_divh = Float64[]
    nn = Int[]
    push!(ru,0.)
    push!(rp,0.)

    modelfine = generate_rectangle(8)
    setup_model_labels_rectangle_channel!(modelfine)
    uref,pref,_,_,_ = solve_ForchheimerNoExact(modelfine; k=k, generate_output=true)

 

    for nk in 1:nkmax
        println("******** Refinement step: $nk")
        model=generate_rectangle(nk-1)
        setup_model_labels_rectangle_channel!(model)
        uh,ph,loss_divh, dΩ, Uh, Ph, ndofs=solve_ForchheimerNoExact(model; k=k, generate_output=generate_output)
        error_u,error_p = compute_errorNoExact(uref,pref,uh,ph,dΩ,Uh,Ph)
        push!(nn,ndofs)
        push!(hh,sqrt(2)/(2^nk))
        println("******** Total DoFs: ", nn[nk])
        
        push!(eu,error_u)
        push!(ep,error_p)
        push!(losses_divh,loss_divh)

        if nk>1
            push!(ru, log(eu[nk]/eu[nk-1])/log(hh[nk]/hh[nk-1]))
            push!(rp, log(ep[nk]/ep[nk-1])/log(hh[nk]/hh[nk-1]))
        end
    end
    println("=======================================================")
    println("   DoF  &    h   &   e(u)   &  r(u) &   e(p)   &  r(p) & loss_divh ")
    println("=======================================================")
    for nk in 1:nkmax
        @printf("%7d & %.4f & %.2e & %.3f & %.2e & %.3f & %.2e\n", 
                nn[nk], hh[nk], eu[nk], ru[nk], ep[nk], rp[nk], losses_divh[nk]);
    end
    println("=======================================================")
  end
  convergence_test(;nkmax=7,k=0,generate_output=false)
end
