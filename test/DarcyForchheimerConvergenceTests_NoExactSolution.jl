module DarcyForchheimerConvergenceTests_NoExactSolution
  using PertubedSaddlePointPreconditioners
  using Gridap
  import Gridap: ∇
  using Printf
  using Test
  using LineSearches: BackTracking
  using LinearAlgebra
  using Gridap.CellData
  using Krylov
  using LinearOperators

  const κ0 = 1.0E-8 # permeability
  const r  = 3.5 # Forchheimer exponent
  const s  = 1.5 # exponent for pressure preconditioner
  const FF = 1.0e4 # Forchheimer coefficient

  function normu(v)
    (v⋅v>0)*(sqrt(v⋅v))^(r-2)
  end

  function deri(v) #derivative of |v| wrt v
    (v⋅v>0)*(r-2)*(sqrt(v⋅v))^(r-4)*v
  end

  κ(x) = κ0*(1+exp(-0.5*(10*x[2]-5.0-sin(10.0*x[1]))^2))
  κinv(x)=  1.0/κ(x)
  weight_κFF(x) = 1.0/(κinv(x)+FF)

  f_ex = VectorValue(0,0) 
  g_ex = 0.0
  zerov = VectorValue(0.0,0.0)
  uin(x) = VectorValue(2.5*x[2]*(1-x[2]),0.0)
  pout  = 0.0

  function solve_ForchheimerNoExact(model; k=k, generate_output=false)
    reffe_u = ReferenceFE(raviart_thomas,Float64,k)
    reffe_p = ReferenceFE(lagrangian,Float64,k)

    Uh_ = TestFESpace(model,reffe_u,dirichlet_tags=["Gamma_in","Gamma_topbot"],conformity=:HDiv)
    Qh_ = TestFESpace(model,reffe_p,conformity=:L2)

    Uh = TrialFESpace(Uh_,[uin,zerov])
    Qh = TrialFESpace(Qh_)

    Yh = MultiFieldFESpace([Uh_,Qh_])
    Xh = MultiFieldFESpace([Uh,Qh])

    Ω = Triangulation(model)
    dΩ = Measure(Ω,2*(k+2))
    Λ = SkeletonTriangulation(model)
    dΛ = Measure(Λ,2*(k+2)-1)
    h_e = CellField(get_array(∫(1)dΛ), Λ)
    n_Λ = get_normal_vector(Λ)

    Γout = BoundaryTriangulation(model, tags = "Gamma_out")
    n_Γout = get_normal_vector(Γout)
    dΓout = Measure(Γout,2*(k+2)-1)
    h_e_Γout = CellField(get_array(∫(1)dΓout), Γout)

    a(u,v)    = ∫( κinv*(u⋅v))*dΩ
    c(u,w,v)  = ∫(FF*v⋅((normu∘(u))*w))dΩ
    d1(u,w,v) = ∫(FF*v⋅((normu∘(u))*w))dΩ
    d2(u,w,v) = ∫(FF*(u⋅v)*((deri∘(u))⋅w))dΩ 
    b(v,q)    = ∫(-(∇⋅v)*q)*dΩ
    F(v)      = ∫(f_ex⋅v)*dΩ -∫(pout*(v⋅n_Γout))dΓout
    G(q)      = ∫((-1.0)*(g_ex*q))dΩ

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

    uh, ph, loss_divh, dΩ, dΛ, dΓout, h_e, h_e_Γout, Uh, Qh, Qh_, Gridap.FESpaces.num_free_dofs(Xh)
  end

  function compute_errorNoExact_simpleNorms(euh,eph,dΩ)
    eu = (sum(∫((euh⋅euh).^(0.5*3.0))dΩ))^(1.0/3.0) + sqrt(sum(∫((∇⋅euh)*(∇⋅euh))*dΩ))
    ep = sqrt(sum(∫(eph*eph)*dΩ))
    eu, ep
  end

  function compute_errorNoExact_weightedNorms(euh,eph,dΩ)
    eu = sqrt(sum(∫(κinv*euh⋅euh)dΩ)) + (sum(∫(FF*(euh⋅euh).^(0.5*3.0))dΩ))^(1.0/3.0) + sqrt(sum(∫((∇⋅euh)*(∇⋅euh))*dΩ))
    ep = sqrt(sum(∫(weight_κFF*eph*eph)*dΩ))
    eu, ep
  end

  function compute_errorNoExact_preconditionerNorms(euh,eph,pref,Qh,Qh_,dΩ,dΛ,dΓout,h_e,h_e_Γout;k=0)

    app1(p,q) = ∫(p*q)dΩ

    app2(p,q) = ∫(κ*(∇(p))⋅(∇(q)))dΩ +
                ∫(κ/h_e*jump(p)*jump(q))dΛ +
                ∫(κ/h_e_Γout*p*q)dΓout 
         
    if k == 0
      function  app3_low(p,q) 
        ∫(FF^(-1)*(∇(p))⋅(∇(q)))dΩ + 
                  ∫(FF^(-1)/h_e*jump(p)*jump(q))dΛ +
                  ∫(FF^(-1)/h_e_Γout*p*q)dΓout 
      end
      app3 = app3_low  
    else
      small_ = 1.0e-10
      function app3_high(p,q) 
        ∫(FF^(-1)*((∇(pref)⋅∇(pref)).^((s-2)/2) ).*(∇(p)⋅∇(q)))dΩ +
        ∫(FF^(-1)*(s-2)*( (∇(pref)⋅∇(pref)).^((s-4)/2)).*((∇(pref)⋅∇(p)).*(∇(pref)⋅∇(q))))dΩ +
        ∫(FF^(-1)*(s-1)*((jump(pref)*jump(pref)+small_).^((s-2)/2)).*h_e.^(1-s)*jump(p)*jump(q))dΛ +
        ∫(FF^(-1)*(s-1)*((pref*pref).^((s-2)/2)).*h_e_Γout.^(1-s)*p*q)dΓout
      end
      app3 = app3_high
    end

    A22_m1, A22_m2, A22_m3 = assemble_matrix(app1,Qh,Qh_), assemble_matrix(app2,Qh,Qh_), assemble_matrix(app3,Qh,Qh_)

    A22lo = LinearOperator(GridapLinearSolverPreconditioner(A22_m1)) +
            LinearOperator(GridapLinearSolverPreconditioner(A22_m2)) +
            LinearOperator(GridapLinearSolverPreconditioner(A22_m3))       
    
    A22inv1 = inv(Array(A22_m1))
    A22inv2 = inv(Array(A22_m2))
    A22inv3 = inv(Array(A22_m3))
    A22lo_inv = A22inv1 + A22inv2 + A22inv3

    eph_int = interpolate_everywhere(eph, Qh)
    eph_int_vec = get_free_dof_values(eph_int)

    A = A22lo_inv  
    b = eph_int_vec
    r = similar(b)

    rtol=1.0e-14
    function custom_stopping_condition(solver, A, b, r, tol)
      mul!(r, A, solver.x)
      r .-= b                        
      bool = norm(r) ≤ tol*norm(b)   
      @printf("ERR NORM: ||b-Ax||/||b||: %16.7e\n",norm(r)/norm(b))
      return bool
    end
    
    minres_callback(solver) = custom_stopping_condition(solver, A, b, r, rtol)
    (newfreedofs, stats) = minres(A, b, M = A22lo; itmax=2000, verbose=0,
                                 callback=minres_callback,
                                 atol=0.0,
                                 rtol=rtol)

    ep = sqrt(dot(newfreedofs,b))
    eu = sqrt(sum(∫(κinv*euh⋅euh)dΩ)) + 
          (sum(∫(FF*(euh⋅euh).^(0.5*3.0))dΩ))^(1.0/3.0) + 
          sqrt(sum(∫((∇⋅euh)*(∇⋅euh))*dΩ))

    eu, ep
  end

  function  convergence_test(; nkmax, k=k, generate_output=false)
    eu = Float64[]
    ep = Float64[]
    ru = Float64[]
    rp = Float64[]
    euw = Float64[]
    epw = Float64[]
    ruw = Float64[]
    rpw = Float64[]
    eupre = Float64[]
    eppre = Float64[]
    rupre = Float64[]
    rppre = Float64[]
    hh = Float64[]
    losses_divh = Float64[]
    nn = Int[]
    push!(ru,0.)
    push!(rp,0.)
    push!(ruw,0.)
    push!(rpw,0.)
    push!(rupre,0.)
    push!(rppre,0.)

    modelfine = generate_rectangle(7)
    setup_model_labels_rectangle_channel!(modelfine)
    uref,pref,_,_,_,_,_,_ = solve_ForchheimerNoExact(modelfine; k=k, generate_output=false)

    for nk in 1:nkmax
        println("******** Refinement step: $nk")
        model=generate_rectangle(nk-1)
        setup_model_labels_rectangle_channel!(model)
        uh,ph,loss_divh, dΩ, dΛ, dΓout,  h_e, h_e_Γout, Uh, Qh, Qh_, ndofs=solve_ForchheimerNoExact(model; k=k, generate_output=generate_output)
        ur_ = Interpolable(uref)
        pr_ = Interpolable(pref)  
        ur = interpolate_everywhere(ur_, Uh)
        pr = interpolate_everywhere(pr_, Qh)
        
        error_u,error_p = compute_errorNoExact_simpleNorms(ur - uh,pr-ph,dΩ)
        error_uw,error_pw = compute_errorNoExact_weightedNorms(ur-uh,pr-ph,dΩ)
        error_upre,error_ppre = compute_errorNoExact_preconditionerNorms(ur-uh,pr-ph,pr,Qh,Qh_,dΩ,dΛ,dΓout,h_e,h_e_Γout;k=k)

        push!(nn,ndofs)
        push!(hh,sqrt(2)/(2^nk))
        println("******** Total DoFs: ", nn[nk])
        
        push!(eu,error_u)
        push!(ep,error_p)
        push!(euw,error_uw)
        push!(epw,error_pw)
        push!(eupre,error_upre)
        push!(eppre,error_ppre)
        push!(losses_divh,loss_divh)

        if nk>1
            push!(ru, log(eu[nk]/eu[nk-1])/log(hh[nk]/hh[nk-1]))
            push!(rp, log(ep[nk]/ep[nk-1])/log(hh[nk]/hh[nk-1]))
            push!(ruw, log(euw[nk]/euw[nk-1])/log(hh[nk]/hh[nk-1]))
            push!(rpw, log(epw[nk]/epw[nk-1])/log(hh[nk]/hh[nk-1]))
            push!(rupre, log(eupre[nk]/eupre[nk-1])/log(hh[nk]/hh[nk-1]))
            push!(rppre, log(eppre[nk]/eppre[nk-1])/log(hh[nk]/hh[nk-1]))
        end
    end
    println("=======================================================")
    println("   DoF  &    h   &   e(u)   &  r(u) &   e(p)   &  r(p) & loss_divh ")
    println("===================== usual norms ==================================")
    for nk in 1:nkmax
        @printf("%7d & %.4f & %.2e & %.3f & %.2e & %.3f & %.2e\n", 
                nn[nk], hh[nk], eu[nk], ru[nk], ep[nk], rp[nk], losses_divh[nk]);
    end
    println("================== p with weighting =====================================")
    for nk in 1:nkmax
        @printf("%7d & %.4f & %.2e & %.3f & %.2e & %.3f & %.2e\n", 
                nn[nk], hh[nk], euw[nk], ruw[nk], epw[nk], rpw[nk], losses_divh[nk]);
    end
    println("=================== p with preconditioning ====================================")
    for nk in 1:nkmax
        @printf("%7d & %.4f & %.2e & %.3f & %.2e & %.3f & %.2e\n", 
                nn[nk], hh[nk], eupre[nk], rupre[nk], eppre[nk], rppre[nk], losses_divh[nk]);
    end
    println("=======================================================")

  end
  convergence_test(;nkmax=5,k=0,generate_output=false)
end