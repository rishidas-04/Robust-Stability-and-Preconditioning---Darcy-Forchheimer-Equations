module DarcyForchheimerConvergenceTests_mixedBC
  #using GridapPertubedSaddlePointPreconditioners
  using Gridap
  import Gridap: ∇
  using Printf
  using Test
  using LineSearches: BackTracking
  using LinearAlgebra

  # Arbitrary model parameters
  const κ = 1.0E-04
  const r = 3
  const FF = 1.0E04
  
  function generate_model2d(nk)
  domain =(0,1,0,1)
  n      = 2^nk + 1
  # Discrete model
  partition = (n,n)
  CartesianDiscreteModel(domain, partition) |> simplexify
  end

  function setup_model_labels_unit_square!(model)
  labels = get_face_labeling(model)
  add_tag!(labels,"Gamma_p",[6,2,3,4,8]) #6top, right, and other 3 corners
  add_tag!(labels,"Gamma_u",[1,2,3,5,7]) # bottom and 7left and low-left corner
  end  

  function normu(v)
    (v⋅v>0)*(sqrt(v⋅v))^(r-2)
  end

  function deri(v) #derivative of |v| wrt v
    (v⋅v>0)*(r-2)*(sqrt(v⋅v))^(r-4)*v
  end


  u_ex(x)  = VectorValue(cos(π*x[1])*sin(π*x[2]),
                          -sin(π*x[1])*cos(π*x[2]))
  p_ex(x)  = sin(π*x[1])*cos(π*x[2])
  pmin_ex(x)  = sin(π*x[1])*cos(π*x[2])

  f_ex(x) = (1.0/κ)*u_ex(x) + ∇(p_ex)(x) + FF*u_ex(x)*(normu∘u_ex)(x) 
  g_ex(x) = (∇⋅u_ex)(x)


  function solve_Forchheimer(model; k=k, generate_output=false)
    reffe_u = ReferenceFE(raviart_thomas,Float64,k)
    reffe_p = ReferenceFE(lagrangian,Float64,k)

    # FESpaces
    Uh_ = TestFESpace(model,reffe_u,dirichlet_tags="Gamma_u",conformity=:HDiv)
    Qh_ = TestFESpace(model,reffe_p,conformity=:L2)

    Uh = TrialFESpace(Uh_,u_ex)
    Qh = TrialFESpace(Qh_)
   
    Yh = MultiFieldFESpace([Uh_,Qh_])
    Xh = MultiFieldFESpace([Uh,Qh])

    Ω = Triangulation(model)
    dΩ = Measure(Ω,2*(k+2))
    Γp = BoundaryTriangulation(model, tags = "Gamma_p")
    n_Γp = get_normal_vector(Γp)
    dΓp = Measure(Γp,2*(k+2)-1)

    a(u,v)   = ∫( (1/κ)*(u⋅v))*dΩ
    c(u,w,v) = ∫(FF*v⋅((normu∘(u))*w))dΩ
    d1(u,w,v) = ∫(FF*v⋅((normu∘(u))*w))dΩ
    d2(u,w,v) = ∫(FF*(u⋅v)*((deri∘(u))⋅w))dΩ 
    b(v,q)   = ∫(-(∇⋅v)*q)*dΩ
    F(v)     = ∫(f_ex⋅v)*dΩ -∫(p_ex*(v⋅n_Γp))dΓp
    G(q)     = ∫((-1.0)*(g_ex*q))dΩ


    resid((u,p),(v,q)) = a(u,v) + c(u,u,v) + b(v,p) + b(u,q) - F(v) - G(q) 
    jacob((u,p),(du,dp),(v,q)) = a(du,v) + d1(u,du,v) + d2(u,du,v) + b(v,dp) + b(du,q)  
    
    oper = FEOperator(resid,jacob,Xh,Yh)
    nls = NLSolver(show_trace=true, method=:newton, linesearch=BackTracking())
    solver = FESolver(nls)
    x0=0.01*ones(num_free_dofs(Xh)) 
    xh=FEFunction(Xh,x0)
    (uh,ph), _ = solve!(xh,solver,oper)

  uh, ph, dΩ, dΓp
  end

# Assembling the sub-minimization problem
  function assemble_Forchheimer_minimization_problem(model, k, κ, FF, pmin_ex)
    reffe_pmin = ReferenceFE(lagrangian,Float64,k)
    V0 = TestFESpace(model,reffe_pmin,conformity=:L2)
    Ug = TrialFESpace(V0,g_ex)

    Ω = Triangulation(model)
    dΩ = Measure(Ω,2*(k+2))
    Λ  = SkeletonTriangulation(model)
    dΛ = Measure(Λ,2*(k+2)-1)
    # dΩ = Measure(Ω,4)
    Γp = BoundaryTriangulation(model, tags = "Gamma_p")
    n_Γp = get_normal_vector(Γp)
    dΓp = Measure(Γp,2*(k+2)-1)
    # dΓp = Measure(Γp,3)
    h_e = CellField(get_array(∫(1)dΛ),Λ)
    h_e_Γp = CellField(get_array(∫(1)dΓp),Γp)

    uh, ph, dΩ = solve_Forchheimer(model; k=k, generate_output=false)
    aa(p,q)   = ∫(p*q)*dΩ + ∫((κ)*∇(p)⋅∇(q))*dΩ + ∫((FF^(-2.0 / 3.0))*∇(p)⋅∇(q))*dΩ 
    # tried with jump terms:
    # + ∫(κ/1.0*1.0/h_e*jump(p)*jump(q))dΛ + ∫(κ/1.0*1.0/h_e_Γp*p*q)dΓp+ ∫((FF^(-2.0 / 3.0))/1.0*1.0/h_e*jump(p)*jump(q))dΛ + ∫((FF^(-2.0 / 3.0))/1.0*1.0/h_e_Γp*p*q)dΓp
    g1(q) = ∫( (p_ex - ph)*q ) * dΩ 

    opmin = AffineFEOperator(aa,g1,Ug,V0)
    ls = LUSolver()
    solver = LinearFESolver(ls)
    pminh = solve(solver,opmin)
    pminh

  end

  function  convergence_test(; nkmax, k=k, generate_output=false)
    eu = Float64[]
    ep = Float64[]
    ru = Float64[]
    rp = Float64[]
    hh = Float64[]
    nn = Int[]

    push!(ru,0.)
    push!(rp,0.)

    for nk in 1:nkmax
        println("******** Refinement step: $nk")
        model=generate_model2d(nk)
        setup_model_labels_unit_square!(model)
        uh, ph, dΩ, dΓp = solve_Forchheimer(model; k=k, generate_output=false)
        pminh = assemble_Forchheimer_minimization_problem(model, k, κ, FF, pmin_ex)
        euh = u_ex - uh
        eph = p_ex - pminh
        error_u = (sum(∫((FF^(2.0 / 3.0))*(euh⋅euh).^(0.5*3))dΩ))^(2.0/3.0) + sqrt(sum(∫((∇⋅euh)*(∇⋅euh))*dΩ)) + sqrt(sum(∫( (1/κ) *((euh)⋅(euh)))*dΩ))
        error_p = sqrt(sum(∫((p_ex-ph-pminh)*(p_ex-ph-pminh))*dΩ + ∫((κ)*((∇(pminh))⋅(∇(pminh))))*dΩ)) +  (sum(∫((FF^(-2.0 / 3.0))*(∇(pminh)⋅∇(pminh)).^(0.5*3))dΩ))^(2.0/3.0)
        push!(hh,sqrt(2)/(2^nk))
        push!(eu,error_u)
        push!(ep,error_p)
        
        if nk>1
            push!(ru, log(eu[nk]/eu[nk-1])/log(hh[nk]/hh[nk-1]))
            push!(rp, log(ep[nk]/ep[nk-1])/log(hh[nk]/hh[nk-1]))
        end
    end
    println("=======================================================")
    println("       h   &   e(u)   &  r(u) &   e(p)   &  r(p)  ")
    println("=======================================================")
    for nk in 1:nkmax
        @printf(" %.4f & %.2e & %.3f & %.2e & %.3f \n", 
                 hh[nk], eu[nk], ru[nk], ep[nk], rp[nk]);
    end
    println("=======================================================")
  end
  convergence_test(;nkmax=6,k=0,generate_output=false)
end