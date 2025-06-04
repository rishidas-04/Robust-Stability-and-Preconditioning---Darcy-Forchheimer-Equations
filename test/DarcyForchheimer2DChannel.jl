#module DarcyForchheimer2DChannel
  #using GridapPertubedSaddlePointPreconditioners
  using Gridap
  using GridapGmsh
  import Gridap: ∇
  using Printf
  using Test
  using LineSearches: BackTracking
  using LinearAlgebra

  # Model parameters from https://eprints.qut.edu.au/211518/1/Patrick_Hassard_Thesis.pdf

  const κ1 = 5.0E-12    # permeability in x   m^2
  const κ2 = 1.0E-12    # permeability in y   m^2
  const θ  = 0.082      # rotation angle      rad   (=4.7 deg)
  R  = TensorValue(cos(θ),-sin(θ),sin(θ),cos(θ))
  Kr = TensorValue(κ1,0,0,κ2)

  κ = (R⋅Kr)⋅transpose(R)
  κinv = inv(κ)

  const ν = 1.0E-6     # viscosity       m^2s^{-1}
  const r = 3.0        # index           - 
  const FF = 1.0E-3    # Forchheimer     -
  const p_out = 0.0

  source = 0.0
  forcing = VectorValue(1.3E-5,0)   #ms^{-2}
  v0 = VectorValue(0,0)
  uin(x) = VectorValue(1.0E-5*x[1]/(sqrt(x[1]^2+x[2]^2)),1.0E-5*x[2]/(sqrt(x[1]^2+x[2]^2))) # m/s

  function normu(v)
    (v⋅v>0)*(sqrt(v⋅v))^(r-2)
  end

  function deri(v) #derivative of |v| wrt v
    (v⋅v>0)*(r-2)*(sqrt(v⋅v))^(r-4)*v
  end


  model = GmshDiscreteModel("meshes/SevenCylindersInASquare.msh")
  labels = get_face_labeling(model)

  add_tag_from_tags!(labels,"In",["inlet"])
  add_tag_from_tags!(labels,"Out",["outlet"])
  add_tag_from_tags!(labels,"Wall",["walls"])
  add_tag_from_tags!(labels,"Cyl",["cylinders"])

  k = 1
  reffe_u = ReferenceFE(raviart_thomas,Float64,k)
  reffe_p = ReferenceFE(lagrangian,Float64,k)

  # FESpaces
  Uh_ = TestFESpace(model,reffe_u,dirichlet_tags=["In","Wall","Cyl"],conformity=:HDiv)
  Qh_ = TestFESpace(model,reffe_p,conformity=:L2)

  Uh = TrialFESpace(Uh_,[uin,v0,v0])
  Qh = TrialFESpace(Qh_)

  Yh = MultiFieldFESpace([Uh_,Qh_])
  Xh = MultiFieldFESpace([Uh,Qh])

  println("Number of dofs in Xh: ",num_free_dofs(Xh))
  Ω = Triangulation(model)
  dΩ = Measure(Ω,2*(k+2))
  Γp = BoundaryTriangulation(model, tags = "Out")
  n_Γp = get_normal_vector(Γp)
  dΓp = Measure(Γp,2*(k+2)-1)

  a(u,v)   = ∫( ν*(κinv⋅u)⋅v)*dΩ
  c(u,w,v) = ∫(FF*v⋅((normu∘(u))*w))dΩ
  d1(u,w,v) = ∫(FF*v⋅((normu∘(u))*w))dΩ
  d2(u,w,v) = ∫(FF*(u⋅v)*((deri∘(u))⋅w))dΩ 
  b(v,q)   = ∫(-(∇⋅v)*q)*dΩ
  F(v)     = ∫(forcing⋅v)*dΩ -∫(p_out*(v⋅n_Γp))dΓp
  G(q)     = ∫((-1.0)*(source*q))dΩ

  resid((u,p),(v,q)) = a(u,v) + c(u,u,v) + b(v,p) + b(u,q) - F(v) - G(q) 

  jacob((u,p),(du,dp),(v,q)) = a(du,v) + d1(u,du,v) + d2(u,du,v) + b(v,dp) + b(du,q)  

  oper = FEOperator(resid,jacob,Xh,Yh)
  nls = NLSolver(show_trace=true, method=:newton, linesearch=BackTracking())
  solver = FESolver(nls)
  x0=0.01*ones(num_free_dofs(Xh)) 
  xh=FEFunction(Xh,x0)
  (uh,ph), _ = solve!(xh,solver,oper)

  writevtk(Ω,"ChannelFlow",order=1,cellfields=["uh"=>uh,"ph"=>ph])

#end
