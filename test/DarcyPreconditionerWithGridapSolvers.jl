module DarcyGridapSolvers

using GridapPertubedSaddlePointPreconditioners
using Test
using LinearAlgebra
using FillArrays, BlockArrays

using Gridap
using Gridap.ReferenceFEs, Gridap.Algebra, Gridap.Geometry, Gridap.FESpaces
using Gridap.CellData, Gridap.MultiField, Gridap.Algebra
#using PartitionedArrays
#using GridapDistributed

using GridapSolvers
using GridapSolvers.LinearSolvers, GridapSolvers.MultilevelTools
using GridapSolvers.BlockSolvers: LinearSystemBlock, BiformBlock, BlockTriangularSolver

function add_labels_2d!(labels)
  add_tag_from_tags!(labels,"top",[6])
  add_tag_from_tags!(labels,"walls",[1,2,3,4,5,7,8])
end


#function main(distribute,np)
#np = 2
#parts = distribute(LinearIndices((prod(np),)))

  domain = (0,1,0,1)
  model=generate_model2d(5)
  #model = CartesianDiscreteModel(parts,np,domain)
  add_labels! = add_labels_2d!
  add_labels!(get_face_labeling(model))

  order = 1
  qdegree = 2*(order+1)
  reffe_u = ReferenceFE(raviart_thomas,Float64,order)
  reffe_p = ReferenceFE(lagrangian,Float64,order-1)

  u_walls = VectorValue(0.0,0.0)
  u_top = VectorValue(1.0,0.0)

  V = TestFESpace(model,reffe_u,conformity=:HDiv,dirichlet_tags=["walls","top"]);
  U = TrialFESpace(V,[u_walls,u_top]);
  Q = TestFESpace(model,reffe_p;conformity=:L2,constraint=:zeromean)

  mfs = Gridap.MultiField.BlockMultiFieldStyle()
  X = MultiFieldFESpace([U,Q];style=mfs)
  Y = MultiFieldFESpace([V,Q];style=mfs)

  α = 1.e2
  f =  VectorValue(1.0,1.0)  
  Π_Qh = LocalProjectionMap(divergence,Q,qdegree)
  graddiv(u,v)  = ∫(α*(∇⋅v)⋅Π_Qh(u))dΩ
  biform_u(u,v) = ∫(u⋅v)dΩ + graddiv(u,v)
  biform((u,p),(v,q),dΩ) = biform_u(u,v) - ∫(divergence(v)*p)dΩ - ∫(divergence(u)*q)dΩ
  liform((v,q),dΩ) = ∫(v⋅f)dΩ

  Ω = Triangulation(model)
  dΩ = Measure(Ω,qdegree)

  a(u,v) = biform(u,v,dΩ)
  l(v) = liform(v,dΩ)
  op = AffineFEOperator(a,l,X,Y)
  A, b = get_matrix(op), get_vector(op);

  solver_u = LUSolver()
  solver_p = CGSolver(JacobiLinearSolver();maxiter=20,atol=1e-14,rtol=1.e-6)
  solver_p.log.depth = 2

  bblocks = [LinearSystemBlock() LinearSystemBlock();
             LinearSystemBlock() BiformBlock((p,q) -> ∫(-(1.0/α)*p*q)dΩ,Q,Q)]
  coeffs = [1.0 1.0;
            0.0 1.0]
  P = BlockTriangularSolver(bblocks,[solver_u,solver_p],coeffs,:upper)
  solver = FGMRESSolver(20,P;atol=1e-10,rtol=1.e-12)#,verbose=i_am_main(parts))
  ns = numerical_setup(symbolic_setup(solver,A),A)

  x = allocate_in_domain(A); fill!(x,0.0)
  solve!(x,ns,b)

  r = allocate_in_range(A)
  mul!(r,A,x)
  r .-= b
  @test norm(r) < 1.e-7
#end

#a=main(distribute,2)

end # module