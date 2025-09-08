
module DarcyGridapSolvers

using PertubedSaddlePointPreconditioners
using Test
using LinearAlgebra
using FillArrays, BlockArrays

using Gridap
using Gridap.ReferenceFEs, Gridap.Algebra, Gridap.Geometry, Gridap.FESpaces
using Gridap.CellData, Gridap.MultiField, Gridap.Algebra
using GridapSolvers
using GridapSolvers.LinearSolvers, GridapSolvers.MultilevelTools
using GridapSolvers.BlockSolvers: LinearSystemBlock, BiformBlock, BlockTriangularSolver

function add_labels_2d!(labels)
  add_tag_from_tags!(labels,"top",[6])
  add_tag_from_tags!(labels,"walls",[1,2,3,4,5,7,8])
end

domain = (0,1,0,1)
model=generate_model2d(5)
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
# α = 0
κ = 1e-2
f =  VectorValue(1.0,1.0)  
Π_Qh = LocalProjectionMap(divergence,Q,qdegree)
graddiv(u,v)  = ∫(α*(∇⋅v)⋅Π_Qh(u))dΩ
biform_u(u,v) = ∫((1.0/κ)*(u⋅v))dΩ + graddiv(u,v)
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
solver = FGMRESSolver(20,P;atol=1e-10,rtol=1.e-12)
ns = numerical_setup(symbolic_setup(solver,A),A)

x = allocate_in_domain(A); fill!(x,0.0)
solve!(x,ns,b)

r = allocate_in_range(A)
mul!(r,A,x)
r .-= b
@test norm(r) < 1.e-7

# Computation of condition numbers

function preconditioned_mul!(y, x)
    tmp = similar(x)
    mul!(tmp, A, x)        # tmp = A*x
    solve!(y, ns, tmp)     # y = P⁻¹*tmp
end

# iteration for λmax
function estimate_lambda_max(n; maxiter=2)
    x = randn(n); x ./= norm(x)
    λ = 0.0
    for _ in 1:maxiter
        y = similar(x)
        preconditioned_mul!(y, x)
        λ = dot(x,y) / dot(x,x)
        x .= y ./ norm(y)
    end
    return abs(λ)
end

# Approximate λmin with inverse iteration 
function estimate_lambda_min(n; maxiter=2)
    x = randn(n); x ./= norm(x)
    λ = 0.0
    for _ in 1:maxiter
        y = similar(x)
        preconditioned_mul!(y, x)  # apply operator
        λ = dot(x,y) / dot(x,x)
        x .= y ./ norm(y)
    end
    return abs(λ)
end

n = size(A,1)
λmax = estimate_lambda_max(n)
λmin = estimate_lambda_min(n)
cond_est = λmax/λmin

println("Lightweight condition number estimate of P⁻¹A = ", cond_est)

end 
