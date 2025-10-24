
module BaerlandDarcyForchheimerPreconditionerTests
  using PertubedSaddlePointPreconditioners
  using Gridap
  import Gridap: ∇
  using Gridap.Algebra
  using Printf
  using Test
  using Krylov
  using LinearAlgebra
  using LinearOperators
# using IterativeSolvers

function normu(v)
    (v⋅v>0)*(sqrt(v⋅v))^(r-2) 
  end

function deri(v) 
    (v⋅v>0)*(r-2)*(sqrt(v⋅v))^(r-4)*v   
  end

const r = 4

u_ex(x)  = VectorValue(0.001*cos(π*x[1])*sin(π*x[2]),
                       -0.001*sin(π*x[1])*cos(π*x[2]))

p_ex(x)  = sin(π*x[1])*cos(π*x[2])

f_rhs(x) = (1.0/κ)*u_ex(x) + ∇(p_ex)(x) + FF*u_ex(x)*((normu∘u_ex)(x))^(r-2)
g_rhs(x) = (∇⋅u_ex)(x)

κF(x) = ((1.0/κ) + FF*((normu(u_ex(x)))^(r-2)))
κinv(x) = 1.0 / κF(x)


function assemble_darcy_forchheimer_noExact(model, k, κ, FF, f_rhs, g_rhs, u_ex, p_ex)
  reffe_u = ReferenceFE(raviart_thomas,Float64,k)
  reffe_p = ReferenceFE(lagrangian,Float64,k)

  # FESpaces
  Uh_ = TestFESpace(model,reffe_u,dirichlet_tags = "Gamma_u",conformity=:HDiv)#,dirichlet_tags="boundary")
  Qh_ = TestFESpace(model,reffe_p,conformity=:L2)
  
  Uh = TrialFESpace(Uh_,u_ex)#VectorValue(0,0))
  Qh = TrialFESpace(Qh_)
  
  Yh = MultiFieldFESpace([Uh_,Qh_])
  Xh = MultiFieldFESpace([Uh,Qh])

  Ω = Triangulation(model)
  dΩ = Measure(Ω,2*(k+2))
  Γ = BoundaryTriangulation(model, tags = "Gamma_p")
  dΓ = Measure(Γ,2*(k+2)-1)
  n_Γ = get_normal_vector(Γ)

  a((u, p),(v, q)) = ∫( (1/κ)*(u⋅v) )*dΩ - ∫(q*(∇⋅u))*dΩ - ∫((∇⋅v)*p)*dΩ + ∫(κF*(v⋅u))dΩ #+ (FF)*(4-2)*∫((x -> (normu(u_ex(x)))^(4-4)) * (u_ex ⋅ u)  * (u_ex ⋅ v))dΩ#+ ∫(FF*v⋅((normu∘(u))*u))*dΩ + ∫(FF*v⋅((normu∘(u))*u))*dΩ#
  b((v, q)) = ∫(f_rhs⋅v)*dΩ -∫(p_ex*(v⋅n_Γ))dΓ - ∫(q*g_rhs)*dΩ

  op = AffineFEOperator(a,b,Xh,Yh)
  op
end

 
function assemble_baerland_preconditioner(op, dΩ, dΓ, dΛ, h_e, h_e_Γ, κ, FF; prec_variant=:prec_variant)
    X1,X2 = op.trial
    Y1,Y2 = op.test

    N  = num_free_dofs(op.trial)
    Ns = [num_free_dofs(U) for U in op.trial] 
    Np = zeros(Int,length(Ns)+1)
    Np[1]=1
      for i=2:length(Ns)+1
          Np[i]=Np[i-1]+Ns[i-1]
      end
    Pinv=zeros(N,N); 
    @show size(Pinv)
    Adense = Array(op.op.matrix)
    range1=Np[1]:Np[2]-1
    range2=Np[2]:Np[3]-1
    
    @assert prec_variant in (:B1,:B2)
    if (prec_variant==:B2)
    
        a11_B2(u,v) = ∫(κF*(u⋅v) + (∇⋅u)*(∇⋅v))dΩ
        a22a_B2(p,q)= ∫(p*q)dΩ
        a22b_B2(p,q) = ∫(κinv*(∇(p))⋅∇(q))dΩ + ∫((κinv/h_e)*jump(p)*jump(q))dΛ + ∫((κinv/h_e_Γ)*p*q)dΓ
    
     A11_,A22a_,A22b_ = assemble_matrix(a11_B2,X1,Y1),assemble_matrix(a22a_B2,X2,Y2),assemble_matrix(a22b_B2,X2,Y2)
     A22=LinearOperator(GridapLinearSolverPreconditioner(A22a_))+
             LinearOperator(GridapLinearSolverPreconditioner(A22b_))

      A22ainv=inv(Array(A22a_));
      A22binv=inv(Array(A22b_));  

    Pinv[range2,range2]=A22ainv+A22binv;         
    
    else #For B1
     
      a11_B1(u,v)=∫(κF*(u⋅v) + κF*(∇⋅u)*(∇⋅v))dΩ
      a22_B1(p,q)= ∫(κinv*p*q)dΩ


     A11_,A22_= assemble_matrix(a11_B1,X1,Y1),assemble_matrix(a22_B1,X2,Y2)  
    A22=LinearOperator(GridapLinearSolverPreconditioner(A22_)) 

      A22inv=inv(Array(A22_));  
   Pinv[range2,range2]=A22inv;  
end

    A11=LinearOperator(GridapLinearSolverPreconditioner(A11_))
    A11inv=inv(Array(A11_));  
    
    Pinv[range1,range1]=A11inv;
    
    evals=eigvals(Pinv*Adense)
    @printf("condition number of the preconditioned system: %1.3e\n", maximum(broadcast(abs,evals))/minimum(broadcast(abs,evals)))

    
    return BlockDiagonalOperator(A11,A22)
end

k=0; κ = 1e-8; nk = 2; FF = 1e0

  model=generate_model2d(nk)
  setup_model_labels_unit_square!(model)
  Ω = Triangulation(model)
  dΩ = Measure(Ω,2*(k+2))
  Λ = SkeletonTriangulation(model)
  dΛ = Measure(Λ,2*(k+2)-1)
  h_e = CellField(get_array(∫(1)dΛ),Λ)
  Γ = BoundaryTriangulation(model, tags = "Gamma_p")
  dΓ = Measure(Γ,2*(k+2)-1)
  h_e_Γ = CellField(get_array(∫(1)dΓ),Γ)
  
oper  = assemble_darcy_forchheimer_noExact(model, k, κ, FF, f_rhs, g_rhs,u_ex,p_ex)
riesz = assemble_baerland_preconditioner(oper, dΩ, dΓ, dΛ, h_e, h_e_Γ, κ, FF; prec_variant=:B2)

  
end