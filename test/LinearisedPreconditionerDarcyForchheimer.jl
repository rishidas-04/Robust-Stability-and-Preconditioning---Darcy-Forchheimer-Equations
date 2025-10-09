
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
#   using IterativeSolvers

function normu(v)
    (v⋅v>0)*(sqrt(v⋅v))^(r-2) 
  end

  function deri(v) #derivative of |v| wrt v
    (v⋅v>0)*(r-2)*(sqrt(v⋅v))^(r-4)*v   
  end
 const r = 4
  # u_ex(x)  = VectorValue(cos(π*x[1])*sin(π*x[2]),
  #                         -sin(π*x[1])*cos(π*x[2]))
  # p_ex(x)  = sin(π*x[1])*cos(π*x[2])

  u_ex(x)  = VectorValue(1.0, 1.0)    # manufactured solutions
  p_ex(x)  = x[1]*x[2]

  Minv = TensorValue(0.25,0.25,0.25,0.25) # Proper inverse can't be taken, as, the matrix [u_ex * transpose(u_ex)] has determinant = zero. Pseudo inverse has been considered : (1/4)*[u_ex * transpose(u_ex)].

  function assemble_baerland_preconditioner(op, dΩ, dΓ, dΛ, h_e, h_e_Γ, κ, FF, u_ex, p_ex; prec_variant=:prec_variant)
    X1,X2 = op.trial
    Y1,Y2 = op.test

  Ω = Triangulation(model)
  dΩ = Measure(Ω,2*(k+2))
  Γ = BoundaryTriangulation(model, tags = "Gamma_p")
  n_Γ = get_normal_vector(Γ)
  dΓ = Measure(Γ,2*(k+2)-1)
  Λ = SkeletonTriangulation(model)
  n_Λ = get_normal_vector(Λ)
  dΛ = Measure(Λ,2*(k+2)-1)
  h_e = CellField(get_array(∫(1)dΛ),Λ)
  Γ = BoundaryTriangulation(model)
  dΓ = Measure(Γ,2*(k+2)-1)
  h_e_Γ = CellField(get_array(∫(1)dΓ),Γ)


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
    
      # a11_B2 is the bilinear form corresponding to the velocity blobk in the preconditioner (5.5).
    a11_B2(u,v) =∫((1.0/κ)*(u⋅v)+(∇⋅u)*(∇⋅v))dΩ + (FF)*∫((x -> (normu(u_ex(x)))^(r-2))*(v⋅u))dΩ + (FF)*(r-2)*∫((x -> (normu(u_ex(x)))^(r-4)) * (u_ex ⋅ u)  * (u_ex ⋅ v))dΩ
      # a22a_B2 is the bilinear gorm corresponding to the 1st part of the bilinear form : (I)^{-1}
    a22a_B2(p,q)= ∫(p*q)dΩ
      # a22_B2_1 corresponds to (-κΔ)^{-1}
    a22_B2_1(p,q) = ∫(κ*(∇(p))⋅∇(q))dΩ + ∫((κ/h_e)*jump(p)*jump(q))dΛ + ∫((κ/h_e_Γ)*p*q)dΓ
      # a22_B2_2 corresponds to (-F*norm(u_ex)^{r-2}Δ)^{-1}
    a22_B2_2(p,q) = (1/FF)*∫((x -> normu(u_ex(x))^(2-r))*(∇(p)⋅∇(q)))dΩ  + (1/FF)*∫(((x -> normu(u_ex(x))^(2-r))/h_e)*jump(p)*jump(q))dΛ  + (1/FF)*∫(((x -> normu(u_ex(x))^(2-r))/h_e_Γ)*p*q)dΓ
      # a22_B2_3 corresponds to (-[div(A∇)])^{-1}, where A = F*(r-2)*norm(u_ex)^{r-4}
    a22_B2_3(p,q) = (1/FF)*(1/(r-2))*∫((x -> normu(u_ex(x))^(4-r))*((Minv⋅(∇(p))) ⋅ ∇(q)) )dΩ - (1/FF)*(1/(r-2))*∫((x -> normu(u_ex(x))^(4-r))*((mean((Minv ⋅ ∇(p)))) ⋅ jump(q*n_Λ)) )dΛ + (1/FF)*(1/(r-2))*∫((x -> normu(u_ex(x))^(4-r))*((mean((Minv ⋅ ∇(q)))) ⋅ jump(p*n_Λ)) )dΛ + ∫(jump(p)*jump(q))dΛ #+ (1/FF)*(1/(r-2))*∫(jump(Minv ⋅ ∇(p) ⋅ n_Γ)*jump(Minv ⋅ ∇(q) ⋅n_Γ))dΓ
    
    a22b_B2(p,q) = a22_B2_1(p,q) + a22_B2_2(p,q) + a22_B2_3(p,q)
     
      
      # a11_B2(u,v) =∫((1.0/κ)*(u⋅v)+(∇⋅u)*(∇⋅v))dΩ + ∫(FF*v⋅((normu∘(u_ex))*u))dΩ
      # a22a_B2(p,q)= ∫(p*q)dΩ
      # a22b_B2(p,q)= ∫((x -> κ / (1 + κ*FF*normu(u_ex(x))))*(∇(p))⋅∇(q))dΩ + ∫(((x -> κ / (1 + κ*FF*normu(u_ex(x))))/h_e)*jump(p)*jump(q))dΛ + ∫(((x -> κ / (1 + κ*FF*normu(u_ex(x))))/h_e_Γ)*p*q)dΓ
     A11_,A22a_,A22b_ = assemble_matrix(a11_B2,X1,Y1),assemble_matrix(a22a_B2,X2,Y2),assemble_matrix(a22b_B2,X2,Y2)#,assemble_matrix(a33_B2,X3,Y3)
     A22=LinearOperator(GridapLinearSolverPreconditioner(A22a_))+
             LinearOperator(GridapLinearSolverPreconditioner(A22b_))

      A22ainv=inv(Array(A22a_));
      A22binv=inv(Array(A22b_));  

    Pinv[range2,range2]=A22ainv+A22binv;         
    
    else #For B1
      a11_B1(u,v)=∫((1.0/κ)*(u⋅v) + (x ->  (1 + κ*FF*normu(u_ex(x))) / κ)*(∇⋅u)*(∇⋅v))dΩ + ∫(FF*v⋅((normu∘(u_ex))*u))dΩ
      a22_B1(p,q)= ∫((x -> κ / (1 + κ*FF*normu(u_ex(x))))*p*q)dΩ
    
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

# tol = 1e-8
# λs = abs.(eigvals(Pinv * Adense))
# λs = λs[λs .> tol]
# cond_number = maximum(λs) / minimum(λs)
# @printf("condition number of the new preconditioned system: %1.3e\n", cond_number)
# tol = 1e-12
# λsP = abs.(eigvals(Matrix(Pinv)))   
# λsP = λsP[λsP .> tol]               
# cond_Pinv = maximum(λsP) / minimum(λsP)

# @printf("condition number of Pinv: %1.3e\n", cond_Pinv)
    
    return BlockDiagonalOperator(A11,A22)#,A33 = LinearOperator(I, 1, 1)
  end

  

  function solve_using_baerland_preconditioner(op,P;
                                                       rtol=1.0e-6,
                                                       itmax=500, 
                                                       verbose=false)
    A=op.op.matrix
    @show size(A)
    b=op.op.vector
    r=copy(b)
    conv_history = Float64[]
    function custom_stopping_condition(solver, A, b, r, tol)
      mul!(r, A, solver.x)
      r .-= b                       # r := b - Ax
      bool = norm(r) ≤ tol*norm(b)  # tolerance based on the 2-norm of the residual
      if (verbose)
         @printf("||b-Ax||/||b||: %16.7e\n",norm(r)/norm(b))
      end
      push!(conv_history,norm(r)/norm(b))
      return bool
    end
    minres_callback(solver) = custom_stopping_condition(solver, A, b, r, rtol)
    (xdofs, stats) = minres(A, b, M=P; itmax=itmax, verbose=1,
                         callback=minres_callback,
                         atol=0.0,
                         rtol=0.0, 
                         etol=0.0)
    xh=FEFunction(op.trial,xdofs)

    if (verbose)
      println("#iters: $(stats.niter)")
    end

    xh, stats, conv_history
  end

k=0; κ = 1e-0; nk = 2; FF = 1e0
    #  ϵ = 1.0/κ
#  ϵ(x) = (κ/(1+κ*FF*(normu∘(u_ex(x)))))

  # RHS for Darcy
  f_rhs  = VectorValue(0.1,0)
  g_rhs  = 0.0

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
  
  
  oper  = assemble_darcy_forchheimer_noExact(model, k, κ, FF, f_rhs, g_rhs)
  riesz = assemble_baerland_preconditioner(oper, dΩ, dΓ, dΛ, h_e, h_e_Γ, κ, FF, u_ex, p_ex; prec_variant=:B2)
  xh,state = solve_using_baerland_preconditioner(oper,riesz)

  
end