using LinearAlgebra 
#using DataArrays
using Plots
using Statistics
using Printf
using OrdinaryDiffEq
using SparseArrays
import GeometryOps as GO
import GeoInterface as GI


function neighbors(j,D0) #Ou neighbor(j,D0), à vérifier si ça pose problème
    edges=findnz(D0[:,j])[1]

    vp = nothing
    vm = nothing

    for e in edges
        nodes = findnz(D0[e,:])[1]

        # voisin = l'autre sommet de l'arête
        voisin = nodes[1] == j ? nodes[2] : nodes[1] #Si ligne e contient j, alors voisin est l'autre élément de la ligne e, sinon c'est l'élément 1

        if D0[e,j] == -1
            vp = voisin   # j → voisin
        elseif D0[e,j] == 1
            vm = voisin   # voisin → j
        end
    end

    return vp, vm
end
function recalc_all_curvatures!(A, D0)
    n_nodes = size(A, 2)
    for i in 1:n_nodes
        j=neighbors(i, D0)[2]
        k=neighbors(i, D0)[1]
        S=1/2*abs((A[1,i]-A[1,j])*(A[2,k]-A[2,j])-(A[2,i]-A[2,j])*(A[1,k]-A[1,j]))
        la=sqrt((A[1,k]-A[1,j])^2+(A[2,k]-A[2,j])^2)
        lb=sqrt((A[1,k]-A[1,i])^2+(A[2,k]-A[2,i])^2)
        lc=sqrt((A[1,i]-A[1,j])^2+(A[2,i]-A[2,j])^2)
        if abs(4*S) < 1e-14 || la*lb*lc < 1e-14
            A[3,i] = 0
        else
            A[3,i] = 4*S / (la*lb*lc)
        end
    end
end


function laplacian(A, D0)
    star1 = spzeros(size(D0,1), size(D0,1))
    inv = spzeros(size(A,2), size(A,2))
    eps = 1e-14  # Small epsilon to avoid division by zero
    for i in 1:size(D0,1)
        nodes = findall(!iszero, D0[i,:])
        dist = norm(A[1:2, nodes[1]] - A[1:2, nodes[2]])
        star1[i,i] = 1 / max(dist, eps)
    end
    for i in 1:size(A,2)
        dist1 = norm(A[1:2, neighbors(i, D0)[1]] - A[1:2, i])
        dist2 = norm(A[1:2, neighbors(i, D0)[2]] - A[1:2, i])
        inv[i,i] = 2 / max(dist1 + dist2, eps)
    end
    return inv * D0' * star1 * D0
end

function MAJ(A,D0)
    recalc_all_curvatures!(A, D0)
    n_nodes = size(A, 2)
    
    #Normale sur le vertexe (A[4:5,i])
    for i in 1:n_nodes
        j=neighbors(i, D0)[1]
        k=neighbors(i, D0)[2]
        L=norm(A[1:2,j] - A[1:2,i])
        R=A[3,i]
        h=sqrt(1/R^2-(L/2)^2)
        

        nx=-(A[2,j]-A[2,i])/L
        ny=(A[1,j]-A[1,i])/L

        local x_m=(A[1,j]+A[1,i])/2
        local y_m=(A[2,j]+A[2,i])/2
        local x_o=x_m+h*nx
        local y_o=y_m+h*ny
        #=
        n=A[1:2,i]-[x_o,y_o]
        n/=norm(n)
        t=(A[1:2,i]-A[1:2,k])/norm(A[1:2,k]-A[1:2,i])+(A[1:2,j]-A[1:2,i])/norm(A[1:2,j]-A[1:2,i])
        t/=norm(t)
        n_verif = [t[2], -t[1]]

        if dot(n, n_verif) < 0
            n = -n
        end
        
        A[4:5,i]=n
        =#
        A[4:5,i]=(A[1:2,i]-[x_o,y_o])/norm(A[1:2,i]-[x_o,y_o])
        #Tangente du vertexe (A[6:7,i])
        A[6,i]=-A[5,i]
        A[7,i]=A[4,i]
    end
    

end

function MAJ2(A,D0)
    eps = 1e-14
    Δ=laplacian(A, D0)
    ΔX=A[1:2,:]*Δ'
    κ = norm.(eachcol(ΔX))
    A[3,:]=κ'
    den = max.(κ, eps)
    A[4:5,:]=ΔX ./den'

    for i in 1:size(A,2)
        if !isfinite(A[4,i]) || !isfinite(A[5,i])
            A[4,i] = 0.0
            A[5,i] = 0.0
        end
    end


    A[6,:]=-A[5,:]
    A[7,:]=A[4,:]
    return nothing
end



function MAJ3(A,D0)
    Δ=laplacian(A, D0)
    ΔX=A[1:2,:]*Δ'
    κ = norm.(eachcol(ΔX))
    A[3,:]=κ'
    #Normale sur le vertexe (A[4:5,i])
    #Calcul de la normale
    n_nodes = size(A, 2)
    for i in 1:n_nodes
        Xi=A[1:2,i]
        Xp=A[1:2,neighbors(i, D0)[1]]
        Xm=A[1:2,neighbors(i, D0)[2]]

        d_i  = norm(Xp - Xi)
        d_im = norm(Xi - Xm)

        t = (Xp - Xi)/d_i + (Xi - Xm)/d_im
        t /= norm(t)

        A[6:7,i]=t

        # Outward normal for CCW curve: rotate tangent 90° counterclockwise
        n_ref = [t[2], -t[1]]
        
        # Get Laplacian-based normal and ensure it points outward
        n_lap = ΔX[1:2, i]
        n_lap_norm = norm(n_lap)
        
        if n_lap_norm > 1e-14
            n_lap = n_lap / n_lap_norm
            # Flip if pointing opposite to reference direction
            if dot(n_lap, n_ref) < 0
                n_lap = -n_lap
            end
            A[4:5,i] = n_lap
        else
            # Fallback to geometric normal if Laplacian is zero
            A[4:5,i] = n_ref
        end
        #Tangente du vertexe (A[6:7,i])
        A[6,i]=-A[5,i]
        A[7,i]=A[4,i]
    end
end

function MAJ4(A,D0)
    recalc_all_curvatures!(A, D0)

    for i in 1:size(A, 2)
        j=neighbors(i, D0)[2]
        k=neighbors(i, D0)[1]
        dm=norm(A[1:2,j] - A[1:2,i])
        dp=norm(A[1:2,k] - A[1:2,i])

        tm=(A[1:2,i]-A[1:2,j])/dm
        tp=(A[1:2,k]-A[1:2,i])/dp

        t=(tm+tp)/2
        t/=norm(t)
        A[6:7,i]=t
        n=[t[2], -t[1]]
        A[4:5,i]=n
    end

end
function vitesse(x,y)
    u=2*pi*(0.5-y)
    v=2*pi*(x-0.5)
    return u,v
end
#
function velocity(dz,z,p,t)
    N=length(z)÷2
    eps = 1e-14
    
    for i in 1:N
        
        x = z[2*i-1]
        y = z[2*i]

        if !isfinite(x) || !isfinite(y)
            dz[2*i-1] = 0.0
            dz[2*i] = 0.0
            continue
        end

        nx = A[4,i]
        ny = A[5,i]
        nrm = sqrt(nx^2 + ny^2)
        if !isfinite(nx) || !isfinite(ny) || !isfinite(nrm) || nrm < eps
            dz[2*i-1] = 0.0
            dz[2*i] = 0.0
            continue
        end

        u, v = vitesse(x, y)
        dot = u*nx + v*ny

        # vitesse projetée
        dz[2*i-1] = dot * nx
        dz[2*i]   = dot * ny
    end
    return nothing
end
#
#=
function velocity(dz,z,p,t)
    N=length(z)÷2
    for i in 1:N
        dz[2*i-1],dz[2*i]=vitesse(z[2*i-1],z[2*i])
    end
    return nothing
end
=#

function f!(dX, X, p, t)
    A, D0 = p
    Nloc = size(A, 2)

    # Build the current geometric state from X (the actual ODE state)
    Acur = copy(A)
    Acur[1, :] .= view(X, 1:Nloc)
    Acur[2, :] .= view(X, Nloc+1:2Nloc)
    MAJ(Acur, D0)
    Ploc=0
    for i in 1:Nloc
        Ploc+=norm(Acur[1:2, neighbors(i, D0)[1]] - Acur[1:2, i])
    end

    Δ = laplacian(Acur, D0)
    b = zeros(Nloc)
    v = zeros(Nloc, 2)
    star0 = zeros(Nloc)
    for i in 1:Nloc
        dist1 = norm(Acur[1:2, neighbors(i, D0)[1]] - Acur[1:2, i])
        dist2 = norm(Acur[1:2, neighbors(i, D0)[2]] - Acur[1:2, i])
        star0[i] = 0.5 * (dist1 + dist2)
    end
    for i in 1:Nloc
        b[i] = (Ploc /Nloc / star0[i] - 1) * ω
    end

    Δ[1,:].=0
    Δ[1,1]=1
    b[1]=0

    Ψ=Δ\b 

    for i in 1:Nloc
        lp = norm(Acur[1:2, neighbors(i, D0)[1]] - Acur[1:2, i])
        lm = norm(Acur[1:2, i] - Acur[1:2, neighbors(i, D0)[2]])
        Tp=(Acur[1:2, neighbors(i, D0)[1]]-Acur[1:2, i])/lp
        Tm=(Acur[1:2, i]-Acur[1:2, neighbors(i, D0)[2]])/lm
        #v[i, :] = -0.5 * ((Ψ[neighbors(i, D0)[1]] - Ψ[i]).* Tp / lp + (Ψ[i]- Ψ[neighbors(i, D0)[2]] ).* Tm / lm) #.* Acur[6:7, i]
        v[i, :] = -0.5 * ((Ψ[neighbors(i, D0)[1]] - Ψ[i])/ lp + (Ψ[i]- Ψ[neighbors(i, D0)[2]] )/ lm) .* Acur[6:7, i]
    end
    
    dx = view(dX, 1:Nloc)
    dy = view(dX, Nloc+1:2Nloc)

    dx .= v[:, 1]
    dy .= v[:, 2]

    # Update A with new positions for next recomputation
    #A[1, :] .= x
    #A[2, :] .= y

      # Update the error reference for convergence checking
    return nothing
end

#=
function f2!(dX, X, p, t)
    A, D0 = p
    Nloc = size(A, 2)

    # Build the current geometric state from X (the actual ODE state)
    Acur = copy(A)
    Acur[1, :] .= view(X, 1:Nloc)
    Acur[2, :] .= view(X, Nloc+1:2Nloc)
    MAJ3(Acur, D0)
    Ploc=0
    for i in 1:Nloc
        Ploc+=norm(Acur[1:2, neighbors(i, D0)[1]] - Acur[1:2, i])
    end
    
    α=zeros(Nloc)
    for i in 1:Nloc
        lm=norm(Acur[1:2,neighbors(i,D0)[2]]-Acur[1:2,i])
        lp=norm(Acur[1:2,neighbors(i,D0)[1]]-Acur[1:2,i])
        #fi=(2*dl/(lm+lp)-1)*ω
        fi=(2*Ploc/(Nloc*(lm+lp))-1)*ω
        α[i]=α[neighbors(i,D0)[2]]+fi*lm
        
    end
    v = α .* Acur[6:7, :]'

    dx = view(dX, 1:Nloc)
    dy = view(dX, Nloc+1:2Nloc)

    dx .= v[:, 1]
    dy .= v[:, 2]

    return nothing
    
end
=#

function f2!(dX, X, p, t) #Methode alpha et répartition selon longueur
    A, D0 = p
    Nloc = size(A, 2)

    # Build the current geometric state from X (the actual ODE state)
    Acur = copy(A)
    Acur[1, :] .= view(X, 1:Nloc)
    Acur[2, :] .= view(X, Nloc+1:2Nloc)
    MAJ3(Acur, D0)

    f=zeros(Nloc)
    α=zeros(Nloc)
    ell=zeros(Nloc)
    ellstar=zeros(Nloc)
    for i in 1:Nloc
        ell[i]=norm(Acur[1:2,neighbors(i,D0)[1]]-Acur[1:2,i])
    end

    #Dual lenght
    for i in 1:Nloc
        ellstar[i]=0.5*(ell[i]+ell[neighbors(i,D0)[2]])
    end
    L=sum(ell)

    for i in 1:Nloc
        f[i]=(L/(Nloc*ellstar[i])-1)*ω
    end
    
        
    # compatibility correction
    fmean = sum(f .* ellstar) / sum(ellstar)
    f .-= fmean

    for i in 1:Nloc
        ds=norm(Acur[1:2,neighbors(i,D0)[2]]-Acur[1:2,i])
        α[i]=α[neighbors(i,D0)[2]]+f[i]*ds
    end


    # remove arbitrary constant / global drift
    alphamean = sum(α .* ellstar) / sum(ellstar)
    α .-= alphamean

    v = α .* Acur[6:7, :]'

    dx = view(dX, 1:Nloc)
    dy = view(dX, Nloc+1:2Nloc)

    dx .= v[:, 1]
    dy .= v[:, 2]

    return nothing
    
end

function f3!(dX, X, p, t) #Methode laplacien et répartition selon longueur
    A, D0 = p
    Nloc = size(A, 2)

    # Build the current geometric state from X (the actual ODE state)
    Acur = copy(A)
    Acur[1, :] .= view(X, 1:Nloc)
    Acur[2, :] .= view(X, Nloc+1:2Nloc)
    MAJ3(Acur, D0)
    Ploc=0.0
    #Δ = laplacian(Acur, D0)
    b = zeros(Nloc)
    v = zeros(Nloc, 2)
    star0 = zeros(Nloc)
    for i in 1:Nloc
        dist1 = norm(Acur[1:2, neighbors(i, D0)[1]] - Acur[1:2, i])
        dist2 = norm(Acur[1:2, neighbors(i, D0)[2]] - Acur[1:2, i])
        star0[i] = 0.5 * (dist1 + dist2)
        Ploc += dist1
    end
    star1 = zeros(size(D0,1), size(D0,1))
    edge_len = zeros(size(D0,1))
    for i in 1:size(D0,1)
        nodes = findall(!iszero, D0[i,:])
        dist = norm(Acur[1:2, nodes[1]] - Acur[1:2, nodes[2]])
        star1[i,i] = 1 / max(dist, 1e-14)
        edge_len[i] = dist
        
    end
    for i in 1:Nloc
        b[i] = (Ploc / Nloc / star0[i] - 1) * ω
    end
    bmean = sum(b .* star0) / sum(star0)
    b .-= bmean
    b .*= star0
    Δ=D0' * star1 * D0
    Δ[1,:].=0
    Δ[:,1].=0
    Δ[1,1]=1
    b[1]=0

    Ψ=Δ\b #Résolution du système linéaire pour trouver Ψ
    w=star1*D0*Ψ #.*t' #Acur[6:7, :]'
    #v=-star1*D0*Ψ.*Acur[6:7, :]' #.*t'
    #
    wmean = sum(w .* edge_len) / sum(edge_len)
    w .-= wmean
    for i in 1:Nloc
        v[i, :] = -0.5*(w[i]+w[neighbors(i, D0)[2]])* Acur[6:7, i]
    end
    #

    #=
    for i in 1:Nloc
        edges = findall(!iszero, D0[:,i])
        v[i,:]=-0.5*(w[edges[1]]+w[edges[2]])*Acur[6:7, i]
    end
    =#

    dx = view(dX, 1:Nloc)
    dy = view(dX, Nloc+1:2Nloc)

    dx .= v[:, 1]
    dy .= v[:, 2]

    # Update A with new positions for next recomputation
    #A[1, :] .= x
    #A[2, :] .= y

      # Update the error reference for convergence checking
    return nothing
end


function f4!(dX, X, p, t) #Méthode laplacien et répartition selon longueur et courbure
    A, D0 = p
    Nloc = size(A, 2)
    ρ=zeros(Nloc)
    # Build the current geometric state from X (the actual ODE state)
    Acur = copy(A)
    Acur[1, :] .= view(X, 1:Nloc)
    Acur[2, :] .= view(X, Nloc+1:2Nloc)
    MAJ3(Acur, D0)
    Ploc=0.0
    Δ = laplacian(Acur, D0)
    b = zeros(Nloc)
    v = zeros(Nloc, 2)
    star0 = zeros(Nloc)
    edge_len = zeros(size(D0,1))
    for i in 1:Nloc
        dist1 = norm(Acur[1:2, neighbors(i, D0)[1]] - Acur[1:2, i])
        dist2 = norm(Acur[1:2, neighbors(i, D0)[2]] - Acur[1:2, i])
        star0[i] = 0.5 * (dist1 + dist2)
        Ploc += dist1
        ρ[i]=1+β*(Acur[3,i]/mean(Acur[3, :]))^P
    end
    star1 = spzeros(size(D0,1), size(D0,1))
    for i in 1:size(D0,1)
        nodes = findall(!iszero, D0[i,:])
        dist = norm(Acur[1:2, nodes[1]] - Acur[1:2, nodes[2]])
        star1[i,i] = 1 / max(dist, 1e-14)
        edge_len[i] = dist
        
    end
    for i in 1:Nloc
       # b[i] = (Ploc/ Nloc / star0[i] - 1) * ω
        b[i]= ω*((sum(star0.*ρ))/(Nloc*star0[i]*ρ[i])-1)
    end
    bmean = sum(b .* star0) / sum(star0)
    b .-= bmean
    Δ[1,:].=0
    Δ[:,1].=0
    Δ[1,1]=1
    b[1]=0

    Ψ=Δ\b 
    w=star1*D0*Ψ #.*t' #Acur[6:7, :]'
    #v=star1*D0*Ψ.*Acur[6:7, :]' #.*t'
    #
    wmean = sum(w .* edge_len) / sum(edge_len)
    w .-= wmean
    for i in 1:Nloc
        v[i, :] = -0.5*(w[i]+w[neighbors(i, D0)[2]])* Acur[6:7, i]
    end
    #

    dx = view(dX, 1:Nloc)
    dy = view(dX, Nloc+1:2Nloc)

    dx .= v[:, 1]
    dy .= v[:, 2]

    # Update A with new positions for next recomputation
    #A[1, :] .= x
    #A[2, :] .= y

      # Update the error reference for convergence checking
    return nothing
end


function f8!(dX, X, p, t) #Méthode utilisant le laplacien et répartition selon longueur et courbure

    A, D0 = p

    Nloc = size(A, 2)
    
    # Build the current geometric state from X (the actual ODE state)
    Acur = copy(A)
    Acur[1, :] .= view(X, 1:Nloc)
    Acur[2, :] .= view(X, Nloc+1:2Nloc)
    MAJ3(Acur, D0)
    Δ = laplacian(Acur, D0)
    b = zeros(Nloc)
    v = zeros(Nloc, 2)
    Un=ones(Nloc,1)
    star0 = zeros(Nloc,Nloc)
    ρ=zeros(Nloc)
    for i in 1:Nloc
        dist1 = norm(Acur[1:2, neighbors(i, D0)[1]] - Acur[1:2, i])
        dist2 = norm(Acur[1:2, neighbors(i, D0)[2]] - Acur[1:2, i])
        star0[i,i] = 0.5 * (dist1 + dist2)
    end
    κ_moy=(Un'*star0*Acur[3,:])[1]/(Un'*star0*Un)[1]
    ρ .= Un .+ β*(Acur[3,:]./κ_moy)
    star1 = spzeros(size(D0,1), size(D0,1))
    for i in 1:size(D0,1)
        nodes = findall(!iszero, D0[i,:])
        dist = norm(Acur[1:2, nodes[1]] - Acur[1:2, nodes[2]])
        star1[i,i] = 1 / max(dist, 1e-14)
    end
    M=(Un'*star0*ρ)[1]
    for i in 1:Nloc
        b[i]= ω*(M/(Nloc*star0[i,i]*ρ[i])-1)
    end
    bmean = (Un'*star0*b)[1] / (Un'*star0*Un)[1]
    b .-= bmean

    Δ[1,:].=0
    Δ[:,1].=0
    Δ[1,1]=1
    b[1]=0

    Ψ=Δ\b 
    w=star1*D0*Ψ #.*t' #Acur[6:7, :]'
    #v=star1*D0*Ψ.*Acur[6:7, :]' #.*t'
    #
    for i in 1:Nloc
        v[i, :] = -0.5*(w[i]+w[neighbors(i, D0)[2]])* Acur[6:7, i]
    end
    #

    dx = view(dX, 1:Nloc)
    dy = view(dX, Nloc+1:2Nloc)

    dx .= v[:, 1]
    dy .= v[:, 2]

    # Update A with new positions for next recomputation
    #A[1, :] .= x
    #A[2, :] .= y

      # Update the error reference for convergence checking
    return nothing
end


function compute_b_norm(Acur, D0)#, ω)
    Nloc = size(Acur, 2)
    ρ = zeros(Nloc)
    star0 = zeros(Nloc)

    for i in 1:Nloc
        dist1 = norm(Acur[1:2, neighbors(i, D0)[1]] - Acur[1:2, i])
        dist2 = norm(Acur[1:2, neighbors(i, D0)[2]] - Acur[1:2, i])
        star0[i] = 0.5 * (dist1 + dist2)
    end

    for i in 1:Nloc
        ρ[i] = 1 + β * (Acur[3,i] / mean(Acur[3, :]))^P
    end

    b = zeros(Nloc)
    for i in 1:Nloc
        b[i] =  ((sum(star0 .* ρ)) / (Nloc * star0[i] * ρ[i]) - 1)#*ω
    end

    # compatibility correction (weighted mean removal)
    bmean = sum(b .* star0) / sum(star0)
    b .-= bmean

    return maximum(abs.(b))
end


a=.30 #.30 pour l'ellipse
b=.15
N=64
h=(a-b)^2/(a+b)^2
Pa=pi*(a+b)*(1+(3*h)/(10+sqrt(4-3*h)))
Sa=pi*a*b
dl=Pa/N
x0=0.5
y0=0.75

#Constantes pour la répartition tangentielle
ω=ω0=10
β=0
P=1

t=0.0
dt=1E-4 #1E-4 pour l'ellipse et 1E-5 pour le cercle #Pas de temps advection
dt_Mikula=1E-4#1E-4 pour l'ellipse et 1E-5 pour le cercle #Pas de temps pour la méthode de Mikula
nt=0 #Nombre de pas de temps
t_plot=0.0 
Δt_plot=0.01 #Intervalle de temps entre les affichages
Iter_Mikula_Max=200
Iter=0

A=zeros(9,N)
B=zeros(9,1)
D0=spzeros(N,N) #Adjency matrix

for i in 1:N
    θ = 2π*(i-1)/N
    A[1,i] = x0 + a*cos(θ)
    A[2,i] = y0 + b*sin(θ)
end
p=scatter(A[1,:],A[2,:],aspect_ratio=1,label="Marqueur")
display(p)
savefig(p,"Image/Advection_Normale/Initial.png")

for i in 1:N
    D0[i,mod1(i+1,N)]=1
    D0[i,i]=-1
end

MAJ3(A,D0)


Su0=0
Sp0=0
P0=0
P_arc_0=0
for i in 1:N
    k=neighbors(i, D0)[1]
    j=neighbors(i, D0)[2]
    #Surface de l'arc est égal à Secteur Disque- Surface triangle P_i O P_i+1
    local R=1/A[3,i]
    local L= norm(A[1:2,neighbors(i, D0)[1]]-A[1:2,i])
    local Φ= 2*asin(L/(2*R))
    global Su0+= 1/2*(A[1,i]-A[1,k])*(A[2,k]+A[2,i])
    global Sp0+=1/2*R^2*(Φ-sin(Φ))
    global P0+=L   
    global P_arc_0+=R*Φ 
end
S0=Su0+Sp0

p=scatter(A[1,:],A[2,:],aspect_ratio=1,label=false)
quiver!(A[1,:], A[2,:], quiver=(A[4,:]/10, A[5,:]/10), aspect_ratio=1, label="Normales",color=:red)
quiver!(A[1,:], A[2,:], quiver=(A[6,:]/10, A[7,:]/10), aspect_ratio=1, label="Tangentes",color=:blue)


Iter_moy=0
changement=0
#CPU_time = @timed begin
anim=Plots.Animation()

@time while t<1
    dteff=min(dt,1-t)
    MAJ3(A,D0)
    global nt+=1 #Nombre de pas de temps

    #Création d'un nouveau point si trop éloigné ou suppression d'un point si trop proche
    
    #=
    global change=1
    while change!=0
        global change = 0
        
        i = 1
        while i <= N
            Lp = sqrt((A[1,neighbors(i, D0)[1]]-A[1,i])^2 + (A[2,neighbors(i, D0)[1]]-A[2,i])^2)
            Lm = sqrt((A[1,neighbors(i, D0)[2]]-A[1,i])^2 + (A[2,neighbors(i, D0)[2]]-A[2,i])^2)
            j=neighbors(i, D0)[1]   

            ϕp = asin(clamp(Lp*A[3,i]/2, -1, 1))*2
            ϕm = asin(clamp(Lm*A[3,i]/2, -1, 1))*2
            
            if A[3,i] != 0 && (ϕp/A[3,i] < 0.5*dl || ϕm/A[3,i] < 0.5*dl)
                #=
                ip=neighbors(i, D0)[1]
                ipp=neighbors(ip, D0)[1]
                im=neighbors(i, D0)[2]
                imm=neighbors(im, D0)[2]
                poly=GI.Polygon([[A[1:2, imm], A[1:2, im], A[1:2, i], A[1:2, ip], A[1:2, ipp],A[1:2, imm]]])
                S=GO.area(poly)
                d=norm(A[1:2,ipp]-A[1:2,imm])/4
                if !isfinite(d) || d < 1e-14
                    i += 1
                    continue
                end
                h=S/(3*d)

                base = [A[1,ipp]-A[1,imm], A[2,ipp]-A[2,imm]]
                base_norm = norm(base)
                if !isfinite(base_norm) || base_norm < 1e-14
                    i += 1
                    continue
                end

                n=[A[2,ipp]-A[2,imm], A[1,imm]-A[1,ipp]]/base_norm
                tan=base/base_norm
                
                #Mise à jour: ajustement des voisins de l'ancien i
                A[1:2,ip]=A[1:2,ipp]-d*tan+h*n
                A[1:2,im]=A[1:2,imm]+d*tan+h*n
                =#
            

                #println("Suppression du point ", i)
                global change = 1
                remove_idx = i
                neigh1 = neighbors(remove_idx, D0)[1]
                neigh2 = neighbors(remove_idx, D0)[2]
                Lignes_enlever=findall(!iszero, D0[:,remove_idx])
                D0=D0[setdiff(1:N, Lignes_enlever), :]
                D0 = D0[:,setdiff(1:N, remove_idx)]
                A = A[:, setdiff(1:N, remove_idx)]
                mapping = Dict{Int, Int}()
                for i in 1:N
                    if i < remove_idx
                        mapping[i] = i
                    elseif i > remove_idx
                        mapping[i] = i-1
                    end
                end
                neigh1 = mapping[neigh1]
                neigh2 = mapping[neigh2]
                N = N-1
                Nouvelle_arête = spzeros(1, N)
                Nouvelle_arête[1, neigh1] = 1
                Nouvelle_arête[1, neigh2] = -1
                D0 = [D0; Nouvelle_arête]
                global changement+=1
                
                
                # Do not increment i, since indices shifted
            elseif A[3,i] == 0 && (Lp < 0.5*dl || Lm < 0.5*dl)

                #=
                ip=neighbors(i, D0)[1]
                ipp=neighbors(ip, D0)[1]
                im=neighbors(i, D0)[2]
                imm=neighbors(im, D0)[2]
                poly=GI.Polygon([[A[1:2, imm], A[1:2, im], A[1:2, i], A[1:2, ip], A[1:2, ipp],A[1:2, imm]]])
                S=GO.area(poly)
                d=norm(A[1:2,ipp]-A[1:2,imm])/4
                if !isfinite(d) || d < 1e-14
                    i += 1
                    continue
                end
                h=S/(3*d)

                base = [A[1,ipp]-A[1,imm], A[2,ipp]-A[2,imm]]
                base_norm = norm(base)
                if !isfinite(base_norm) || base_norm < 1e-14
                    i += 1
                    continue
                end

                n=[A[2,ipp]-A[2,imm], A[1,imm]-A[1,ipp]]/base_norm
                tan=base/base_norm
                
                #Mise à jour: ajustement des voisins de l'ancien i
                A[1:2,ip]=A[1:2,ipp]-d*tan+h*n
                A[1:2,im]=A[1:2,imm]+d*tan+h*n
                =#
                

                #println("Suppression du point ", i)
                global change = 1
                remove_idx = i
                neigh1 = neighbors(remove_idx, D0)[1]
                neigh2 = neighbors(remove_idx, D0)[2]
                Lignes_enlever=findall(!iszero, D0[:,remove_idx])
                D0=D0[setdiff(1:N, Lignes_enlever), :]
                D0 = D0[:,setdiff(1:N, remove_idx)]
                A = A[:, setdiff(1:N, remove_idx)]
                mapping = Dict{Int, Int}()
                for i in 1:N
                    if i < remove_idx
                        mapping[i] = i
                    elseif i > remove_idx
                        mapping[i] = i-1
                    end
                end
                neigh1 = mapping[neigh1]
                neigh2 = mapping[neigh2]
                N = N-1
                Nouvelle_arête = spzeros(1, N)
                Nouvelle_arête[1, neigh1] = 1
                Nouvelle_arête[1, neigh2] = -1
                D0 = [D0; Nouvelle_arête]

               
                global changement+=1
                # Do not increment i
            elseif A[3,i] != 0 && ϕp / A[3,i] > 1.5*dl
                #println("Création d'un point entre ", i, " et ", neighbors(i,D0)[1])
                
                xm=(A[1,neighbors(i,D0)[1]]+A[1,i])/2
                ym=(A[2,neighbors(i,D0)[1]]+A[2,i])/2

                h1=1/A[3,i]-sqrt(1/(A[3,i]^2)-(Lp/2)^2)
                h2=1/A[3,neighbors(i,D0)[1]]-sqrt(1/(A[3,neighbors(i,D0)[1]]^2)-(Lp/2)^2)

                cx = A[1,neighbors(i,D0)[1]] - A[1,i]
                cy = A[2,neighbors(i,D0)[1]] - A[2,i]

                nx = cy / Lp
                ny = -cx / Lp

                xnp = xm + h1 * nx
                ynp = ym + h1 * ny

                xnm=xm+h2*nx
                ynm=ym+h2*ny

                B.=0
                
                B[1]=(xnp+xnm)/2
                B[2]=(ynp+ynm)/2
                #=
                B[1]=xm
                B[2]=ym
                B[3]=(A[3,neighbors(i,D0)[1]]+A[3,i])/2
                =#
                global A=[A B]
                
                arête_i=findall(!iszero, D0[:,i])
                arête_enlever=nothing
                for e in arête_i
                    nodes = findall(!iszero, D0[e,:])
                    voisin = nodes[1] == i ? nodes[2] : nodes[1]
                    if voisin == j
                        arête_enlever = e
                        break
                    end
                end
                # sens
                sign_i = D0[arête_enlever, i]

                # suppression arête
                D0 = D0[setdiff(1:size(D0,1), arête_enlever), :]

                # ajout colonne
                D0 = [D0 spzeros(size(D0,1))]

                new = size(D0,2)

                # nouvelles arêtes
                e1 = spzeros(new)
                e2 = spzeros(new)

                if sign_i == -1
                    # i → j
                    e1[i] = -1; e1[new] = 1
                    e2[new] = -1; e2[j] = 1
                else #Dans notre cas, on sait que sign_i est égal à -1 
                    # j → i
                    e1[j] = -1; e1[new] = 1
                    e2[new] = -1; e2[i] = 1
                end

                global D0 = vcat(D0, e1', e2')
                global N+=1
                
                global change = 1
                global changement+=1
            else
                i += 1
            end
            MAJ(A,D0)
            
            
        end
    end
    =#
    #

    #----------------------------------------------------------------
    #Point redistribution

    #
    # Compute arc lengths for uniformity check
    arc_lengths = zeros(N)
    for i in 1:N
        j = neighbors(i, D0)[1]
        arc_lengths[i] = sqrt((A[1,j]-A[1,i])^2 + (A[2,j]-A[2,i])^2)
    end
    
    # Measure uniformity: coefficient of variation
    mean_arc = mean(arc_lengths)
    std_arc = std(arc_lengths)
    global uniformity = std_arc / mean_arc
        
    global Iter=0
   
    while uniformity>0.05 && Iter < Iter_Mikula_Max #for t in 1:1e5
        
        global Iter+=1
        global changement+=1
        #println("Iteration ", t)
        local X0 = vcat(A[1,:], A[2,:])
        local prob = ODEProblem(f3!, X0, (0.0, dt), (A, D0)) #f2! alpha et f3! laplacien
        local sol=solve(prob,Euler(),abstol=1e-8,reltol=1e-8,dt=dt_Mikula)

        local Xf = sol.u[end]
        A[1,:] .= view(Xf, 1:N)
        A[2,:] .= view(Xf, N+1:2N)
        MAJ3(A,D0)
        

        # Compute arc lengths for uniformity check
        arc_lengths = zeros(N)
        for i in 1:N
            j = neighbors(i, D0)[1]
            arc_lengths[i] = sqrt((A[1,j]-A[1,i])^2 + (A[2,j]-A[2,i])^2)
        end
        
        # Measure uniformity: coefficient of variation
        mean_arc = mean(arc_lengths)
        std_arc = std(arc_lengths)
        global uniformity = std_arc / mean_arc
        
    end
    #
    #=
    global ω=ω0
    global tol_b=.1
    nredistrib=0
    while true
      
        global changement += 1
        nredistrib+=1
        local X0 = vcat(A[1,:], A[2,:])
        local prob = ODEProblem(f4!, X0, (0.0, dt), (A, D0))
        local sol = solve(prob, Euler(), abstol=1e-8, reltol=1e-8, dt=dt_Mikula)

        local Xf = sol.u[end]
        A[1,:] .= view(Xf, 1:N)
        A[2,:] .= view(Xf, N+1:2N)
        MAJ3(A,D0)

        # Compute convergence measure from the redistribution RHS (b)
        b_norm = compute_b_norm(A, D0)
        if nredistrib==1
            global b_norm_initial=b_norm*ω
        end
        if b_norm*ω <b_norm_initial/10 
            global ω*=1.1
        end
        

        # Stop when redistribution forcing is small
        if b_norm < tol_b
            #println("Converged: b_norm=", b_norm, " at iteration ", nt)
            break
        end

        # Optional diagnostic: compute uniformity (kept for information only)
        arc_lengths = zeros(N)
        for i in 1:N
            j = neighbors(i, D0)[1]
            arc_lengths[i] = sqrt((A[1,j]-A[1,i])^2 + (A[2,j]-A[2,i])^2)
        end
        mean_arc = mean(arc_lengths)
        std_arc = std(arc_lengths)
        uniformity = std_arc / mean_arc
        
    end
    =#
    #
    #----------------------------------------------------------------
    if t >= t_plot
        p=scatter(A[1,:],A[2,:],aspect_ratio=1,label=false, xlims=(0,1), ylims=(0,1))
        #quiver!(A[1,:], A[2,:], quiver=(A[4,:]/10, A[5,:]/10), aspect_ratio=1, label="Normales",color=:red)
        #quiver!(A[1,:], A[2,:], quiver=(A[6,:]/10, A[7,:]/10), aspect_ratio=1, label="Tangentes",color=:blue)
        frame(anim)
        global t_plot += Δt_plot   
    end
    #
    if isapprox(t, 0.0)
        println("t=0")
        local p=scatter(A[1,:],A[2,:],aspect_ratio=1,label=false, xlims=(0,1), ylims=(0,1))
        savefig(p, "Image/Advection_Normale/t=0.png")
    elseif isapprox(t, 0.25)
        println("t=1/4")
        local p=scatter(A[1,:],A[2,:],aspect_ratio=1,label=false, xlims=(0,1), ylims=(0,1))
        savefig(p, "Image/Advection_Normale/t=0,25.png")
    elseif isapprox(t, 0.50)
        println("t=1/2")
        local p=scatter(A[1,:],A[2,:],aspect_ratio=1,label=false, xlims=(0,1), ylims=(0,1))
        savefig(p, "Image/Advection_Normale/t=0,5.png")
    elseif isapprox(t, 0.75)
        println("t=3/4")
        local p=scatter(A[1,:],A[2,:],aspect_ratio=1,label=false, xlims=(0,1), ylims=(0,1))
        savefig(p, "Image/Advection_Normale/t=0,75.png")
    end
    #
    #
    #Ordinary Differential Equation Solver
    z0 = zeros(2*N)
    for i in 1:N
        z0[2*i-1]=A[1,i]
        z0[2*i]=A[2,i]
    end
    prob=ODEProblem(velocity,z0,(0.0,dteff))
    sol=solve(prob,Tsit5(),abstol=1e-12,reltol=1e-12)
    for i in 1:N
        A[1,i]=sol[2*i-1,end]
        A[2,i]=sol[2*i,end]
    end
    #=
    P_new=0
    for i in 1:N #Recalcul de dl
        j=neighbors(i, D0)[1]
        P_new+=norm(A[1:2,j]-A[1:2,i])


    end
    global dl = P_new/N
    =#
    global t+=dteff
    



end

#=
pf=scatter(A[1,:],A[2,:],aspect_ratio=1,label=false, xlims=(0,1), ylims=(0,1))
savefig(pf, "Image/Ellipse_Normale/t=1.png")
=#
#Shoelace + Arc
Su=0
Sp=0
Pf=0
P_arc_f=0
for i in 1:N
    k=neighbors(i, D0)[1]
    if A[3,i] == 0
        local L= sqrt((A[1,k]-A[1,i])^2+(A[2,k]-A[2,i])^2)
        global Su+= 1/2*(A[1,i]-A[1,k])*(A[2,k]+A[2,i])
        global Pf+=L
        global P_arc_f+=L
    else
        #Surface de l'arc est égal à Secteur Disque- Surface triangle P_i O P_i+1
        local R=1/A[3,i]
        local L= sqrt((A[1,k]-A[1,i])^2+(A[2,k]-A[2,i])^2)
        local Φ= 2*asin(L/(2*R))
        global Su+= 1/2*(A[1,i]-A[1,k])*(A[2,k]+A[2,i])
        global Sp+=1/2*R^2*(Φ-sin(Φ))
        global Pf+=L
        global P_arc_f+=R*Φ
    end
end
#Pour une ellipse 

E_f_1=0
for i in 1:N
    global E_f_1+=abs((A[1,i]-x0)^2/a^2+(A[2,i]-y0)^2/b^2-1)
end
E_f_1/=N

E_f_2=0
for i in 1:N
    global E_f_2+=abs((A[1,i]-x0)^2/a^2+(A[2,i]-y0)^2/b^2-1)^2
end
E_f_2=sqrt(E_f_2/N)

E_inf=0
for i in 1:N
    global E_inf=max(E_inf,abs(((A[1,i]-x0)/a)^2+((A[2,i]-y0)/b)^2-1))
end

#Pour un cercle de rayon r=0.15 (using implicit equation like ellipse)
E_f_1_cercle=0
r=b 
for i in 1:N
    global E_f_1_cercle+=abs(((A[1,i]-x0)^2+(A[2,i]-y0)^2)/r^2-1)
    #println("Point ", i, ": ", ((A[1,i]-x0)^2+(A[2,i]-y0)^2)/r^2-1)
end
E_f_1_cercle/=N

E_f_2_cercle=0
for i in 1:N
    global E_f_2_cercle+=abs(((A[1,i]-x0)^2+(A[2,i]-y0)^2)/r^2-1)^2
end
E_f_2_cercle=sqrt(E_f_2_cercle/N)

E_inf_cercle=0
for i in 1:N
    global E_inf_cercle=max(E_inf_cercle,abs(((A[1,i]-x0)^2+(A[2,i]-y0)^2)/r^2-1))
end

Sf=Su+Sp

Ecart=abs(Sf-S0)/S0
println("Nombre de changements : ", changement)
#=
println("E_surf Shoelace= ",abs(Su-Su0)/Su0)
println("E_surf Arc= ",Ecart)
println("Périmètre Shoelace= ",abs(Pf-P0)/P0)
println("Périmètre Arc= ",abs(P_arc_f-P_arc_0)/P_arc_0)
=#
if a==b
    @printf("Erreur forme cercle (moyenne)= %.3e\n", E_f_1_cercle)
    @printf("Erreur forme cercle (RMS)= %.3e\n", E_f_2_cercle)
    @printf("Erreur forme cercle (inf)= %.3e\n", E_inf_cercle)
else
     @printf("Erreur forme ellipse (moyenne)= %.3e\n", E_f_1)
     @printf("Erreur forme ellipse (RMS)= %.3e\n", E_f_2)
     @printf("Erreur forme ellipse (inf)= %.3e\n", E_inf)
end


p3=Plots.scatter(A[1,:], A[2,:],aspect_ratio=1,label="Marqueur")
        #for i in 1:N
        #    Plots.plot!([A[1,i], A[1,neighbors(i,D0)[1]]], [A[2,i], A[2,neighbors(i,D0)[1]]], lw=1,legend=false,color="blue")
        #end

        display(p3)
savefig(p3,"Image/Advection_Normale/Final Mikula.png")
gif(anim, "Image/Advection_Normale/Animation Mikula.gif")
