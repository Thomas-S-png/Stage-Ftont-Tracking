using LinearAlgebra 
#using DataArrays
using Plots
using Statistics
using SparseArrays
using OrdinaryDiffEq
import GeometryOps as GO
import GeoInterface as GI
using Printf


function neighbors(j,D0) #Ou neighbor(j,D0), à vérifier si ça pose problème
    edges = findall(!iszero, D0[:,j])

    vp = nothing
    vm = nothing

    for e in edges
        nodes = findall(!iszero, D0[e,:])

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

function MAJ(A,D0)
    recalc_all_curvatures!(A, D0)
    #Normale sur le vertexe (A[4:5,i])
    for i in 1:size(A, 2)
        j=neighbors(i, D0)[1]
        L=sqrt((A[1,j]-A[1,i])^2+(A[2,j]-A[2,i])^2)
        h=sqrt(1/A[3,i]^2-(L/2)^2)
        nx=-(A[2,j]-A[2,i])/L
        ny=(A[1,j]-A[1,i])/L

        local x_m=(A[1,j]+A[1,i])/2
        local y_m=(A[2,j]+A[2,i])/2
        local x_o=x_m+h*nx
        local y_o=y_m+h*ny

        A[4,i]=(A[1,i]-x_o)/norm(A[1:2,i]-[x_o,y_o])
        A[5,i]=(A[2,i]-y_o)/norm(A[1:2,i]-[x_o,y_o])
    end
    #Tangente du vertexe (A[6:7,i])

    for i in 1:size(A, 2)
        A[6,i]=-A[5,i]
        A[7,i]=A[4,i]
    end

end

function MAJ2(A,D0)
    recalc_all_curvatures!(A, D0)
    #Normale sur le vertexe (A[4:5,i])
    #Calcul de la normale
    for i in 1:size(A, 2)
        Xi=A[1:2,i]
        Xp=A[1:2,neighbors(i, D0)[1]]
        Xm=A[1:2,neighbors(i, D0)[2]]

        d_i  = norm(Xp - Xi)
        d_im = norm(Xi - Xm)

        t = (Xp - Xi)/d_i + (Xi - Xm)/d_im
        t /= norm(t)

        A[6:7,i]=t

        n = [t[2], -t[1]]

        A[4:5,i]=n

    end
    
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

    end
end

#=
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

        n = [t[2], -t[1]]

        A[4:5,i]=n

    end
    
end
=#
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

function f!(dX, X, p, t)
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
    star0 = zeros(Nloc)
    for i in 1:Nloc
        dist1 = norm(Acur[1:2, neighbors(i, D0)[1]] - Acur[1:2, i])
        dist2 = norm(Acur[1:2, neighbors(i, D0)[2]] - Acur[1:2, i])
        star0[i] = 0.5 * (dist1 + dist2)
    end
    for i in 1:Nloc
        b[i] = (dl / star0[i] - 1) * ω
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
        v[i, :] = -0.5 * ((Ψ[neighbors(i, D0)[1]] - Ψ[i]).* Tp / lp + (Ψ[i]- Ψ[neighbors(i, D0)[2]] ).* Tm / lm) #.* Acur[6:7, i]
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


function f3!(dX, X, p, t)
    A, D0 = p
    Nloc = size(A, 2)

    # Build the current geometric state from X (the actual ODE state)
    Acur = copy(A)
    Acur[1, :] .= view(X, 1:Nloc)
    Acur[2, :] .= view(X, Nloc+1:2Nloc)
    MAJ3(Acur, D0)
    Ploc=0.0
    Δ = laplacian(Acur, D0)
    b = zeros(Nloc)
    v = zeros(Nloc, 2)
    t=zeros(Nloc,2)
    star0 = zeros(Nloc)
    for i in 1:Nloc
        dist1 = norm(Acur[1:2, neighbors(i, D0)[1]] - Acur[1:2, i])
        dist2 = norm(Acur[1:2, neighbors(i, D0)[2]] - Acur[1:2, i])
        star0[i] = 0.5 * (dist1 + dist2)
        Ploc += dist1
    end
    star1 = spzeros(size(D0,1), size(D0,1))
    for i in 1:size(D0,1)
        nodes = findall(!iszero, D0[i,:])
        dist = norm(Acur[1:2, nodes[1]] - Acur[1:2, nodes[2]])
        star1[i,i] = 1 / max(dist, 1e-14)
    end
    for i in 1:Nloc
        b[i] = (Ploc/ N / star0[i] - 1) * ω
    end

    Δ[1,:].=0
    Δ[1,1]=1
    b[1]=0

    Ψ=Δ\b 
    for i in 1:Nloc
        t[i,:]=(Acur[1:2, neighbors(i, D0)[1]]-Acur[1:2, i])'
        t[i,:]=t[i,:]/norm(t[i,:])
    end
    w=star1*D0*Ψ #.*t #Acur[6:7, :]'
    for i in 1:Nloc
        v[i, :] = -0.5*(w[i]+w[neighbors(i, D0)[2]])* Acur[6:7, i]
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

x_o=0.5   #Centre du cercle
y_o=0.75

a=.30
b=.15
N=32
h=(a-b)^2/(a+b)^2
P=pi*(a+b)*(1+(3*h)/(10+sqrt(4-3*h)))
Sa=pi*a*b
dl=P/N
R=(a+b)/2

ω=10

B=zeros(9,1)
A=zeros(9,N)
D0=spzeros(N,N) #Adjency matrix
change=1

#Initialisation
for i in 1:N
    local θ = 2π*(i-1)/N
    A[1,i] = x_o + a*cos(θ)
    A[2,i] = y_o + b*sin(θ)
end


for i in 1:N
    D0[i,mod1(i+1,N)]=1
    D0[i,i]=-1
end

MAJ3(A,D0)

p1 = scatter(A[1,:], A[2,:],
            aspect_ratio=1,
            label="Marqueurs")

quiver!(A[1,:], A[2,:],
        quiver=(A[4,:]/10, A[5,:]/10),
        label="Normales")

display(p1)


#Calcul de surface

Su0=0
Sp0=0
P_0=0
P_arc_0=0
for i in 1:N
    j=neighbors(i, D0)[1]
    #Surface de l'arc est égal à Secteur Disque- Surface triangle P_i O P_i+1
    local R=1/A[3,i]
    local L= sqrt((A[1,j]-A[1,i])^2+(A[2,j]-A[2,i])^2)
    local Φ= 2*asin(L/(2*R))
    global Su0+=1/2*(A[1,i]-A[1,j])*(A[2,j]+A[2,i])
    global Sp0+=1/2*R^2*(Φ-sin(Φ))
    global P_0+=L
    global P_arc_0+=R*Φ

end
Su0=abs(Su0)
Sp0=abs(Sp0)
S0=Su0+Sp0

#Initialisation avec stretching

for i in 1:N

    s = (i-1)/N
    s_stretch = 0.5*(1 - cos(pi*s))   # stretching

    local θ = 2*pi*s_stretch
    A[1,i] = x_o + a*cos(θ)
    A[2,i] = y_o + b*sin(θ)

end
p=scatter(A[1,:],A[2,:],aspect_ratio=1)
display(p)

for i in 1:N
    D0[i,mod1(i+1,N)]=1
    D0[i,i]=-1
end

p2 = scatter(A[1,:], A[2,:],aspect_ratio=1,label="Marqueurs")
display(p2)

changement=0
#=
@time while change!=0
    global change = 0
    
    local i = 1
    while i <= N
        Lp = sqrt((A[1,neighbors(i, D0)[1]]-A[1,i])^2 + (A[2,neighbors(i, D0)[1]]-A[2,i])^2)
        Lm = sqrt((A[1,neighbors(i, D0)[2]]-A[1,i])^2 + (A[2,neighbors(i, D0)[2]]-A[2,i])^2)
        j=neighbors(i, D0)[1]   

        ϕp = asin(clamp(Lp*A[3,i]/2, -1, 1))*2
        ϕm = asin(clamp(Lm*A[3,i]/2, -1, 1))*2
        
        if A[3,i] != 0 && (ϕp/A[3,i] < dl*0.5 || ϕm/A[3,i] <dl*0.5)
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
            local h=S/(3*d)

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
        elseif A[3,i] == 0 && (Lp < dl*0.5 || Lm < dl*0.5)

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
            local h=S/(3*d)

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
        elseif Lp > 2*dl
            #println("Création d'un point entre ", i, " et ", neighbors(i, D0)[1])
            
            #
            xm=(A[1,neighbors(i, D0)[1]]+A[1,i])/2
            ym=(A[2,neighbors(i, D0)[1]]+A[2,i])/2

            h1=1/A[3,i]-sqrt(1/(A[3,i]^2)-(Lp/2)^2)
            h2=1/A[3,neighbors(i, D0)[1]]-sqrt(1/(A[3,neighbors(i, D0)[1]]^2)-(Lp/2)^2)

            cx = A[1,neighbors(i, D0)[1]] - A[1,i]
            cy = A[2,neighbors(i, D0)[1]] - A[2,i]

            nx = cy / Lp
            ny = -cx / Lp

            xnp = xm + h1 * nx
            ynp = ym + h1 * ny

            xnm=xm+h2*nx
            ynm=ym+h2*ny

            B.=0
            B[1]=(xnp+xnm)/2
            B[2]=(ynp+ynm)/2
            B[3]=(A[3,neighbors(i, D0)[1]]+A[3,i])/2
            #
            #=
            xm=(A[1,neighbors(i, D0)[1]]+A[1,i])/2
            ym=(A[2,neighbors(i, D0)[1]]+A[2,i])/2
            B.=0
            B[1]=xm
            B[2]=ym
            B[3]=(A[3,neighbors(i, D0)[1]]+A[3,i])/2
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
            
            global change=1
            global changement+=1
        else
            i += 1
        end
        MAJ(A,D0)
        
        
    end
end
println("Nombre de changements : ", changement)
=#


#
println("Début de la répartition")
uniformity = 1.0
nt=0
anim=Animation()
@time while uniformity>0.05#for t in 1:1e5
    local dt=1e-5
    global nt+=1
    #println("Iteration ", t)
    local X0 = vcat(A[1,:], A[2,:])
    local prob = ODEProblem(f3!, X0, (0.0, dt), (A, D0))
    local sol=solve(prob,Euler(),abstol=1e-8,reltol=1e-8,dt=dt)

    local Xf = sol.u[end]
    A[1,:] .= view(Xf, 1:N)
    A[2,:] .= view(Xf, N+1:2N)
    MAJ3(A,D0)
     if nt % 500 == 0
        p3=scatter(A[1,:],A[2,:],aspect_ratio=1,label=false)
        display(p3)
        frame(anim)
     end
    #p3=scatter(A[1,:],A[2,:],aspect_ratio=1,label=false)
    #display(p3)

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
    
    #println("Iteration $t: uniformity = $uniformity")
    #=
    if uniformity <= 0.05 #0.27
        println("Even spacing achieved at iteration ", nt)
        break
    end
    =#
end
println("Nombre d'itérations : ", nt)
#
gif(anim,"Image/Repartition_Immobile/Animation.gif",fps=60)




MAJ3(A,D0)
#Shoelace + Arc
Su=0
Sp=0
P_f_arc=0
for i in 1:N
    j=neighbors(i, D0)[1]
    #Surface de l'arc est égal à Secteur Disque- Surface triangle P_i O P_i+1
    local R=1/A[3,i]
    local L= sqrt((A[1,j]-A[1,i])^2+(A[2,j]-A[2,i])^2)
    local Φ= 2*asin(L/(2*R))
    global Su+=1/2*(A[1,i]-A[1,j])*(A[2,j]+A[2,i])
    global Sp+=1/2*R^2*(Φ-sin(Φ))
    global P_f_arc+=Φ*R

end
Su=abs(Su)
Scf=Su+abs(Sp)


#Shoelace
Su=0
for i in 1:N 
    j=neighbors(i, D0)[1]

    global Su+=1/2*(A[1,i]-A[1,j])*(A[2,j]+A[2,i])

end
Su=abs(Su)


Erreur2=abs(Scf-S0)/S0 #Ecart entre la surface réelle de l'ellipse et la surface calculée par shoelace + arc

P_f_shoe=0
for i in 1:N
    j=neighbors(i, D0)[1]
    global P_f_shoe+=sqrt((A[1,j]-A[1,i])^2+(A[2,j]-A[2,i])^2)
end



#Test de comvergence de la courbure de l'arrête
κ_err=0
for i in 1:N
    #local c = clamp((A[1,i]-x0)/a, -1.0, 1.0)
    #local θ = acos(c)
    local θ = atan((A[2,i]-y_o)/b, (A[1,i]-x_o)/a)
    κ_th=a*b/(sqrt(a^2*sin(θ)^2+b^2*cos(θ)^2))^3
    global κ_err=max(κ_err,abs(κ_th-A[3,i])/κ_th)
end

E_f_1=0
for i in 1:N
    global E_f_1+=abs((A[1,i]-x_o)^2/a^2+(A[2,i]-y_o)^2/b^2-1)
end
E_f_1/=N

E_f_2=0
for i in 1:N
    global E_f_2+=abs((A[1,i]-x_o)^2/a^2+(A[2,i]-y_o)^2/b^2-1)^2
end
E_f_2=sqrt(E_f_2/N)

E_inf=0
for i in 1:N
    global E_inf=max(E_inf,abs(((A[1,i]-x_o)/a)^2+((A[2,i]-y_o)/b)^2-1))
end

#Erreur de forme
dist=0
for i in 1:N
    global dist=max(dist,abs(((A[1,i]-x_o)/a)^2+((A[2,i]-y_o)/b)^2-1))
end


println("E_surf shoe : ", abs(Su-Su0)/Su0)
println("E_surf arc : ", Erreur2)

println("E_P Arête : ", abs(P_f_shoe-P_0)/P_0)
println("E_Perimètre Arc : ", abs(P_f_arc-P_arc_0)/P_arc_0)
println("E_curv : ", κ_err)
@printf("E_f_1 : %.5e\n", E_f_1)
@printf("E_f_2 : %.5e\n", E_f_2)
@printf("E_inf : %.5e\n", E_inf)
println("E_Forme : ", dist)
p3 = scatter(A[1,:], A[2,:],aspect_ratio=1,label="Marqueurs")
display(p3)
