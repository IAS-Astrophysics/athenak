#!/usr/bin/env python3
"""Explicit seven-wave decomposition of a native GRMHD primitive symbol.

Offline diagnostic, NOT a GPU boundary replacement. Uses the covariant ideal-MHD
quartic for speeds and SVD nullspaces for a complete left/right basis. Coalescing
families are returned as a grouped eigenspace: their individual polarizations
are nonunique, but their sum and incoming/outgoing projection are well defined.
The normal CT magnetic field is a parameter, not an eighth propagating mode.
"""
import numpy as np
from numpy.polynomial import Polynomial as P

FAMILIES = ('fast-', 'Alfven-', 'slow-', 'entropy', 'slow+', 'Alfven+', 'fast+')

def speeds(q, bn, g, alpha, beta, axis, gamma=5/3):
    """Covariant RMHD dispersion relation in actual coordinate time/normal.

    a = W(v^n-x), G = gamma^{nn}-x^2, b.phi = B^n/W+B.v*a,
    x=(lambda+beta^n)/alpha. The magnetosonic polynomial is
    rho*h*(1-cs^2)*a^4 -(rho*h*cs^2+b^2)*a^2*G
        +cs^2*(b.phi)^2*G = 0 (Anton et al. 2010).
    """
    q=np.asarray(q,float);g=np.asarray(g,float)
    u=q[2:5];W=np.sqrt(1+u@g@u);v=u/W
    B=np.empty(3);B[axis]=bn;B[(axis+1)%3]=q[5];B[(axis+2)%3]=q[6]
    Bv=B@g@v;bsq=B@g@B/W**2+Bv**2
    H=q[0]+gamma*q[1]/(gamma-1);cs2=gamma*q[1]/H
    gi=np.linalg.inv(g)[axis,axis];a=W*P([v[axis],-1]);G=P([gi,0,-1])
    bphi=P([B[axis]/W])+Bv*a
    polynomial=H*(1-cs2)*a**4-(H*cs2+bsq)*a*a*G+cs2*bphi*bphi*G
    roots=polynomial.roots()
    # Multiple quartic roots can acquire tiny imaginary parts. Reject genuinely
    # complex roots rather than silently taking the real part of any spectrum.
    if max(abs(roots.imag))>2e-6*max(1,np.sqrt(gi)):
        raise ValueError(f'Nonreal magnetosonic roots: {roots}')
    ms=np.sort(roots.real)
    h=np.sqrt(H+bsq)
    alf=np.sort([(h*W*v[axis]-s*(B[axis]/W+W*Bv*v[axis]))/(h*W-s*W*Bv) for s in (-1,1)])
    physical=np.array([ms[0],alf[0],ms[1],v[axis],ms[2],alf[1],ms[3]])
    return alpha*physical-np.asarray(beta)[axis]

def decompose(A, eigenvalues, light_speed, cluster_tolerance=2e-6):
    """Return R,L and group metadata, with L R = I and q = R (L q).

    A acts on dimensionless primitive perturbations. SVD handles exact/near
    degeneracies without dividing by B_n or the fast/slow speed separation.
    A grouped near-degeneracy is an approximation bounded by its speed spread;
    do not interpret its arbitrary orthonormal basis as unique physical waves.
    """
    A=np.asarray(A,float);lam=np.asarray(eigenvalues,float)
    order=np.argsort(lam);groups=[]
    for j in order:
        if not groups or lam[j]-lam[groups[-1][0]]>cluster_tolerance*light_speed:
            groups.append([int(j)])
        else:groups[-1].append(int(j))
    R=np.empty((7,7));effective=lam.copy();metadata=[]
    for group in groups:
        center=float(np.mean(lam[group]));_,sv,vh=np.linalg.svd(A-center*np.eye(7))
        R[:,group]=vh[-len(group):].T
        effective[group]=center
        metadata.append(dict(families=[FAMILIES[j] for j in group],indices=group,speed=center,
                             speed_spread=float(np.ptp(lam[group])),null_residual=float(sv[-len(group)])))
    condition=float(np.linalg.cond(R))
    if not np.isfinite(condition) or condition>1e7:
        raise ValueError(f'Ill-conditioned eigenbasis: {condition}')
    L=np.linalg.inv(R)
    residual=float(np.max(abs(A@R-R*effective[None,:]))/light_speed)
    if residual>2e-5:
        raise ValueError(f'Inaccurate native eigenbasis: {residual}')
    return dict(speeds=lam,effective_speeds=effective,R=R,L=L,groups=metadata,
                condition=condition,residual=residual)

def outgoing_projector(basis,outward=1):
    """Stationary modes go outward, as in the native matrix-sign convention."""
    lam=basis['effective_speeds'];keep=(outward*lam>=0).astype(float)
    return (basis['R']*keep[None,:])@basis['L']
