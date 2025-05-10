from pyscf import dft
import numpy as np
ni = dft.numint

def eval_rho(mol, ao, dm, non0tab=None, xctype='LDA', hermi=0, verbose=None):
    dat = {4 : [0,0], 5 : [0, 1], 6 : [0, 2], 7 : [1, 1], 8 : [1, 2], 9 : [2, 2]}

    xctype = xctype.upper()
    ngrids, nao = ao[0].shape

    shls_slice = (0, mol.nbas)
    ao_loc = mol.ao_loc_nr()
    rho = np.empty((7,ngrids))
    tau_idx = 5
    c0 = ni._dot_ao_dm(mol, ao[0], dm, non0tab, shls_slice, ao_loc)
    #:rho[0] = np.einsum('pi,pi->p', ao[0], c0)
    rho[0] = ni._contract_rho(ao[0], c0)

    rho[5] = 0
    for i in range(1, 4):
        c1 = ni._dot_ao_dm(mol, ao[i], dm, non0tab, shls_slice, ao_loc)
        #:rho[5] += np.einsum('pi,pi->p', c1, ao[i])
        rho[5] += ni._contract_rho(ao[i], c1)

        #:rho[i] = np.einsum('pi,pi->p', c0, ao[i])
        rho[i] = ni._contract_rho(ao[i], c0)
        if hermi:
            rho[i] *= 2
        else:
            rho[i] += ni._contract_rho(c1, ao[0])

### Laplacian
    XX, YY, ZZ = 4, 7, 9
    ao2 = ao[XX] + ao[YY] + ao[ZZ]
    # \nabla^2 rho
    #:rho[4] = np.einsum('pi,pi->p', c0, ao2)
    rho[4] = ni._contract_rho(ao2, c0)
    rho[4] += rho[5]
    if hermi:
        rho[4] *= 2
    else:
        c2 = ni._dot_ao_dm(mol, ao2, dm, non0tab, shls_slice, ao_loc)
        rho[4] += ni._contract_rho(ao[0], c2)
        rho[4] += rho[5]

    # tau = 1/2 (\nabla f)^2
    rho[5] *= .5

### w
    if not hermi:
        raise ValueError('Not implemented w for non hermi')
    tmp = np.empty((3,3,ngrids))
    for i in range(4, 10):
        c1 = ni._dot_ao_dm(mol, ao[dat[i][0] + 1], dm, non0tab, shls_slice, ao_loc)
        l = ni._contract_rho(ao[i], c0) * 2 + ni._contract_rho(ao[dat[i][1] + 1], c1) * 2
        tmp[dat[i][0]][dat[i][1]] = l
        tmp[dat[i][1]][dat[i][0]] = l

    rho[6] = np.einsum("ip,ijp,jp->p", rho[1:4], tmp, rho[1:4]) ## 17/12
    return rho

def _format_uks_dm(dms):
    if isinstance(dms, np.ndarray) and dms.ndim == 2:  # RHF DM
        dma = dmb = dms * .5
    else:
        dma, dmb = dms
    return dma, dmb
