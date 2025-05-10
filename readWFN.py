import numpy as np
from pyscf import gto, lib, scf, dft
import os
import re
import sys
from iodata import load_one
import tutil
import h5py
import cclib
from multiprocessing import Process, Queue
import shutil
import subprocess
import argparse
from density_functional_approximation_dm21 import compute_hfx_density

from config import *

import mod_NumInt as mni
ni = dft.numint.NumInt()

lib.num_threads(24)

def read2mol(wfn_fn, log_fn, basis_name):
    data_log = cclib.io.ccread(log_fn)

    mol_iod = load_one(wfn_fn)
    atom = np.column_stack((mol_iod.atnums.astype(int), mol_iod.atcoords.astype(str)))
    atom = str(atom).replace("[", "").replace("]", "").replace("'", "")
    print(data_log.charge, data_log.mult, data_log)
    mol = gto.Mole(atom = atom, charge = data_log.charge, spin = data_log.mult - 1, basis = basis_name, unit = "Bohr")
    mol.build()

    del mol_iod
    return mol


def create_dataset(out_path, ans):
    for key in ans:
        print("Saving feature ", key, "; shape = ", ans[key].shape)
        with h5py.File(f"{out_path}_{key}.h5", "w") as f:
            ds = f.create_dataset(key, data=ans[key])


def read_grid(path):
    ans = []
    for key in ["coords", "weights"]:
        with h5py.File(f"{path}_{key}.h5", "r") as f:
            ans.append(f[key][:])

    return ans


def make_feature_ans(dm, mol, mf, old_grids = None):

    grids = mf.grids

    if old_grids is None:
        pass
    else:
        grids.coords = old_grids[0]
        grids.weights = old_grids[1]

    ans = {}

    if COMPUTE_BASE:
        rho_a, rho_b = make_rho(dm, mol, grids)

        sigmaaa = rho_a[1] * rho_a[1] + rho_a[2] * rho_a[2] + rho_a[3] * rho_a[3]
        sigmaab2 = (rho_a[1] + rho_b[1])**2 + (rho_a[2] + rho_b[2])**2 + (rho_a[3] + rho_b[3])**2
        sigmabb = rho_b[1] * rho_b[1] + rho_b[2] * rho_b[2] + rho_b[3] * rho_b[3]

        ex_lda, _ = ni.eval_xc("LDA_X", (rho_a[:-1], rho_b[:-1]), spin=1,
                                            relativity=0, deriv=1,
                                            verbose=4)[:2]

        tmp_ans = {
            "rhoa" : rho_a,
            "rhob" : rho_b,
            "norm_grad_a" : sigmaaa,
            "norm_grad_b" : sigmabb,
            "norm_grad" : sigmaab2,
            "tau_a" : rho_a[5],
            "tau_b" : rho_b[5],
            "w_a" : rho_a[6],
            "w_b" : rho_b[6],
            "e_lda" : ex_lda,
            }
        ans = ans | tmp_ans
    
    if COMPUTE_LHF:
        ao = ni.eval_ao(mol, grids.coords, deriv=0)
        exxa, exxb = [], []
        fxxa, fxxb = [], []
        for omega in [0.0, 0.4]:
            hfx_results = compute_hfx_density.get_hf_density(mol, dm, coords=grids.coords, omega=omega, deriv=1, ao=ao)
            exxa.append(hfx_results.exx[0])
            exxb.append(hfx_results.exx[1])
            #fxxa.append(hfx_results.fxx[0])
            #fxxb.append(hfx_results.fxx[1])

        tmp_ans = {
            "exxa00" : exxa[0],
            "exxb00" : exxb[0],
            "exxa04" : exxa[1],
            "exxb04" : exxb[1],
            #"fxxa00" : fxxa[0],
            #"fxxb00" : fxxb[0],
            #"fxxa04" : fxxa[1],
            #"fxxb04" : fxxb[1],
            }
        ans = ans | tmp_ans

    return ans


def make_rho(dm, mol, grids):
    xctype = "MGGA"

    dma, dmb = mni._format_uks_dm(dm)

    dma = lib.hermi_sum(dma, axes=(0,2,1)) * .5
    dmb = lib.hermi_sum(dmb, axes=(0,2,1)) * .5
    hermi = 1

    nao = dma.shape[-1]

    aow = None

    ao_deriv = 2
    rho_a = np.array([])
    rho_b = np.array([])
    for ao, mask, weight, coords \
            in ni.block_loop(mol, grids, nao, ao_deriv, max_memory):
        aow = np.ndarray(ao[0].shape, order='F', buffer=aow)
        if len(rho_a) == 0:
            rho_a = mni.eval_rho(mol, ao, dma, mask, xctype, hermi)
            rho_b = mni.eval_rho(mol, ao, dmb, mask, xctype, hermi)
        else:
            rho_a = np.column_stack([rho_a, mni.eval_rho(mol, ao, dma, mask, xctype, hermi)])
            rho_b = np.column_stack([rho_b, mni.eval_rho(mol, ao, dmb, mask, xctype, hermi)])

    return rho_a, rho_b


def run_scf(path, out_path, basis_name):
    folder = os.path.basename(path)
    wfn_fn = path + ".wfn"
    log_fn = path + POSTFIX
    print("Hello!", wfn_fn, log_fn)

    print(wfn_fn, log_fn)

    if os.path.isfile(wfn_fn) and os.path.isfile(log_fn):
        print("Out path for substance: ", out_path)
        os.makedirs(out_path, exist_ok=True)
        mol = read2mol(wfn_fn, log_fn, basis_name)
        mol.verbose = verbose
        mf = scf.UKS(mol)
        mf.xc = xc_functional
        mf.grids.level = grid_level
        mf.grids.radi_method = radi_method
        mf.max_cycle = 0
        mo_coeff, mo_occ, mo_energy, dm = tutil.readwfn(wfn_fn, mol, makerdm=True)
        if os.path.isfile("%s_coords.h5" % os.path.join(out_path, folder)):
            print("I start from grid")
            old_grid = read_grid(os.path.join(out_path, folder))
            ans = make_feature_ans(dm, mol, mf, old_grids = old_grid)
            create_dataset(os.path.join(out_path, folder), ans)
        else:
            print("I successfully read wfn parameters!")
            print("mo_coeff shape: ", mo_coeff.shape)
            print("mo_occ shape: ", mo_occ.shape)
            print("mo_energy shape: ", mo_energy.shape)
            print("dm shape: ", dm.shape)

            mf.conv_tol = 1e-7
            mf.run(dm)
            if True:
                mDFTe = mf.e_tot - mf.scf_summary["exc"]

                energy_ans = {
                    "eTot" : mf.e_tot,
                    "mDFTe" : mDFTe,
                    "coords" : mf.grids.coords,
                    "weights" : mf.grids.weights,
                }
                create_dataset(os.path.join(out_path, folder), energy_ans)
                ans = make_feature_ans(dm, mol, mf)
                create_dataset(os.path.join(out_path, folder), ans)
    else:
        raise ValueError('log or wfn file not found')


