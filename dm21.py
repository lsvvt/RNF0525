import numpy as np
import plotly.graph_objects as go
import density_functional_approximation_dm21 as dm21


# PySCF modules
from pyscf import gto, dft

def compute_h2_dissociation_dm21(basis='cc-pVDZ', start=0.5, end=3.0, steps=20):
    """
    Compute H2 total energies for bond distances from 'start' to 'end' (Å)
    using the DM21 functional (if available in PySCF).
    
    Parameters:
    -----------
    basis : str
        Gaussian basis set name (e.g., 'cc-pVDZ', 'sto-3g', etc.)
    start : float
        Starting H-H distance in angstrom
    end : float
        Final H-H distance in angstrom
    steps : int
        Number of points for scanning the dissociation curve
    
    Returns:
    --------
    bond_lengths : np.ndarray
        Array of H-H bond lengths in angstrom
    energies : np.ndarray
        Computed total energies in Hartree at each bond length
    """
    bond_lengths = np.linspace(start, end, steps)
    energies = []

    for R in bond_lengths:
        # 1. Build the molecule
        mol = gto.Mole()
        mol.atom = f"""
        H 0 0 0
        H 0 0 {R}
        """
        mol.unit = 'Angstrom'
        mol.basis = basis
        mol.spin = 0   # (Z-alpha) - (Z-beta) = 0 for H2
        mol.charge = 0
        mol.build()

        # 2. Set up DFT calculation with DM21 functional
        mf = dft.RKS(mol)
        mf._numint = dm21.NeuralNumInt(dm21.Functional.DM21)

        # 3. Run the SCF calculation
        energy = mf.kernel()
        energies.append(energy)

    return bond_lengths, np.array(energies)

if __name__ == "__main__":
    # Compute H2 dissociation curve
    bond_lengths, energies = compute_h2_dissociation_dm21(
        basis='cc-pVDZ', 
        start=0.5, 
        end=3.0, 
        steps=30
    )

    # Create a Plotly figure
    fig = go.Figure()

    # Add a scatter trace for the energies
    fig.add_trace(
        go.Scatter(
            x=bond_lengths,
            y=energies,
            mode='lines+markers',
            name='DM21 Energy',
            marker=dict(color='blue')
        )
    )

    # Customize layout
    fig.update_layout(
        title="H₂ Dissociation Curve (DM21)",
        xaxis_title="Bond Length (Å)",
        yaxis_title="Total Energy (Hartree)",
        template="plotly_white"
    )

    # Display the interactive figure
    fig.show()
