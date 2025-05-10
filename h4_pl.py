import pandas as pd
import plotly.express as px

# Load the user‑provided data
df = pd.read_csv('h4_new_data.csv')

# Compute relative energies (dissociation curves) so all functionals start at 0 Ha
df['e_rel'] = df.groupby('xc')['e'].transform(lambda x: x - x.min())

# Create an interactive Plotly line chart
fig = px.scatter(
    df,
    x='r',
    y='e_rel',
    color='xc',
    title='H₄ Dissociation Curves (relative energy, all functionals)',
    labels={'r': 'Bond‑length r (Å)', 'e_rel': 'ΔE (Hartree)'}
)

fig.show()
