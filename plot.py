import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px


# Read in the data
df = pd.read_csv("data.csv")

# Filter to the basis sets of interest
df = df[df["basis_set"].isin(["cc-pVQZ", "cc-pVTZ", "cc-pVDZ"])]

fig = make_subplots(rows=1, cols=2, shared_yaxes=True)

for i, basis_set in enumerate(["cc-pVQZ", "cc-pVTZ", "cc-pVDZ"]):
    df_h = df[(df["is_hybrid"] == "hybrid") & (df["basis_set"] == basis_set)]
    df_nh = df[(df["is_hybrid"] == "non hybrid") & (df["basis_set"] == basis_set)]

    color = px.colors.qualitative.Safe

    fig.add_trace(
        go.Bar(x=df_h.xc, y=df_h.mae, name=basis_set, marker_color = color[i]), 1, 1
    )
    fig.add_trace(
        go.Bar(x=df_nh.xc, y=df_nh.mae, name=basis_set, marker_color = color[i]), 1, 2,
    )


# fig.show()

# Update the overall layout and styling
fig.update_layout(barmode='group')
fig.update_layout(
    template="simple_white",
    font=dict(size=16, family="Arial"),
    legend_title_text="Basis set",
    # You might prefer a figure title or leave it out if using a figure caption
    title="CTB22-12ds",
    width=900,  # adjust as needed
    height=600
)

# # Update axis labels and tick formatting
# fig.update_xaxes(
#     title_text="Exchange-Correlation Functional",
#     tickangle=-45,
# )
fig.update_xaxes(title_text="Hybrid", row=1, col=1)
fig.update_xaxes(title_text="Non Hybrid", row=1, col=2)
fig.update_yaxes(title_text="MAE (kJ/mol)")

fig.show()
