import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import numpy as np

def plot_mu_vs_code(code, labels, mu_class, mu, gamma):
    fig = go.Figure()

    fig.add_trace( 
        go.Scatter3d(
            x=code[0,:,:,0].flatten(),
            y=code[0,:,:,1].flatten(), 
            z=code[0,:,:,2].flatten(),
            name='class',
            mode='markers',
            marker=dict(
                size=0.8, 
                #color=labels.astype(int),
                colorscale='Rainbow'
            ) 
        )
    )

    fig.add_trace(
        go.Scatter3d(
            x=mu[:,0],
            y=mu[:,1], 
            z=mu[:,2],
            name='centers',
            mode='markers',
            marker=dict(
                color=mu_class,
                size=gamma*20,
                opacity=0.8,
                colorscale='Rainbow'
            ) 
        )
    )

    fig.update_layout(template="plotly_dark", title="latent space")

    fig.show()