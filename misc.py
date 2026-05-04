import numpy as np
import plotly.graph_objects as go


def plot_partial_tangent():
    # 1. Setup the Surface
    x = np.linspace(-np.pi, np.pi, 100)
    y = np.linspace(-np.pi, np.pi, 100)
    X, Y = np.meshgrid(x, y)
    Z = np.sin(X) + np.cos(Y)

    # 2. Define the Point of Tangency
    # You can change these values to test different locations
    x0 = np.pi / 2  # Point in x (0.5π approx)
    y0 = np.pi / 4  # Point in y (Fixed value for the slice)

    # Calculate Height at the point
    z0 = np.sin(x0) + np.cos(y0)

    # Calculate the Slope (Partial Derivative wrt x)
    slope = np.cos(x0)  # dz/dx at x0

    # 3. Define the Tangent Line Range
    # We take a window around x0 to see the line clearly
    x_line = np.linspace(x0 - 1.5, x0 + 1.5, 100)
    y_line = np.full_like(x_line, y0)  # Keep y constant

    # Tangent Line Z-coordinates (Linear Approximation)
    z_line = z0 + slope * (x_line - x0)

    # 4. Create the Plot
    fig = go.Figure()

    # Add Surface
    fig.add_trace(go.Surface(
        x=X, y=Y, z=Z,
        opacity=0.8,
        colorscale='Viridis',
        showscale=False,
        name='Surface $f(x,y) = \\sin x + \\cos y$'
    ))

    # Add Tangent Line
    fig.add_trace(go.Scatter3d(
        x=x_line,
        y=y_line,
        z=z_line,
        mode='lines',
        line=dict(color='red', width=5),
        name=f'Tangent at {x0:.2f} (slope={slope:.2f})'
    ))

    # Add Marker for the Point of Contact
    fig.add_trace(go.Scatter3d(
        x=[x0], y=[y0], z=[z0],
        mode='markers',
        marker=dict(size=5, color='gold'),
        name='Contact Point'
    ))

    # Annotations
    fig.update_layout(
        title='Partial Derivative Visualization (dx)',
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z (Height)'
        ),
        margin={'l': 0, 'r': 0, 'b': 0, 't': 50}
    )

    fig.show()


if __name__ == "__main__":
    plot_partial_tangent()