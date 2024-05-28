import plotly.graph_objs as go
from utils import get_history


BACKGROUND_COLOR = "#002b36"
GRID_COLOR = "#2b6472"
FONT_COLOR = "#839496"
YAXIS_COLOR_1 = "#baffc9"
YAXIS_COLOR_2 = "#ffb3ba"


def plot_history(client, market_symbol, step_sec, start_dt, end_dt, title=""):
    df = get_history(client, market_symbol, step_sec, start_dt, end_dt)
    fig = go.Figure()

    # Add the btceur_close line
    fig.add_trace(
        go.Scatter(x=df.datetime, y=df.close, name="close", line=dict(color=YAXIS_COLOR_1))
    )

    # Add the btceur_volume line with a secondary y-axis
    fig.add_trace(
        go.Scatter(
            x=df.datetime,
            y=df.volume,
            name="volume",
            yaxis="y2",
            line=dict(dash="dot", color=YAXIS_COLOR_2),
        )
    )

    # Set the layout with two y-axes
    fig.update_layout(
        title=title,
        yaxis=dict(title="close", color=YAXIS_COLOR_1, gridcolor=GRID_COLOR),
        yaxis2=dict(title="volume", overlaying="y", side="right", color=YAXIS_COLOR_2),
        legend=dict(traceorder="reversed", font=dict(color=FONT_COLOR)),
        plot_bgcolor=BACKGROUND_COLOR,
        paper_bgcolor=BACKGROUND_COLOR,
        font=dict(color=FONT_COLOR),
        yaxis2_showgrid=False,
        xaxis=dict(gridcolor=GRID_COLOR),
        xaxis_rangeslider_visible=True,
    )

    fig.show()


def plot_candlestick_history(client, market_symbol, step_sec, start_dt, end_dt, title=""):
    df = get_history(client, market_symbol, step_sec, start_dt, end_dt)
    fig = go.Figure(
        data=[
            go.Candlestick(
                x=df["datetime"],
                open=df["open"],
                high=df["high"],
                low=df["low"],
                close=df["close"],
                increasing_line_color=YAXIS_COLOR_1,
                decreasing_line_color=YAXIS_COLOR_2,
            )
        ]
    )

    fig.update_layout(
        title=title,
        yaxis=dict(title="Price", color=YAXIS_COLOR_1, gridcolor=GRID_COLOR, autorange=True),
        legend=dict(traceorder="reversed", font=dict(color=FONT_COLOR)),
        plot_bgcolor=BACKGROUND_COLOR,
        paper_bgcolor=BACKGROUND_COLOR,
        font=dict(color=FONT_COLOR),
        xaxis=dict(title="Time", gridcolor=GRID_COLOR),
        xaxis_rangeslider_visible=False,
    )

    fig.show()
