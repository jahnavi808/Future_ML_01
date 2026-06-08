import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# -----------------------------------
# PAGE CONFIG
# -----------------------------------

st.set_page_config(
    page_title="Sales Forecasting Dashboard",
    page_icon="📈",
    layout="wide"
)

# -----------------------------------
# TITLE
# -----------------------------------

st.title("📈 Sales & Demand Forecasting Dashboard")
st.markdown(
    """
    Predict future sales using historical business data.
    This dashboard helps business owners plan inventory,
    staffing, and financial decisions.
    """
)

# -----------------------------------
# FILE UPLOAD
# -----------------------------------

uploaded_file = st.file_uploader(
    "Upload Sales Dataset (CSV)",
    type=["csv"]
)

if uploaded_file is not None:

    # Load Data
    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # -----------------------------------
    # DATA CLEANING
    # -----------------------------------

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")

    df = df.dropna()

    # -----------------------------------
    # FEATURE ENGINEERING
    # -----------------------------------

    df["Month"] = df["Date"].dt.month
    df["Year"] = df["Date"].dt.year
    df["Day"] = df["Date"].dt.day

    df["TimeIndex"] = np.arange(len(df))

    X = df[["TimeIndex"]]
    y = df["Sales"]

    # -----------------------------------
    # MODEL TRAINING
    # -----------------------------------

    model = LinearRegression()
    model.fit(X, y)

    predictions = model.predict(X)

    # -----------------------------------
    # METRICS
    # -----------------------------------

    mae = mean_absolute_error(y, predictions)
    mse = mean_squared_error(y, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y, predictions)

    st.subheader("📊 Model Performance")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("MAE", f"{mae:.2f}")
    col2.metric("RMSE", f"{rmse:.2f}")
    col3.metric("R² Score", f"{r2:.2f}")
    col4.metric("Records", len(df))

    # -----------------------------------
    # ACTUAL VS PREDICTED
    # -----------------------------------

    st.subheader("Actual vs Predicted Sales")

    fig, ax = plt.subplots(figsize=(10,5))

    ax.plot(
        df["Date"],
        y,
        marker='o',
        label="Actual Sales"
    )

    ax.plot(
        df["Date"],
        predictions,
        linestyle='--',
        label="Predicted Sales"
    )

    ax.set_xlabel("Date")
    ax.set_ylabel("Sales")
    ax.legend()

    st.pyplot(fig)

    # -----------------------------------
    # FUTURE FORECAST
    # -----------------------------------

    st.subheader("🔮 Future Forecast")

    future_periods = st.slider(
        "Months to Forecast",
        1,
        12,
        6
    )

    future_index = np.arange(
        len(df),
        len(df) + future_periods
    ).reshape(-1,1)

    future_sales = model.predict(future_index)

    last_date = df["Date"].max()

    future_dates = pd.date_range(
    start=last_date,
    periods=future_periods+1,
    freq='ME'
)[1:]

    forecast_df = pd.DataFrame({
        "Date": future_dates,
        "Forecasted Sales": future_sales
    })

    st.dataframe(forecast_df)

    # -----------------------------------
    # FORECAST CHART
    # -----------------------------------

    fig2, ax2 = plt.subplots(figsize=(10,5))

    ax2.plot(
        df["Date"],
        df["Sales"],
        marker='o',
        label="Historical Sales"
    )

    ax2.plot(
        forecast_df["Date"],
        forecast_df["Forecasted Sales"],
        marker='o',
        linestyle='--',
        label="Forecast"
    )

    ax2.set_title("Future Sales Forecast")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Sales")

    ax2.legend()

    st.pyplot(fig2)

    # -----------------------------------
    # BUSINESS INSIGHTS
    # -----------------------------------

    st.subheader("💡 Business Insights")

    growth = (
        (forecast_df["Forecasted Sales"].iloc[-1]
         - df["Sales"].iloc[-1])
        / df["Sales"].iloc[-1]
    ) * 100

    st.success(
        f"Expected sales change over forecast period: {growth:.2f}%"
    )

    st.info(
        """
        Business Use Cases:

        • Inventory Planning

        • Demand Forecasting

        • Budget Preparation

        • Staff Scheduling

        • Revenue Projection
        """
    )

else:
    st.warning("Please upload a CSV file containing Date and Sales columns.")