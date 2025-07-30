import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Policy Pattern Explorer", layout="wide")

st.title("Insurance Policy Pattern Explorer (AI-Ready)")

# File Upload - CSV or Excel
uploaded_file = st.file_uploader("Upload Policy-Level Data (.csv or .xlsx)", type=["csv", "xlsx"])
if uploaded_file:
    # Read based on file type
    if uploaded_file.name.endswith(".csv"):
        
        df = pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith(".xlsx"):
        df = pd.read_excel(uploaded_file)
    st.write("📋 Columns found in file:", df.columns.tolist())
    df.columns = df.columns.str.strip().str.upper()
    df["NZ_RES_IF_94"] = pd.to_numeric(df["NZ_RES_IF_94"], errors="coerce")



    # Define relevant columns
    rating_factors = ["CL_PBAND", "CL_PFREQ", "CL_PPT", "CL_STATUS", "ANNUAL_PREM", "ENTRY_MONTH", "ENTRY_YEAR"]
    outcome_vars = ["NZ_RES_IF_94", "RES_GP_PUPS", "PREM_GP_PUPS"]

    # Add calculated columns
    df["RATIO_PUP"] = df["RES_GP_PUPS"] / (df["PREM_GP_PUPS"] + 1e-6)
    df["DIFF_PUP"] = df["RES_GP_PUPS"] - df["PREM_GP_PUPS"]

    st.markdown("### 👇 Select a pattern you want to explore")

    pattern = st.selectbox("Choose Analysis Type", [
        "Top Annual Premium bands by NZ_RES_IF_94",
        "Which PFREQ + PBAND combinations have high (RES - PREM)?",
        "Policies with highest RES_GP_PUPS / PREM_GP_PUPS ratio",
        "Which rating factor has highest variance?",
        "Show top 10 clusters using KMeans (preview)"
    ])

    if pattern == "Top Annual Premium bands by NZ_RES_IF_94":
        result = df.groupby("ANNUAL_PREM")["NZ_RES_IF_94"].mean().sort_values(ascending=False).head(10)
        st.bar_chart(result)

    elif pattern == "Which PFREQ + PBAND combinations have high (RES - PREM)?":
        result = df.groupby(["CL_PFREQ", "CL_PBAND"])["DIFF_PUP"].mean().sort_values(ascending=False).head(10)
        st.dataframe(result)

    elif pattern == "Policies with highest RES_GP_PUPS / PREM_GP_PUPS ratio":
        result = df[["POL_NUMBER", "RATIO_PUP"]].sort_values(by="RATIO_PUP", ascending=False).head(10)
        st.dataframe(result)

    elif pattern == "Which rating factor has highest variance?":
        numeric_ratings = df[rating_factors].select_dtypes(include=[np.number])
        result = numeric_ratings.var().sort_values(ascending=False)
        st.bar_chart(result)

    elif pattern == "Show top 10 clusters using KMeans (preview)":
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler

        numeric_cols = df[["ANNUAL_PREM", "RATIO_PUP", "NZ_RES_IF_94", "ENTRY_MONTH", "ENTRY_YEAR"]]
        scaled = StandardScaler().fit_transform(numeric_cols)
        kmeans = KMeans(n_clusters=3, random_state=42)
        df["Cluster"] = kmeans.fit_predict(scaled)
        st.dataframe(df[["POL_NUMBER", "ANNUAL_PREM", "RATIO_PUP", "NZ_RES_IF_94", "Cluster"]].head(20))
        st.markdown("✅ Cluster labels added as `Cluster` column in DataFrame.")

    st.markdown("---")
    st.dataframe(df.head(50))

else:
    st.info("Please upload your `.csv` or `.xlsx` file to begin analysis.")
