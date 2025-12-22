
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    f1_score, roc_auc_score, fbeta_score,
    roc_curve, precision_recall_curve, average_precision_score,
    precision_score, recall_score
)
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.calibration import CalibratedClassifierCV
import os

# =========================
# Load & prepare data
# =========================

# Correct path relative to src/streamlit/
DATA_PATH = "../../data/raw/insurance_claims.csv"

# Check if file exists, if not try absolute path or warn
if not os.path.exists(DATA_PATH):
    # Try alternate location if running from root
    if os.path.exists("data/raw/insurance_claims.csv"):
        DATA_PATH = "data/raw/insurance_claims.csv"
    else:
        st.error(f"Data file not found at {DATA_PATH}. Please check the path.")
        st.stop()

df = pd.read_csv(DATA_PATH)
df_raw = pd.read_csv(DATA_PATH)

df['witnesses'] = pd.to_numeric(df['witnesses'])
df['policy_bind_date'] = pd.to_datetime(df['policy_bind_date'])
df['bind_year']  = df['policy_bind_date'].dt.year
df['bind_month'] = df['policy_bind_date'].dt.month
df['incident_date'] = pd.to_datetime(df['incident_date'])
df['incident_weekday'] = df['incident_date'].dt.day_name()

df['is_weekend'] = df['incident_date'].dt.dayofweek >= 5
holiday_days = ['2015-01-01 00:00:00', '2015-01-20 00:00:00', '2015-02-17 00:00:00', '2015-03-26 00:00:00']
holiday_days = pd.to_datetime(holiday_days)

df['is_holiday'] = (
    (df['incident_date'].dt.dayofweek >= 5) |
    (df['incident_date'].isin(holiday_days))
)

df['days_between'] = (df['incident_date'] - df['policy_bind_date']).dt.days
df[['csl_person', 'csl_accident']] = df['policy_csl'].str.split('/', expand=True)
df['csl_person']   = pd.to_numeric(df['csl_person'], errors='coerce')
df['csl_accident'] = pd.to_numeric(df['csl_accident'], errors='coerce')
df["umbrella_limit"] = df["umbrella_limit"].abs()
df['incident_place'] = df['incident_city'] + '_' + df['incident_state']
df['loc_type'] = df['incident_location'].apply(lambda x: x.split()[-1])
df['vehicle_age'] = 2015 - df['auto_year']

df['incident_time_category'] = pd.cut(
    df['incident_hour_of_the_day'],
    bins=[0, 6, 12, 18, 24],
    labels=["Night", "Morning", "Afternoon", "Evening"],
    right=False
)

premium_models = {
    "M5", "X5", "X6", "3 Series",
    "E400", "C300", "ML350",
    "A3", "A5"
}
low_models = {"92x", "93", "95"}

def map_vehicle_tier(model):
    if model in premium_models:
        return "premium"
    elif model in low_models:
        return "low"
    else:
        return "mid"

df["vehicle_tier"] = df["auto_model"].apply(map_vehicle_tier)

df["age_group"] = pd.cut(
    df["age"],
    bins=[0, 25, 35, 50, 70, 100],
    labels=["0-25", "26-35", "36-50", "51-70", "70+"]
)

df["vehicle_age_group"] = pd.cut(
    df["vehicle_age"],
    [-1, 3, 7, 12, 30],
    labels=["0-3", "4-7", "8-12", "13+"]
)

df["csl_ratio"] = df["csl_accident"] / df["csl_person"]

df["sev_collision_mismatch"] = (
    (df["collision_type"] == "Rear Collision") & (df["incident_severity"] == "Trivial Damage")
) | (
    (df["collision_type"] == "Side Collision") & (df["incident_severity"] == "Minor Damage")
)
df["sev_collision_mismatch"] = df["sev_collision_mismatch"].astype(int)

df["inj_claim_unusual"] = (
    (df["bodily_injuries"] >= 2) & (df["incident_severity"] == "Minor Damage")
).astype(int)

df.police_report_available.replace({'?': 'Missing'}, inplace=True)
df['property_damage'] = np.where(
    (df['property_damage'] == '?') & (df['property_claim'] > 0),
    'YES',
    df['property_damage']
)
df['property_damage'] = np.where(
    (df['property_damage'] == '?') & (df['property_claim'] == 0),
    'NO',
    df['property_damage']
)

df.loc[(df['incident_type'] == 'Vehicle Theft') | (df['incident_type'] == 'Parked Car'), 'collision_type'] = 'Other'
df['authorities_contacted'] = df['authorities_contacted'].fillna('?')
df.loc[(df['incident_type'] == 'Vehicle Theft') | (df['incident_type'] == 'Parked Car'), 'authorities_contacted'] = 'Police'

cols = ['is_holiday', 'incident_time_category', 'is_weekend', 'vehicle_age_group', "age_group"]
df[cols] = df[cols].astype('object')

df.drop(
    columns=[
        'policy_number','insured_zip', 'total_claim_amount','_c39','incident_date',
        'is_weekend','auto_year','policy_bind_date','incident_state','incident_city',
        'auto_model', 'auto_make','incident_hour_of_the_day','incident_location'
    ],
    axis=1,
    inplace=True
)

# =========================
# Streamlit layout
# =========================
st.set_page_config(page_title="Insurance Fraud Detection", layout="wide")

st.title("Insurance Claim Fraud Detection")
st.sidebar.title("Table of contents")
pages=["Exploration", "Data Visualization", "Modelling", "SHAP Analysis", "Claim Risk Analyzer"]
page = st.sidebar.radio("Go to", pages)

st.sidebar.markdown(
    """
    ### 🔗 External links
    - [🚗 Auto Claim Fraud Detection App](https://lowentropy.works/pages/auto_claim_fraud.html)
    - [🤗 HuggingFace demo](https://huggingface.co/spaces/MyNameIsTatiBond/fraud-detector)
    """
)

# =========================
# Utility Functions
# =========================
def create_summary_statistics(df, fraud_col='fraud_reported'):
    """Generate comprehensive summary statistics"""
    summary = {
        'Total Claims': len(df),
        'Fraud Claims': (df[fraud_col] == 'Y').sum(),
        'Fraud Rate': f"{(df[fraud_col] == 'Y').mean():.2%}",
        'Features': len(df.columns),
        'Missing Values': df.isna().sum().sum(),
        'Categorical Features': len(df.select_dtypes(include=['object']).columns),
        'Numerical Features': len(df.select_dtypes(include=['int64', 'float64']).columns)
    }
    return summary

# =========================
# Page 1: Exploration
# =========================
if page == pages[0]:
    st.write("## 🏠 Data Exploration & Analysis")

    # Summary statistics in cards
    summary = create_summary_statistics(df_raw)
    cols = st.columns(4)
    with cols[0]:
        st.markdown(f'<div class="metric-card"><h3>Total Claims</h3><h2>{summary["Total Claims"]:,}</h2></div>', unsafe_allow_html=True)
    with cols[1]:
        st.markdown(f'<div class="metric-card"><h3>Fraud Rate</h3><h2>{summary["Fraud Rate"]}</h2></div>', unsafe_allow_html=True)
    with cols[2]:
        st.markdown(f'<div class="metric-card"><h3>Features</h3><h2>{summary["Features"]}</h2></div>', unsafe_allow_html=True)
    with cols[3]:
        st.markdown(f'<div class="metric-card"><h3>Missing Values</h3><h2>{summary["Missing Values"]:,}</h2></div>', unsafe_allow_html=True)

    # Data preview with tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📋 Data Sample", "📊 Statistics", "🎯 Target Analysis", "🔍 Data Quality"])

    with tab1:
        st.subheader("Sample of the dataset")
        sample_size = st.slider("Sample size", 5, 100, 10)
        st.dataframe(df_raw.head(sample_size), use_container_width=True)

        if st.button("📥 Download sample data"):
            csv = df_raw.head(100).to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name="insurance_claims_sample.csv",
                mime="text/csv"
            )

    with tab2:
        st.subheader("Descriptive Statistics")
        stat_type = st.radio("Statistics type", ["All data", "Fraud cases only", "Non-fraud cases only"])

        if stat_type == "Fraud cases only":
            stats_df = df_raw[df_raw['fraud_reported'] == 'Y'].describe()
        elif stat_type == "Non-fraud cases only":
            stats_df = df_raw[df_raw['fraud_reported'] == 'N'].describe()
        else:
            stats_df = df_raw.describe()

        st.dataframe(stats_df, use_container_width=True)

    with tab3:
        st.subheader("Target Variable Distribution")

        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots(figsize=(4, 3))
            sns.countplot(x='fraud_reported', data=df_raw, ax=ax)
            ax.set_title("Distribution of Reported Fraud")
            st.pyplot(fig)

        with col2:
            fraud_stats = df_raw['fraud_reported'].value_counts()
            st.dataframe(fraud_stats, use_container_width=True)


    with tab4:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Data Quality Check")
            # Missing values analysis
            missing_values = df_raw.isna().sum()
            missing_percentage = (missing_values / len(df_raw) * 100).round(2)

            missing_df = pd.DataFrame({
                'Column': missing_values.index,
                'Missing Count': missing_values.values,
                'Missing %': missing_percentage.values
            }).sort_values('Missing Count', ascending=False)

            st.dataframe(missing_df[missing_df['Missing Count'] > 0], use_container_width=True)
        with col2:
            # Data types overview
            st.subheader("Data Types Overview")
            dtype_counts = df_raw.dtypes.value_counts()
            st.dataframe(dtype_counts, use_container_width=True)

# =========================
# Page 2: Data Visualization
# =========================
elif page == pages[1]:
    st.write("### 📊 Interactive Data Visualization")

    # -------------------------
    # Column categorization
    # -------------------------
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    num_cols = df.select_dtypes(include=['int64', 'float64', 'int32']).columns.tolist()

    plot_type = st.selectbox(
        "Select plot type",
        [
            "Count plot (categorical)",
            "Fraud rate bar plot",
            "Box plot (numerical vs fraud)",
            "Histogram",
            "Scatter plot",
            "Facet plot",
            "Categorical association (Cramér’s V)",
            "Correlation heatmap (numerical)"
        ]
    )

    st.markdown("---")

    # =========================
    # COUNT PLOT
    # =========================
    if plot_type == "Count plot (categorical)":
        x_col = st.selectbox("Select categorical variable", cat_cols)

        fig, ax = plt.subplots(figsize=(5, 3))
        sns.countplot(
            x=x_col,
            hue='fraud_reported',
            data=df,
            ax=ax
        )
        ax.set_title(f"Fraud distribution by {x_col}")
        ax.tick_params(axis='x', rotation=45)
        st.pyplot(fig, use_container_width=False, clear_figure=True)

    # =========================
    # FRAUD RATE BAR PLOT
    # =========================
    elif plot_type == "Fraud rate bar plot":
        x_col = st.selectbox("Select categorical variable", cat_cols)

        rate_df = (
            df.groupby(x_col)['fraud_reported']
              .apply(lambda x: (x == 'Y').mean()
              .reset_index(name='fraud_rate')
        ))

        # Fix potentially broken groupby apply reset index logic
        # Re-doing properly for safety
        rate_series = df.groupby(x_col)['fraud_reported'].apply(lambda x: (x == 'Y').mean())
        if isinstance(rate_series, pd.Series):
             rate_df = rate_series.reset_index(name='fraud_rate')
        else:
             rate_df = rate_series

        fig, ax = plt.subplots(figsize=(5, 3))
        sns.barplot(
            x=x_col,
            y='fraud_rate',
            data=rate_df,
            ax=ax
        )
        ax.set_ylim(0, 1)
        ax.set_title(f"Fraud rate by {x_col}")
        ax.tick_params(axis='x', rotation=45)
        st.pyplot(fig, use_container_width=False, clear_figure=True)

    # =========================
    # BOX PLOT
    # =========================
    elif plot_type == "Box plot (numerical vs fraud)":
        y_col = st.selectbox("Select numerical variable", num_cols)

        fig, ax = plt.subplots(figsize=(5, 3))
        sns.boxplot(
            x='fraud_reported',
            y=y_col,
            data=df,
            ax=ax
        )
        ax.set_title(f"{y_col} vs Fraud")
        st.pyplot(fig, use_container_width=False, clear_figure=True)

    # =========================
    # HISTOGRAM
    # =========================
    elif plot_type == "Histogram":
        x_col = st.selectbox("Select numerical variable", num_cols)
        bins = st.slider("Number of bins", 10, 100, 30)

        fig, ax = plt.subplots(figsize=(5, 3))
        sns.histplot(
            df[x_col],
            bins=bins,
            kde=True,
            ax=ax
        )
        ax.set_title(f"Distribution of {x_col}")
        st.pyplot(fig, use_container_width=False, clear_figure=True)

    # =========================
    # SCATTER PLOT
    # =========================
    elif plot_type == "Scatter plot":
        x_col = st.selectbox("X-axis (numerical)", num_cols)
        y_col = st.selectbox("Y-axis (numerical)", num_cols)

        fig, ax = plt.subplots(figsize=(5, 3))
        sns.scatterplot(
            x=x_col,
            y=y_col,
            hue='fraud_reported',
            data=df,
            alpha=0.6,
            ax=ax
        )
        ax.set_title(f"{x_col} vs {y_col}")
        st.pyplot(fig, use_container_width=False, clear_figure=True)


    elif plot_type == "Facet plot":
        st.markdown("### 📐 Facet Plot")

        facet_kind = st.selectbox(
            "Facet plot type",
            ["Count plot", "Fraud rate bar", "Box plot"]
        )

        facet_col = st.selectbox(
            "Facet by (column)",
            cat_cols
        )

        # =========================
        # FACET COUNT PLOT
        # =========================
        if facet_kind == "Count plot":
            x_col = st.selectbox("X-axis (categorical)", cat_cols)

            g = sns.catplot(
                data=df,
                x=x_col,
                hue="fraud_reported",
                col=facet_col,
                kind="count",
                col_wrap=3,
                height=3,
                aspect=1
            )

            g.set_titles(col_template="{col_name}")
            g.set_xticklabels(rotation=45)
            st.pyplot(g.fig, use_container_width=False, clear_figure=True)

        # =========================
        # FACET FRAUD RATE
        # =========================
        elif facet_kind == "Fraud rate bar":
            x_col = st.selectbox("X-axis (categorical)", cat_cols)

            rate_df = (
                df.groupby([facet_col, x_col])['fraud_reported']
                  .apply(lambda x: (x == 'Y').mean())
                  .reset_index(name='fraud_rate')
            )

            g = sns.catplot(
                data=rate_df,
                x=x_col,
                y="fraud_rate",
                col=facet_col,
                kind="bar",
                col_wrap=3,
                height=3,
                aspect=1
            )

            g.set_titles(col_template="{col_name}")
            g.set(ylim=(0, 1))
            g.set_xticklabels(rotation=45)
            st.pyplot(g.fig, use_container_width=False, clear_figure=True)

        # =========================
        # FACET BOX PLOT
        # =========================
        elif facet_kind == "Box plot":
            y_col = st.selectbox("Numerical variable", num_cols)

            g = sns.catplot(
                data=df,
                x="fraud_reported",
                y=y_col,
                col=facet_col,
                kind="box",
                col_wrap=3,
                height=3,
                aspect=1
            )

            g.set_titles(col_template="{col_name}")
            st.pyplot(g.fig, use_container_width=False, clear_figure=True)

    elif plot_type == "Categorical association (Cramér’s V)":
        st.markdown("### 🔗 Categorical Association (Cramér’s V)")
        st.write(
            "Cramér’s V measures the **strength of association** between categorical variables "
            "(0 = no association, 1 = perfect association)."
        )

        from scipy.stats import chi2_contingency

        # =========================
        # Cramér’s V function
        # =========================
        def cramers_v(x, y):
            if isinstance(x, pd.DataFrame):
                x = x.iloc[:, 0]
            if isinstance(y, pd.DataFrame):
                y = y.iloc[:, 0]

            valid = pd.DataFrame({'x': x, 'y': y}).dropna()
            if valid.empty or valid['x'].nunique() < 2 or valid['y'].nunique() < 2:
                return np.nan

            confusion = pd.crosstab(valid['x'].astype(str), valid['y'].astype(str))
            chi2 = chi2_contingency(confusion)[0]
            n = confusion.values.sum()
            phi2 = chi2 / n
            r, k = confusion.shape

            phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
            rcorr = r - ((r-1)**2)/(n-1)
            kcorr = k - ((k-1)**2)/(n-1)
            denom = min((kcorr-1), (rcorr-1))

            return np.sqrt(phi2corr / denom) if denom > 0 else np.nan

        # =========================
        # Controls
        # =========================
        cat_cols = df.select_dtypes(include='object').columns.tolist()

        selected_cols = st.multiselect(
            "Select categorical variables",
            cat_cols,
            default=cat_cols[:8]  # sensible default
        )

        min_strength = st.slider(
            "Highlight associations stronger than",
            0.0, 1.0, 0.2, 0.05
        )

        if len(selected_cols) < 2:
            st.warning("Select at least two categorical variables.")
            st.stop()

        # =========================
        # Compute matrix
        # =========================
        cramer_matrix = pd.DataFrame(
            index=selected_cols,
            columns=selected_cols,
            dtype=float
        )

        with st.spinner("Computing Cramér’s V matrix…"):
            for c1 in selected_cols:
                for c2 in selected_cols:
                    cramer_matrix.loc[c1, c2] = cramers_v(df[c1], df[c2])

        # Mask weak associations
        masked_matrix = cramer_matrix.where(cramer_matrix >= min_strength)

        # =========================
        # Heatmap
        # =========================
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(
            masked_matrix.astype(float),
            cmap="coolwarm",
            vmin=0,
            vmax=1,
            square=True,
            linewidths=0.5,
            ax=ax
        )

        ax.set_title("Cramér’s V – Categorical Associations", fontsize=14)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
        st.pyplot(fig, use_container_width=False, clear_figure=True)

        # =========================
        # Interpretation helper
        # =========================
        st.markdown("""
        **Interpretation guide:**
        - **0.00 – 0.10** → very weak
        - **0.10 – 0.30** → weak
        - **0.30 – 0.50** → moderate
        - **> 0.50** → strong association

        Use this to:
        - detect redundant categorical features
        - identify strong fraud-related signals
        - guide feature selection & encoding
        """)


    elif plot_type == "Correlation heatmap (numerical)":
        st.markdown("### 🔥 Correlation Heatmap (Numerical Features)")

        # Select numerical columns
        num_cols = df.select_dtypes(include=['int64', 'float64', 'int32']).columns.tolist()

        selected_nums = st.multiselect(
            "Select numerical variables",
            num_cols,
            default=num_cols[:10]  # avoid huge plots by default
        )

        if len(selected_nums) < 2:
            st.warning("Please select at least two numerical variables.")
            st.stop()

        corr_method = st.radio(
            "Correlation method",
            ["pearson", "spearman"],
            horizontal=True
        )

        # Compute correlation
        corr = df[selected_nums].corr(method=corr_method)

        # Plot
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(
            corr,
            cmap="coolwarm",
            annot=False,
            linewidths=0.5,
            square=True,
            cbar_kws={"label": "Correlation"}
        )

        ax.set_title(f"{corr_method.capitalize()} Correlation Heatmap", fontsize=14)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

        st.pyplot(fig, use_container_width=False, clear_figure=True)



# =========================
# Page 3: Modelling
# =========================
elif page == pages[2]:
    st.write("### Modelling")

    # =========================
    # Sidebar controls
    # =========================
    st.sidebar.subheader("Modelling options")

    test_size = st.sidebar.slider("Test size", 0.1, 0.4, 0.2, 0.05)
    threshold_default = st.sidebar.slider(
        "Initial decision threshold", 0.05, 0.95, 0.2, 0.01
    )

    # =========================
    # 1) Preprocessing (NO SMOTE)
    # =========================
    X = df.drop(columns=['fraud_reported'], axis=1)
    y = df['fraud_reported'].map({'Y': 1, 'N': 0})

    num_cols = X.select_dtypes(include=['int64', 'float64', 'int32']).columns
    cat_cols = X.select_dtypes(include=['object']).columns

    ohe = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
    X_cat = pd.DataFrame(
        ohe.fit_transform(X[cat_cols]),
        columns=ohe.get_feature_names_out(cat_cols),
        index=X.index
    )

    X_num = X[num_cols].copy()
    X_final = pd.concat([X_num, X_cat], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(
        X_final, y, test_size=test_size, random_state=42, stratify=y
    )

    # =========================
    # 2) Model selection
    # =========================

    Best_parameters = {'n_estimators': 400, 'max_depth': 5, 'min_samples_split': 17, 'min_samples_leaf': 1, 'max_features': None, 'bootstrap': True}
    best_rf = RandomForestClassifier(
        #**study.best_params,
        **Best_parameters,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )


    best_parameters = {'n_estimators': 800, 'learning_rate': 0.009395986539899289, 'max_depth': 6, 'min_child_weight': 2.155649428219426, 'gamma': 4.178059046900583, 'subsample': 0.8286853576043973,
    'colsample_bytree': 0.9556448135919854, 'reg_alpha': 4.857224881320965, 'reg_lambda': 12.48248260697908, 'scale_pos_weight': 3.1873147325141504}
    best_xgb = XGBClassifier(
        **best_parameters,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
        tree_method="hist"
    )

    choice = ['XGB Classifier', 'Random Forest']
    option = st.selectbox('Choice of the model', choice)
    st.write('The chosen model is:', option)

    if option == 'XGB Classifier':
        base_model = best_xgb
    else:
        base_model = best_rf

    # =========================
    # 3) Calibration
    # =========================
    clf = CalibratedClassifierCV(
        base_model,
        method="sigmoid",
        cv=5
    )

    clf.fit(X_train, y_train)

    # =========================
    # 3) Threshold analysis (TRAIN set)
    # =========================
    y_proba_train = clf.predict_proba(X_train)[:, 1]
    prec_tr, rec_tr, pr_thresh = precision_recall_curve(y_train, y_proba_train)

    eps = 1e-12
    f1_scores_tr = 2 * (prec_tr * rec_tr) / (prec_tr + rec_tr + eps)
    f2_scores_tr = (5 * prec_tr * rec_tr) / (4 * prec_tr + rec_tr + eps)

    thr = pr_thresh
    f1_for_plot = f1_scores_tr[1:]
    f2_for_plot = f2_scores_tr[1:]

    # =========================
    # 4) Threshold & metrics (TEST set)
    # =========================
    st.subheader("Calibrated threshold selection and performance on test set")

    threshold = st.slider(
        "Decision threshold (applied on test set)",
        0.05, 0.95, float(threshold_default), 0.01
    )

    y_proba_test = clf.predict_proba(X_test)[:, 1]
    y_pred_test = (y_proba_test >= threshold).astype(int)

    acc = accuracy_score(y_test, y_pred_test)
    f1 = f1_score(y_test, y_pred_test)
    f2 = fbeta_score(y_test, y_pred_test, beta=2)
    prec = precision_score(y_test, y_pred_test)
    rec = recall_score(y_test, y_pred_test)
    roc = roc_auc_score(y_test, y_proba_test)

    mcol1, mcol2, mcol3, col4 = st.columns([1,1,1,3])
    with mcol1:
        st.metric("Accuracy", f"{acc:.3f}")
        st.metric("ROC-AUC", f"{roc:.3f}")
    with mcol2:
        st.metric("F1 score", f"{f1:.3f}")
        st.metric("F2 score", f"{f2:.3f}")
    with mcol3:
        st.metric("Precision", f"{prec:.3f}")
        st.metric("Recall", f"{rec:.3f}")

    # =========================
    # 5) Visualizations
    # =========================
    display = st.radio(
        'What do you want to show ?',
        (
            'Confusion matrix',
            'ROC curve',
            'Precision–Recall curve',
            'F1/F2 vs threshold',
            'Classification report'
        )
    )

    if display == 'Confusion matrix':
        cm = confusion_matrix(y_test, y_pred_test)
        fig_cm, ax_cm = plt.subplots(figsize=(3, 2))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax_cm)
        ax_cm.set_xlabel("Predicted")
        ax_cm.set_ylabel("True")
        st.pyplot(fig_cm, use_container_width=False, clear_figure=True)

    elif display == 'ROC curve':
        fpr, tpr, _ = roc_curve(y_test, y_proba_test)
        fig_roc, ax_roc = plt.subplots(figsize=(4, 3), dpi=120)
        ax_roc.plot(fpr, tpr, label=f"ROC AUC = {roc:.3f}")
        ax_roc.plot([0, 1], [0, 1], "k--")
        ax_roc.set_xlabel("False Positive Rate")
        ax_roc.set_ylabel("True Positive Rate")
        ax_roc.legend()
        st.pyplot(fig_roc, use_container_width=False, clear_figure=True)

    elif display == 'Precision–Recall curve':
        prec_te, rec_te, _ = precision_recall_curve(y_test, y_proba_test)
        ap = average_precision_score(y_test, y_proba_test)
        fig_pr, ax_pr = plt.subplots(figsize=(4, 3), dpi=120)
        ax_pr.plot(rec_te, prec_te, label=f"AP = {ap:.3f}")
        baseline = (y_test == 1).mean()
        ax_pr.hlines(baseline, 0, 1, linestyles='--')
        ax_pr.set_xlabel("Recall")
        ax_pr.set_ylabel("Precision")
        ax_pr.legend()
        st.pyplot(fig_pr, use_container_width=False, clear_figure=True)

    elif display == 'F1/F2 vs threshold':
        fig_thr, ax_thr = plt.subplots(figsize=(4, 3), dpi=120)
        ax_thr.plot(thr, f1_for_plot, label='F1')
        ax_thr.plot(thr, f2_for_plot, label='F2')
        ax_thr.set_xlabel('Decision Threshold')
        ax_thr.set_ylabel('Score')
        ax_thr.legend()
        ax_thr.grid(True, linestyle='--', alpha=0.6)
        st.pyplot(fig_thr, use_container_width=False, clear_figure=True)

    elif display == 'Classification report':
        report = classification_report(y_test, y_pred_test, digits=3)
        st.text(report)

# =========================
# Page 4: SHAP Analysis
# =========================
elif page == pages[3]:
    st.write("### Explainability with SHAP")
    st.write("This page explains predictions using SHAP for each trained model.")

    import shap
    shap.initjs()

    # =========================
    # Model selection
    # =========================
    model_choice = st.selectbox(
        "Select model for SHAP analysis",
        ["XGB Classifier", "Random Forest"]
    )

    # =========================
    # Rebuild data (same preprocessing, NO SMOTE)
    # =========================
    X = df.drop(columns=['fraud_reported'], axis=1)
    y = df['fraud_reported'].map({'Y': 1, 'N': 0})

    num_cols = X.select_dtypes(include=['int64', 'float64', 'int32']).columns
    cat_cols = X.select_dtypes(include=['object']).columns

    ohe = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
    X_cat = pd.DataFrame(
        ohe.fit_transform(X[cat_cols]),
        columns=ohe.get_feature_names_out(cat_cols),
        index=X.index
    )

    X_num = X[num_cols].copy()
    X_final = pd.concat([X_num, X_cat], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(
        X_final, y, test_size=0.2, random_state=42, stratify=y
    )

    # =========================
    # Define base models
    # =========================
    rf_params = {
        "n_estimators": 400,
        "max_depth": 5,
        "min_samples_split": 17,
        "min_samples_leaf": 1,
        "max_features": None,
        "bootstrap": True
    }

    xgb_params = {
        "n_estimators": 800,
        "learning_rate": 0.0094,
        "max_depth": 6,
        "min_child_weight": 2.15,
        "gamma": 4.18,
        "subsample": 0.83,
        "colsample_bytree": 0.96,
        "reg_alpha": 4.86,
        "reg_lambda": 12.48,
        "scale_pos_weight": 3.19
    }

    if model_choice == "XGB Classifier":
        base_model = XGBClassifier(
            **xgb_params,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42,
            tree_method="hist"
        )
    else:
        base_model = RandomForestClassifier(
            **rf_params,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        )

    # =========================
    # Fit BASE model (SHAP explains base, not calibrated)
    # =========================
    base_model.fit(X_train, y_train)

    # =========================
    # SHAP sample size
    # =========================
    shap_sample_size = st.slider(
        "SHAP sample size (higher = slower)",
        min_value=50,
        max_value=min(500, len(X_test)),
        value=200,
        step=50
    )

    X_shap = X_test.sample(shap_sample_size, random_state=42)

    st.info("Computing SHAP values…")

    # =========================
    # SHAP explainer
    # =========================
    explainer = shap.TreeExplainer(base_model)
    shap_values = explainer.shap_values(X_shap)

    # -------------------------
    # UNIVERSAL FIX
    # -------------------------
    if isinstance(shap_values, list):
        # Old RF style: list of arrays
        shap_vals = shap_values[1]
        expected_val = explainer.expected_value[1]

    elif shap_values.ndim == 3:
        # New SHAP style: (n_samples, n_features, n_classes)
        shap_vals = shap_values[:, :, 1]
        expected_val = explainer.expected_value[1]

    else:
        # XGBoost classic
        shap_vals = shap_values
        expected_val = explainer.expected_value

    # Force scalar expected value
    expected_val = float(np.asarray(expected_val).ravel()[0])


    # =========================
    # 🔑 FIX: convert SHAP values to DataFrame
    # =========================
    shap_df = pd.DataFrame(
        shap_vals,
        columns=X_shap.columns,
        index=X_shap.index
    )

    # =========================
    # GLOBAL EXPLANATIONS
    # =========================
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### SHAP Summary (impact & direction)")
        plt.figure(figsize=(5, 3), dpi=120)
        shap.summary_plot(
            shap_df.values,
            X_shap,
            show=False,
            max_display=10
        )
        st.pyplot(plt.gcf())
        plt.clf()


    with col2:
        st.markdown("#### SHAP Feature Importance")
        plt.figure(figsize=(5, 3), dpi=120)
        shap.summary_plot(
            shap_df.values,
            X_shap,
            plot_type="bar",
            show=False,
            max_display=10
        )
        st.pyplot(plt.gcf())
        plt.clf()


    # =========================
    # INDIVIDUAL EXPLANATION
    # =========================
    st.markdown("---")
    st.subheader("Explain a single prediction")

    idx = st.slider(
        "Select test instance",
        0,
        len(X_shap) - 1,
        0
    )

    sample = X_shap.iloc[[idx]]

    shap_single = shap_df.loc[sample.index[0]].values

    col3, col4 = st.columns(2)
    with col3:
        plt.figure(figsize=(6, 3), dpi=120)
        shap.plots._waterfall.waterfall_legacy(
            expected_val,
            shap_single,
            feature_names=list(X_shap.columns),
            max_display=10
        )
        st.pyplot(plt.gcf())
        plt.clf()

    with col4:
        st.markdown("#### Feature values for this instance")
        st.dataframe(sample.T)

    # =========================
    # FORCE PLOT (single instance)
    # =========================
    from streamlit.components.v1 import html
    st.subheader("SHAP Force Plot (single prediction)")

    # select instance (reuse the same idx)
    sample = X_shap.iloc[[idx]]

    # SHAP values for this instance (1D array)
    shap_single = shap_df.loc[sample.index[0]].values

    # Create force plot
    force_plot = shap.force_plot(
        expected_val,
        shap_single,
        sample.iloc[0],
        feature_names=X_shap.columns,
        matplotlib=False   # IMPORTANT → JS version
    )

    # Render in Streamlit
    html(
        f"<head>{shap.getjs()}</head><body>{force_plot.html()}</body>",
        height=300,
    )

elif page == pages[4]:
    st.write("### 🧪 Interactive Claim Risk Analyzer")
    st.write(
        "Enter claim details below and click **Analyze Claim** to assess fraud risk "
        "using calibrated machine learning models."
    )
    # =========================
    # Input mode selector
    # =========================
    input_mode = st.radio(
        "Choose input source",
        ["Manual input", "Use existing claim from dataset"]
    )

    selected_row = None

    if input_mode == "Use existing claim from dataset":
        row_idx = st.slider(
            "Select claim index from dataset",
            min_value=0,
            max_value=len(df) - 1,
            value=0,
            step=1
        )
        selected_row = df.iloc[row_idx]
        st.caption("You can still edit the values below before analysis.")



    # =========================
    # Reuse preprocessing logic
    # =========================
    X = df.drop(columns=['fraud_reported'], axis=1)
    y = df['fraud_reported'].map({'Y': 1, 'N': 0})

    num_cols = X.select_dtypes(include=['int64', 'float64', 'int32']).columns.tolist()
    cat_cols = X.select_dtypes(include=['object']).columns.tolist()

    # =========================
    # User input form
    # =========================
    st.subheader("Claim information")

    user_input = {}

    with st.form("claim_form"):
        col1, col2, col3, col4 = st.columns(4)

        # -------------------------
        # Numerical features
        # -------------------------
        with col1:
            st.markdown("**Numerical features**")
            for col in num_cols[:len(num_cols)//2]:
                default_val = (
                    float(selected_row[col])
                    if selected_row is not None
                    else float(X[col].median())
                )

                user_input[col] = st.number_input(
                    col,
                    value=default_val,
                    step=1.0
                )

        with col4:
            st.markdown("**Categorical features**")
            for col in cat_cols[:len(cat_cols)//2]:
                options = sorted(X[col].dropna().unique().tolist())

                default_val = (
                    selected_row[col]
                    if selected_row is not None and selected_row[col] in options
                    else options[0]
                )

                user_input[col] = st.selectbox(
                    col,
                    options,
                    index=options.index(default_val)
                )

        with col2:
            st.markdown("**Numerical features**")
            for col in num_cols[len(num_cols)//2:]:
                default_val = (
                    float(selected_row[col])
                    if selected_row is not None
                    else float(X[col].median())
                )

                user_input[col] = st.number_input(
                    col,
                    value=default_val,
                    step=1.0
                )

        with col3:
            st.markdown("**Categorical features**")
            for col in cat_cols[len(cat_cols)//2:]:
                options = sorted(X[col].dropna().unique().tolist())

                default_val = (
                    selected_row[col]
                    if selected_row is not None and selected_row[col] in options
                    else options[0]
                )

                user_input[col] = st.selectbox(
                    col,
                    options,
                    index=options.index(default_val)
                )
        submitted = st.form_submit_button("🔍 Analyze Claim")


    if not submitted:
        st.stop()

    # =========================
    # Convert input to DataFrame
    # =========================
    input_df = pd.DataFrame([user_input])

    # =========================
    # One-hot encode (same as training)
    # =========================
    ohe = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
    X_cat = pd.DataFrame(
        ohe.fit_transform(X[cat_cols]),
        columns=ohe.get_feature_names_out(cat_cols),
        index=X.index
    )

    X_num = X[num_cols].copy()
    X_final = pd.concat([X_num, X_cat], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(
        X_final, y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    X_cat_input = pd.DataFrame(
        ohe.transform(input_df[cat_cols]),
        columns=X_cat.columns
    )

    X_num_input = input_df[num_cols].reset_index(drop=True)

    X_input_final = pd.concat([X_num_input, X_cat_input], axis=1)

    # Align columns (important!)
    X_input_final = X_input_final.reindex(
        columns=pd.concat([X[num_cols], X_cat], axis=1).columns,
        fill_value=0
    )

    # =========================
    # Train & calibrate models
    # =========================

    Best_parameters = {'n_estimators': 400, 'max_depth': 5, 'min_samples_split': 17, 'min_samples_leaf': 1, 'max_features': None, 'bootstrap': True}
    best_parameters = {'n_estimators': 800, 'learning_rate': 0.009395986539899289, 'max_depth': 6, 'min_child_weight': 2.155649428219426, 'gamma': 4.178059046900583, 'subsample': 0.8286853576043973,
    'colsample_bytree': 0.9556448135919854, 'reg_alpha': 4.857224881320965, 'reg_lambda': 12.48248260697908, 'scale_pos_weight': 3.1873147325141504}

    best_rf = RandomForestClassifier(
        **Best_parameters,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    best_xgb = XGBClassifier(
        **best_parameters,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
        tree_method="hist"
    )

    calibrated_rf = CalibratedClassifierCV(best_rf, method="sigmoid", cv=5)
    calibrated_xgb = CalibratedClassifierCV(best_xgb, method="sigmoid", cv=5)
    if "calibrated_rf" not in st.session_state:
        calibrated_rf.fit(X_train, y_train)
        st.session_state.calibrated_rf = calibrated_rf

    if "calibrated_xgb" not in st.session_state:
        calibrated_xgb.fit(X_train, y_train)
        st.session_state.calibrated_xgb = calibrated_xgb


    # =========================
    # Predict probabilities
    # =========================
    calibrated_rf  = st.session_state.calibrated_rf
    calibrated_xgb = st.session_state.calibrated_xgb

    prob_rf = calibrated_rf.predict_proba(X_input_final)[0, 1]
    prob_xgb = calibrated_xgb.predict_proba(X_input_final)[0, 1]

    # =========================
    # Risk labeling function
    # =========================
    def risk_label(p):
        if p < 0.30:
            return "🟢 Low risk"
        elif p < 0.60:
            return "🟠 Medium risk"
        else:
            return "🔴 High risk"

    # =========================
    # Display results
    # =========================
    st.markdown("---")
    st.subheader("📊 Fraud Risk Assessment")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🌲 Random Forest")
        st.metric("Fraud probability", f"{prob_rf:.3f}")
        st.write("Risk level:", risk_label(prob_rf))

    with col2:
        st.markdown("### ⚡ XGBoost")
        st.metric("Fraud probability", f"{prob_xgb:.3f}")
        st.write("Risk level:", risk_label(prob_xgb))

    st.markdown("---")
    st.info(
        "Probabilities are **calibrated**, meaning they represent true estimated fraud likelihoods."
    )

st.sidebar.markdown("---")
st.sidebar.caption("© 2025 – Insurance Claim Fraud Detection Project")
