import dash
from dash import dcc, html, Input, Output, State, ctx
import pandas as pd
import plotly.express as px
import dash_bootstrap_components as dbc
import base64
import io

# Load predefined datasets
def load_datasets():
    # Original datasets
    airquality = pd.DataFrame({
        "Ozone": [41, 36, 12, 18, 23, 19, 8, 16, 11, 14],
        "Solar.R": [190, 118, 149, 313, None, None, 299, 99, 19, 194],
        "Wind": [7.4, 8.0, 12.6, 11.5, 14.3, 8.6, 13.8, 20.1, 8.0, 10.9],
        "Temp": [67, 72, 74, 62, 56, 66, 65, 59, 61, 69],
        "Month": [5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
        "Day": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    })

    mtcars = pd.read_csv("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/mpg.csv").head(32)
    iris = px.data.iris()
    plantgrowth = pd.DataFrame({"group": ["ctrl", "trt1", "trt2"], "weight": [5.0, 4.9, 5.3]})
    usarrests = px.data.gapminder().query("year==2007")[["country", "pop", "gdpPercap"]]
    energy = pd.read_csv("sustainableEnergyData.csv")

    original_datasets = {
        "mtcars": mtcars,
        "iris": iris,
        "airquality": airquality,
        "PlantGrowth": plantgrowth,
        "USArrests": usarrests,
        "Sustainable Energy": energy,
    }

    # Filtered datasets (for other tabs)
    filtered_datasets = {name: df.dropna() for name, df in original_datasets.items()}

    return original_datasets, filtered_datasets

# Initialize datasets
original_datasets, datasets = load_datasets()


# App initialization
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Dynamic Plotting App"

# App layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H2("Dataset Selector"),
            dcc.Dropdown(
                id="dataset-dropdown",
                options=[{"label": name, "value": name} for name in datasets.keys()],
                value=list(datasets.keys())[0]
            ),
            html.Hr(),
            html.H4("Upload Your Dataset"),
            dcc.Upload(
                id="upload-data",
                children=html.Div(["Drag and Drop or ", html.A("Select File")]),
                style={
                    "width": "100%", "height": "60px", "lineHeight": "60px",
                    "borderWidth": "1px", "borderStyle": "dashed",
                    "borderRadius": "5px", "textAlign": "center",
                    "margin": "10px"
                },
                multiple=False
            ),
            html.Div(id="upload-message"),
            html.Hr(),
            html.H4("Variable Selection"),
            dcc.Dropdown(id="x-axis-dropdown", placeholder="Select X-axis variable"),
            dcc.Dropdown(id="y-axis-dropdown", placeholder="Select Y-axis variable"),
        ], width=3, style={"backgroundColor": "#f8f9fa", "padding": "20px"}),
        dbc.Col([
            dcc.Tabs(id="main-tabs", value="plotting", children=[
    dcc.Tab(label="Global Data on Sustainable Energy (2000-2020)", value="sustainable-energy", children=[
        html.H3("Global Data on Sustainable Energy (2000-2020)", style={"marginTop": "20px"}),
        html.P("This dataset showcases sustainable energy indicators and other useful factors across all countries from 2000 to 2020. "
               "Vital aspects such as electricity access, renewable energy, carbon emissions, energy intensity, financial flows, "
               "and economic growth are shown. One can compare nations, track progress towards Sustainable Development Goal 7, "
               "and gain insights into global energy consumption patterns over time."),
        html.A("Access the Dataset Source Here",
               href="https://www.kaggle.com/datasets/anshtanwar/global-data-on-sustainable-energy?select=global-data-on-sustainable-energy+%281%29.csv",
               target="_blank", style={"fontWeight": "bold", "color": "#007bff"})
    ]),
    dcc.Tab(label="Plotting", value="plotting", children=[
        dcc.Tabs(id="plot-tabs", value="scatter", children=[
            dcc.Tab(label="Scatterplot", value="scatter"),
            dcc.Tab(label="Bar Chart", value="bar"),
            dcc.Tab(label="Histogram", value="histogram"),
            dcc.Tab(label="Boxplot", value="box"),
            dcc.Tab(label="Line Chart", value="line"),
        ]),
        html.Div(id="plot-area")
    ]),
    dcc.Tab(label="Plotting with Faceting", value="faceting", children=[
    html.Div([
        html.H4("Facet Options"),
        dcc.Dropdown(
            id="facet-col-dropdown",
            placeholder="Select Facet Column",
            style={"marginBottom": "10px"}
        ),
        dcc.Dropdown(
            id="facet-row-dropdown",
            placeholder="Select Facet Row",
            style={"marginBottom": "20px"}
        ),
    ]),
    dcc.Tabs(id="facet-plot-tabs", value="faceted-scatter", children=[
        dcc.Tab(label="Faceted Scatterplot", value="faceted-scatter"),
        dcc.Tab(label="Faceted Histogram", value="faceted-histogram"),
        dcc.Tab(label="Faceted Boxplot", value="faceted-box"),
    ]),
    html.Div(id="facet-plot-area")
]),
    dcc.Tab(label="Dataset Info", value="dataset-info", children=[
    html.Div([
        html.H3("Structure of the Dataset"),
        html.Div(id="dataset-structure"),
        html.Hr(),
        html.H3("Summary of the Dataset"),
        html.Div(id="dataset-summary"),
        html.Hr(),
        html.H3("Python Code"),
        html.Pre("""
# Structure of the dataset
structure = df.info()

# Summary of the dataset
summary = df.describe(include='all')
        """)
    ])
])
,
    dcc.Tab(label="Missing Values", value="missing-values", children=[
    html.Div([
        html.H3("Overall Missing Values"),
        html.Div(id="overall-missing-values"),
        html.Hr(),
        html.Pre("""
# Total number of missing values
total_missing = df.isnull().sum().sum()

# Number of rows with missing values
rows_with_missing = df.isnull().any(axis=1).sum()

# Number of columns with missing values
columns_with_missing = df.isnull().any().sum()

# Create a summary table
overall_summary = pd.DataFrame({
    "Metric": ["Total Missing Values", "Rows with Missing Values", "Columns with Missing Values"],
    "Value": [total_missing, rows_with_missing, columns_with_missing]
})
        """),
        html.Hr(),
        html.H3("Column-wise Missing Values"),
        html.Div(id="column-missing-values"),
        html.Hr(),
        html.Pre("""
# Column-wise missing values
column_missing = df.isnull().sum().reset_index()
column_missing.columns = ["Column Name", "Missing Values"]
        """)
    ])
]),
    dcc.Tab(label="Variable Pairs", value="variable-pairs", children=[
    html.Div([
        html.H3("Variance-Covariance Matrix Heatmap"),
        dcc.Graph(id="variance-covariance-heatmap"),
        html.Hr(),
        html.Pre("""
# Variance-Covariance Matrix Heatmap
cov_matrix = df.cov()
fig = px.imshow(cov_matrix, text_auto=True, aspect="auto", color_continuous_scale="Viridis")
        """),
        html.Hr(),
        html.H3("Correlation Matrix Heatmap"),
        dcc.Graph(id="correlation-heatmap"),
        html.Hr(),
        html.Pre("""
# Correlation Matrix Heatmap
corr_matrix = df.corr()
fig = px.imshow(corr_matrix, text_auto=True, aspect="auto", color_continuous_scale="Viridis")
        """),
        html.Hr(),
        html.H3("Pairwise Scatterplots"),
        dcc.Graph(id="pairwise-scatterplots"),
        html.Hr(),
        html.Pre("""
# Pairwise Scatterplots
fig = px.scatter_matrix(df, dimensions=df.columns, color=df.columns[0])
        """)
    ])
]),
    dcc.Tab(label="GLM Modeling", value="glm-modeling", children=[
    html.Div([
        html.H3("Generalized Linear Model (GLM)"),
        html.Hr(),
        html.Div([
            html.Label("Select Model Family:"),
            dcc.Dropdown(
                id="glm-family-dropdown",
                options=[
                    {"label": "Gaussian", "value": "Gaussian"},
                    {"label": "Binomial", "value": "Binomial"},
                    {"label": "Poisson", "value": "Poisson"},
                    {"label": "Gamma", "value": "Gamma"}
                ],
                value="Gaussian",
                placeholder="Select a model family"
            ),
            html.Br(),
            html.Label("Select Response Variable:"),
            dcc.Dropdown(id="glm-response-dropdown", placeholder="Select Response Variable"),
            html.Br(),
            html.Label("Select Predictor Variables:"),
            dcc.Dropdown(id="glm-predictors-dropdown", multi=True, placeholder="Select Predictor Variables"),
        ]),
        html.Hr(),
        html.H3("Model Summary"),
        html.Div(id="glm-summary"),
        html.Hr(),
        html.Pre("""
# Fit the GLM
import statsmodels.api as sm
import statsmodels.formula.api as smf

formula = "response ~ predictors"
family = sm.families.Gaussian()  # Adjust based on selected family
model = smf.glm(formula=formula, data=df, family=family).fit()
summary = model.summary()
        """),
        html.Hr(),
        html.H3("Predictions"),
        html.Div(id="glm-predictions"),
        html.Hr(),
        html.Pre("""
# Predictions
predictions = model.predict(df)
        """),
        html.Hr(),
        html.H3("Model Visualization: Predicted vs Actual"),
        dcc.Graph(id="glm-visualization"),
        html.Hr(),
        html.Pre("""
# Predicted vs Actual Plot
import plotly.express as px
fig = px.scatter(x=df["response"], y=predictions, labels={"x": "Actual", "y": "Predicted"})
        """)
    ])
]),
    dcc.Tab(label="Model Selection", value="model-selection", children=[
    html.Div([
        html.H3("Model Selection Methods"),
        html.Label("Choose a Model Selection Method:"),
        dcc.Dropdown(
            id="model-selection-method-dropdown",
            options=[
                {"label": "Forward Selection", "value": "forward"},
                {"label": "Backward Elimination", "value": "backward"},
                {"label": "Stepwise Selection", "value": "stepwise"},
                {"label": "All Possible Subset Selection", "value": "all-subsets"}
            ],
            value="forward",
            placeholder="Select a model selection method"
        ),
        html.Br(),
        html.Label("Select Response Variable:"),
        dcc.Dropdown(id="model-selection-response-dropdown", placeholder="Select Response Variable"),
        html.Br(),
        html.Label("Select Predictor Variables:"),
        dcc.Dropdown(id="model-selection-predictors-dropdown", multi=True, placeholder="Select Predictor Variables"),
        html.Hr(),
        html.H3("Selected Model Summary"),
        html.Div(id="model-selection-summary"),
        html.Hr(),
        html.Pre("""
# Example: Forward Selection
import statsmodels.api as sm

def forward_selection(data, response):
    remaining_predictors = set(data.columns) - {response}
    selected_predictors = []
    current_score, best_score = float('inf'), float('inf')
    formula_template = "{response} ~ {predictors}"
    while remaining_predictors:
        scores_with_candidates = []
        for candidate in remaining_predictors:
            formula = formula_template.format(
                response=response,
                predictors=" + ".join(selected_predictors + [candidate])
            )
            score = sm.OLS.from_formula(formula, data).fit().aic
            scores_with_candidates.append((score, candidate))
        scores_with_candidates.sort()
        best_new_score, best_candidate = scores_with_candidates[0]
        if best_new_score < current_score:
            remaining_predictors.remove(best_candidate)
            selected_predictors.append(best_candidate)
            current_score = best_new_score
        else:
            break
    formula = formula_template.format(
        response=response,
        predictors=" + ".join(selected_predictors)
    )
    model = sm.OLS.from_formula(formula, data).fit()
    return model
        """),
    ])
]),
    dcc.Tab(label="First Two Principal Components", value="pca-tab", children=[
    html.Div([
        html.H3("Principal Component Analysis (PCA)"),
        html.P("Principal Component Analysis (PCA) is a dimensionality reduction technique that transforms correlated "
               "variables into a smaller set of uncorrelated variables called principal components. The first principal "
               "component (PCA1) explains the largest variance in the data, and subsequent components explain the remaining "
               "variance. This technique is useful for visualization and noise reduction."),
        html.Hr(),
        html.H4("PCA1 and PCA2 as Linear Combinations"),
        html.Div(id="pca-linear-combination-table"),
        html.Pre("""
# PCA Linear Combinations
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca.fit(df)
components = pd.DataFrame(pca.components_, columns=df.columns, index=["PCA1", "PCA2"])
        """),
        html.Hr(),
        html.H4("Variance Explained by PCA1 and PCA2"),
        html.Div(id="pca-variance-explained-table"),
        html.Pre("""
# Variance Explained
explained_variance = pca.explained_variance_ratio_
variance_table = pd.DataFrame({
    "PCA": ["PCA1", "PCA2"],
    "Variance": explained_variance,
    "Percentage": explained_variance * 100
})
        """),
        html.Hr(),
        html.H4("PCA1 vs PCA2"),
        dcc.Graph(id="pca-scatter-plot"),
        html.Pre("""
# PCA Scatter Plot
pca_transformed = pca.transform(df)
pca_df = pd.DataFrame(pca_transformed, columns=["PCA1", "PCA2"])
fig = px.scatter(pca_df, x="PCA1", y="PCA2", title="PCA1 vs PCA2")
        """)
    ])
])

# dcc.Tabs list ends here
])


        ], width=9)
    ])
], fluid=True)
# dcc.Tabs ends here

from sklearn.decomposition import PCA

from sklearn.decomposition import PCA

@app.callback(
    [Output("pca-linear-combination-table", "children"),
     Output("pca-variance-explained-table", "children"),
     Output("pca-scatter-plot", "figure")],
    [Input("dataset-dropdown", "value")]
)
def update_pca_tab(selected_dataset):
    if not selected_dataset or selected_dataset not in datasets:
        return "No dataset selected.", "No dataset selected.", {}

    df = datasets[selected_dataset].select_dtypes(include=["number"])  # Use only numeric columns
    if df.shape[1] < 2:
        return "Not enough numeric columns for PCA.", "Not enough numeric columns for PCA.", {}

    try:
        # Perform PCA
        pca = PCA(n_components=2)
        pca.fit(df)

        # PCA1 and PCA2 as Linear Combinations
        components = pd.DataFrame(pca.components_, columns=df.columns, index=["PCA1", "PCA2"])
        components_table = dbc.Table.from_dataframe(components, striped=True, bordered=True, hover=True)

        # Variance Explained
        explained_variance = pca.explained_variance_ratio_
        variance_table = pd.DataFrame({
            "PCA": ["PCA1", "PCA2"],
            "Variance": explained_variance,
            "Percentage": explained_variance * 100
        })
        variance_explained_table = dbc.Table.from_dataframe(variance_table, striped=True, bordered=True, hover=True)

        # PCA Scatter Plot
        pca_transformed = pca.transform(df)
        pca_df = pd.DataFrame(pca_transformed, columns=["PCA1", "PCA2"])
        fig = px.scatter(pca_df, x="PCA1", y="PCA2", title="PCA1 vs PCA2")

        return components_table, variance_explained_table, fig

    except Exception as e:
        return f"Error performing PCA: {e}", f"Error performing PCA: {e}", {}



@app.callback(
    Output("model-selection-summary", "children"),
    [Input("dataset-dropdown", "value"),
     Input("model-selection-method-dropdown", "value"),
     Input("model-selection-response-dropdown", "value"),
     Input("model-selection-predictors-dropdown", "value")]
)
def perform_model_selection(selected_dataset, method, response, predictors):
    if not selected_dataset or selected_dataset not in datasets:
        return "No dataset selected."

    df = datasets[selected_dataset]
    if not response or not predictors:
        return "Please select a response and predictors."

    try:
        if method == "forward":
            model = forward_selection(df, response)
        elif method == "backward":
            model = backward_elimination(df, response, predictors)
        elif method == "stepwise":
            model = stepwise_selection(df, response)
        elif method == "all-subsets":
            model = all_possible_subsets_selection(df, response, predictors)
        else:
            return "Invalid model selection method."

        # Parse Model Summary
        summary_table = model.summary().tables[1].as_html()
        summary_df = pd.read_html(summary_table, header=0, index_col=0)[0].reset_index()
        summary_html = dbc.Table.from_dataframe(summary_df, striped=True, bordered=True, hover=True)

        return summary_html
    except Exception as e:
        return f"Error during model selection: {e}"

def forward_selection(data, response):
    remaining_predictors = list(data.columns.difference([response]))
    selected_predictors = []
    current_score, best_new_score = float("inf"), float("inf")
    formula_template = "{response} ~ {predictors}"

    while remaining_predictors:
        scores_with_candidates = []
        for candidate in remaining_predictors:
            formula = formula_template.format(
                response=response,
                predictors=" + ".join(selected_predictors + [candidate])
            )
            model = smf.ols(formula=formula, data=data).fit()
            scores_with_candidates.append((model.aic, candidate))

        scores_with_candidates.sort()
        best_new_score, best_candidate = scores_with_candidates[0]
        if best_new_score < current_score:
            remaining_predictors.remove(best_candidate)
            selected_predictors.append(best_candidate)
            current_score = best_new_score
        else:
            break

    final_formula = formula_template.format(
        response=response,
        predictors=" + ".join(selected_predictors)
    )
    final_model = smf.ols(formula=final_formula, data=data).fit()
    return final_model


def backward_elimination(data, response, predictors):
    selected_predictors = list(predictors)
    formula_template = "{response} ~ {predictors}"

    while len(selected_predictors) > 0:
        formula = formula_template.format(
            response=response,
            predictors=" + ".join(selected_predictors)
        )
        model = smf.ols(formula=formula, data=data).fit()
        p_values = model.pvalues.iloc[1:]  # Exclude intercept
        worst_p_value = p_values.max()
        if worst_p_value > 0.05:  # Significance level
            worst_predictor = p_values.idxmax()
            selected_predictors.remove(worst_predictor)
        else:
            break

    final_formula = formula_template.format(
        response=response,
        predictors=" + ".join(selected_predictors)
    )
    final_model = smf.ols(formula=final_formula, data=data).fit()
    return final_model


def stepwise_selection(data, response):
    remaining_predictors = list(data.columns.difference([response]))
    selected_predictors = []
    current_score, best_new_score = float("inf"), float("inf")
    formula_template = "{response} ~ {predictors}"

    while remaining_predictors or selected_predictors:
        # Forward Step
        scores_with_candidates = []
        for candidate in remaining_predictors:
            formula = formula_template.format(
                response=response,
                predictors=" + ".join(selected_predictors + [candidate])
            )
            model = smf.ols(formula=formula, data=data).fit()
            scores_with_candidates.append((model.aic, candidate))

        scores_with_candidates.sort()
        best_new_score, best_candidate = scores_with_candidates[0]
        if best_new_score < current_score:
            remaining_predictors.remove(best_candidate)
            selected_predictors.append(best_candidate)
            current_score = best_new_score
        else:
            break

        # Backward Step
        p_values = model.pvalues.iloc[1:]  # Exclude intercept
        worst_p_value = p_values.max()
        if worst_p_value > 0.05:  # Significance level
            worst_predictor = p_values.idxmax()
            selected_predictors.remove(worst_predictor)
            remaining_predictors.append(worst_predictor)

    final_formula = formula_template.format(
        response=response,
        predictors=" + ".join(selected_predictors)
    )
    final_model = smf.ols(formula=final_formula, data=data).fit()
    return final_model


from itertools import combinations

def all_possible_subsets_selection(data, response, predictors):
    best_score = float("inf")
    best_model = None
    formula_template = "{response} ~ {predictors}"

    for k in range(1, len(predictors) + 1):
        for subset in combinations(predictors, k):
            formula = formula_template.format(
                response=response,
                predictors=" + ".join(subset)
            )
            model = smf.ols(formula=formula, data=data).fit()
            if model.aic < best_score:
                best_score = model.aic
                best_model = model

    return best_model


@app.callback(
    [Output("model-selection-response-dropdown", "options"),
     Output("model-selection-predictors-dropdown", "options")],
    [Input("dataset-dropdown", "value")]
)
def update_model_selection_options(selected_dataset):
    if not selected_dataset or selected_dataset not in datasets:
        return [], []

    df = datasets[selected_dataset]
    options = [{"label": col, "value": col} for col in df.columns]
    return options, options


from statsmodels.api import families
import statsmodels.formula.api as smf

@app.callback(
    [Output("glm-response-dropdown", "options"),
     Output("glm-predictors-dropdown", "options")],
    [Input("dataset-dropdown", "value")]
)
def update_glm_variable_options(selected_dataset):
    if not selected_dataset or selected_dataset not in datasets:
        return [], []

    df = datasets[selected_dataset]
    options = [{"label": col, "value": col} for col in df.columns]
    return options, options


@app.callback(
    [Output("glm-summary", "children"),
     Output("glm-predictions", "children"),
     Output("glm-visualization", "figure")],
    [Input("dataset-dropdown", "value"),
     Input("glm-family-dropdown", "value"),
     Input("glm-response-dropdown", "value"),
     Input("glm-predictors-dropdown", "value")]
)
def update_glm_model(selected_dataset, family, response, predictors):
    if not selected_dataset or selected_dataset not in datasets:
        return "No dataset selected.", "No predictions available.", {}

    df = datasets[selected_dataset]
    if not response or not predictors:
        return "Please select a response and predictors.", "No predictions available.", {}

    # Prepare formula
    formula = f"{response} ~ {' + '.join(predictors)}"

    # Map family to statsmodels families
    family_mapping = {
        "Gaussian": families.Gaussian(),
        "Binomial": families.Binomial(),
        "Poisson": families.Poisson(),
        "Gamma": families.Gamma()
    }

    try:
        # Fit GLM model
        model = smf.glm(formula=formula, data=df, family=family_mapping[family]).fit()

        # Parse Model Summary
        summary_table = model.summary().tables[1].as_html()
        summary_df = pd.read_html(summary_table, header=0, index_col=0)[0].reset_index()
        summary_html = dbc.Table.from_dataframe(summary_df, striped=True, bordered=True, hover=True)

        # Predictions
        predictions = model.predict(df)
        predictions_table = pd.DataFrame({response: df[response], "Predicted": predictions}).head(8)
        predictions_table_html = dbc.Table.from_dataframe(predictions_table, striped=True, bordered=True, hover=True)

        # Visualization: Predicted vs Actual
        fig = px.scatter(x=df[response], y=predictions, labels={"x": "Actual", "y": "Predicted"},
                         title="Predicted vs Actual")
        fig.add_shape(type="line", x0=min(df[response]), y0=min(df[response]),
                      x1=max(df[response]), y1=max(df[response]), line=dict(dash="dash"))

        return summary_html, predictions_table_html, fig

    except Exception as e:
        return f"Error fitting model: {e}", "No predictions available.", {}



@app.callback(
    [Output("variance-covariance-heatmap", "figure"),
     Output("correlation-heatmap", "figure"),
     Output("pairwise-scatterplots", "figure")],
    [Input("dataset-dropdown", "value")]
)
def update_variable_pairs(selected_dataset):
    if not selected_dataset or selected_dataset not in datasets:
        return {}, {}, {}

    df = datasets[selected_dataset]  # Use filtered dataset
    numeric_df = df.select_dtypes(include=["number"])  # Exclude non-numeric columns

    # Variance-Covariance Matrix Heatmap
    try:
        cov_matrix = numeric_df.cov()
        cov_fig = px.imshow(cov_matrix, text_auto=True, aspect="auto",
                            color_continuous_scale="Viridis",
                            labels={"color": "Variance-Covariance"})
    except Exception as e:
        cov_fig = px.imshow([[0]], text_auto=True, title=f"Error: {e}")

    # Correlation Matrix Heatmap
    try:
        corr_matrix = numeric_df.corr()
        corr_fig = px.imshow(corr_matrix, text_auto=True, aspect="auto",
                             color_continuous_scale="Viridis",
                             labels={"color": "Correlation"})
    except Exception as e:
        corr_fig = px.imshow([[0]], text_auto=True, title=f"Error: {e}")

    # Pairwise Scatterplots
    try:
        scatter_fig = px.scatter_matrix(numeric_df, dimensions=numeric_df.columns,
                                        color=numeric_df.columns[0] if len(numeric_df.columns) > 0 else None)
    except Exception as e:
        scatter_fig = px.scatter([[0, 0]], title=f"Error: {e}")

    return cov_fig, corr_fig, scatter_fig



@app.callback(
    [Output("overall-missing-values", "children"),
     Output("column-missing-values", "children")],
    [Input("dataset-dropdown", "value")]
)
def update_missing_values_tab(selected_dataset):
    if not selected_dataset or selected_dataset not in original_datasets:
        return html.Div("No dataset selected or available."), html.Div("No dataset selected or available.")

    df = original_datasets[selected_dataset]  # Use the original dataset for missing values analysis

    # Overall missing values
    total_missing = df.isnull().sum().sum()
    rows_with_missing = df.isnull().any(axis=1).sum()
    columns_with_missing = df.isnull().any().sum()

    overall_summary = pd.DataFrame({
        "Metric": ["Total Missing Values", "Rows with Missing Values", "Columns with Missing Values"],
        "Value": [total_missing, rows_with_missing, columns_with_missing]
    })

    # Column-wise missing values
    column_missing = df.isnull().sum().reset_index()
    column_missing.columns = ["Column Name", "Missing Values"]

    # Tables for output
    overall_table = dbc.Table.from_dataframe(overall_summary, striped=True, bordered=True, hover=True)
    column_table = dbc.Table.from_dataframe(column_missing, striped=True, bordered=True, hover=True)

    return overall_table, column_table



# Callbacks for dataset selection and file upload
@app.callback(
    [Output("x-axis-dropdown", "options"),
     Output("y-axis-dropdown", "options"),
     Output("x-axis-dropdown", "value"),
     Output("y-axis-dropdown", "value"),
     Output("upload-message", "children")],
    [Input("dataset-dropdown", "value"),
     Input("upload-data", "contents")],
    [State("upload-data", "filename")]
)
def update_dropdowns(selected_dataset, uploaded_file_content, uploaded_file_name):
    if ctx.triggered_id == "upload-data" and uploaded_file_content is not None:
        content_type, content_string = uploaded_file_content.split(',')
        decoded = base64.b64decode(content_string)
        try:
            if uploaded_file_name.endswith(".csv"):
                df = pd.read_csv(io.StringIO(decoded.decode('utf-8'))).dropna()
            elif uploaded_file_name.endswith(".xlsx"):
                df = pd.read_excel(io.BytesIO(decoded)).dropna()
            else:
                return [], [], None, None, "Unsupported file type"
        except Exception as e:
            return [], [], None, None, f"Error loading file: {e}"
        datasets["Uploaded File"] = df
        selected_dataset = "Uploaded File"

    df = datasets[selected_dataset]
    options = [{"label": col, "value": col} for col in df.columns]
    default_x = df.columns[0] if len(df.columns) > 0 else None
    default_y = df.columns[1] if len(df.columns) > 1 else None
    return options, options, default_x, default_y, f"Loaded dataset: {selected_dataset} (filtered missing values)"

# Callback to update plot
@app.callback(
    Output("plot-area", "children"),
    [Input("plot-tabs", "value"),
     Input("x-axis-dropdown", "value"),
     Input("y-axis-dropdown", "value"),
     Input("dataset-dropdown", "value")]
)
def update_plot(plot_type, x_axis, y_axis, selected_dataset):
    if not selected_dataset or selected_dataset not in datasets:
        return html.Div("No dataset selected or available.")
    df = datasets[selected_dataset]

    if not x_axis or not y_axis:
        return html.Div("Please select variables for both X and Y axes.")

    try:
        plot_code = ""
        if plot_type == "scatter":
            fig = px.scatter(df, x=x_axis, y=y_axis)
            plot_code = f"fig = px.scatter(df, x='{x_axis}', y='{y_axis}')"
        elif plot_type == "bar":
            fig = px.bar(df, x=x_axis, y=y_axis)
            plot_code = f"fig = px.bar(df, x='{x_axis}', y='{y_axis}')"
        elif plot_type == "histogram":
            fig = px.histogram(df, x=x_axis)
            plot_code = f"fig = px.histogram(df, x='{x_axis}')"
        elif plot_type == "box":
            fig = px.box(df, x=x_axis, y=y_axis)
            plot_code = f"fig = px.box(df, x='{x_axis}', y='{y_axis}')"
        elif plot_type == "line":
            fig = px.line(df, x=x_axis, y=y_axis)
            plot_code = f"fig = px.line(df, x='{x_axis}', y='{y_axis}')"
        else:
            return html.Div("Invalid plot type selected.")
        
        return html.Div([
            dcc.Graph(figure=fig),
            html.Hr(),
            html.Div([
                html.H4("Plot Code"),
                html.Pre(plot_code)  # Use html.Pre for plain text formatting
            ])
        ])
    except Exception as e:
        return html.Div(f"Error generating plot: {e}")

@app.callback(
    [Output("facet-col-dropdown", "options"),
     Output("facet-row-dropdown", "options"),
     Output("facet-col-dropdown", "value"),
     Output("facet-row-dropdown", "value")],
    [Input("dataset-dropdown", "value")]
)
def update_facet_options(selected_dataset):
    if not selected_dataset or selected_dataset not in datasets:
        return [], [], None, None

    df = datasets[selected_dataset]
    options = [{"label": col, "value": col} for col in df.columns]
    default_col = df.columns[0] if len(df.columns) > 0 else None
    default_row = df.columns[1] if len(df.columns) > 1 else None
    return options, options, default_col, default_row


@app.callback(
    Output("facet-plot-area", "children"),
    [Input("facet-plot-tabs", "value"),
     Input("x-axis-dropdown", "value"),
     Input("y-axis-dropdown", "value"),
     Input("dataset-dropdown", "value"),
     Input("facet-col-dropdown", "value"),
     Input("facet-row-dropdown", "value")]
)
def update_facet_plot(facet_plot_type, x_axis, y_axis, selected_dataset, facet_col, facet_row):
    if not selected_dataset or selected_dataset not in datasets:
        return html.Div("No dataset selected or available.")
    df = datasets[selected_dataset]

    if not x_axis or (facet_plot_type != "faceted-histogram" and not y_axis):
        return html.Div("Please select variables for the X and Y axes.")
    
    if not facet_col and not facet_row:
        return html.Div("Please select facet columns for the plot.")

    try:
        facet_code = ""
        if facet_plot_type == "faceted-scatter":
            fig = px.scatter(df, x=x_axis, y=y_axis, facet_col=facet_col, facet_row=facet_row)
            facet_code = f"fig = px.scatter(df, x='{x_axis}', y='{y_axis}', facet_col='{facet_col}', facet_row='{facet_row}')"
        elif facet_plot_type == "faceted-histogram":
            fig = px.histogram(df, x=x_axis, facet_col=facet_col)
            facet_code = f"fig = px.histogram(df, x='{x_axis}', facet_col='{facet_col}')"
        elif facet_plot_type == "faceted-box":
            fig = px.box(df, x=x_axis, y=y_axis, facet_col=facet_col)
            facet_code = f"fig = px.box(df, x='{x_axis}', y='{y_axis}', facet_col='{facet_col}')"
        else:
            return html.Div("Invalid plot type selected.")

        return html.Div([
            dcc.Graph(figure=fig),
            html.Hr(),
            html.Div([
                html.H4("Plot Code"),
                html.Pre(facet_code)  # Display code as plain text
            ])
        ])
    except Exception as e:
        return html.Div(f"Error generating plot: {e}")

@app.callback(
    [Output("dataset-structure", "children"),
     Output("dataset-summary", "children")],
    [Input("dataset-dropdown", "value")]
)
def update_dataset_info(selected_dataset):
    if not selected_dataset or selected_dataset not in datasets:
        return html.Div("No dataset selected or available."), html.Div("No dataset selected or available.")

    df = datasets[selected_dataset]

    # Generate structure of the dataset
    structure = pd.DataFrame({
        "Column Name": df.columns,
        "Data Type": df.dtypes.astype(str),  # Convert dtypes to strings for JSON serialization
        "Non-Null Count": [df[col].notnull().sum() for col in df.columns]
    })

    # Generate summary of the dataset
    summary = df.describe(include='all').reset_index()
    summary = summary.astype(str)  # Convert all data to strings for JSON serialization

    return (
        # Convert structure to a table
        dbc.Table.from_dataframe(structure, striped=True, bordered=True, hover=True),
        # Convert summary to a table
        dbc.Table.from_dataframe(summary, striped=True, bordered=True, hover=True)
    )




if __name__ == "__main__":
    app.run_server(debug=True)
