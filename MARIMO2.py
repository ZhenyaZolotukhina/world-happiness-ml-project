import marimo

__generated_with = "0.18.3"
app = marimo.App(layout_file="layouts/MARIMO2.slides.json")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Homework 2. Team 4, MLBA group 3.
    World Happiness Report 2015–2024: regression task

    Eugenia Zolotukhina, Natalia Murchich & Fidan Akhundova
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Import libraries
    """)
    return


@app.cell
def _():
    # magic command not supported in marimo; please file an issue to add support
    # %%capture
    # pip install mlba
    return


@app.cell
def _():
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import xgboost as xgb

    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.linear_model import LinearRegression
    from mlba import regressionSummary
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import KFold, cross_validate
    from sklearn.model_selection import GridSearchCV
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.preprocessing import LabelEncoder
    return (
        GridSearchCV,
        KFold,
        LabelEncoder,
        LinearRegression,
        RandomForestRegressor,
        StandardScaler,
        cross_validate,
        mean_squared_error,
        np,
        pd,
        plt,
        r2_score,
        regressionSummary,
        sns,
        train_test_split,
        xgb,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Load dataset
    """)
    return


@app.cell
def _(pd):
    df = pd.read_csv('world_happiness_combined.csv', sep = ';', decimal = ',')
    df.head()
    return (df,)


@app.cell
def _(df):
    print('Dataset shape:', df.shape)
    return


@app.cell
def _(df):
    print('\nColumn information:')
    print(df.info())
    return


@app.cell
def _(df):
    print('\nMissing values per column:')
    print(df.isnull().sum())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Exploratory data analysis (EDA)
    """)
    return


@app.cell
def _(df):
    # 3.1 Basic statistics for numerical features
    df.describe().round(2)
    return


@app.cell
def _(df):
    # 3.2 Distribution by year
    print('\nDistribution by year:')
    print(df['Year'].value_counts().sort_index())
    return


@app.cell
def _(df):
    # 3.3 Distribution by region (including missing values)
    print('\nDistribution by region (including missing values):')
    print(df['Regional indicator'].value_counts(dropna = False))
    return


@app.cell
def _(df, plt, sns):
    # 3.4 Correlation heatmap for numerical features
    _numeric_cols = ['Happiness score', 'GDP per capita', 'Social support', 'Healthy life expectancy', 'Freedom to make life choices', 'Generosity', 'Perceptions of corruption', 'Year']
    plt.figure(figsize=(8, 6))
    sns.heatmap(df[_numeric_cols].corr(), annot=True, fmt='.2f', cmap='coolwarm')
    plt.title('Correlation heatmap for numerical features')
    return


@app.cell
def _(df, plt, sns):
    # 3.5 Distribution of Happiness score
    plt.figure(figsize = (6,4))
    sns.histplot(df['Happiness score'], bins = 20, kde = True)
    plt.title('Distribution of Happiness score')
    plt.xlabel('Happiness score');
    return


@app.cell
def _(df, np, plt, sns):
    # 3.6 Boxplots for all numerical features
    _numeric_cols = df.select_dtypes(include=['number']).columns
    n_cols = 3
    n_rows = int(np.ceil(len(_numeric_cols) / n_cols))
    plt.figure(figsize=(5 * n_cols, 4 * n_rows))
    for i, _col in enumerate(_numeric_cols, start=1):
        plt.subplot(n_rows, n_cols, i)
        sns.boxplot(y=df[_col])
        plt.title(_col)
        plt.xlabel('')
    return


@app.cell
def _(df, plt, sns):
    plt.figure(figsize = (10,5))
    sns.boxplot(data=df, x = 'Regional indicator', y = 'Happiness score')
    plt.xticks(rotation = 45, ha = 'right')
    plt.title('Happiness score by region');
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Data preprocessing and feature selection
    """)
    return


@app.cell
def _(LabelEncoder, df):

    country_region_map = {'Greece': 'Southern Europe', 'Cyprus': 'Middle East and North Africa', 'Gambia': 'West Africa'}
    for _country, region in country_region_map.items():
        df.loc[df['Country'] == _country, 'Regional indicator'] = region
    target_col = 'Happiness score'
    numeric_features = ['GDP per capita', 'Social support', 'Healthy life expectancy', 'Freedom to make life choices', 'Generosity', 'Perceptions of corruption', 'Year', 'Country_encoded']
    cat_feature = 'Regional indicator'
    le_country = LabelEncoder()

    df['Country_encoded'] = le_country.fit_transform(df['Country'])
    return cat_feature, le_country, numeric_features, target_col


@app.cell
def _(StandardScaler, cat_feature, df, numeric_features, pd, target_col):
    scaler = StandardScaler()
    X_num_scaled = scaler.fit_transform(df[numeric_features])
    _X_cat = pd.get_dummies(df[cat_feature], prefix='region')
    X_num_scaled = pd.DataFrame(X_num_scaled, columns=numeric_features, index=df.index)
    X = pd.concat([X_num_scaled, _X_cat], axis=1)
    y = df[target_col]
    X
    return X, X_num_scaled, y


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Train–test split
    """)
    return


@app.cell
def _(X, train_test_split, y):

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size = 0.3,
        random_state = 42
    )
    return X_test, X_train, y_test, y_train


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Model training
    """)
    return


@app.cell
def _(LinearRegression, X_train, y_train):
    model = LinearRegression()
    model.fit(X_train,y_train)
    return (model,)


@app.cell
def _(X_train, model, pd, y_train):
    train_pred = model.predict(X_train)
    train_results = pd.DataFrame({
        'Happiness_score': y_train,
        'predicted': train_pred,
        'residual': y_train - train_pred,
    })

    train_results.head()
    return (train_results,)


@app.cell
def _(X_test, model, pd, y_test):
    holdout_pred = model.predict(X_test)
    holdout_results = pd.DataFrame({
        'Happiness_score': y_test,
        'predicted': holdout_pred,
        'residual': y_test - holdout_pred,
    })
    holdout_results.head()
    return (holdout_results,)


@app.cell
def _(X_num_scaled, model):
    print('Coefficients:')
    print('Intercept:', model.intercept_)
    for _feature, coef in zip(X_num_scaled, model.coef_):
        print(f'{_feature}: {coef:.4f}')
    return


@app.cell
def _(holdout_results, regressionSummary, train_results):
    print("\nTraining Set")
    regressionSummary(y_true=train_results.Happiness_score, y_pred=train_results.predicted)
    print("\nHoldout Set")
    regressionSummary(y_true=holdout_results.Happiness_score, y_pred=holdout_results.predicted)
    return


@app.cell
def _(
    RandomForestRegressor,
    X_test,
    X_train,
    regressionSummary,
    y_test,
    y_train,
):
    rf_model = RandomForestRegressor(
        n_estimators = 300,
        max_depth = None,
        random_state = 42,
        n_jobs = -1
    )

    rf_model.fit(X_train, y_train)

    y_train_pred_rf = rf_model.predict(X_train)
    y_test_pred_rf = rf_model.predict(X_test)

    print('Random Forest - train set metrics:')
    regressionSummary(y_true = y_train, y_pred = y_train_pred_rf)

    print('\nRandom Forest - holdout set metrics:')
    regressionSummary(y_true = y_test, y_pred = y_test_pred_rf)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## K-fold cross-validation for Linear Regression and Random Forest
    """)
    return


@app.cell
def _(
    KFold,
    LinearRegression,
    RandomForestRegressor,
    X_train,
    cross_validate,
    y_train,
):
    _scoring = {'neg_RMSE': 'neg_root_mean_squared_error', 'neg_MAE': 'neg_mean_absolute_error', 'R2': 'r2'}
    _cv = KFold(n_splits=5, shuffle=True, random_state=42)
    lin_reg_cv = LinearRegression()
    cv_results_lr = cross_validate(lin_reg_cv, X_train, y_train, cv=_cv, scoring=_scoring, n_jobs=-1, return_train_score=False)
    rmse_lr = -cv_results_lr['test_neg_RMSE'].mean()
    mae_lr = -cv_results_lr['test_neg_MAE'].mean()
    r2_lr = cv_results_lr['test_R2'].mean()
    rf_cv = RandomForestRegressor(n_estimators=300, max_depth=None, random_state=42, n_jobs=-1)
    cv_results_rf = cross_validate(rf_cv, X_train, y_train, cv=_cv, scoring=_scoring, n_jobs=-1, return_train_score=False)
    rmse_rf = -cv_results_rf['test_neg_RMSE'].mean()
    mae_rf = -cv_results_rf['test_neg_MAE'].mean()
    r2_rf = cv_results_rf['test_R2'].mean()
    print('5-fold cross-validation results (on training set)\n')
    print('Linear Regression')
    print(f'  RMSE (mean over 5 folds): {rmse_lr:.3f}')
    print(f'  MAE  (mean over 5 folds): {mae_lr:.3f}')
    print(f'  R^2  (mean over 5 folds): {r2_lr:.3f}\n')
    print('Random Forest')
    print(f'  RMSE (mean over 5 folds): {rmse_rf:.3f}')
    print(f'  MAE  (mean over 5 folds): {mae_rf:.3f}')
    print(f'  R^2  (mean over 5 folds): {r2_rf:.3f}')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Hyperparameter tuning for Random Forest (optional but useful)
    """)
    return


@app.cell
def _(
    GridSearchCV,
    KFold,
    RandomForestRegressor,
    X_test,
    X_train,
    regressionSummary,
    y_test,
    y_train,
):
    param_grid = {'n_estimators': [100, 300, 500], 'max_depth': [None, 8, 12], 'min_samples_leaf': [1, 3, 5]}
    _cv = KFold(n_splits=5, shuffle=True, random_state=42)
    _scoring = {'neg_RMSE': 'neg_root_mean_squared_error', 'neg_MAE': 'neg_mean_absolute_error', 'R2': 'r2'}
    rf_base = RandomForestRegressor(random_state=42, n_jobs=-1)
    grid_search = GridSearchCV(estimator=rf_base, param_grid=param_grid, scoring='neg_root_mean_squared_error', cv=_cv, n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_
    best_rmse_cv = -grid_search.best_score_
    print('Random Forest hyperparameter tuning (5-fold CV on training set)')
    print(f'  Best params: {best_params}')
    print(f'  Best CV RMSE: {best_rmse_cv:.3f}')
    rf_best = grid_search.best_estimator_
    y_test_pred_rf_best = rf_best.predict(X_test)
    print('\nRandom Forest (tuned)-test set metrics:')
    regressionSummary(y_true=y_test, y_pred=y_test_pred_rf_best)
    return (rf_best,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Forecasring happiness score with XGBoost
    """)
    return


@app.cell
def _(df):
    df_1 = df.sort_values(['Country', 'Year']).reset_index(drop=True)
    lag_features = ['Happiness score', 'GDP per capita', 'Social support']
    for lag in [1, 2, 3]:
        for _feature in lag_features:
            df_1[f'{_feature}_lag_{lag}'] = df_1.groupby('Country')[_feature].shift(lag)
    for _feature in lag_features:
        df_1[f'{_feature}_ma_3'] = df_1.groupby('Country')[_feature].rolling(3, min_periods=1).mean().values
    return (df_1,)


@app.cell
def _(df_1):
    #Preparing numeric and categorical features
    numeric_features_1 = ['GDP per capita', 'Social support', 'Healthy life expectancy', 'Freedom to make life choices', 'Generosity', 'Perceptions of corruption', 'Year', 'Country_encoded', 'Happiness score_lag_1', 'GDP per capita_lag_1', 'Social support_lag_1', 'Happiness score_lag_2', 'GDP per capita_lag_2', 'Social support_lag_2', 'Happiness score_ma_3', 'GDP per capita_ma_3', 'Social support_ma_3']
    cat_feature_1 = 'Regional indicator'
    # Removing rows with NaN (caused by lags)
    df_model = df_1.dropna(subset=numeric_features_1 + ['Happiness score']).copy()  # Lags  # MA
    return cat_feature_1, df_model, numeric_features_1


@app.cell
def _(StandardScaler, cat_feature_1, df_model, numeric_features_1, pd):
    X_1 = df_model[numeric_features_1]
    y_1 = df_model['Happiness score']
    _X_cat = pd.get_dummies(df_model[cat_feature_1], prefix='region')
    X_1 = pd.concat([X_1, _X_cat], axis=1)
    scaler_1 = StandardScaler()
    X_scaled = scaler_1.fit_transform(X_1)
    return X_1, X_scaled, scaler_1, y_1


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Train/test split
    """)
    return


@app.cell
def _(X_scaled, y_1):
    split_idx = int(len(X_scaled) * 0.7)
    X_train_1 = X_scaled[:split_idx]
    X_test_1 = X_scaled[split_idx:]
    y_train_1 = y_1.iloc[:split_idx]
    y_test_1 = y_1.iloc[split_idx:]
    print(f'Train: {X_train_1.shape[0]} observations (first {split_idx} lines)')
    print(f'Test: {X_test_1.shape[0]} observations')
    return X_test_1, X_train_1, y_test_1, y_train_1


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Model training
    """)
    return


@app.cell
def _(X_train_1, xgb, y_train_1):
    model_xgb = xgb.XGBRegressor(n_estimators=100, max_depth=4, random_state=42)
    model_xgb.fit(X_train_1, y_train_1)
    return (model_xgb,)


@app.cell
def _(
    X_test_1,
    X_train_1,
    mean_squared_error,
    model_xgb,
    np,
    r2_score,
    y_test_1,
    y_train_1,
):
    y_pred_train = model_xgb.predict(X_train_1)
    y_pred_test = model_xgb.predict(X_test_1)
    print(f'Train R²: {r2_score(y_train_1, y_pred_train):.4f}')
    print(f'Test R²:  {r2_score(y_test_1, y_pred_test):.4f}')
    print(f'Train RMSE: {np.sqrt(mean_squared_error(y_train_1, y_pred_train)):.4f}')
    print(f'Test RMSE: {np.sqrt(mean_squared_error(y_test_1, y_pred_test)):.4f}')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Happiness score prediction by 2030
    """)
    return


@app.cell
def _(X_1, df_1, le_country, model_xgb, np, numeric_features_1, pd, scaler_1):
    future_rows = []
    last_year = df_1['Year'].max()
    future_years = range(last_year + 1, 2031)
    countries = df_1['Country'].unique()
    for _country in countries:
        df_c = df_1[df_1['Country'] == _country].sort_values('Year').copy()
        for year in future_years:
            row = {'Country': _country, 'Year': year, 'Regional indicator': df_c['Regional indicator'].iloc[-1]}
            for _col in ['GDP per capita', 'Social support', 'Healthy life expectancy', 'Freedom to make life choices', 'Generosity', 'Perceptions of corruption']:
                x = df_c['Year'].values
                ycol = df_c[_col].values
                if len(np.unique(ycol)) < 2:
                    slope, intercept = (0, ycol[-1])
                else:
                    slope, intercept = np.polyfit(x, ycol, 1)
                row[_col] = slope * year + intercept
            last = df_c.iloc[-1]
            row['Happiness score_lag_1'] = last['Happiness score']
            row['GDP per capita_lag_1'] = last['GDP per capita']
            row['Social support_lag_1'] = last['Social support']
            if len(df_c) >= 2:
                last2 = df_c.iloc[-2]
                row['Happiness score_lag_2'] = last2['Happiness score']
                row['GDP per capita_lag_2'] = last2['GDP per capita']
                row['Social support_lag_2'] = last2['Social support']
            else:
                row['Happiness score_lag_2'] = last['Happiness score']
                row['GDP per capita_lag_2'] = last['GDP per capita']
                row['Social support_lag_2'] = last['Social support']
            row['Happiness score_ma_3'] = df_c['Happiness score'].tail(3).mean()
            row['GDP per capita_ma_3'] = df_c['GDP per capita'].tail(3).mean()
            row['Social support_ma_3'] = df_c['Social support'].tail(3).mean()
            row['Country_encoded'] = le_country.transform([_country])[0]
            future_rows.append(row)
    future_df = pd.DataFrame(future_rows)
    future_cat = pd.get_dummies(future_df['Regional indicator'], prefix='region')
    future_X = pd.concat([future_df[numeric_features_1], future_cat], axis=1)
    for _col in X_1.columns:
        if _col not in future_X.columns:
            future_X[_col] = 0
    future_X = future_X[X_1.columns]
    future_X_scaled = scaler_1.transform(future_X)
    future_df['Predicted_happiness'] = model_xgb.predict(future_X_scaled)
    future_df = future_df[['Country', 'Year', 'Predicted_happiness']]
    print(future_df.head(20))
    return (future_df,)


@app.cell
def _(future_df, plt):
    plt.figure(figsize=(14, 6))
    for _country in future_df['Country'].unique():
        country_data = future_df[future_df['Country'] == _country].sort_values('Year')
        plt.plot(country_data['Year'], country_data['Predicted_happiness'], alpha=0.7)
    plt.title('Happiness score prediction by 2030', fontsize=14)
    plt.xlabel('Year')
    plt.ylabel('Happiness score')
    plt.grid(True, alpha=0.3)
    plt.show()
    return


@app.cell
def _(df, future_df, mo):

    df_forecast = future_df  

    PRED_COL = "Predicted_happiness"  


    regions = sorted(df["Country"].dropna().unique())

    ts_region_dropdown = mo.ui.dropdown(
        options=regions,
        label="country",
        value=regions[0],
    )

    ts_horizon_slider = mo.ui.slider(
        start=1,
        stop=5,
        step=1,
        value=3,
        label="Forecast horizon (years ahead)",
        show_value=True,
    )

    mo.vstack(
        [
            mo.md("### Time series controls"),
            ts_region_dropdown,
            ts_horizon_slider,
        ]
    )

    return (
        PRED_COL,
        df_forecast,
        regions,
        ts_horizon_slider,
        ts_region_dropdown,
    )


@app.cell
def _(PRED_COL, df, df_forecast, mo, ts_horizon_slider, ts_region_dropdown):

    selected_region = ts_region_dropdown.value
    forecast_years = ts_horizon_slider.value

    hist_data = df[df["Country"] == selected_region].sort_values("Year")

    forecast_data = df_forecast[
        (df_forecast["Country"] == selected_region) & 
        (df_forecast["Year"] <= df["Year"].max() + forecast_years)
    ].sort_values("Year")

    import plotly.graph_objects as go

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=hist_data["Year"],
        y=hist_data["Happiness score"],
        mode='lines+markers',
        name='Historical Data',
        line=dict(color='blue', width=2),
        marker=dict(size=6)
    ))

    fig.add_trace(go.Scatter(
        x=forecast_data["Year"],
        y=forecast_data[PRED_COL],
        mode='lines+markers',
        name='Forecast',
        line=dict(color='red', width=2, dash='dash'),
        marker=dict(size=6)
    ))

    fig.update_layout(
        title=f"Happiness Forecast for {selected_region}",
        xaxis_title="Year",
        yaxis_title="Happiness Score",
        hovermode='x unified',
        height=500,
        template='plotly_white'
    )


    recent_hist = hist_data.tail(5)[["Year", "Happiness score", "GDP per capita", "Social support"]].round(3)

  
    forecast_table = forecast_data[
        forecast_data["Year"] > df["Year"].max()
    ][["Year", PRED_COL]].head(forecast_years).round(3)
    forecast_table = forecast_table.rename(columns={PRED_COL: "Predicted Happiness"})

    mo.vstack([
        mo.md(f"## Dashboard: {selected_region}"),
        mo.md(f"**Forecast horizon:** {forecast_years} years ahead"),
    
        mo.md("### Time Series Visualization"),
        mo.ui.plotly(fig),
    
        mo.hstack([
            mo.vstack([
                mo.md("### Recent Historical Data"),
                mo.ui.table(recent_hist, selection=None)
            ]),
            mo.vstack([
                mo.md("### Forecast"),
                mo.ui.table(forecast_table, selection=None)
            ])
        ], justify="space-around")
    ])
    return


@app.cell
def _(mo, regions):

    scenario_country_dropdown = mo.ui.dropdown(
        options=regions,
        label="Select country for scenario planning",
        value=regions[0],
    )

    # Слайдеры для ключевых факторов
    gdp_slider = mo.ui.slider(
        start=0.5,
        stop=2.0,
        step=0.1,
        value=1.0,
        label="GDP per capita multiplier",
        show_value=True,
    )

    social_support_slider = mo.ui.slider(
        start=0.5,
        stop=1.5,
        step=0.05,
        value=1.0,
        label="Social support multiplier",
        show_value=True,
    )

    life_expectancy_slider = mo.ui.slider(
        start=0.8,
        stop=1.2,
        step=0.05,
        value=1.0,
        label="Life expectancy multiplier",
        show_value=True,
    )

    freedom_slider = mo.ui.slider(
        start=0.7,
        stop=1.3,
        step=0.05,
        value=1.0,
        label="Freedom multiplier",
        show_value=True,
    )

    generosity_slider = mo.ui.slider(
        start=0.5,
        stop=1.5,
        step=0.1,
        value=1.0,
        label="Generosity multiplier",
        show_value=True,
    )

    corruption_slider = mo.ui.slider(
        start=0.7,
        stop=1.3,
        step=0.05,
        value=1.0,
        label="Corruption perception multiplier",
        show_value=True,
    )


    mo.vstack([
        mo.md("## Scenario Planning: Feature Sliders"),
        mo.md("Adjust key indicators to model different scenarios and assess market sensitivity to external shocks."),
    
        scenario_country_dropdown,
    
        mo.md("### Economic & Social Factors"),
        mo.hstack([
            mo.vstack([gdp_slider, social_support_slider]),
            mo.vstack([life_expectancy_slider, freedom_slider]),
        ]),
    
        mo.md("### Governance Factors"),
        mo.hstack([
            generosity_slider,
            corruption_slider,
        ]),
    ])
    return (
        corruption_slider,
        freedom_slider,
        gdp_slider,
        generosity_slider,
        life_expectancy_slider,
        scenario_country_dropdown,
        social_support_slider,
    )


@app.cell
def _(
    corruption_slider,
    df,
    freedom_slider,
    gdp_slider,
    generosity_slider,
    life_expectancy_slider,
    mo,
    np,
    plt,
    rf_best,
    scenario_country_dropdown,
    social_support_slider,
    y,
):
    def scenario_result():
        country = scenario_country_dropdown.value
        country_data = df[df['Country'] == country].sort_values('Year').iloc[-1]
    
        features = [
            country_data['GDP per capita'] * gdp_slider.value,
            country_data['Social support'] * social_support_slider.value,
            country_data['Healthy life expectancy'] * life_expectancy_slider.value,
            country_data['Freedom to make life choices'] * freedom_slider.value,
            country_data['Generosity'] * generosity_slider.value,
            country_data['Perceptions of corruption'] * corruption_slider.value,
            country_data['Year'],  # текущий год страны (НЕ слайдер!)
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0
        ]
    
        pred = rf_best.predict(np.array(features).reshape(1, -1))[0]
        avg = y.mean()
 
        plt.figure(figsize=(8, 5))
        bars = plt.bar(['Average', f'{country}'], [avg, pred], color=['blue', 'orange'], alpha=0.7)
        plt.ylabel('Happiness Score')
        plt.title(f'{country}')
        plt.ylim(0, max(avg, pred)*1.2)
        plt.grid(True, alpha=0.3)

        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                    f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
    
        plt.show()
    
        return mo.md(f"**Prediction: {pred:.3f}** | Average: {avg:.3f} | **Δ{(pred-avg):+.3f}**")

    mo.vstack([mo.md("## Random Forest"), scenario_result()])

    return


@app.cell
def _(df, mo):
    country_forecast_selector = mo.ui.dropdown(
        options=sorted(df['Country'].unique()),
        label="Choose the country",
        value=sorted(df['Country'].unique())[0],
    )
    country_forecast_selector
    return (country_forecast_selector,)


@app.cell
def _(country_forecast_selector, df, future_df, mo, plt):
    def plot_selected_forecast():
        selected_country = country_forecast_selector.value
    
        history_data = df[df['Country'] == selected_country].sort_values('Year')
        forecast_data = future_df[future_df['Country'] == selected_country].sort_values('Year')
    
        plt.figure(figsize=(14, 7))
        plt.plot(history_data['Year'], history_data['Happiness score'], 
                 linewidth=3, marker='o', markersize=8, color='gray', alpha=0.8,
                 label='History')
    

        plt.plot(forecast_data['Year'], forecast_data['Predicted_happiness'], 
                 linewidth=3, marker='o', markersize=8, color='red', alpha=1.0,
                 label='Forecast')
    
        if len(history_data) > 0 and len(forecast_data) > 0:
            last_history = history_data.iloc[-1]
            first_forecast = forecast_data.iloc[0]
            plt.plot([last_history['Year'], first_forecast['Year']], 
                    [last_history['Happiness score'], first_forecast['Predicted_happiness']], 
                    color='orange', linewidth=2, linestyle='-', alpha=0.6, label='Transition')
    
        for i, row in history_data.iterrows():
            plt.annotate(f'{row["Happiness score"]:.2f}', 
                        (row['Year'], row["Happiness score"]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=9)
    
        for i, row in forecast_data.iterrows():
            plt.annotate(f'{row["Predicted_happiness"]:.2f}', 
                        (row['Year'], row["Predicted_happiness"]), 
                        xytext=(5, 5), textcoords='offset points', 
                        fontsize=9, fontweight='bold', color='red')
    
        plt.title(f'{selected_country}: History → Forecast', fontsize=18, fontweight='bold', pad=20)
        plt.xlabel('Year', fontsize=14)
        plt.ylabel('Happiness Score', fontsize=14)
        plt.legend(loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
        current = history_data['Happiness score'].iloc[-1]
        pred_2030 = forecast_data[forecast_data['Year'] == 2030]['Predicted_happiness'].iloc[0]
        change = pred_2030 - current
    
        return mo.md(f"""
    **{selected_country}:**
    History: **{current:.3f}** → Forecast 2030: **{pred_2030:.3f}**
    **Δ{change:+.3f}** ({change/current*100:+.1f}%)
        """)

    mo.vstack([
        country_forecast_selector,
        plot_selected_forecast()
    ])

    return


if __name__ == "__main__":
    app.run()
