from prophet import Prophet


def forecast_load(df):

    # Prepare data for Prophet
    forecast_df = df[['Date', 'Total_System_Load']].copy()

    # Rename columns for Prophet
    forecast_df.columns = ['ds', 'y']

    # Ensure datetime format
    forecast_df['ds'] = forecast_df['ds'].astype('datetime64[ns]')

    # Remove missing values
    forecast_df = forecast_df.dropna()

    # Create Prophet model
    model = Prophet(
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=True
    )

    # Train model
    model.fit(forecast_df)

    # Create future dataframe
    future = model.make_future_dataframe(periods=30)

    # Generate forecast
    forecast = model.predict(future)

    return forecast
