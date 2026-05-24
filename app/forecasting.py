from prophet import Prophet

def forecast_load(df):

    # Prepare data for Prophet
    forecast_df = df[['Date', 'Total_System_Load']].copy()

    # Prophet requires ds and y
    forecast_df.columns = ['ds', 'y']

    # Create model
    model = Prophet()

    # Train model
    model.fit(forecast_df)

    # Future dates
    future = model.make_future_dataframe(periods=30)

    # Predict
    forecast = model.predict(future)

    return forecast