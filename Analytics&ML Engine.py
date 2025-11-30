import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.tsa.seasonal import seasonal_decompose
from prophet import Prophet
import matplotlib.pyplot as plt
import seaborn as sns 
import io
import base64

class AnalyticsService:
    def __init__(self):
        self.data = None

    def load_data(self, gee_data_list):
        """
        Ingests real data from Google Earth Engine.
        Expected format: List of dictionaries (JSON).
        Example: [{'system:time_start': 1672531200000, 'NDVI': 0.45, 'Temperature': 25.0, 'Wind_Speed': 5.0}, ...]
        """
        try:
            # 1. Convert list of dicts to DataFrame
            df = pd.DataFrame(gee_data_list)
            
            # 2. Rename GEE specific keys to readable names and convert to datetime
            if 'system:time_start' in df.columns:
                df['Date'] = pd.to_datetime(df['system:time_start'], unit='ms')
            elif 'date' in df.columns:
                df['Date'] = pd.to_datetime(df['date'])
            else:
                raise ValueError("Data missing 'system:time_start' or 'date' column")

            # 3. Set Index
            df.set_index('Date', inplace=True)
            
            # 4. Handle Missing Values (Clouds/Masked pixels often result in NaNs)
            # Interpolate fills the gaps smoothly for all relevant columns (NDVI + Climatic Variables)
            cols_to_interpolate = ['NDVI', 'Temperature', 'Wind_Speed', 'Pressure']
            for col in cols_to_interpolate:
                if col in df.columns:
                    # Only interpolate if the column exists
                    df[col] = df[col].interpolate(method='time')
            
            self.data = df.sort_index()
            return True
        except Exception as e:
            # Keep the data loading error message clean
            print(f"Error loading data: {e}")
            return False

    def generate_correlation_heatmap(self):
        """
        Generates a Seaborn correlation heatmap plot, Base64 encoded for the API response.
        """
        if self.data is None:
            return None

        # Select numerical columns for correlation (NDVI and climatic variables)
        available_cols = [col for col in ['NDVI', 'Temperature', 'Wind_Speed', 'Pressure'] if col in self.data.columns]
        if len(available_cols) < 2:
            # Cannot compute correlation with fewer than 2 columns
            return None 

        corr_data = self.data[available_cols].copy()
        corr_matrix = corr_data.corr()

        # Create the plot
        img = io.BytesIO()
        plt.figure(figsize=(8, 7))
        sns.heatmap(
            corr_matrix, 
            annot=True, 
            cmap='coolwarm', 
            fmt=".2f", 
            linewidths=.5, 
            cbar_kws={'label': 'Pearson Correlation Coefficient'}
        )
        plt.title('Correlation Heatmap: Vegetation vs. Environmental Variables')
        plt.tight_layout()
        plt.savefig(img, format='png')
        plt.close()
        
        img.seek(0)
        return base64.b64encode(img.getvalue()).decode()


    def perform_full_analysis(self, forecast_years=2):
        """
        Runs all analysis (stats, correlation, forecasting, plots) and returns 
        a JSON-serializable dictionary.
        """
        if self.data is None or len(self.data) < 12:
            return {"error": "Not enough data for analysis (need at least 12 months)"}

        response_payload = {}

        # --- A. Trend Stats ---
        current_ndvi = self.data['NDVI'].iloc[-1]
        start_ndvi = self.data['NDVI'].iloc[0]
        trend_direction = "Increasing" if current_ndvi > start_ndvi else "Decreasing"
        
        response_payload['summary'] = {
            'trend_direction': trend_direction,
            'change_percentage': round(((current_ndvi - start_ndvi) / start_ndvi) * 100, 2),
            'latest_ndvi': round(current_ndvi, 3)
        }

        # --- B. Correlation Analysis ---
        if 'Temperature' in self.data.columns:
            corr, p_val = stats.pearsonr(self.data['NDVI'], self.data['Temperature'])
            response_payload['correlation'] = {
                'ndvi_temp_coefficient': round(corr, 4),
                'significance': "Significant" if p_val < 0.05 else "Not Significant"
            }

        # --- B.1 Correlation Heatmap Plot  ---
        response_payload['correlation_heatmap_base64'] = self.generate_correlation_heatmap()


        # --- C. Forecasting (Prophet) ---
        df_prophet = self.data.reset_index()[['Date', 'NDVI']].rename(columns={'Date': 'ds', 'NDVI': 'y'})
        
        m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
        m.fit(df_prophet)
        
        future = m.make_future_dataframe(periods=forecast_years * 12, freq='M')
        forecast = m.predict(future)
        
        future_forecast = forecast.tail(forecast_years * 12)
        response_payload['forecast'] = future_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_dict(orient='records')

        # --- D. Generate Plot Image (Historical & Forecast) ---
        img = io.BytesIO()
        plt.figure(figsize=(10, 6))
        
        # Plot Historical data
        plt.plot(self.data.index, self.data['NDVI'], label='Historical NDVI', color='blue')
        # Plot Forecast data
        plt.plot(future_forecast['ds'], future_forecast['yhat'], label='Forecast NDVI', linestyle='--', color='orange')
        
        plt.title('NDVI Vegetation Forecast')
        plt.xlabel('Date')
        plt.ylabel('NDVI Value')
        plt.legend()
        plt.savefig(img, format='png')
        img.seek(0)
        response_payload['plot_image_base64'] = base64.b64encode(img.getvalue()).decode()
        plt.close()

        return response_payload