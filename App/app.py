from flask import Flask, render_template, request
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

# Initialize Flask app
app = Flask(__name__)

# Load and prepare data (replace with your dataset path or logic)
data = pd.read_csv('waste.csv')
data['Date'] = data['Month'] + " " + data['Year'].astype(str)
data['Date'] = pd.to_datetime(data['Date'], format='%b %Y')
data = data.drop(columns=['Month', 'Year'])
data = data.sort_values(by='Date')
data = data.resample('Y', on='Date').agg({
    'Waste Generated (Metric Tons)': 'sum',
    'Population': 'mean'
}).reset_index()
annual_waste_data = data.set_index('Date')['Waste Generated (Metric Tons)']

# Fit ARIMA model
model = ARIMA(annual_waste_data, order=(2, 1, 2))
model_fit = model.fit()

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        # Get user input from form
        target_year = int(request.form['years'])
        latest_year = annual_waste_data.index[-1].year

        # Calculate steps ahead
        years_ahead = target_year - latest_year

        if years_ahead < 0:
            prediction = "Cannot predict for years in the past."
        else:
            try:
                # Generate prediction
                forecast = model_fit.forecast(steps=years_ahead)
                prediction = forecast.iloc[-1]
            except Exception as e:
                prediction = f"Error: {str(e)}"

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
