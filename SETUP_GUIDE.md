# Setup Guide for AI-Powered Inventory Optimization

## 🚀 Quick Setup Steps

### 1. API Keys Configuration

#### Option A: Google Gemini (Recommended - Free)

1. Go to [Google AI Studio](https://aistudio.google.com/)
2. Create a free account
3. Generate an API key
4. In `app.py`, replace `YOUR_GEMINI_API_KEY` with your actual key
5. Free tier: 15 requests/minute, 1500 requests/day

#### Option B: OpenAI (Paid but reliable)

1. Go to [OpenAI Platform](https://platform.openai.com/)
2. Create account and add payment method
3. Generate API key
4. Install: `pip install openai`
5. Replace `YOUR_OPENAI_API_KEY` in code

#### Option C: Hugging Face (Free)

1. Go to [Hugging Face](https://huggingface.co/settings/tokens)
2. Create free account
3. Generate access token
4. Replace `YOUR_HUGGINGFACE_API_KEY` in code

### 2. Configuration Settings

Edit the `CONFIG` dictionary in `app.py`:

```python
CONFIG = {
    # Set your API keys
    "GEMINI_API_KEY": "AIzaSyDD07J4iNMK_-U_xEZz_B-8mFyiyiT-HAw",

    # Optimization parameters
    "HOLDING_COST_RATE": 0.05,  # Adjust based on your business
    "SHORTAGE_COST_MULTIPLIER": 2.0,  # Penalty for stockouts

    # Enable ML model (requires historical data)
    "USE_ML_PREDICTION": False,  # Set to True when you have data
}
```

### 3. Machine Learning Setup

#### Current State: Rule-Based

- Uses business type heuristics
- Random demand variation
- No historical data required
- Good for proof of concept

#### To Enable ML Model:

1. Set `CONFIG["USE_ML_PREDICTION"] = True`
2. Prepare historical sales data in format:
   ```
   date, item, demand, business_type
   2023-01-01, Product_A, 45, Retail
   2023-01-02, Product_A, 52, Retail
   ```
3. Replace `simulate_historical_data()` with your data loading function

### 4. Required Packages

Already installed in your environment:

- ✅ streamlit
- ✅ pandas
- ✅ numpy
- ✅ gurobipy
- ✅ scikit-learn
- ✅ requests

Additional for OpenAI: `pip install openai`

### 5. Gurobi License

Your environment already has Gurobi installed. For production:

- Academic: Free license available
- Commercial: Requires paid license
- Alternative: Replace with open-source solver (CVXPY, PuLP)

## 🎯 Current Optimization Model

### Input Parameters:

- Business type (Retail, Manufacturing, E-commerce, etc.)
- Location
- Monthly budget
- Storage capacity
- Number of SKUs

### Optimization Process:

1. **Demand Prediction**: Based on business type heuristics
2. **Mathematical Optimization**: Gurobi solver
3. **Objective**: Minimize total cost
4. **Constraints**: Budget and storage limits

### Output:

- Optimal order quantities
- Expected stock levels
- Cost breakdown
- Shortage predictions

## 🔧 Customization Options

### Business Rules:

- Modify cost parameters in `CONFIG`
- Adjust demand patterns in `predict_demand()`
- Add seasonality factors
- Include supplier lead times

### AI Responses:

- Enhance `simulate_ai_response()` with more business logic
- Add industry-specific insights
- Include market trend analysis

### UI Improvements:

- Add data upload for historical sales
- Include charts and visualizations
- Add export functionality for recommendations

## 🚨 Important Notes

1. **API Costs**: Monitor usage to avoid unexpected charges
2. **Data Privacy**: Ensure sensitive business data is protected
3. **Model Accuracy**: Validate predictions against actual sales
4. **Scalability**: Consider database integration for larger datasets

## 🔄 Next Steps

1. Set up API key (Gemini recommended for start)
2. Test with sample business data
3. Collect historical sales data
4. Enable ML model for better predictions
5. Customize for specific industry needs
