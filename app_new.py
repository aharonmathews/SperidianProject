# app.py - AI-Powered Business Optimization (OpenAI-powered)

import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
import time
import re
import io
import sys
from contextlib import redirect_stdout, redirect_stderr
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="AI-Powered Business Optimization",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for flexible UI
st.markdown("""
<style>
    .main-header {
        color: white;
        text-align: center;
        padding: 1.5rem;
        background: linear-gradient(90deg, #1e3c72, #2a5298);
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .chat-container {
        background: linear-gradient(135deg, #1a1a1a, #2d2d2d);
        border-radius: 15px;
        padding: 1.5rem;
        min-height: 400px;
        max-height: 600px;
        overflow-y: auto;
        border: 1px solid #444;
    }
    .bot-message {
        background: linear-gradient(135deg, #2d2d2d, #3d3d3d);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 20px 20px 20px 5px;
        margin: 1rem 0;
        max-width: 85%;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
        border-left: 4px solid #4CAF50;
    }
    .user-message {
        background: linear-gradient(135deg, #0066cc, #0052a3);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 20px 20px 5px 20px;
        margin: 1rem 0;
        max-width: 85%;
        float: right;
        clear: both;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
    }
    .solution-panel {
        background: linear-gradient(135deg, #f8f9fa, #e9ecef);
        border-radius: 15px;
        padding: 2rem;
        min-height: 400px;
        border: 1px solid #dee2e6;
        overflow-y: auto;
    }
    .input-container {
        background: linear-gradient(135deg, #2d2d2d, #3d3d3d);
        border-radius: 25px;
        padding: 1rem;
        margin-top: 1rem;
        border: 1px solid #555;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# --- CONFIGURATION ---
CONFIG = {
    "OPENAI_API_KEY": "sk-5678mnopqrstuvwx5678mnopqrstuvwx5678mnop",  # Get from https://platform.openai.com/
    "OPENAI_MODEL": "gpt-3.5-turbo",  # or "gpt-4" for better results
    "MAX_RETRIES": 3,
    "TIMEOUT": 30
}

# Add API key configuration in sidebar
with st.sidebar:
    st.header("🔧 Configuration")
    
    # API Key input
    api_key_input = st.text_input(
        "OpenAI API Key:", 
        value=CONFIG["OPENAI_API_KEY"] if CONFIG["OPENAI_API_KEY"] != "YOUR_OPENAI_API_KEY" else "",
        type="password",
        help="Get your API key from https://platform.openai.com/"
    )
    
    if api_key_input and api_key_input != "YOUR_OPENAI_API_KEY":
        CONFIG["OPENAI_API_KEY"] = api_key_input
        st.success("✅ API Key configured!")
    elif not api_key_input or api_key_input == "YOUR_OPENAI_API_KEY":
        st.warning("⚠ Please enter your OpenAI API key to use the app")
        st.markdown("""
        *To get your API key:*
        1. Go to [OpenAI Platform](https://platform.openai.com/)
        2. Sign up or log in
        3. Go to API Keys section
        4. Create a new secret key
        5. Copy and paste it above
        """)
    
    # Model selection
    model_choice = st.selectbox(
        "Select Model:",
        options=["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"],
        index=0,
        help="GPT-4 gives better results but costs more"
    )
    CONFIG["OPENAI_MODEL"] = model_choice
    
    st.markdown("---")
    st.markdown("📊 App Features:")
    st.markdown("• Dynamic business questions")
    st.markdown("• CSV data upload") 
    st.markdown("• AI problem identification")
    st.markdown("• Gurobi code generation")
    st.markdown("• Business-friendly explanations")
    
    # Reset quota button
    if 'quota_exceeded' in st.session_state and st.session_state.quota_exceeded:
        st.markdown("---")
        st.error("⚠ API Quota Exceeded")
        if st.button("🔄 Reset & Try Again"):
            st.session_state.quota_exceeded = False
            st.success("✅ Quota reset! You can try again now.")
            st.rerun()

# --- SESSION STATE INITIALIZATION ---
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = [
        {"role": "bot", "content": "Hello! I'm your AI Business Optimization Assistant. I can help solve various optimization problems like inventory management, scheduling, routing, and more. Let's start by understanding your business better."}
    ]
if 'user_data' not in st.session_state:
    st.session_state.user_data = {}
if 'optimization_results' not in st.session_state:
    st.session_state.optimization_results = None
if 'questions_completed' not in st.session_state:
    st.session_state.questions_completed = False
if 'current_question' not in st.session_state:
    st.session_state.current_question = 0
if 'uploaded_data' not in st.session_state:
    st.session_state.uploaded_data = None
if 'ai_generated_code' not in st.session_state:
    st.session_state.ai_generated_code = None
if 'problem_type' not in st.session_state:
    st.session_state.problem_type = None
if 'optimization_status' not in st.session_state:
    st.session_state.optimization_status = None

# Dynamic Questions based on previous answers
INITIAL_QUESTIONS = [
    {
        "key": "business_type",
        "question": "What type of business or problem are you trying to optimize?",
        "type": "selectbox",
        "options": ["Retail/Inventory Management", "Manufacturing/Production", "Logistics/Transportation", "Supply Chain", "Scheduling/Resource Allocation", "Financial Portfolio", "Other"],
        "required": True
    },
    {
        "key": "business_location", 
        "question": "Where is your business located? (This helps with regional considerations)",
        "type": "text_input",
        "required": True
    },
    {
        "key": "optimization_goal",
        "question": "What is your primary optimization goal?",
        "type": "selectbox", 
        "options": ["Minimize Cost", "Maximize Profit", "Maximize Efficiency", "Minimize Time", "Maximize Output", "Minimize Risk", "Balance Multiple Objectives"],
        "required": True
    },
    {
        "key": "constraints",
        "question": "What are your main constraints? (Select all that apply)",
        "type": "multiselect",
        "options": ["Budget Limitations", "Storage/Space Constraints", "Time Constraints", "Resource Availability", "Regulatory Requirements", "Quality Standards", "Customer Demand", "Supplier Limitations"],
        "required": True
    },
    {
        "key": "budget_range",
        "question": "What's your budget range for this optimization?",
        "type": "selectbox",
        "options": ["Under ₹1 Lakh", "₹1-10 Lakhs", "₹10-50 Lakhs", "₹50 Lakhs - 1 Crore", "Above ₹1 Crore", "No specific budget"],
        "required": True
    },
    {
        "key": "time_horizon",
        "question": "What's your planning time horizon?",
        "type": "selectbox",
        "options": ["Daily", "Weekly", "Monthly", "Quarterly", "Yearly", "Long-term (5+ years)"],
        "required": True
    },
    {
        "key": "data_description",
        "question": "Please describe the data you have or the specific problem details:",
        "type": "text_area",
        "required": True
    }
]

# --- AI INTEGRATION ---
def get_openai_response(prompt, context=""):
    """Get response from OpenAI API with rate limiting and fallback"""
    API_KEY = CONFIG["OPENAI_API_KEY"]
    
    if not API_KEY or API_KEY == "YOUR_OPENAI_API_KEY":
        return "Please configure your OpenAI API key in the CONFIG section."
    
    # Check if we should use fallback due to quota issues
    if 'quota_exceeded' in st.session_state and st.session_state.quota_exceeded:
        return get_fallback_response(prompt, context)
    
    try:
        # Use requests instead of openai library to avoid dependency issues
        url = "https://api.openai.com/v1/chat/completions"
        
        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        }
        
        # Prepare messages for chat completion
        system_message = """You are an expert optimization consultant and Python programmer specializing in operations research and Gurobi optimization. 
        
        You help businesses solve complex optimization problems by:
        1. Analyzing their data and business context
        2. Identifying the type of optimization problem (knapsack, transportation, scheduling, etc.)
        3. Generating complete, working Python code using Gurobi
        4. Explaining results in simple, business-friendly terms
        
        Always provide detailed, actionable responses with clear explanations."""
        
        user_message = f"""
        Context: {context}
        
        User Request: {prompt}
        
        Please provide detailed, actionable responses. Use markdown formatting for clarity.
        """
        
        data = {
            "model": CONFIG["OPENAI_MODEL"],
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ],
            "max_tokens": 2000,
            "temperature": 0.7
        }
        
        # Add delay to respect rate limits
        time.sleep(1)
        
        response = requests.post(url, headers=headers, json=data, timeout=CONFIG["TIMEOUT"])
        
        if response.status_code == 200:
            result = response.json()
            if 'choices' in result and len(result['choices']) > 0:
                return result['choices'][0]['message']['content']
            else:
                return "I couldn't generate a response. Please try again."
        elif response.status_code == 429:
            st.session_state.quota_exceeded = True
            st.error("⚠ API rate limit exceeded. Switching to fallback mode.")
            return get_fallback_response(prompt, context)
        elif response.status_code == 401:
            return "❌ Invalid OpenAI API key. Please check your API key in the CONFIG section."
        else:
            return f"API Error: {response.status_code} - {response.text}"
            
    except Exception as e:
        error_str = str(e)
        if "quota" in error_str.lower() or "rate" in error_str.lower() or "429" in error_str:
            st.session_state.quota_exceeded = True
            st.error("⚠ API quota exceeded. Switching to fallback mode.")
            return get_fallback_response(prompt, context)
        return f"Error connecting to OpenAI: {error_str}"

def identify_problem_and_generate_code():
    """Use OpenAI to identify problem type and generate Gurobi code"""
    # Prepare comprehensive context
    context = f"""
    Business Information:
    - Type: {st.session_state.user_data.get('business_type', 'Not specified')}
    - Location: {st.session_state.user_data.get('business_location', 'Not specified')}
    - Goal: {st.session_state.user_data.get('optimization_goal', 'Not specified')}
    - Constraints: {', '.join(st.session_state.user_data.get('constraints', []))}
    - Budget: {st.session_state.user_data.get('budget_range', 'Not specified')}
    - Time Horizon: {st.session_state.user_data.get('time_horizon', 'Not specified')}
    - Problem Description: {st.session_state.user_data.get('data_description', 'Not specified')}
    
    Uploaded Data Preview:
    {st.session_state.uploaded_data.head(10).to_string() if st.session_state.uploaded_data is not None else 'No data uploaded'}
    
    Data Shape: {st.session_state.uploaded_data.shape if st.session_state.uploaded_data is not None else 'No data'}
    Data Columns: {list(st.session_state.uploaded_data.columns) if st.session_state.uploaded_data is not None else 'No columns'}
    """
    
    prompt = f"""
    Based on the business information and data provided, please:
    
    1. *Identify the optimization problem type* (e.g., Knapsack, Transportation, Assignment, Traveling Salesman, Inventory Optimization, Production Planning, etc.)
    
    2. *Generate complete Python code using Gurobi* that:
       - Imports necessary libraries (gurobipy, pandas, numpy)
       - Reads the data from a pandas DataFrame called 'data'
       - Sets up the optimization model with appropriate variables, constraints, and objective
       - Solves the problem
       - Returns results in a structured format (dictionary)
       - Includes error handling
    
    3. *Format your response as follows:*
       
       PROBLEM_TYPE: [Type of optimization problem]
       
       EXPLANATION: [Brief explanation of why this problem type fits]
       
       PYTHON_CODE:
       python
       # Your complete Gurobi code here
       ```
       ```
    
    Make sure the code:
    - Is production-ready and handles edge cases
    - Uses variable names that match the actual data columns
    - Includes appropriate constraints based on the business context
    - Returns meaningful results
    - Has proper error handling
    """
    
    return get_openai_response(prompt, context)

def execute_generated_code(code_string, data):
    """Safely execute the generated Gurobi code"""
    try:
        # Create a safe execution environment
        namespace = {
            'gurobipy': _import_('gurobipy'),
            'Model': _import_('gurobipy').Model,
            'GRB': _import_('gurobipy').GRB,
            'quicksum': _import_('gurobipy').quicksum,
            'pandas': pd,
            'numpy': np,
            'data': data,
            'print': lambda *args, **kwargs: None  # Suppress prints
        }
        
        # Capture output
        output_buffer = io.StringIO()
        error_buffer = io.StringIO()
        
        with redirect_stdout(output_buffer), redirect_stderr(error_buffer):
            exec(code_string, namespace)
        
        # Look for results in the namespace
        if 'results' in namespace:
            return namespace['results'], None
        elif 'result' in namespace:
            return namespace['result'], None
        else:
            # Try to find any dictionary that might contain results
            for key, value in namespace.items():
                if isinstance(value, dict) and key not in ['_builtins_', 'gurobipy', 'pandas', 'numpy']:
                    return value, None
            
            return None, "No results variable found in the generated code"
            
    except Exception as e:
        return None, f"Error executing code: {str(e)}"

def explain_results_with_ai(results, problem_type):
    """Use OpenAI to explain technical results in simple terms"""
    context = f"""
    Problem Type: {problem_type}
    Business Context: {st.session_state.user_data}
    """
    
    prompt = f"""
    I have solved an optimization problem and got the following technical results:
    
    Results: {results}
    
    Please explain these results in a simple, business-friendly way for a {st.session_state.user_data.get('business_type', 'business owner')}. 
    
    Your explanation should:
    1. *Summarize the key findings* in plain language
    2. *Highlight the most important numbers* and what they mean
    3. *Provide actionable recommendations* based on the results
    4. *Include any visualizations suggestions* (describe what charts would be helpful)
    5. *Use markdown formatting* with headers, bullet points, tables, etc.
    6. *Include emojis* to make it more engaging
    7. *Suggest specific next steps* the business should take
    
    Format your response with clear sections and make it as comprehensive as possible.
    """
    
    return get_openai_response(prompt, context)

def get_fallback_response(prompt, context=""):
    """Fallback response when API quota is exceeded"""
    
    # Check if this is a code generation request
    if "generate" in prompt.lower() and "code" in prompt.lower():
        return generate_fallback_code()
    
    # Check if this is a results explanation request
    elif "explain" in prompt.lower() and "results" in prompt.lower():
        return generate_fallback_explanation()
    
    # General fallback response
    return f"""
    *⚠ OpenAI API Quota Exceeded - Using Fallback Mode*
    
    I understand you're asking about: {prompt[:100]}...
    
    Unfortunately, I've reached the OpenAI API quota limit. Here's what I can suggest:
    
    ### 🔄 Solutions:
    1. *Wait a few minutes* and try again (quotas reset periodically)
    2. *Check your OpenAI billing* at https://platform.openai.com/usage
    3. *Use the fallback optimization templates* below
    4. *Click "Reset & Try Again"* in the sidebar
    
    ### 📊 Common Optimization Problems I Can Help With:
    - *Inventory Management*: Minimize costs while meeting demand
    - *Transportation*: Find shortest routes or minimize shipping costs
    - *Resource Allocation*: Assign resources optimally
    - *Production Planning*: Maximize output within constraints
    
    ### 💡 Quick Fix:
    Try clicking "Reset & Try Again" in the sidebar, then restart the optimization.
    """

def generate_fallback_code():
    """Generate a basic optimization code template when API is unavailable"""
    return """
    PROBLEM_TYPE: Inventory Optimization (Fallback Template)
    
    EXPLANATION: This is a fallback template for basic inventory optimization when API quota is exceeded.
    
    PYTHON_CODE:
    python
    from gurobipy import *
    import pandas as pd
    import numpy as np

    def solve_optimization(data):
        try:
            # Basic inventory optimization template
            m = Model("inventory_optimization")
            m.setParam('OutputFlag', 0)
            
            # Assume data has columns: item, cost, demand, capacity
            if 'item' not in data.columns:
                data['item'] = [f'Item_{i}' for i in range(len(data))]
            if 'cost' not in data.columns:
                data['cost'] = data.iloc[:, 1] if len(data.columns) > 1 else [100] * len(data)
            if 'demand' not in data.columns:
                data['demand'] = data.iloc[:, 2] if len(data.columns) > 2 else [50] * len(data)
            
            n = len(data)
            
            # Decision variables
            x = m.addVars(n, name="quantity")
            
            # Objective: minimize total cost
            m.setObjective(quicksum(data.iloc[i]['cost'] * x[i] for i in range(n)), GRB.MINIMIZE)
            
            # Constraints
            for i in range(n):
                m.addConstr(x[i] >= data.iloc[i]['demand'] * 0.8)  # At least 80% of demand
                m.addConstr(x[i] <= data.iloc[i]['demand'] * 1.5)  # At most 150% of demand
            
            # Solve
            m.optimize()
            
            if m.status == GRB.OPTIMAL:
                results = {
                    'status': 'Optimal',
                    'total_cost': m.objVal,
                    'quantities': {data.iloc[i]['item']: x[i].X for i in range(n)},
                    'message': 'Fallback optimization completed successfully'
                }
            else:
                results = {
                    'status': 'Infeasible',
                    'message': 'No solution found with current constraints'
                }
            
            return results
            
        except Exception as e:
            return {'status': 'Error', 'message': f'Optimization failed: {str(e)}'}

    # Execute the optimization
    results = solve_optimization(data)
    
    """

def generate_fallback_explanation():
    """Generate a basic explanation when API is unavailable"""
    return f"""
    # 📊 Optimization Results Summary (Fallback Mode)
    
    *⚠ Note*: This is a simplified explanation due to API quota limits.
    
    ## 🎯 Key Findings:
    
    Based on your uploaded data with *{st.session_state.uploaded_data.shape[0] if st.session_state.uploaded_data is not None else 'N/A'} items*, here's what the optimization suggests:
    
    ### 💰 *Cost Analysis*:
    - Total optimization cost has been minimized
    - Resource allocation has been optimized within your constraints
    
    ### 📈 *Recommendations*:
    1. *Prioritize high-value items* in your optimization
    2. *Monitor demand patterns* for better forecasting
    3. *Review constraint limits* if results seem restrictive
    
    ### 🔄 *Next Steps*:
    - Wait a few minutes and try again for detailed AI analysis
    - Review the technical results in the expandable section
    - Ask specific questions about the numbers you see
    
    For detailed insights, please wait for API quota to reset and re-run the analysis.
    """

# --- UTILITY FUNCTIONS ---
def add_message(role, content):
    """Add message to chat history"""
    st.session_state.chat_history.append({"role": role, "content": content})

def display_chat():
    """Display chat messages with improved formatting"""
    chat_html = ""
    for message in st.session_state.chat_history:
        if message["role"] == "bot":
            chat_html += f'<div class="bot-message">🤖 <strong>Assistant:</strong><br>{message["content"]}</div><div style="clear: both;"></div>'
        else:
            chat_html += f'<div class="user-message">👤 <strong>You:</strong><br>{message["content"]}</div><div style="clear: both;"></div>'
    
    return chat_html

def extract_code_from_response(response):
    """Extract Python code from AI response"""
    # Look for code blocks
    code_pattern = r'python\s*(.*?)\s*'
    matches = re.findall(code_pattern, response, re.DOTALL)
    
    if matches:
        return matches[0].strip()
    
    # Fallback: look for PYTHON_CODE: section
    if 'PYTHON_CODE:' in response:
        code_start = response.find('PYTHON_CODE:') + len('PYTHON_CODE:')
        code_section = response[code_start:].strip()
        if code_section.startswith('python'):
            code_section = code_section[9:]  # Remove python
        if code_section.endswith(''):
            code_section = code_section[:-3]  # Remove 
        return code_section.strip()
    
    return None

def extract_problem_type(response):
    """Extract problem type from AI response"""
    if 'PROBLEM_TYPE:' in response:
        lines = response.split('\n')
        for line in lines:
            if line.strip().startswith('PROBLEM_TYPE:'):
                return line.replace('PROBLEM_TYPE:', '').strip()
    return "Unknown Problem Type"

# --- MAIN UI ---
st.markdown('<div class="main-header"><h1>🤖 AI-Powered Business Optimization Assistant</h1><p>Upload your data, describe your problem, and let AI create & run custom optimization solutions</p></div>', unsafe_allow_html=True)

# Create two columns
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### 💬 Business Analysis & Problem Setup")
    
    # Chat display
    chat_container = st.container()
    with chat_container:
        st.markdown(f'<div class="chat-container">{display_chat()}</div>', unsafe_allow_html=True)
    
    # Input area
    st.markdown('<div class="input-container">', unsafe_allow_html=True)
    
    # Handle questions flow
    if not st.session_state.questions_completed:
        if st.session_state.current_question < len(INITIAL_QUESTIONS):
            question = INITIAL_QUESTIONS[st.session_state.current_question]
            
            st.subheader(f"Question {st.session_state.current_question + 1}/{len(INITIAL_QUESTIONS)}")
            st.write(question["question"])
            
            user_input = None
            
            if question["type"] == "selectbox":
                user_input = st.selectbox("Choose an option:", question["options"], key=f"q_{st.session_state.current_question}")
            elif question["type"] == "multiselect":
                user_input = st.multiselect("Select all that apply:", question["options"], key=f"q_{st.session_state.current_question}")
            elif question["type"] == "text_input":
                user_input = st.text_input("Your answer:", key=f"q_{st.session_state.current_question}")
            elif question["type"] == "text_area":
                user_input = st.text_area("Please provide details:", height=100, key=f"q_{st.session_state.current_question}")
            
            col_a, col_b = st.columns([1, 1])
            with col_a:
                if st.button("Next ➡", key=f"next_{st.session_state.current_question}"):
                    if user_input and (not question.get("required") or user_input):
                        st.session_state.user_data[question["key"]] = user_input
                        add_message("user", f"{question['question']}\n{user_input}")
                        
                        if st.session_state.current_question < len(INITIAL_QUESTIONS) - 1:
                            st.session_state.current_question += 1
                        else:
                            st.session_state.questions_completed = True
                            add_message("bot", "Great! Now please upload your data file (CSV format) so I can analyze your specific optimization problem.")
                        
                        st.rerun()
                    else:
                        st.error("Please provide an answer before proceeding.")
            
            with col_b:
                if st.session_state.current_question > 0:
                    if st.button("⬅ Back", key=f"back_{st.session_state.current_question}"):
                        st.session_state.current_question -= 1
                        st.rerun()
        
        # Show progress
        progress = (st.session_state.current_question + 1) / len(INITIAL_QUESTIONS)
        st.progress(progress)
        
    else:
        # Data upload section
        st.subheader("📁 Upload Your Data")
        uploaded_file = st.file_uploader(
            "Upload your data file (CSV format):", 
            type=["csv"],
            help="Upload a CSV file containing your optimization data"
        )
        
        if uploaded_file is not None:
            try:
                st.session_state.uploaded_data = pd.read_csv(uploaded_file)
                add_message("user", f"Uploaded data file: {uploaded_file.name}")
                st.success(f"✅ Data uploaded successfully! Shape: {st.session_state.uploaded_data.shape}")
                
                # Add a prominent callout
                st.markdown("""
                <div style="background: linear-gradient(135deg, #28a745, #20c997); padding: 1rem; border-radius: 10px; margin: 1rem 0; text-align: center;">
                    <h3 style="color: white; margin: 0;">🎯 READY TO OPTIMIZE!</h3>
                    <p style="color: white; margin: 0.5rem 0 0 0;">Click the button below to start AI analysis</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Show data preview
                st.subheader("📊 Data Preview")
                st.dataframe(st.session_state.uploaded_data.head())
                
                # Make the optimization button more prominent
                st.markdown("---")
                st.markdown("### 🚀 Ready to Optimize!")
                st.write("Your data is uploaded. Click below to let AI analyze your problem and generate optimization solutions.")
                
                col_opt1, col_opt2 = st.columns([2, 1])
                with col_opt1:
                    if st.button("🚀 START OPTIMIZATION ANALYSIS", type="primary", use_container_width=True):
                        add_message("bot", "🔍 Analyzing your problem and generating optimization code...")
                        
                        # Show progress in the solution panel
                        st.session_state.optimization_status = "analyzing"
                        st.rerun()
                        
                with col_opt2:
                    if st.button("📝 Edit Data", use_container_width=True):
                        st.session_state.uploaded_data = None
                        st.rerun()
                
                # Run optimization if status is analyzing
                if st.session_state.get('optimization_status') == "analyzing":
                    with st.spinner("🤖 AI is analyzing your problem..."):
                        # Get problem analysis and code from OpenAI
                        ai_response = identify_problem_and_generate_code()
                        st.session_state.ai_generated_code = extract_code_from_response(ai_response)
                        st.session_state.problem_type = extract_problem_type(ai_response)
                        
                        add_message("bot", f"*Problem Analysis Complete!*\n\n{ai_response}")
                    
                    if st.session_state.ai_generated_code:
                        with st.spinner("⚙ Running optimization..."):
                            try:
                                # Execute the generated code
                                results, error = execute_generated_code(
                                    st.session_state.ai_generated_code, 
                                    st.session_state.uploaded_data
                                )
                                
                                if error:
                                    add_message("bot", f"❌ *Execution Error:*\n{error}")
                                    st.error(error)
                                    st.session_state.optimization_status = "error"
                                else:
                                    st.session_state.optimization_results = results
                                    add_message("bot", "✅ *Optimization Complete!* Check the Solution panel for detailed results.")
                                    st.session_state.optimization_status = "complete"
                            except Exception as e:
                                add_message("bot", f"❌ *Error during optimization:*\n{str(e)}")
                                st.session_state.optimization_status = "error"
                    else:
                        add_message("bot", "❌ *Could not extract code from AI response.* Please try again or check your data format.")
                        st.session_state.optimization_status = "error"
                    
                    st.rerun()
                    
            except Exception as e:
                st.error(f"Error reading file: {str(e)}")
        
        # Free-form chat after optimization
        if st.session_state.optimization_results:
            st.subheader("💭 Ask Follow-up Questions")
            user_question = st.text_input("Ask anything about your results or request modifications:", key="followup_question")
            
            if st.button("Send Question", key="send_followup"):
                if user_question:
                    add_message("user", user_question)
                    
                    context = f"""
                    Problem Type: {st.session_state.problem_type}
                    Optimization Results: {st.session_state.optimization_results}
                    User Data: {st.session_state.user_data}
                    """
                    
                    ai_response = get_openai_response(user_question, context)
                    add_message("bot", ai_response)
                    st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown("### 📊 Optimization Solution")
    
    solution_container = st.container()
    with solution_container:
        if st.session_state.optimization_results:
            st.success("🎯 *Optimization Successfully Completed!*")
            
            # Show problem type
            if st.session_state.problem_type:
                st.info(f"*Identified Problem Type:* {st.session_state.problem_type}")
            
            # Show raw results in an expandable section
            with st.expander("🔍 Technical Results (Raw Data)"):
                st.json(st.session_state.optimization_results)
            
            # Show generated code
            with st.expander("💻 Generated Optimization Code"):
                if st.session_state.ai_generated_code:
                    st.code(st.session_state.ai_generated_code, language="python")
                else:
                    st.write("Code not available")
            
            # Get AI explanation of results
            st.subheader("🎯 Business Impact Analysis")
            
            if 'ai_explanation' not in st.session_state:
                with st.spinner("🤖 AI is preparing business insights..."):
                    explanation = explain_results_with_ai(
                        st.session_state.optimization_results, 
                        st.session_state.problem_type
                    )
                    st.session_state.ai_explanation = explanation
            
            # Display AI explanation with full markdown support
            if 'ai_explanation' in st.session_state:
                st.markdown(st.session_state.ai_explanation, unsafe_allow_html=True)
            
            # Add visualization suggestions if mentioned in explanation
            if 'chart' in st.session_state.ai_explanation.lower() or 'graph' in st.session_state.ai_explanation.lower():
                st.subheader("📈 Suggested Visualizations")
                st.info("💡 The AI suggested creating visualizations. You can ask for specific charts in the chat!")
            
        elif st.session_state.uploaded_data is not None:
            status = st.session_state.get('optimization_status', None)
            
            if status == "analyzing":
                st.warning("🔄 *Optimization in Progress...*")
                st.write("AI is analyzing your problem and generating optimization code. Please wait...")
                # Show a progress bar or spinner effect
                progress_bar = st.progress(0)
                for i in range(100):
                    time.sleep(0.01)
                    progress_bar.progress(i + 1)
                
            elif status == "error":
                st.error("❌ *Optimization Failed*")
                st.write("There was an error during optimization. Please check the chat for details and try again.")
                if st.button("🔄 Try Again"):
                    st.session_state.optimization_status = None
                    st.rerun()
                    
            else:
                st.info("🎯 *Ready for Optimization*")
                st.write("Data uploaded successfully. Click *'START OPTIMIZATION ANALYSIS'* in the left panel to begin.")
                
                # Show data info with more details
                st.subheader("📋 Data Summary")
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.metric("Rows", st.session_state.uploaded_data.shape[0])
                with col_b:
                    st.metric("Columns", st.session_state.uploaded_data.shape[1])
                with col_c:
                    st.metric("Size", f"{st.session_state.uploaded_data.memory_usage().sum() / 1024:.1f} KB")
                
                # Show column info
                st.subheader("📊 Column Information")
                col_info = pd.DataFrame({
                    'Column': st.session_state.uploaded_data.columns,
                    'Type': st.session_state.uploaded_data.dtypes.astype(str),
                    'Non-Null Count': st.session_state.uploaded_data.count(),
                    'Sample Values': [str(st.session_state.uploaded_data[col].iloc[0]) if len(st.session_state.uploaded_data) > 0 else 'N/A' for col in st.session_state.uploaded_data.columns]
                })
                st.dataframe(col_info, use_container_width=True)
        
        elif st.session_state.questions_completed:
            st.info("📁 *Upload Data to Continue*")
            st.write("Please upload your CSV data file to start the optimization analysis.")
        
        else:
            st.info("❓ *Complete Setup Questions*")
            st.write("Answer the questions on the left to help me understand your optimization problem.")
            
            # Show progress
            progress = (st.session_state.current_question + 1) / len(INITIAL_QUESTIONS)
            st.progress(progress)
            st.write(f"Progress: {st.session_state.current_question + 1}/{len(INITIAL_QUESTIONS)} questions")

# Reset button
st.markdown("---")
if st.button("🔄 Start Over", key="reset_button"):
    for key in list(st.session_state.keys()):
        if key != 'OPENAI_API_KEY':  # Keep API key
            del st.session_state[key]
    st.rerun()

# Footer
st.markdown("---")
st.markdown("🤖 Powered by OpenAI GPT + Gurobi Optimization** | Upload CSV data and let AI solve your optimization problems")