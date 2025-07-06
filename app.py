# app_gemini_powered.py

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
    "GEMINI_API_KEY": "AIzaSyAeu8ExwlQe_ntdW3gidAruxPqM51vKato",
    "MAX_RETRIES": 3,
    "TIMEOUT": 30
}

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
if 'gemini_generated_code' not in st.session_state:
    st.session_state.gemini_generated_code = None
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
def get_gemini_response(prompt, context=""):
    """Get response from Gemini API"""
    API_KEY = CONFIG["GEMINI_API_KEY"]
    
    if not API_KEY or API_KEY == "YOUR_GEMINI_API_KEY":
        return "Please configure your Gemini API key in the CONFIG section."
    
    try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={API_KEY}"
        
        headers = {'Content-Type': 'application/json'}
        
        full_prompt = f"""
        You are an expert optimization consultant and Python programmer specializing in operations research and Gurobi optimization.
        
        Context: {context}
        
        User Request: {prompt}
        
        Please provide detailed, actionable responses. Use markdown formatting for clarity.
        """
        
        data = {
            "contents": [{
                "parts": [{
                    "text": full_prompt
                }]
            }]
        }
        
        response = requests.post(url, headers=headers, json=data, timeout=CONFIG["TIMEOUT"])
        
        if response.status_code == 200:
            result = response.json()
            if 'candidates' in result and len(result['candidates']) > 0:
                return result['candidates'][0]['content']['parts'][0]['text']
            else:
                return "I couldn't generate a response. Please try again."
        else:
            return f"API Error: {response.status_code} - {response.text}"
            
    except Exception as e:
        return f"Error connecting to Gemini: {str(e)}"

def identify_problem_and_generate_code():
    """Use Gemini to identify problem type and generate Gurobi code"""
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
    
    1. **Identify the optimization problem type** (e.g., Knapsack, Transportation, Assignment, Traveling Salesman, Inventory Optimization, Production Planning, etc.)
    
    2. **Generate complete Python code using Gurobi** that:
       - Imports necessary libraries (gurobipy, pandas, numpy)
       - Reads the data from a pandas DataFrame called 'data'
       - Sets up the optimization model with appropriate variables, constraints, and objective
       - Solves the problem
       - Returns results in a structured format (dictionary)
       - Includes error handling
    
    3. **Format your response as follows:**
       ```
       PROBLEM_TYPE: [Type of optimization problem]
       
       EXPLANATION: [Brief explanation of why this problem type fits]
       
       PYTHON_CODE:
       ```python
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
    
    return get_gemini_response(prompt, context)

def execute_generated_code(code_string, data):
    try:
        namespace = {
            'gurobipy': __import__('gurobipy'),
            'Model': __import__('gurobipy').Model,
            'GRB': __import__('gurobipy').GRB,
            'quicksum': __import__('gurobipy').quicksum,
            'pd': pd,
            'np': np,
            'data': data,
        }
        # Execute the function definition
        exec(code_string, namespace)
        # Find the function name (assume it's the first 'def ...' in the code)
        match = re.search(r'def (\\w+)\\(', code_string)
        if match:
            func_name = match.group(1)
            result = namespace[func_name](data)
            return result, None
        else:
            return None, "No function found in generated code."
    except Exception as e:
        return None, f"Error executing code: {str(e)}"

def explain_results_with_gemini(results, problem_type):
    """Use Gemini to explain technical results in simple terms"""
    context = f"""
    Problem Type: {problem_type}
    Business Context: {st.session_state.user_data}
    """
    
    prompt = f"""
    I have solved an optimization problem and got the following technical results:
    
    Results: {results}
    
    Please explain these results in a simple, business-friendly way for a {st.session_state.user_data.get('business_type', 'business owner')}. 
    
    Your explanation should:
    1. **Summarize the key findings** in plain language
    2. **Highlight the most important numbers** and what they mean
    3. **Provide actionable recommendations** based on the results
    4. **Include any visualizations suggestions** (describe what charts would be helpful)
    5. **Use markdown formatting** with headers, bullet points, tables, etc.
    6. **Include emojis** to make it more engaging
    7. **Suggest specific next steps** the business should take
    
    Format your response with clear sections and make it as comprehensive as possible.
    """
    
    return get_gemini_response(prompt, context)

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
    # Extract the first python code block
    code_pattern = r'```python\\s*(.*?)\\s*```'
    matches = re.findall(code_pattern, response, re.DOTALL)
    if matches:
        code = matches[0].strip()
        # Remove any __main__ block and print statements
        code = re.sub(r"if __name__ == .+?\\n(.|\\n)*", "", code)
        code = re.sub(r"print\\(.*?\\)", "", code)
        return code
    return None
    
def extract_problem_type(response):
    """Extract problem type from Gemini's response"""
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
                if st.button("Next ➡️", key=f"next_{st.session_state.current_question}"):
                    if user_input and (not question.get("required") or user_input):
                        st.session_state.user_data[question["key"]] = user_input
                        add_message("user", f"**{question['question']}**\n{user_input}")
                        
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
                    if st.button("⬅️ Back", key=f"back_{st.session_state.current_question}"):
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
                        # Get problem analysis and code from Gemini
                        gemini_response = identify_problem_and_generate_code()
                        st.session_state.gemini_generated_code = extract_code_from_response(gemini_response)
                        st.session_state.problem_type = extract_problem_type(gemini_response)
                        
                        add_message("bot", f"**Problem Analysis Complete!**\n\n{gemini_response}")
                    
                    if st.session_state.gemini_generated_code:
                        with st.spinner("⚙️ Running optimization..."):
                            try:
                                # Execute the generated code
                                results, error = execute_generated_code(
                                    st.session_state.gemini_generated_code, 
                                    st.session_state.uploaded_data
                                )
                                
                                if error:
                                    add_message("bot", f"❌ **Execution Error:**\n{error}")
                                    st.error(error)
                                    st.session_state.optimization_status = "error"
                                else:
                                    st.session_state.optimization_results = results
                                    add_message("bot", "✅ **Optimization Complete!** Check the Solution panel for detailed results.")
                                    st.session_state.optimization_status = "complete"
                            except Exception as e:
                                add_message("bot", f"❌ **Error during optimization:**\n{str(e)}")
                                st.session_state.optimization_status = "error"
                    else:
                        add_message("bot", "❌ **Could not extract code from AI response.** Please try again or check your data format.")
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
                    
                    ai_response = get_gemini_response(user_question, context)
                    add_message("bot", ai_response)
                    st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown("### 📊 Optimization Solution")
    
    solution_container = st.container()
    with solution_container:
        if st.session_state.optimization_results:
            st.success("🎯 **Optimization Successfully Completed!**")
            
            # Show problem type
            if st.session_state.problem_type:
                st.info(f"**Identified Problem Type:** {st.session_state.problem_type}")
            
            # Show raw results in an expandable section
            with st.expander("🔍 Technical Results (Raw Data)"):
                st.json(st.session_state.optimization_results)
            
            # Show generated code
            with st.expander("💻 Generated Optimization Code"):
                if st.session_state.gemini_generated_code:
                    st.code(st.session_state.gemini_generated_code, language="python")
                else:
                    st.write("Code not available")
            
            # Get AI explanation of results
            st.subheader("🎯 Business Impact Analysis")
            
            if 'ai_explanation' not in st.session_state:
                with st.spinner("🤖 AI is preparing business insights..."):
                    explanation = explain_results_with_gemini(
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
                st.warning("🔄 **Optimization in Progress...**")
                st.write("AI is analyzing your problem and generating optimization code. Please wait...")
                # Show a progress bar or spinner effect
                progress_bar = st.progress(0)
                for i in range(100):
                    time.sleep(0.01)
                    progress_bar.progress(i + 1)
                
            elif status == "error":
                st.error("❌ **Optimization Failed**")
                st.write("There was an error during optimization. Please check the chat for details and try again.")
                if st.button("🔄 Try Again"):
                    st.session_state.optimization_status = None
                    st.rerun()
                    
            else:
                st.info("🎯 **Ready for Optimization**")
                st.write("Data uploaded successfully. Click **'START OPTIMIZATION ANALYSIS'** in the left panel to begin.")
                
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
            st.info("📁 **Upload Data to Continue**")
            st.write("Please upload your CSV data file to start the optimization analysis.")
        
        else:
            st.info("❓ **Complete Setup Questions**")
            st.write("Answer the questions on the left to help me understand your optimization problem.")
            
            # Show progress
            progress = (st.session_state.current_question + 1) / len(INITIAL_QUESTIONS)
            st.progress(progress)
            st.write(f"Progress: {st.session_state.current_question + 1}/{len(INITIAL_QUESTIONS)} questions")

# Reset button
st.markdown("---")
if st.button("🔄 Start Over", key="reset_button"):
    for key in list(st.session_state.keys()):
        if key != 'GEMINI_API_KEY':  # Keep API key
            del st.session_state[key]
    st.rerun()

# Footer
st.markdown("---")
st.markdown("**🤖 Powered by Google Gemini AI + Gurobi Optimization** | *Upload CSV data and let AI solve your optimization problems*")
