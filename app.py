import os
from flask import Flask, request, jsonify, render_template, url_for, redirect, flash, session
from transformers import AutoModelForCausalLM, AutoTokenizer
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from flask_cors import CORS
from flask_mail import Mail, Message
import torch

# Increase timeout for Hugging Face downloads
os.environ['HF_HUB_HTTP_TIMEOUT'] = '600'  # 10 minutes

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configuration for the Flask app
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', os.urandom(24))
app.config['MAIL_SERVER'] = 'smtp.gmail.com'  # Use for Gmail
app.config['MAIL_PORT'] = 465
app.config['MAIL_USERNAME'] = os.getenv('MAIL_USERNAME', 'nynightcompany@gmail.com')  # Replace with your email
app.config['MAIL_PASSWORD'] = os.getenv('MAIL_PASSWORD', '@lex951Mps')  # Replace with your email password
app.config['MAIL_USE_TLS'] = False
app.config['MAIL_USE_SSL'] = True
app.config['MAIL_DEFAULT_SENDER'] = os.getenv('MAIL_USERNAME', 'nynightcompany@gmail.com')  # Replace with your email

# Initialize extensions
db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
mail = Mail(app)

# Load DeepSeek model and tokenizer
model_name = "DeepSeek-R1-Distill-Llama"  # Use a smaller DeepSeek model
try:
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    print("Tokenizer loaded successfully!")

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained("distilgpt2", torch_dtype=torch.float32)  # Use float32 for CPU
    device = torch.device("cpu")  # Force CPU usage
    model.to(device)
    print("Model loaded successfully!")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
except Exception as e:
    print(f"Error loading model: {e}")
    model = None
    tokenizer = None

# User model for the database
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)

# Create the database if it doesn't exist
with app.app_context():
    if not os.path.exists('users.db'):
        db.create_all()

# Helper function to format prompts for DeepSeek
def format_prompt(query):
    return f"### Instruction:\n{query}\n### Response:\n"

@app.route('/')
def index():
    is_logged_in = 'user' in session
    return render_template('index.html', is_logged_in=is_logged_in)

# Route: Login page
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')

        if not email or not password:
            flash("Both email and password are required!", "error")
            return redirect(url_for('login'))

        user = User.query.filter_by(email=email).first()
        if user and bcrypt.check_password_hash(user.password, password):
            session['user'] = user.email
            flash(f"Welcome back, {user.email}!", "success")
            return redirect(url_for('index'))
        else:
            flash("Invalid email or password!", "error")
            return redirect(url_for('login'))

    return render_template('login.html')

# Route: Registration page
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')

        if not email or not password:
            flash("All fields are required!", "error")
            return redirect(url_for('register'))

        if password != confirm_password:
            flash("Passwords do not match!", "error")
            return redirect(url_for('register'))

        existing_user = User.query.filter_by(email=email).first()
        if existing_user:
            flash("Email already registered!", "error")
            return redirect(url_for('register'))

        hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
        new_user = User(email=email, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()
        flash("Registration successful! Please log in.", "success")
        return redirect(url_for('login'))

    return render_template('register.html')

@app.route('/process', methods=['POST'])
def process():
    if model is None or tokenizer is None:
        return jsonify({'response': 'Model is not available. Please check the server logs.'}), 500

    data = request.get_json()
    query = data.get('query', '')

    if not query:
        return jsonify({'response': 'No query provided.'}), 400

    try:
        # Format the prompt for DeepSeek
        formatted_prompt = format_prompt(query)
        
        # Tokenize with proper formatting and generate attention mask
        inputs = tokenizer(formatted_prompt, return_tensors="pt", padding=True, truncation=True)
        inputs = {key: value.to(device) for key, value in inputs.items()}  # Move inputs to the correct device
        
        # Generate response with adjusted parameters
        outputs = model.generate(
            inputs['input_ids'],  # Access input_ids using square brackets
            attention_mask=inputs['attention_mask'],  # Access attention_mask using square brackets
            max_new_tokens=150,  # Reduce max tokens for CPU
            temperature=0.7,      # Adjust for creativity
            top_p=0.9,            # Adjust for diversity
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            num_return_sequences=1,
            use_cache=True  # Use caching for better performance
        )
        
        # Decode and clean up response
        full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract only the response part after "### Response:"
        response = full_response.split("### Response:\n")[-1].strip()

        return jsonify({'response': response})
    
    except Exception as e:
        return jsonify({'response': f"Error generating response: {str(e)}"}), 500

# Route: Dashboard page
@app.route('/dashboard')
def dashboard():
    if 'user' not in session:
        flash("Please log in to access the dashboard.", "error")
        return redirect(url_for('login'))
    return render_template('index.html')

# Route: Logout
@app.route('/logout')
def logout():
    session.pop('user', None)  # Remove user from session
    return redirect(url_for('index'))

@app.route('/ny_ai')
def ny_ai():
    return render_template('Ny_AI.html')

@app.route('/contact')
def contact_page():
    return render_template('contact.html')

@app.route('/software')
def software():
    return render_template('software.html')

@app.route('/ios')
def ios():
    return render_template('ios.html')

@app.route('/android')
def android():
    return render_template('android.html')

@app.route('/web')
def web():
    return render_template('web.html')

@app.route('/jobs')
def jobs():
    return render_template('jobs.html')

@app.route('/AIP')
def AIP():
    return render_template('AIP.html')

@app.route('/send-email', methods=['POST'])
def send_email():
    try:
        data = request.get_json()

        # Get message details
        to_email = data['to']
        subject = data['subject']
        message_body = data['message']

        # Create the email message
        msg = Message(
            subject=subject,
            recipients=[to_email],
            body=message_body
        )

        # Send the email
        mail.send(msg)

        return jsonify({'status': 'success', 'message': 'Message Sent!'})
    except Exception as e:
        # Log the error for debugging
        print(f"Error sending email: {str(e)}")
        return jsonify({'status': 'error', 'message': f"Error: {str(e)}"})

if __name__ == '__main__':
    app.run(debug=True)