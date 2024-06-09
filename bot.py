from flask import Flask, request, jsonify
import random
from difflib import get_close_matches
from flask_cors import CORS
from langchain import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import fitz  # PyMuPDF

app = Flask(__name__)
CORS(app)

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text += page.get_text()
    return text

# Load your dataset from PDF
pdf_text = extract_text_from_pdf('dataset.pdf')

# Assume that each question-answer pair in the PDF is separated by a specific delimiter, such as a newline or any other delimiter.
# Here we assume that each pair is separated by double newline and question-answer are separated by a single newline.
pairs = pdf_text.strip().split("\n\n")
dataset = [tuple(pair.split("\n")) for pair in pairs if "\n" in pair]

# Initialize the GPT-4 API through LangChain
llm = OpenAI(model="gpt-4", api_key="openai_api_key")

# Define intents, entities, and actions
intents = {
    "greet": ["hi", "hello", "hey", "hi there", "good morning"],
    "goodbye": ["bye", "goodbye", "see you", "talk to you later"],
    "ask_purpose": ["what is the purpose?", "why are you here?", "what can you do?"],
    "ask_features": ["what all features do you have?", "what can you help me with?"],
    "ask_syllabus": ["give me the syllabus for data science?", "what topics are covered in data science?"],
    "ask_packages": ["what are the packages for data science?", "which libraries are used in data science?"],
    "ask_ml": ["what is machine learning?", "explain machine learning"],
    "ask_dl": ["what is deep learning?", "explain deep learning"],
    "ask_nn": ["what is a neural network?", "explain neural networks"],
    "ask_supervised_learning": ["how does supervised learning work?", "explain supervised learning"],
    "ask_unsupervised_learning": ["what is unsupervised learning?", "explain unsupervised learning"],
    "ask_reinforcement_learning": ["what is reinforcement learning?", "explain reinforcement learning"],
    "ask_overfitting": ["what is overfitting?", "explain overfitting"],
    "ask_underfitting": ["what is underfitting?", "explain underfitting"],
    "ask_model_performance": ["how to improve model performance?", "what are ways to improve model performance?"],
    "ask_python_help": ["can you help me with python programming?", "do you know python?"]
}

actions = {
    "greet": "Hi! How can I help you today?",
    "goodbye": "Goodbye! Have a great day!",
    "ask_purpose": "The purpose of this chatbot is to help you prepare for data science interviews by providing answers to common questions.",
    "ask_features": "I can provide answers to data science questions, give explanations on various topics, and help you understand key concepts.",
    "ask_syllabus": "The syllabus for data science typically includes topics such as statistics, machine learning, data visualization, data wrangling, and deep learning.",
    "ask_packages": "Common packages for data science include NumPy, Pandas, Matplotlib, Scikit-Learn, TensorFlow, and PyTorch.",
    "ask_ml": "Machine learning is a field of artificial intelligence that involves training algorithms to learn patterns from data and make predictions or decisions based on that data.",
    "ask_dl": "Deep learning is a subset of machine learning that uses neural networks with many layers (deep neural networks) to model complex patterns in data.",
    "ask_nn": "A neural network is a computational model inspired by the human brain, consisting of interconnected layers of nodes (neurons) that process data and learn to make predictions.",
    "ask_supervised_learning": "Supervised learning involves training a model on labeled data, where the input data is paired with the correct output, to learn a mapping from inputs to outputs.",
    "ask_unsupervised_learning": "Unsupervised learning involves training a model on unlabeled data to discover hidden patterns or structures in the data, such as clustering or dimensionality reduction.",
    "ask_reinforcement_learning": "Reinforcement learning is a type of machine learning where an agent learns to make decisions by receiving rewards or penalties for its actions in an environment.",
    "ask_overfitting": "Overfitting occurs when a model learns the training data too well, capturing noise and outliers, and performs poorly on new, unseen data.",
    "ask_underfitting": "Underfitting occurs when a model is too simple to capture the underlying patterns in the data, resulting in poor performance on both training and test data.",
    "ask_model_performance": "To improve model performance, you can try techniques such as cross-validation, hyperparameter tuning, feature engineering, and using more advanced algorithms.",
    "ask_python_help": "Yes, I can help you with Python programming. Feel free to ask any Python-related questions."
}

# List of default responses
default_responses = [
    "I'm not sure about that. Can you rephrase your question?",
    "I don't have the answer to that right now. Could you ask something else?",
    "That's a good question! Can you ask it in a different way or ask another question?"
]

# Predefined responses for common questions
predefined_responses = {
    "what is the purpose?": "The purpose of this chatbot is to help you prepare for data science interviews by providing answers to common questions.",
    "what all questions can you answer?": "I can answer a wide range of data science questions, including those related to supervised learning, neural networks, clustering, and more.",
    "are you a chatbot?": "Yes, I am a chatbot designed to assist you with data science interview preparation.",
    "what all features do you have?": "I can provide answers to data science questions, give explanations on various topics, and help you understand key concepts.",
    "give me the syllabus for data science?": "The syllabus for data science typically includes topics such as statistics, machine learning, data visualization, data wrangling, and deep learning.",
    "what are the packages for data science?": "Common packages for data science include NumPy, Pandas, Matplotlib, Scikit-Learn, TensorFlow, and PyTorch.",
    "what is master's in data science?": "A Master's in Data Science is an advanced degree focusing on data analysis, machine learning, statistical modeling, and big data technologies.",
    "how are you?": "I am a chatbot, so I don't have feelings, but I'm here to help you with your data science questions!",
    "what do you do?": "I assist users in preparing for data science interviews by answering questions and explaining concepts.",
    "how can I help you?": "You can help me by asking any data science-related questions you have, and I'll do my best to provide useful answers.",
    "can you help me with python programming?": "Yes, I can help you with Python programming. Feel free to ask any Python-related questions.",
    "what is machine learning?": "Machine learning is a field of artificial intelligence that involves training algorithms to learn patterns from data and make predictions or decisions based on that data.",
    "what is deep learning?": "Deep learning is a subset of machine learning that uses neural networks with many layers (deep neural networks) to model complex patterns in data.",
    "what is a neural network?": "A neural network is a computational model inspired by the human brain, consisting of interconnected layers of nodes (neurons) that process data and learn to make predictions.",
    "how does supervised learning work?": "Supervised learning involves training a model on labeled data, where the input data is paired with the correct output, to learn a mapping from inputs to outputs.",
    "what is unsupervised learning?": "Unsupervised learning involves training a model on unlabeled data to discover hidden patterns or structures in the data, such as clustering or dimensionality reduction.",
    "what is reinforcement learning?": "Reinforcement learning is a type of machine learning where an agent learns to make decisions by receiving rewards or penalties for its actions in an environment.",
    "what is overfitting?": "Overfitting occurs when a model learns the training data too well, capturing noise and outliers, and performs poorly on new, unseen data.",
    "what is underfitting?": "Underfitting occurs when a model is too simple to capture the underlying patterns in the data, resulting in poor performance on both training and test data.",
    "how to improve model performance?": "To improve model performance, you can try techniques such as cross-validation, hyperparameter tuning, feature engineering, and using more advanced algorithms."
}

# Function to generate responses using the GPT-4 API
def generate_response(prompt, dataset):
    # Identify intent based on the prompt
    intent = None
    for key, phrases in intents.items():
        if any(phrase in prompt.lower() for phrase in phrases):
            intent = key
            break
    
    # Execute action based on identified intent
    if intent in actions:
        return actions[intent]

    # Check for partial matches in predefined responses
    close_matches = get_close_matches(prompt.lower(), predefined_responses.keys(), n=1, cutoff=0.6)
    if close_matches:
        return predefined_responses[close_matches[0]]

    # Check if the prompt matches any question in the dataset
    for question, answer in dataset:
        if question.lower() in prompt.lower():
            return answer

    # If no match is found, use the GPT-4 API to generate a response
    prompt_template = PromptTemplate(input_variables=["question"], template="Q: {question}\nA:")
    chain = LLMChain(prompt_template=prompt_template, llm=llm)
    response = chain.run({"question": prompt})

    # If the generated response is empty or non-informative, return a default response
    if not response.strip():
        response = random.choice(default_responses)
    
    return response

@app.route('/get_response', methods=['POST'])
def get_response():
    user_input = request.json['user_input']
    response = generate_response(user_input, dataset)
    return jsonify({'bot_response': response})

if __name__ == '__main__':
    app.run(host="0.0.0.0", port="8000", debug=True)
