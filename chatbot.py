import tkinter as tk
from tkinter import scrolledtext, ttk, messagebox
import random
import json
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import threading
import time
from datetime import datetime

# Ensure NLTK data is downloaded
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

class AdvancedChatbot:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.knowledge_base = self.load_knowledge_base()
        self.prepare_knowledge_base()
        self.user_history = []
        self.session_start = datetime.now()
        
        # Personality traits
        self.personality = {
            'name': 'Aurora',
            'mood': 'neutral',
            'responsiveness': 0.8,
            'verbosity': 0.7
        }
        
        # Initialize GUI
        self.init_gui()
    
    def load_knowledge_base(self):
        try:
            with open('knowledge_base.json', 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            # Default knowledge base if file doesn't exist
            return {
                "intents": [
                    {
                        "tag": "greeting",
                        "patterns": ["Hi", "Hello", "Hey", "Good morning", "Good afternoon"],
                        "responses": ["Hello! How can I assist you today?", "Hi there! What can I do for you?", "Greetings! How may I help you?"]
                    },
                    {
                        "tag": "goodbye",
                        "patterns": ["Bye", "Goodbye", "See you later", "Have a nice day"],
                        "responses": ["Goodbye! Have a great day!", "See you soon!", "Until next time!"]
                    },
                    {
                        "tag": "thanks",
                        "patterns": ["Thanks", "Thank you", "That's helpful", "Appreciate it"],
                        "responses": ["You're welcome!", "My pleasure!", "Glad I could help!"]
                    },
                    {
                        "tag": "about",
                        "patterns": ["Who are you", "What are you", "Tell me about yourself"],
                        "responses": ["I'm Aurora, an advanced AI chatbot designed to assist you.", "I'm an intelligent virtual assistant here to help with your queries."]
                    },
                    {
                        "tag": "time",
                        "patterns": ["What time is it", "Current time", "Time now"],
                        "responses": [f"The current time is {datetime.now().strftime('%H:%M:%S')}"]
                    }
                ],
                "context": {
                    "last_intent": None,
                    "follow_up": None
                }
            }
    
    def save_knowledge_base(self):
        with open('knowledge_base.json', 'w') as f:
            json.dump(self.knowledge_base, f, indent=4)
    
    def prepare_knowledge_base(self):
        # Prepare data for TF-IDF vectorization
        self.all_patterns = []
        self.all_tags = []
        
        for intent in self.knowledge_base['intents']:
            for pattern in intent['patterns']:
                self.all_patterns.append(self.preprocess_text(pattern))
                self.all_tags.append(intent['tag'])
        
        # Train TF-IDF vectorizer
        if self.all_patterns:
            self.X = self.vectorizer.fit_transform(self.all_patterns)
    
    def preprocess_text(self, text):
        # Tokenize and lemmatize
        tokens = nltk.word_tokenize(text.lower())
        lemmatized = [self.lemmatizer.lemmatize(token) for token in tokens]
        return ' '.join(lemmatized)
    
    def get_response(self, user_input):
        # Update user history
        self.user_history.append({
            'input': user_input,
            'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })
        
        # Preprocess input
        processed_input = self.preprocess_text(user_input)
        
        # Vectorize input
        input_vec = self.vectorizer.transform([processed_input])
        
        # Calculate similarity scores
        if hasattr(self, 'X'):
            similarities = cosine_similarity(input_vec, self.X)
            max_sim_idx = np.argmax(similarities)
            max_sim = similarities[0, max_sim_idx]
            
            # Threshold for considering it a match
            if max_sim > 0.5:
                matched_tag = self.all_tags[max_sim_idx]
                
                # Find the intent with this tag
                for intent in self.knowledge_base['intents']:
                    if intent['tag'] == matched_tag:
                        # Update context
                        self.knowledge_base['context']['last_intent'] = intent['tag']
                        
                        # Select response based on personality
                        response = random.choice(intent['responses'])
                        return self.adjust_response_tone(response)
        
        # Fallback response if no match found
        fallbacks = [
            "I'm not sure I understand. Could you rephrase that?",
            "That's an interesting point. Could you elaborate?",
            "I'm still learning. Can you ask me something else?",
            "I don't have enough information to answer that properly."
        ]
        
        # Check if we're in a follow-up context
        if self.knowledge_base['context']['follow_up']:
            response = self.handle_follow_up(user_input)
            if response:
                return response
        
        return self.adjust_response_tone(random.choice(fallbacks))
    
    def adjust_response_tone(self, response):
        """Adjust response based on personality traits"""
        # Adjust based on mood
        if self.personality['mood'] == 'happy':
            response = response.replace('.', '!').replace('?', '?!')
        elif self.personality['mood'] == 'serious':
            response = response.capitalize()
        
        # Adjust verbosity
        if self.personality['verbosity'] < 0.5 and len(response.split()) > 10:
            response = ' '.join(response.split()[:10]) + '...'
        
        return response
    
    def handle_follow_up(self, user_input):
        """Handle follow-up questions based on context"""
        # This can be expanded based on specific needs
        last_intent = self.knowledge_base['context']['last_intent']
        
        if last_intent == 'time':
            if 'difference' in user_input.lower() or 'zone' in user_input.lower():
                return "I currently don't have timezone conversion capabilities."
        
        self.knowledge_base['context']['follow_up'] = None
        return None
    
    def init_gui(self):
        """Initialize the graphical user interface"""
        self.root = tk.Tk()
        self.root.title("Advanced AI Chatbot - Designed & Developed by Jilson C J")
        self.root.geometry("800x600")
        self.root.resizable(True, True)
        self.root.configure(bg='#f0f0f0')
        
        # Style configuration
        self.style = ttk.Style()
        self.style.theme_use('clam')
        self.style.configure('TFrame', background='#f0f0f0')
        self.style.configure('TButton', font=('Arial', 10), padding=5)
        self.style.configure('TLabel', font=('Arial', 10), background='#f0f0f0')
        self.style.configure('Header.TLabel', font=('Arial', 14, 'bold'))
        
        # Create header
        header_frame = ttk.Frame(self.root)
        header_frame.pack(pady=10)
        
        ttk.Label(header_frame, text="AURORA AI CHATBOT", style='Header.TLabel').pack()
        ttk.Label(header_frame, text="Advanced Conversational AI", style='TLabel').pack()
        
        # Create chat display area
        chat_frame = ttk.Frame(self.root)
        chat_frame.pack(pady=10, padx=20, fill=tk.BOTH, expand=True)
        
        self.chat_display = scrolledtext.ScrolledText(
            chat_frame,
            wrap=tk.WORD,
            width=60,
            height=20,
            font=('Arial', 10),
            bg='white',
            fg='black'
        )
        self.chat_display.pack(fill=tk.BOTH, expand=True)
        self.chat_display.config(state=tk.DISABLED)
        
        # Create input area
        input_frame = ttk.Frame(self.root)
        input_frame.pack(pady=10, padx=20, fill=tk.X)
        
        self.user_input = ttk.Entry(input_frame, font=('Arial', 12))
        self.user_input.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        self.user_input.bind('<Return>', lambda event: self.send_message())
        
        send_button = ttk.Button(input_frame, text="Send", command=self.send_message)
        send_button.pack(side=tk.LEFT)
        
        # Create footer
        footer_frame = ttk.Frame(self.root)
        footer_frame.pack(pady=5)
        
        ttk.Label(footer_frame, text="Designed & Developed by Jilson C J", style='TLabel').pack()
        
        # Add welcome message
        self.display_message("Aurora", "Hello! I'm Aurora, your advanced AI assistant. How can I help you today?")
    
    def send_message(self):
        """Send user message and get response"""
        message = self.user_input.get().strip()
        if not message:
            return
        
        # Display user message
        self.display_message("You", message)
        self.user_input.delete(0, tk.END)
        
        # Get response in a separate thread to prevent GUI freezing
        threading.Thread(target=self.process_and_display_response, args=(message,), daemon=True).start()
    
    def process_and_display_response(self, message):
        """Process message and display response with typing indicator"""
        # Show typing indicator
        self.display_typing_indicator(True)
        
        # Simulate processing time
        time.sleep(random.uniform(0.5, 1.5))
        
        # Get response
        response = self.get_response(message)
        
        # Hide typing indicator and show response
        self.root.after(0, lambda: [
            self.display_typing_indicator(False),
            self.display_message("Aurora", response)
        ])
    
    def display_message(self, sender, message):
        """Display a message in the chat window"""
        self.chat_display.config(state=tk.NORMAL)
        
        # Configure tags for different senders
        self.chat_display.tag_config('user', foreground='blue', font=('Arial', 10, 'bold'))
        self.chat_display.tag_config('bot', foreground='green', font=('Arial', 10, 'bold'))
        self.chat_display.tag_config('time', foreground='gray', font=('Arial', 8))
        
        # Insert message with appropriate tag
        current_time = datetime.now().strftime('%H:%M:%S')
        self.chat_display.insert(tk.END, f"\n{sender} ({current_time}):\n", 'time')
        tag = 'user' if sender == "You" else 'bot'
        self.chat_display.insert(tk.END, f"{message}\n", tag)
        
        self.chat_display.config(state=tk.DISABLED)
        self.chat_display.see(tk.END)
    
    def display_typing_indicator(self, show):
        """Show or hide typing indicator"""
        self.chat_display.config(state=tk.NORMAL)
        
        if show:
            # Add typing indicator
            self.chat_display.insert(tk.END, "\nAurora is typing...\n", 'typing')
            self.typing_indicator_id = self.chat_display.index(tk.END)
        elif hasattr(self, 'typing_indicator_id'):
            # Remove typing indicator if it exists
            start_pos = f"{self.typing_indicator_id.split('.')[0]}.0"
            self.chat_display.delete(start_pos, self.typing_indicator_id)
        
        self.chat_display.config(state=tk.DISABLED)
        self.chat_display.see(tk.END)
    
    def run(self):
        """Run the application"""
        self.root.mainloop()

if __name__ == "__main__":
    # Create and run the chatbot
    chatbot = AdvancedChatbot()
    chatbot.run()