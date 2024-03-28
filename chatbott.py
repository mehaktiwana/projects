from flask import Flask, request, jsonify
from transformers import GPT2LMHeadModel, GPT2Tokenizer

app = Flask(__name__)

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

@app.route('/get_chatbot_response', methods=['POST'])
def get_chatbot_response():
    user_input = request.json['user_input']
    
    # Generate chatbot response using the pre-trained model
    input_ids = tokenizer.encode(f"User: {user_input}\nChatbot:", return_tensors='pt')
    output = model.generate(input_ids, max_length=150, temperature=0.7)
    chatbot_response = tokenizer.decode(output[0], skip_special_tokens=True)
    
    return jsonify({'chatbot_response': chatbot_response})

if __name__ == '__main__':
    app.run(debug=True)