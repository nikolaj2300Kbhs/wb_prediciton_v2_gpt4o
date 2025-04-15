from flask import Flask, request, jsonify
from openai import OpenAI
import os
from dotenv import load_dotenv
import logging

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=OPENAI_API_KEY)

def predict_box_intake(historical_data, future_box_info, context_text):
    try:
        prompt = f"""
        First, read and understand the following context document to ground your analysis:
        {context_text}

        Now, you are an expert in evaluating Goodiebox welcome boxes for their ability to attract new members. 
        Below is historical data showing, for each past box, its features, the achieved CAC (Customer Acquisition Cost in euros), 
        and the corresponding daily intake (new members per day). Use this data to predict the daily intake for the future 
        welcome box, given its features and a CAC of 17.5 EUR. Consider factors such as the number of products, total retail 
        value, number of unique categories, number of full-size products, number of premium products (> €20), total weight, 
        average product rating, average brand rating, average category rating, and niche products, as well as the insights from 
        the context document (e.g., niche vs. non-niche products, free gifts, box weight). Return only the numerical value of the 
        predicted daily intake (e.g., 150.0).

        Historical Data:
        {historical_data}

        Future Box:
        {future_box_info}
        """
        intakes = []
        for _ in range(5):  # 5 runs for averaging
            logger.info("Sending request to OpenAI API")
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are an expert in predicting Goodiebox performance, skilled at analyzing historical trends."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                seed=42
            )
            intake = response.choices[0].message.content.strip()
            logger.info(f"Run response: {intake}")
            if not intake:
                raise ValueError("Empty response from model")
            intake_float = float(intake)
            if intake_float < 0:
                raise ValueError("Intake cannot be negative")
            intakes.append(intake_float)
        if not intakes:
            raise ValueError("No valid intake values collected")
        avg_intake = sum(intakes) / len(intakes)
        logger.info(f"Averaged intake from 5 runs: {avg_intake}")
        return avg_intake
    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        raise

@app.route('/predict_box_score', methods=['POST'])
def box_score():
    try:
        data = request.get_json()
        if not data or 'future_box_info' not in data or 'context' not in data:
            logger.error("Missing future_box_info or context")
            return jsonify({'error': 'Missing future_box_info or context'}), 400
        historical_data = data.get('historical_data', 'No historical data provided')
        future_box_info = data['future_box_info']
        context_text = data['context']
        intake = predict_box_intake(historical_data, future_box_info, context_text)
        return jsonify({'predicted_intake': intake})
    except Exception as e:
        logger.error(f"Endpoint error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY not set")
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
