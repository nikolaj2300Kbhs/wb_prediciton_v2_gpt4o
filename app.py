from flask import Flask, request, jsonify
from openai import OpenAI
import os
from dotenv import load_dotenv
import logging
import re

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()

# Load and verify environment variables
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
if not OPENAI_API_KEY:
    logger.error("OPENAI_API_KEY is not set")
    raise ValueError("OPENAI_API_KEY is not set")

# Initialize OpenAI client
logger.info("Initializing OpenAI client...")
try:
    client = OpenAI(
        api_key=OPENAI_API_KEY,
        http_client=None
    )
    logger.info("OpenAI client initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize OpenAI client: {str(e)}")
    raise

def predict_box_intake(historical_data, future_box_info, context_text):
    try:
        logger.info("Constructing prompt for GPT-4o...")
        prompt = f"""
        First, read and understand the following context document to ground your analysis:
        {context_text}

        Now, you are an expert in evaluating Goodiebox welcome boxes for their ability to attract new members. 
        Below is historical data showing, for each past box, its features, the achieved CAC (Customer Acquisition Cost in euros), 
        and the corresponding daily intake (new members per day). Use this data to predict the daily intake for the future 
        welcome box, given its features and a CAC of 17.5 EUR. Consider factors such as the number of products, total retail 
        value, number of unique categories, number of full-size products, number of premium products (> €20), total weight, 
        average product rating, average brand rating, average category rating, and niche products, as well as the insights from 
        the context document (e.g., niche vs. non-niche products, free gifts, box weight).

        Historical Data:
        {historical_data}

        Future Box:
        {future_box_info}

        Return **only** the numerical value of the predicted daily intake as a float (e.g., 150.0). Do not include any explanations, text, or units. If you cannot provide a numerical value, return 0.0.
        """
        intakes = []
        for i in range(5):  # 5 runs for averaging
            logger.info(f"Sending request to OpenAI API (run {i+1}/5)")
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
            logger.info(f"Run {i+1} response: {intake}")
            if not intake:
                logger.error("Empty response from model")
                raise ValueError("Empty response from model")
            try:
                # Try to extract a number using regex as a fallback
                match = re.search(r'\d+\.\d+', intake)
                if match:
                    intake_float = float(match.group())
                else:
                    intake_float = float(intake)
                if intake_float < 0:
                    logger.error("Negative intake value received")
                    raise ValueError("Intake cannot be negative")
                intakes.append(intake_float)
            except ValueError as e:
                logger.error(f"Invalid intake format: {intake}, error: {str(e)}")
                raise
        if not intakes:
            logger.error("No valid intake values collected")
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
            logger.error("Missing future_box_info or context in request")
            return jsonify({'error': 'Missing future_box_info or context'}), 400
        historical_data = data.get('historical_data', 'No historical data provided')
        future_box_info = data['future_box_info']
        context_text = data['context']
        logger.info("Received request to predict box intake")
        intake = predict_box_intake(historical_data, future_box_info, context_text)
        logger.info(f"Returning predicted intake: {intake}")
        return jsonify({'predicted_intake': intake})
    except Exception as e:
        logger.error(f"Endpoint error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    logger.info("Health check requested")
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    logger.info("Starting Flask app locally...")
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
