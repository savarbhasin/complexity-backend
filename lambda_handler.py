import json
import os
from mangum import Mangum
from main import app

# Configure Mangum for Lambda Function URLs
handler = Mangum(app, lifespan="off")

def lambda_handler(event, context):
    """
    Handle Lambda Function URL events
    """
    response = handler(event, context)
    
    # For Function URLs, Lambda expects the exact same format
    # but the URL service should unwrap it automatically
    # If you're still seeing the wrapped format, it means
    # the response is being double-wrapped
    return response
