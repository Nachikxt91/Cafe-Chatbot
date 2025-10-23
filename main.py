from agent_controller import AgentController
import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Any
import logging
import time
import os

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

app = FastAPI()

# CRITICAL: Add CORS middleware for React Native
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your app's domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

agent_controller = AgentController()

@app.middleware("http")
async def log_requests(request: Request, call_next):
    request_id = f"req_{int(time.time() * 1000)}"
    logger.info(f"{request_id} Started - {request.method} {request.url.path}")
    
    start_time = time.time()
    response = await call_next(request)
    duration = (time.time() - start_time) * 1000
    
    logger.info(f"{request_id} Completed - Status: {response.status_code} Duration: {duration:.2f}ms")
    return response

@app.post("/query")
async def handle_query(request: Dict[Any, Any]):
    """Handle agent queries - Main endpoint for React Native"""
    logger.info("=" * 50)
    logger.info("INCOMING REQUEST")
    logger.info(f"Input data: {request}")
    logger.info("=" * 50)
    
    try:
        logger.info("Calling agent_controller.get_response")
        response = agent_controller.get_response(request)
        
        logger.info("=" * 50)
        logger.info("AGENT RESPONSE")
        logger.info(f"Response: {response}")
        logger.info("=" * 50)
        
        return {"response": response}
    except Exception as e:
        logger.error(f"ERROR: {str(e)}", exc_info=True)
        raise

@app.get("/")
async def root():
    """Health check endpoint"""
    logger.info("Root endpoint accessed")
    return {"status": "Coffee Shop Agent API is running", "version": "1.0"}

@app.get("/health")
async def health_check():
    """Health check for monitoring services"""
    return {"status": "healthy", "service": "cafe-chatbot-api"}

if __name__ == "__main__":
    # Use PORT environment variable from Render
    port = int(os.getenv("PORT", 8000))
    logger.info(f"Starting Coffee Shop API server on port {port}...")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="debug")
