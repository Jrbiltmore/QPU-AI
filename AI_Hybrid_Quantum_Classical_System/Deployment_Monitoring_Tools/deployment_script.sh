
# /Deployment_Monitoring_Tools/deployment_script.sh

#!/bin/bash

# Script to deploy the AI system to a production environment

echo "Starting deployment process..."

# Step 1: Check dependencies
echo "Checking dependencies..."
if ! command -v docker &> /dev/null
then
    echo "Docker could not be found, please install Docker."
    exit 1
fi

# Step 2: Build Docker container
echo "Building Docker container..."
docker build -t ai_system .

# Step 3: Run the container
echo "Running Docker container on port 8000..."
docker run -p 8000:8000 ai_system

echo "Deployment complete. The AI system is now running on port 8000."
