#!/bin/bash

# Script to deploy the AI system to a production environment

echo "Starting deployment process..."

# Function to log messages with timestamps
log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log "Checking dependencies..."
# Step 1: Check for Docker and Docker Compose
if ! command -v docker &> /dev/null; then
    log "Docker could not be found, please install Docker."
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    log "Docker Compose is not installed, please install it."
    exit 1
fi

log "Dependencies are installed."

# Step 2: Load configuration
CONFIG_FILE="deployment_config.env"
if [ -f "$CONFIG_FILE" ]; then
    log "Loading configuration from $CONFIG_FILE"
    source $CONFIG_FILE
else
    log "Configuration file $CONFIG_FILE not found, using default settings."
fi

# Step 3: Build Docker container
log "Building Docker container..."
docker build -t ai_system .

if [ $? -eq 0 ]; then
    log "Docker container built successfully."
else
    log "Failed to build Docker container."
    exit 1
fi

# Step 4: Run the container
CONTAINER_PORT=${CONTAINER_PORT:-8000}
HOST_PORT=${HOST_PORT:-8000}

log "Running Docker container on port $HOST_PORT..."
docker run -d -p $HOST_PORT:$CONTAINER_PORT ai_system

if [ $? -eq 0 ]; then
    log "Deployment complete. The AI system is now running on port $HOST_PORT."
else
    log "Failed to start Docker container."
    exit 1
fi
