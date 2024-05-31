#!/bin/bash

# Function to kill the server process
kill_server() {
  echo "Killing server process..."
  kill $P1
}

# Catch the SIGINT and SIGTERM signals and kill the server process
trap kill_server SIGINT SIGTERM

PYTHONPATH=$(dirname "$0") python server/start.py &
P1=$!
echo "Server started with PID: $P1"

# Wait for the server to start
while ! curl -s http://127.0.0.1:5000/ > /dev/null
do
  echo "Waiting for server to start..."
  sleep 5
done

# Server has started, send the POST request and save the response to a JSON file
curl -s -X POST -H "Content-Type: application/json" -d '{"master_key": "YourSuperSecretMasterKey"}' http://127.0.0.1:5000/generate_tokens > response.json
echo "Tokens saved in response.json"
wait $P1
