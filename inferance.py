# 1. Import the library
from inference_sdk import InferenceHTTPClient

# 2. Connect to your local server
client = InferenceHTTPClient(
    api_url="http://localhost:9001", # Local server address
    api_key="**********"
)

# 3. Run your workflow on an image
result = client.run_workflow(
    workspace_name="iot-en67p",
    workflow_id="custom-workflow-2",
    images={
        "image": "YOUR_IMAGE.jpg" # Path to your image file
    },
    use_cache=True # Speeds up repeated requests
)

# 4. Get your results
print(result)
