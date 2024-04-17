# Custom Deployment for Mistral 7B VLLM Gradient

Original Source from: https://github.com/mistralai/mistral-src/
Deploy: https://github.com/mistralai/mistral-src/tree/main/deploy 

## Add Model and Host as ENV:
- Made minor changes to `entrypoint.sh` 
```
model=$MODEL
host=$HOST

exec python3 -u -m vllm.entrypoints.openai.api_server --model $model --host $host "$@"
```
- Building docker image with the following command, takes nearly 90 mins, but built successfully
`sudo docker build -t test_img .`

- Run docker local:
```
docker run --gpus all \
    -e HF_TOKEN=$HF_TOKEN -p 8000:8000 \
    ghcr.io/mistralai/mistral-src/vllm:latest \
    --host 0.0.0.0 \
    --model mistralai/Mistral-7B-Instruct-v0.2
```

