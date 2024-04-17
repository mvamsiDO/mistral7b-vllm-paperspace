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
- Building docker image with the following command:
`sudo docker build -t test_img .`