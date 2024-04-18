# Mistral7B-Instruct - API - Docker - Core
In this gist, we will go over: 
- the steps to create and setup a new Paperspace core machine,
- Run a VLLM docker image of Mistral 7b model 
- Test with sample REST requests using Postman.

## Assumptions for this demo:
- Model being used is: `mistralai/Mistral-7B-Instruct-v0.2`
- GPU being used is: A5000 (why? the model size in GPU is about 18GB, so any GPU with 24GB RAM can be used, cheapest GPU with this config is A5000)

## Setup - Run Docker - Test API

### Create Machine
- Ensure you sign up and login to [paperspace console](https://console.paperspace.com/) and navigate to `Core Platform` on the top left corner
- You can [follow the steps in this tutorial](https://docs.digitalocean.com/products/paperspace/machines/how-to/create-machines/) to create a new `Ubunut22.04 - A5000 - StaticIP` machine, make using a static IP Address makes it easier later on to test the REST API  
- Once the machine is up and runnning, You can [connect to the machine by ssh](https://docs.digitalocean.com/products/paperspace/machines/how-to/connect-using-ssh/) 

### Setup Machine
- In order for us to use the machine properly, we have to install a bunch of packages, example: nvidia drivers, cuda toolkit, docker etc.
- It takes roughly 20-30 minutes to setup. And might require a reboot. 
1. Create some Env variables:
```
export APT_INSTALL="apt-get install -y --no-install-recommends"
export PIP_INSTALL="python3 -m pip --no-cache-dir install --upgrade"
export GIT_CLONE="git clone --depth 10"
``` 

2. Install common pkgs
```
sudo apt-get update && \
        sudo $APT_INSTALL \
        gcc \
        make \
        pkg-config \
        apt-transport-https \
        build-essential \
        apt-utils \
        ca-certificates \
        wget \
        rsync \
        git \
        vim \
        mlocate \
        libssl-dev \
        curl \
        openssh-client \
        unzip \
        unrar \
        zip \
        awscli \
        csvkit \
        emacs \
        joe \
        jq \
        dialog \
        man-db \
        manpages \
        manpages-dev \
        manpages-posix \
        manpages-posix-dev \
        nano \
        iputils-ping \
        sudo \
        ffmpeg \
        libsm6 \
        libxext6 \
        libboost-all-dev \
        gnupg \
        cifs-utils \
        zlib1g \
        software-properties-common
```

3. Install docker.io:
```
sudo apt install docker.io
```

4. Now the Nvidia stuff:
```
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/12.2.0/local_installers/cuda-repo-ubuntu2204-12-2-local_12.2.0-535.54.03-1_amd64.deb

sudo dpkg -i cuda-repo-ubuntu2204-12-2-local_12.2.0-535.54.03-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2204-12-2-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda

# install drivers:
sudo apt install nvidia-driver-535  
```

5. `nvidia-smi` command should work from here on! - might require reboot sometimes - output should look something like this. 
```
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.54.03              Driver Version: 535.54.03    CUDA Version: 12.2     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  NVIDIA RTX A5000               On  | 00000000:00:05.0 Off |                  Off |
| 30%   30C    P8              14W / 230W |      0MiB / 24564MiB |      0%      Default |
|                                         |                      |                  N/A |
+-----------------------------------------+----------------------+----------------------+
                                                                                         
+---------------------------------------------------------------------------------------+
| Processes:                                                                            |
|  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
|        ID   ID                                                             Usage      |
|=======================================================================================|
|                                                                                       |
+---------------------------------------------------------------------------------------+
```

6. Install compatable CUDNN
```
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub 

sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/ /" 

sudo apt-get update 

sudo $APT_INSTALL libcudnn8=8.9.7.29-1+cuda12.2  libcudnn8-dev=8.9.7.29-1+cuda12.2

export PATH=$PATH:/usr/local/cuda/bin
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

7. Install nvidia docker: (this allows the docker images to use the host GPUs)
```
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

# Install the NVIDIA docker toolkit
sudo apt-get update
sudo apt-get install -y nvidia-docker2

sudo systemctl restart docker
```

**Note: This is required to be done once, and you can create a template of this setup, so next time on you can spin up new machines using this template. 
More on [templates here](https://docs.digitalocean.com/products/paperspace/machines/how-to/create-template/).**

### Run Docker Image
- There is a [public docker image](https://github.com/orgs/mistralai/packages/container/package/mistral-src%2Fvllm) created by [Mistral folks using vLLM](https://docs.mistral.ai/deployment/self-deployment/vllm/) that can be used to deploy the LLM on any machine.
- The only dependency is to have a [HuggingFace Read Access Key](https://www.youtube.com/watch?v=jo_fTD2H4xA) and also [request access to the model in HuggingFace](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2)
- Once you have these ready, just run the following docker command:
```
sudo docker run --gpus all \
    -e HF_TOKEN="<your hf read access token>" -p 8000:8000 \
    ghcr.io/mistralai/mistral-src/vllm:latest \
    --host 0.0.0.0 \
    --model mistralai/Mistral-7B-Instruct-v0.2
```
- This will download the `docker image`, download the model and prepare the API server. The logs are super readable. 
- If everything goes well, you will see logs like this and the server is ready to accept requests.
```
Initializing an LLM engine with config: model='mistralai/Mistral-7B-Instruct-v0.2', tokenizer='mistralai/Mistral-7B-Instruct-v0.2', tokenizer_mode=auto, revision=None, tokenizer_revision=None, trust_remote_code=False, dtype=torch.bfloat16, max_seq_len=32768, download_dir=None, load_format=auto, tensor_parallel_size=1, quantization=None, seed=0)
tokenizer_config.json: 100%|██████████| 1.46k/1.46k [00:00<00:00, 11.2MB/s]
tokenizer.model: 100%|██████████| 493k/493k [00:00<00:00, 11.6MB/s]
tokenizer.json: 100%|██████████| 1.80M/1.80M [00:00<00:00, 10.3MB/s]
special_tokens_map.json: 100%|██████████| 72.0/72.0 [00:00<00:00, 471kB/s]
model-00002-of-00003.safetensors: 100%|██████████| 5.00G/5.00G [00:25<00:00, 200MB/s] 
model-00003-of-00003.safetensors: 100%|██████████| 4.54G/4.54G [00:25<00:00, 179MB/s] 
model-00001-of-00003.safetensors: 100%|██████████| 4.94G/4.94G [00:26<00:00, 184MB/s] 
INFO 04-18 13:41:20 llm_engine.py:222] # GPU blocks: 1730, # CPU blocks: 2048126MB/s]
INFO 04-18 13:41:22 api_server.py:113] Using default chat template:
INFO 04-18 13:41:22 api_server.py:113] {{ bos_token }}{% for message in messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if message['role'] == 'user' %}{{ '[INST] ' + message['content'] + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ message['content'] + eos_token}}{% else %}{{ raise_exception('Only user and assistant roles are supported!') }}{% endif %}{% endfor %}
INFO:     Started server process [1]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
``` 
**Note: Just add `-d` option in the docker run command to run it in background.**
`sudo docker run -d --gpus all -e HF_TOKEN="<your hf read access token>" -p 8000:8000 ghcr.io/mistralai/mistral-src/vllm:latest --host 0.0.0.0 --model mistralai/Mistral-7B-Instruct-v0.2`

### Test API
1. Use the static IP of your machine and add `:8000` to create your base URL. `http://staticIP:8000/`
2. You can verify the models that are loaded in Inference Server by making a `GET` request to `http://staticIP:8000/v1/models`
3. Curl simple test: 
```
<!-- Request -->
curl --location 'http://staticIP:8000/v1/chat/completions' \
--header 'Content-Type: application/json' \
--data '{
  "model": "mistralai/Mistral-7B-Instruct-v0.2",
  "messages": [
    {
      "role": "user",
      "content": "Keep your answers to 20 words or less. What is the plot of movie Lion king ? "
    }
  ],
  "temperature": 0.7,
  "top_p": 1,
  "max_tokens": 512,
  "stream": false,
  "safe_prompt": false,
  "random_seed": 1337
}'

<!-- Response -->
{
    "id": "cmpl-59272b12aa7c445a865d0cef2e41615b",
    "object": "chat.completion",
    "created": 36345,
    "model": "mistralai/Mistral-7B-Instruct-v0.2",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": " A lion prince is cast out from his Pride Rock home. He grows up among a herd of antelopes, returns to claim his throne. With help from friends, he confronts his past and earns respect."
            },
            "finish_reason": "stop"
        }
    ],
    "usage": {
        "prompt_tokens": 30,
        "total_tokens": 77,
        "completion_tokens": 47
    }
}

```
4. Also find the Postman Collection file: `VLLM-MistralTest.postman_collection.json`, Be sure to change the url field with your own endpoint. [Here is a tutorial on how to import a postman collection](https://www.geeksforgeeks.org/how-to-import-export-collections-in-postman/#1-how-to-importexport-collections-using-the-postman-dashboard) 
The collection has examples of:
    - LLM answering simple q and a (like the one in above image),
    - LLM summarises an article with follow up q and a about the article
    - LLM responds with json only



## References:
- Machine Setup Ref: https://github.com/gradient-ai/base-container/blob/main/pt211-tf215-cudatk120-py311/Dockerfile
- VLLM Mistral Ref: https://docs.mistral.ai/self-deployment/vllm/
- VLLM Ref: https://docs.vllm.ai/en/latest/getting_started/quickstart.html#using-openai-completions-api-with-vllm