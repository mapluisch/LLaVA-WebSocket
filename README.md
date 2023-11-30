# LLaVA-WebSocket
Python-based WebSocket for CLI LLaVA inference. The WebSocket server receives prompts and images from other clients and returns the CLI-inference results via the socket connection.

## Overview
This project is a quick and simple implementation of LLaVA ([Website](https://llava-vl.github.io/), [GitHub](https://github.com/haotian-liu/LLaVA)) CLI-based inference via a Python WebSocket. It is based on LLaVA's own `cli.py`, adding WebSocket capabilites.

When running `python llava-websocket.py`, the checkpoint shards are loaded and stay in cache while a WebSocket server is started. Clients in your local network can send new prompts and images for inference without having to re-load checkpoint shards and without using Gradio.

This project, in its current form, is not designed for conversation, but rather for one-time prompt processing, enabling the inference of various images and prompts based on individual requests. I might add conversation capabilities in the near future.

This project was tested on Ubuntu.

## Setup
You should follow the LLaVA tutorial, so that you have the pretrained model / checkpoint shards ready. Then, put my script into your LLaVA directory and start it while in the LLaVA conda-environment (`conda activate llava`).


## Usage 
```
python llava-websocket.py [ARGS]
```

### Arguments

Given that this project is based on LLaVA's `cli.py`, the same arguments can be specified
```
--model-path, default="liuhaotian/llava-v1.5-13b"
--model-base, default=None
--device, default="cuda"
--conv-mode, default=None
--temperature, default=0.2
--max-new-tokens, default=512
--load-8bit, action="store_true"
--load-4bit, action="store_true"
--debug, action="store_true"
```

I've additionally added two more args that you can specify
```
--port, default=1995
--verbose, action="store_true"
--json, action="store_true"
```
Using `--port [int]`, you can specify your own WebSocket port.

Using `--verbose`, you will receive verbose output on the server-side console (WebSocket connection info, transmitted inference results).

Using `--json`, the WebSocket responses will be formatted as JSON, containing a timestamp and the inference result: 

```json
{
    "time": "11:48:52.632415",
    "result": "The image features a wooden pier (...)"
}
```

## WebSocket Communication
In your local LAN, clients can access the WebSocket via `ws://[your-ip]:1995`. Specify your own port when calling the python script via `--port [int]`.

The WebSocket server waits for a JSON object:

```json
{
    "prompt": "Here goes your prompt (e.g., describe this image.)",
    "image": "URL to an image file, or BASE64-encoded string of an image file"
}
```

and returns the written inference result via the socket connection.

### Demo
https://github.com/mapluisch/LLaVA-WebSocket-Server/assets/31780571/8d8daaae-92e2-4da1-bfbd-bae873d51716

The input image is LLaVA's test image: 

<img src="https://llava-vl.github.io/static/images/view.jpg" alt="LLaVA's test image" style="width:50%;"/>
(Source: https://llava-vl.github.io/static/images/view.jpg)

## Disclaimer
This project is a prototype and serves as a basic example of using a WebSocket for LLaVA's CLI inference. Feel free to create a PR.

