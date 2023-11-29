# LLaVA-WebSocket
Python-based WebSocket for CLI LLaVA inference. The WebSocket server receives prompts and images from other clients and returns the CLI-inference results via the socket connection.

## Overview
This project is a quick and simple implementation of LLaVA ([Website](https://llava-vl.github.io/), [GitHub](https://github.com/haotian-liu/LLaVA)) CLI-based inference via a Python WebSocket. It is based on LLaVA's own `cli.py`, adding WebSocket capabilites.

When you run `python llava-websocket.py`, the checkpoint shards are loaded and stay in cache while a WebSocket server is started. Clients in your local network can send new prompts and images for inference without having to re-load checkpoint shards and without using Gradio.

This project was tested on Ubuntu.

## Setup
You should follow the LLaVA tutorial, so that you have the pretrained model / checkpoint shards ready. Then, put my script into your LLaVA directory and start it while in the LLaVA conda-environment (`conda activate llava`).

`python llava-websocket.py --model-path liuhaotian/llava-v1.5-13b --load-4bit`

8-bit quantization should also work, but I have not tested it due to RAM constraints.

## WebSocket Communication
In your local LAN, clients can access the WebSocket via `ws://[your-ip]:1995`. You can specify your own port when calling the python script via `--websocket-port [int]`.

The WebSocket server waits for a JSON object:

```json
{
    "prompt": "Here goes your prompt (e.g., describe this image.)",
    "image": "URL to an image file, or BASE64-encoded string of an image file"
}
```

and returns the written inference result via the socket connection.

## Disclaimer
This project is a prototype and serves as a basic example of using a WebSocket for LLaVA's CLI inference. Feel free to create a PR.

