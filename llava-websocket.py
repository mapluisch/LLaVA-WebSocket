import argparse
import asyncio
import base64
import json
import requests
import torch
import websockets
import socket
from io import BytesIO
from PIL import Image

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from llava.utils import disable_torch_init

def load_image(image_data):
    if image_data.startswith('http://') or image_data.startswith('https://'):
        response = requests.get(image_data)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(BytesIO(base64.b64decode(image_data))).convert('RGB')
    return image

async def inference(websocket, path, args, model, tokenizer, image_processor, model_config, model_device):
    client_ip = websocket.remote_address[0]
    client_port = websocket.remote_address[1]
    if args.verbose:
        print(f"Client connected: {client_ip}:{client_port}")

    async for message in websocket:
        try:
            data = json.loads(message)
        except json.JSONDecodeError:
            await websocket.send("Error: Malformed JSON object.")
            continue

        prompt_text = data.get("prompt", "")
        image_data = data.get("image", "")
        
        if not prompt_text or not image_data:
            await websocket.send("Error: Missing 'prompt' or 'image' in JSON object.")
            continue

        image = load_image(image_data)
        image_tensor = process_images([image], image_processor, model_config)

        if type(image_tensor) is list:
            image_tensor = [image.to(model_device, dtype=torch.float16) for image in image_tensor]
        else:
            image_tensor = image_tensor.to(model_device, dtype=torch.float16)

        conv = conv_templates[args.conv_mode].copy()
        if model_config.mm_use_im_start_end:
            prompt_text = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + prompt_text
        else:
            prompt_text = DEFAULT_IMAGE_TOKEN + '\n' + prompt_text

        conv.append_message('user', prompt_text)
        conv.append_message('assistant', None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model_device)
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                max_new_tokens=args.max_new_tokens,
                use_cache=True,
                stopping_criteria=[stopping_criteria])

        outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
        if args.verbose:
            print("Sending: " + outputs)
        await websocket.send(outputs)

def main(args):
    disable_torch_init()
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name, args.load_8bit, args.load_4bit, device=args.device)

    if args.conv_mode is None:
        if 'llama-2' in model_name.lower():
            args.conv_mode = "llava_llama_2"
        elif "v1" in model_name.lower():
            args.conv_mode = "llava_v1"
        elif "mpt" in model_name.lower():
            args.conv_mode = "mpt"
        else:
            args.conv_mode = "llava_v0"

    start_server = websockets.serve(lambda ws, path: inference(ws, path, args, model, tokenizer, image_processor, model.config, model.device), "0.0.0.0", args.port, max_size=None)
    print(f"WebSocket server started at ws://{socket.gethostbyname(socket.gethostname())}:{args.port}")
    asyncio.get_event_loop().run_until_complete(start_server)
    asyncio.get_event_loop().run_forever()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="liuhaotian/llava-v1.5-13b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--port", type=int, default=1995)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    main(args)