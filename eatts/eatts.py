import os
import pika
import soundfile as sf
import json
from pika.adapters.blocking_connection import BlockingChannel
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts


MODEL_PATH = os.path.join(os.path.dirname(__file__), "../resource/eatts_checkpoint/tts_model_4in1")
SAMPLE_RATE = 22000  # 22000 32750

def setup_tts(checkpoint_dir: str):
    config = XttsConfig()
    # "/Users/elnuralimirzayev/Thesis/saved_model3/GPT_XTTS_v2.0_LJSpeech_FT-August-30-2024_12+00AM-5099586"
    config.load_json(os.path.join(checkpoint_dir, "config.json"))
    model = Xtts.init_from_config(config)
    model.load_checkpoint(
        config,
        checkpoint_dir=checkpoint_dir,
        eval=True,
        vocab_path=os.path.join(checkpoint_dir, "vocab.json"),
    )
    model.cpu()
    return model, config

model, config = setup_tts(MODEL_PATH)

def synthesize(emo: str, text: str, output_path = None):
    outputs = model.synthesize(
        text,
        config,
        speaker_wav=os.path.join(MODEL_PATH, emo + ".wav"),
        gpt_cond_len=3,
        language="en",
    )
    raw_audio = outputs["wav"]
    if output_path is None:
        output_path = os.path.join(os.path.dirname(__file__), "../tts_output/voice.wav")
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    sf.write(output_path, raw_audio, SAMPLE_RATE)

def callback(ch, method, properties, body):
    try:
        feed = json.loads(body.decode())
        for message in feed:
            emo = message.get("emo")
            text = message.get("text")
            output_path = message.get("output_wav")
            if not emo or not text:
                raise ValueError("Missing 'emo' or 'text' in the message.")
            print(f"Received message: {message}")
            synthesize(emo, text, output_path)
        # Send finish signal to tts_output queue
        finish_message = json.dumps({"status": "finished"})
        connection = pika.BlockingConnection(pika.ConnectionParameters(host='localhost', port=5672))
        channel = connection.channel()
        channel.basic_publish(exchange='tts_output_exchange', routing_key='output', body=finish_message)
    except (json.JSONDecodeError, ValueError) as e:
        print(f"Error processing message: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

def listen(channel: BlockingChannel):
    channel.exchange_declare(exchange='tts_input_exchange', exchange_type='direct')
    channel.exchange_declare(exchange='tts_output_exchange', exchange_type='direct')
    channel.queue_declare(queue='tts_input')
    channel.queue_declare(queue='tts_output')
    channel.queue_purge(queue='tts_input')
    channel.queue_purge(queue='tts_output')
    channel.queue_bind(exchange='tts_input_exchange', queue='tts_input', routing_key='input')
    channel.queue_bind(exchange='tts_output_exchange', queue='tts_output', routing_key='output')
    channel.basic_consume(queue='tts_input', on_message_callback=callback, auto_ack=True)
    channel.start_consuming()

def start_server():
    connection = pika.BlockingConnection(pika.ConnectionParameters(host='localhost', port=5672))
    channel = connection.channel()
    listen(channel)

if __name__ == "__main__":
    start_server()
