import pika
import json

parameters = pika.ConnectionParameters(host='localhost', port=5672)
connection = pika.BlockingConnection(parameters)
channel = connection.channel()

# Function to ensure the channel is open
def ensure_channel():
    global connection, channel
    if channel.is_closed:
        connection = pika.BlockingConnection(parameters)
        channel = connection.channel()

# emo: [neu], [hap], [ang], [sad]
def call_tts(msg):
    ensure_channel()
    message_json = json.dumps(msg)
    channel.basic_publish(exchange='tts_input_exchange', routing_key='input', body=message_json)

def wait_for_finish(callback):
    ensure_channel()
    channel.basic_consume(queue='tts_output', on_message_callback=callback, auto_ack=True)
    channel.start_consuming()
