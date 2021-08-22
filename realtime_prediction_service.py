import keras.models
import numpy as np
import matplotlib.pyplot as plt
from pandas import read_csv
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from sklearn.preprocessing import MinMaxScaler
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from sklearn.metrics import mean_squared_error
import pandas as pd
from confluent_kafka import Consumer, KafkaError, KafkaException
import sys

model = keras.models.load_model('1LSTM_total.h5')


def load_dataset():
    # x = np.load('testX_1LSTM.npy')
    # y = np.load('testY_1LSTM.npy')
    x = np.load('trainX_1LSTM.npy')
    y = np.load('trainY_1LSTM.npy')
    return x, y


def prepare_initial_queue():
    x, y = load_dataset()
    data_queue = list(x[165][0])
    return data_queue


data_queue = prepare_initial_queue()


def make_prediction(daily_total):
    transformed_total = daily_total / 42038
    data_queue.pop(0)
    data_queue.append(transformed_total)
    transformed_x = np.array([[data_queue]])
    print(transformed_x)
    return model.predict(transformed_x)


prediction_list = []
real_list = [item * 42038 for item in data_queue[-12:]]
initial_real_list_size = len(real_list)


def basic_consume_loop(consumer, topics):
    try:
        consumer.subscribe(topics)

        while True:
            msg = consumer.poll(timeout=1.0)
            if msg is None:
                continue
            else:
                print('key:{}, value:{}'.format(msg.key().decode('utf-8'), msg.value().decode('utf-8')))
                value_dict = eval(msg.value().decode('utf-8'))
                daily_total = value_dict.get('total')

                # Append the real daily data
                real_list.append(daily_total)
                plt.plot(list(range(1, len(real_list) + 1)), real_list, color='blue', label='Real')
                prediction_result = make_prediction(daily_total)
                print(prediction_result)

                # Append the prediction data
                prediction_list.append(prediction_result[0][0] * 42038)
                plt.plot(list(range(initial_real_list_size + 1, initial_real_list_size + len(prediction_list) + 1)),
                         prediction_list, color='red', label='Prediction')
                plt.xlabel('Day')
                plt.ylabel('Number of Infected People')
                plt.title('Realtime Covid Prediction')
                plt.legend()
                plt.savefig('/usr/share/nginx/html/test.png')
                plt.close()

            if msg.error():
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    # End of partition event
                    sys.stderr.write('%% %s [%d] reached end at offset %d\n' %
                                     (msg.topic(), msg.partition(), msg.offset()))
                elif msg.error():
                    raise KafkaException(msg.error())
            else:
                # msg_process(msg)
                pass
    finally:
        # Close down consumer to commit final offsets.
        consumer.close()


if __name__ == '__main__':
    conf = {'bootstrap.servers': '',# Set the server here
            'security.protocol': 'SASL_SSL',
            'sasl.mechanisms': 'PLAIN',
            'sasl.username': '', # Username
            'sasl.password': '', # Password
            'group.id': 'python_consumer',
            'client.id': 'p1',
            'enable.auto.commit': False,
            # 'auto.offset.reset': 'earliest',
            'auto.offset.reset': 'latest',
            }

    consumer = Consumer(conf)

    plt.plot([])
    plt.savefig('/usr/share/nginx/html/test.png')
    basic_consume_loop(consumer, ['input'])
