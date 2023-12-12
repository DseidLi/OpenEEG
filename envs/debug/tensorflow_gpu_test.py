import tensorflow as tf

print("TensorFlow version:", tf.__version__)
print("Is GPU available? ", tf.test.is_gpu_available())

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

for device in tf.config.experimental.list_physical_devices():
    print(device)
