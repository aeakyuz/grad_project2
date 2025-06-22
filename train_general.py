from google.colab import drive
drive.mount('/content/drive')
import tensorflow as tf
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Conv2DTranspose,
    BatchNormalization, LeakyReLU, Activation, Dropout
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam # Changed from .legacy.Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau
import datetime
from skimage.metrics import peak_signal_noise_ratio as skimage_psnr
from skimage.metrics import structural_similarity as skimage_ssim
import random

# --- GPU Check ---
print(f"TensorFlow Version: {tf.__version__}")
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"Num GPUs Available: {len(gpus)}")
    for gpu_device in gpus: print(f"GPU device: {gpu_device}")
    try:
        for gpu in gpus: tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e: print(f"RuntimeError in GPU memory growth setup: {e}")
else:
    print("TensorFlow is NOT using the GPU. It will run on CPU.")
print("-" * 30)

def resize(input_image, real_image, height, width):
  input_image = tf.image.resize(input_image, [height, width],
                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
  real_image = tf.image.resize(real_image, [height, width],
                               method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
  return input_image, real_image

def random_crop(input_image, real_image):
  stacked_image = tf.stack([input_image, real_image], axis=0)
  cropped_image = tf.image.random_crop(
      stacked_image, size=[2, IMG_HEIGHT, IMG_WIDTH, 3])
  return cropped_image[0], cropped_image[1]

def random_jitter(input_image, real_image):
  # Resizing to 286x286
  input_image, real_image = resize(input_image, real_image, 286, 286)

  # Randomly cropping to 256x256
  input_image, real_image = random_crop(input_image, real_image)

  # Random mirroring
  if tf.random.uniform(()) > 0.5:
    input_image = tf.image.flip_left_right(input_image)
    real_image = tf.image.flip_left_right(real_image)

  return input_image, real_image

# --- Modify your existing functions ---

# This function will now be used for TRAINING data
def load_image_train(original_path_tensor, target_path_tensor):
    # This part is the same as your old load_and_preprocess_image
    original_image = tf.io.read_file(original_path_tensor)
    original_image = tf.image.decode_jpeg(original_image, channels=3)
    target_image = tf.io.read_file(target_path_tensor)
    target_image = tf.image.decode_jpeg(target_image, channels=3)

    original_image = tf.cast(original_image, tf.float32)
    target_image = tf.cast(target_image, tf.float32)

    # Apply the random jitter
    original_image, target_image = random_jitter(original_image, target_image)

    # Normalize images to [-1, 1]
    original_image = (original_image / 127.5) - 1.0
    target_image = (target_image / 127.5) - 1.0

    return original_image, target_image

# This function will now be used for VALIDATION/TESTING data (no jitter)
def load_image_test(original_path_tensor, target_path_tensor):
    original_image = tf.io.read_file(original_path_tensor)
    original_image = tf.image.decode_jpeg(original_image, channels=3)
    target_image = tf.io.read_file(target_path_tensor)
    target_image = tf.image.decode_jpeg(target_image, channels=3)

    original_image = tf.cast(original_image, tf.float32)
    target_image = tf.cast(target_image, tf.float32)

    # Just resize, no jitter
    original_image, target_image = resize(original_image, target_image, IMG_HEIGHT, IMG_WIDTH)

    # Normalize images to [-1, 1]
    original_image = (original_image / 127.5) - 1.0
    target_image = (target_image / 127.5) - 1.0

    return original_image, target_image

INPUT_IMAGES_PATH = "/content/drive/My Drive/adobe5k/raw"
GROUND_TRUTH_IMAGES_PATH = "/content/drive/My Drive/adobe5k/d"

# --- General Parameters ---
# Set NUM_IMAGES_TO_USE to -1 to use all available images
NUM_IMAGES_TO_USE = 1000
FILTER_NAME_TAG = "expert_d" # A custom name for this training run

# --- Model & Training Parameters ---
IMG_HEIGHT = 256
IMG_WIDTH = 256
OUTPUT_CHANNELS = 3
BATCH_SIZE = 1
EPOCHS = 200
LEARNING_RATE_G = 0.0002
LEARNING_RATE_D = 0.0002
BETA_1 = 0.5
LAMBDA_L1 = 100

NORMALIZE_TO_PLUS_MINUS_ONE = True
assert NORMALIZE_TO_PLUS_MINUS_ONE, "Pix2Pix typically uses [-1,1] normalization."

SSIM_WIN_SIZE = 7

data_size_tag = "all" if NUM_IMAGES_TO_USE == -1 else str(NUM_IMAGES_TO_USE)
FILTER_NAME = f"{FILTER_NAME_TAG}_lambda{LAMBDA_L1}_data{data_size_tag}"
MODEL_SAVE_DIR = f"/content/drive/My Drive/aea_grad/models_{FILTER_NAME}" # Saves to a project folder in Drive
GENERATOR_BEST_MODEL_NAME = f"generator_{FILTER_NAME}_best.keras"
GENERATOR_FINAL_MODEL_NAME = f"generator_{FILTER_NAME}_final.keras"
CHECKPOINT_MODEL_DIR_BASE = f"pix2pix_checkpoints_{FILTER_NAME}"

SAVE_GENERATOR_PATIENCE = 30 # <<<< INCREASED EARLY STOPPING PATIENCE
REDUCE_LR_PATIENCE = 15    # <<<< PATIENCE FOR LR REDUCTION

# --- Loss Functions for GAN ---
bce = BinaryCrossentropy(from_logits=False)

def discriminator_loss_fn(real_output, fake_output):
    real_loss = bce(tf.ones_like(real_output), real_output)
    fake_loss = bce(tf.zeros_like(fake_output), fake_output)
    total_disc_loss = real_loss + fake_loss
    return total_disc_loss

def generator_loss_fn(disc_fake_output, gen_output, target_image):
    gan_loss = bce(tf.ones_like(disc_fake_output), disc_fake_output)
    l1 = tf.reduce_mean(tf.abs(target_image - gen_output))
    total_gen_loss = gan_loss + (LAMBDA_L1 * l1)
    return total_gen_loss, gan_loss, l1

def prepare_dataset_paths(input_dir, target_dir, num_images_to_select):
    """
    Prepares lists of corresponding image paths from input and target directories.
    If num_images_to_select is -1, all images will be used.
    """
    print(f"Preparing dataset paths...")
    print(f"Input Directory: {input_dir}")
    print(f"Target Directory: {target_dir}")

    if not os.path.isdir(input_dir):
        print(f"Error: Input directory not found: {input_dir}")
        return [], []
    if not os.path.isdir(target_dir):
        print(f"Error: Target directory not found: {target_dir}")
        return [], []

    # Find all image files in the input directory
    input_image_files = []
    for ext in ('*.jpg', '*.jpeg', '*.JPG', '*.JPEG', '*.png', '*.PNG'):
        input_image_files.extend(glob.glob(os.path.join(input_dir, ext)))

    if not input_image_files:
        print(f"No images found in {input_dir}")
        return [], []

    # Shuffle all available input files to get a random subset if needed
    random.seed(42)
    random.shuffle(input_image_files)

    original_paths = []
    target_paths = []
    processed_count = 0

    # Determine if we should limit the number of images
    limit_images = num_images_to_select != -1

    # Iterate through the shuffled full list
    for input_img_path in input_image_files:
        # If we are limiting images, check if the limit has been reached
        if limit_images and processed_count >= num_images_to_select:
            break

        img_filename = os.path.basename(input_img_path)
        target_img_path = os.path.join(target_dir, img_filename)

        # Add the pair if the corresponding target image exists
        if os.path.exists(target_img_path):
            original_paths.append(input_img_path)
            target_paths.append(target_img_path)
            processed_count += 1

    total_or_requested = "all" if not limit_images else str(num_images_to_select)
    print(f"Found and paired {len(original_paths)} input/target images out of {total_or_requested} requested.")

    if limit_images and len(original_paths) < num_images_to_select:
        print(f"Warning: Could only find {len(original_paths)} pairs, which is less than the requested {num_images_to_select}.")

    return original_paths, target_paths

# --- 2. Data Loading and Preprocessing ---
def load_and_preprocess_image(path, normalize_pm_one=NORMALIZE_TO_PLUS_MINUS_ONE):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [IMG_HEIGHT, IMG_WIDTH], method=tf.image.ResizeMethod.BILINEAR)
    image_float = tf.cast(image, tf.float32)
    if normalize_pm_one: processed_image = (image_float / 127.5) - 1.0
    else: processed_image = image_float / 255.0
    return processed_image

def load_image_pair(original_path_tensor, target_path_tensor):
    original_image = load_and_preprocess_image(original_path_tensor)
    target_image = load_and_preprocess_image(target_path_tensor)
    return original_image, target_image

def create_tf_dataset(orig_paths, target_paths, batch_size, shuffle=True):
    dataset = tf.data.Dataset.from_tensor_slices((orig_paths, target_paths))
    dataset = dataset.map(load_image_pair, num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle: dataset = dataset.shuffle(buffer_size=max(1000, len(orig_paths)//10), seed=42) # Ensure buffer_size is reasonable
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset

# --- 3. Model Definitions (Generator & Discriminator) ---
# ... 그대로 ...
# Modified generator with Dropout layers in the decoder/upsampling path
def build_generator(input_shape=(IMG_HEIGHT, IMG_WIDTH, OUTPUT_CHANNELS)):
    inputs = Input(input_shape)

    # --- Encoder ---
    c1 = Conv2D(32, (3,3),activation='relu',kernel_initializer='he_normal',padding='same')(inputs)
    c1 = Conv2D(32, (3,3),activation='relu',kernel_initializer='he_normal',padding='same')(c1)
    p1 = MaxPooling2D((2,2))(c1)

    c2 = Conv2D(64, (3,3),activation='relu',kernel_initializer='he_normal',padding='same')(p1)
    c2 = Conv2D(64, (3,3),activation='relu',kernel_initializer='he_normal',padding='same')(c2)
    p2 = MaxPooling2D((2,2))(c2)

    c3 = Conv2D(128,(3,3),activation='relu',kernel_initializer='he_normal',padding='same')(p2)
    c3 = Conv2D(128,(3,3),activation='relu',kernel_initializer='he_normal',padding='same')(c3)
    p3 = MaxPooling2D((2,2))(c3)

    # --- Bottleneck ---
    b = Conv2D(256,(3,3),activation='relu',kernel_initializer='he_normal',padding='same')(p3)
    b = Conv2D(256,(3,3),activation='relu',kernel_initializer='he_normal',padding='same')(b)

    # --- Decoder (with Dropout) ---
    u1 = Conv2DTranspose(128,(2,2),strides=(2,2),padding='same')(b)
    u1 = concatenate([u1,c3])
    c4 = Conv2D(128,(3,3),activation='relu',kernel_initializer='he_normal',padding='same')(u1)
    c4 = Conv2D(128,(3,3),activation='relu',kernel_initializer='he_normal',padding='same')(c4)
    c4 = Dropout(0.5)(c4) # <--- ADDED DROPOUT

    u2 = Conv2DTranspose(64,(2,2),strides=(2,2),padding='same')(c4)
    u2 = concatenate([u2,c2])
    c5 = Conv2D(64,(3,3),activation='relu',kernel_initializer='he_normal',padding='same')(u2)
    c5 = Conv2D(64,(3,3),activation='relu',kernel_initializer='he_normal',padding='same')(c5)
    c5 = Dropout(0.5)(c5) # <--- ADDED DROPOUT

    u3 = Conv2DTranspose(32,(2,2),strides=(2,2),padding='same')(c5)
    u3 = concatenate([u3,c1])
    c6 = Conv2D(32,(3,3),activation='relu',kernel_initializer='he_normal',padding='same')(u3)
    c6 = Conv2D(32,(3,3),activation='relu',kernel_initializer='he_normal',padding='same')(c6)

    # --- Output Layer ---
    outputs = Conv2D(OUTPUT_CHANNELS,(1,1),activation='tanh')(c6)

    model = Model(inputs=[inputs],outputs=[outputs], name="generator")
    return model

def build_discriminator(input_shape=(IMG_HEIGHT, IMG_WIDTH, OUTPUT_CHANNELS)):
    input_a = Input(shape=input_shape, name='input_A')
    input_b = Input(shape=input_shape, name='input_B')
    merged = concatenate([input_a, input_b])
    d = Conv2D(64, (4, 4), strides=(2, 2), padding='same', kernel_initializer='he_normal')(merged)
    d = LeakyReLU(alpha=0.2)(d)
    d = Conv2D(128, (4, 4), strides=(2, 2), padding='same', kernel_initializer='he_normal')(d)
    d = BatchNormalization()(d); d = LeakyReLU(alpha=0.2)(d)
    d = Conv2D(256, (4, 4), strides=(2, 2), padding='same', kernel_initializer='he_normal')(d)
    d = BatchNormalization()(d); d = LeakyReLU(alpha=0.2)(d)
    d = Conv2D(512, (4, 4), strides=(1, 1), padding='same', kernel_initializer='he_normal')(d) # Stride 1
    d = BatchNormalization()(d); d = LeakyReLU(alpha=0.2)(d)
    output_patch = Conv2D(1, (4, 4), strides=(1, 1), padding='same', kernel_initializer='he_normal', activation='sigmoid')(d) # Stride 1
    model = Model([input_a, input_b], output_patch, name="discriminator")
    return model

# --- 4. Pix2Pix Keras Model with Custom Training Loop ---
# ... 그대로 ...
class Pix2Pix(Model):
    def __init__(self, generator, discriminator, lambda_l1=LAMBDA_L1):
        super(Pix2Pix, self).__init__()
        self.generator = generator; self.discriminator = discriminator
        self.lambda_l1 = lambda_l1
        self.gen_total_loss_tracker = tf.keras.metrics.Mean(name="gen_total_loss")
        self.gen_gan_loss_tracker = tf.keras.metrics.Mean(name="gen_gan_loss")
        self.gen_l1_loss_tracker = tf.keras.metrics.Mean(name="gen_l1_loss")
        self.disc_loss_tracker = tf.keras.metrics.Mean(name="disc_loss")

    def compile(self, gen_optimizer, disc_optimizer, gen_loss_fn_global, disc_loss_fn_global, dummy_loss_fn):
        super(Pix2Pix, self).compile(loss=dummy_loss_fn)
        self.gen_optimizer = gen_optimizer; self.disc_optimizer = disc_optimizer
        self.gen_loss_fn_global = gen_loss_fn_global; self.disc_loss_fn_global = disc_loss_fn_global

    def call(self, inputs, training=False): return self.generator(inputs, training=training)
    @property
    def metrics(self):
        return [self.gen_total_loss_tracker, self.gen_gan_loss_tracker, self.gen_l1_loss_tracker, self.disc_loss_tracker]

    def train_step(self, data):
        input_image, target_image = data
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_image = self.generator(input_image, training=True)
            real_output = self.discriminator([input_image, target_image], training=True)
            fake_output = self.discriminator([input_image, generated_image], training=True)
            gen_total_loss, gen_gan_loss, gen_l1_loss = self.gen_loss_fn_global(fake_output, generated_image, target_image)
            disc_loss = self.disc_loss_fn_global(real_output, fake_output)
        generator_gradients = gen_tape.gradient(gen_total_loss, self.generator.trainable_variables)
        discriminator_gradients = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
        self.gen_optimizer.apply_gradients(zip(generator_gradients, self.generator.trainable_variables))
        self.disc_optimizer.apply_gradients(zip(discriminator_gradients, self.discriminator.trainable_variables))
        self.gen_total_loss_tracker.update_state(gen_total_loss); self.gen_gan_loss_tracker.update_state(gen_gan_loss)
        self.gen_l1_loss_tracker.update_state(gen_l1_loss); self.disc_loss_tracker.update_state(disc_loss)
        return {m.name: m.result() for m in self.metrics}

    def save_generator(self, filepath, overwrite=True, save_format=None, **kwargs):
        self.generator.save(filepath,overwrite=overwrite,save_format=save_format,**kwargs)
    def load_generator(self, filepath, **kwargs): self.generator = tf.keras.models.load_model(filepath, **kwargs)

# --- Dummy Loss Function for Keras Compile ---
def dummy_loss(y_true, y_pred): return tf.reduce_mean(0.0 * y_pred)

# --- Custom Callback to Save Generator and Early Stop ---
class SaveGeneratorOnValL1(tf.keras.callbacks.Callback):
    def __init__(self, val_data, generator_save_path, patience=10, monitor_metric_name='val_gen_l1_loss'):
        super().__init__(); self.val_data = val_data; self.generator_save_path = generator_save_path
        self.best_val_l1_loss = float('inf'); self.wait = 0; self.patience = patience
        self.stopped_epoch = 0; self.monitor_metric_name = monitor_metric_name

    # on_epoch_begin removed for brevity - no changes needed there for this pass
    def on_epoch_end(self, epoch, logs=None):
        current_val_l1_sum = 0; num_batches = 0
        for input_img_val, target_img_val in self.val_data:
            generated_img_val = self.model.generator(input_img_val, training=False)
            l1_val = tf.reduce_mean(tf.abs(target_img_val - generated_img_val))
            current_val_l1_sum += l1_val; num_batches += 1
        current_val_l1_avg = current_val_l1_sum / num_batches if num_batches > 0 else float('inf')
        logs = logs or {}; logs[self.monitor_metric_name] = current_val_l1_avg
        print(f"Epoch {epoch+1}: {self.monitor_metric_name}: {current_val_l1_avg:.4f} (best: {self.best_val_l1_loss:.4f})")
        if current_val_l1_avg < self.best_val_l1_loss:
            print(f"{self.monitor_metric_name} improved from {self.best_val_l1_loss:.4f} to {current_val_l1_avg:.4f}. Saving generator to {self.generator_save_path}")
            self.model.save_generator(self.generator_save_path)
            self.best_val_l1_loss = current_val_l1_avg; self.wait = 0
        else:
            self.wait +=1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch; self.model.stop_training = True
                print(f"{self.monitor_metric_name} did not improve for {self.patience} epochs. Stopping training at epoch {self.stopped_epoch + 1}.")

# --- 5. Training ---
def train_pix2pix_model(orig_paths, target_paths):
    if not orig_paths or not target_paths:
        print(f"Error: No image paths for training {FILTER_NAME}."); return None, None, None

    orig_train, orig_val, target_train, target_val = train_test_split(
        orig_paths, target_paths, test_size=0.1, random_state=42 # 10% for validation
    )
    print(f"Training {FILTER_NAME} with {len(orig_train)} images, validating with {len(orig_val)} images.")

    train_dataset = tf.data.Dataset.from_tensor_slices((orig_train, target_train))
    train_dataset = train_dataset.map(load_image_train, num_parallel_calls=tf.data.AUTOTUNE)
    train_dataset = train_dataset.shuffle(buffer_size=max(1000, len(orig_train)//10), seed=42)
    train_dataset = train_dataset.batch(BATCH_SIZE)
    train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)


    # val_dataset = create_tf_dataset(orig_val, target_val, BATCH_SIZE, shuffle=False)
    # This should now be:
    val_dataset = tf.data.Dataset.from_tensor_slices((orig_val, target_val))
    val_dataset = val_dataset.map(load_image_test) # No jitter for validation
    val_dataset = val_dataset.batch(BATCH_SIZE)

    generator = build_generator(input_shape=(IMG_HEIGHT, IMG_WIDTH, OUTPUT_CHANNELS))
    discriminator = build_discriminator(input_shape=(IMG_HEIGHT, IMG_WIDTH, OUTPUT_CHANNELS))
    # print("--- Generator Summary ---"); generator.summary() # Less verbose for multiple runs
    # print("\n--- Discriminator Summary ---"); discriminator.summary()

    pix2pix_model = Pix2Pix(generator, discriminator, lambda_l1=LAMBDA_L1)

    generator_optimizer = Adam(learning_rate=LEARNING_RATE_G, beta_1=BETA_1)
    discriminator_optimizer = Adam(learning_rate=LEARNING_RATE_D, beta_1=BETA_1)

    pix2pix_model.compile(
        gen_optimizer=generator_optimizer, disc_optimizer=discriminator_optimizer,
        gen_loss_fn_global=generator_loss_fn, disc_loss_fn_global=discriminator_loss_fn,
        dummy_loss_fn=dummy_loss
    )

    if not os.path.exists(MODEL_SAVE_DIR): os.makedirs(MODEL_SAVE_DIR)
    log_dir = os.path.join(MODEL_SAVE_DIR, "logs/fit/", datetime.datetime.now().strftime("%Y%m%d-%H%M%S")) # Unique log dir
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1, update_freq='epoch') # Keep update_freq for GANs

    save_generator_callback = SaveGeneratorOnValL1(
        val_dataset,
        os.path.join(MODEL_SAVE_DIR, GENERATOR_BEST_MODEL_NAME),
        patience=SAVE_GENERATOR_PATIENCE
    )

    checkpoint_dir = os.path.join(MODEL_SAVE_DIR, CHECKPOINT_MODEL_DIR_BASE) # Unique due to FILTER_NAME
    if not os.path.exists(checkpoint_dir): os.makedirs(checkpoint_dir)
    checkpoint_filepath_template = os.path.join(checkpoint_dir, "ckpt_epoch-{epoch:02d}.keras")

    checkpoint_callback = ModelCheckpoint( # Saves the whole Pix2Pix model
        filepath=checkpoint_filepath_template, monitor='val_gen_l1_loss', # Monitor val_gen_l1_loss
        save_best_only=True, save_weights_only=False, save_freq='epoch', verbose=1
    )

    reduce_lr_gen_callback = ReduceLROnPlateau(
        monitor='val_gen_l1_loss', # <<<< MONITORING VAL L1 FOR LR REDUCTION
        factor=0.5,
        patience=REDUCE_LR_PATIENCE,
        min_lr=0.000005,
        verbose=1
    )

    initial_epoch_to_start = 0
    # Checkpoint loading: Since FILTER_NAME now includes NUM_IMAGES_TO_USE,
    # it will by default start fresh unless a checkpoint for this specific data size exists.
    latest_checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)
    if latest_checkpoint_file:
        print(f"Resuming training from Pix2Pix checkpoint: {latest_checkpoint_file}")
        try:
            pix2pix_model.load_weights(latest_checkpoint_file)
            print(f"  Current Gen LR before load: {pix2pix_model.gen_optimizer.learning_rate.numpy()}") # Debug print
            # Reload to ensure optimizer state is updated by load_weights if possible
            # pix2pix_model.load_weights(latest_checkpoint_file) # Keras should handle this
            print(f"  Current Gen LR after load: {pix2pix_model.gen_optimizer.learning_rate.numpy()}") # Debug print
            try:
                epoch_str = latest_checkpoint_file.split('epoch-')[-1].split('.')[0].split('_')[0] # Handle potential extra suffixes
                initial_epoch_to_start = int(epoch_str)
                print(f"Successfully loaded weights. Will resume from epoch {initial_epoch_to_start}.")
            except Exception as e_epoch:
                print(f"Could not parse epoch from '{latest_checkpoint_file}': {e_epoch}. Starting epoch counter from 0, weights loaded.")
        except Exception as e_load:
            print(f"Error loading weights from checkpoint {latest_checkpoint_file}: {e_load}. Starting training from scratch.")
            initial_epoch_to_start = 0
    else:
        print(f"No Pix2Pix checkpoint found for {FILTER_NAME} at {checkpoint_dir}. Starting training from scratch.")

    print(f"Starting training for {FILTER_NAME} Pix2Pix model (Epochs: {EPOCHS}, Initial Epoch: {initial_epoch_to_start})...")
    history = pix2pix_model.fit(
        train_dataset,
        epochs=EPOCHS,
        initial_epoch=initial_epoch_to_start,
        callbacks=[tensorboard_callback, save_generator_callback, checkpoint_callback, reduce_lr_gen_callback],
        validation_data=val_dataset,
        verbose=1
    )

    pix2pix_model.save_generator(os.path.join(MODEL_SAVE_DIR, GENERATOR_FINAL_MODEL_NAME))
    print(f"Training complete. Final generator saved as {GENERATOR_FINAL_MODEL_NAME}")

    return pix2pix_model.generator, history, val_dataset


# --- 6. Evaluation ---
# ... 그대로 ...
def evaluate_pix2pix_generator(generator_to_eval, dataset_to_eval):
    print(f"\nEvaluating Generator performance for {FILTER_NAME}...")
    all_maes, all_psnrs, all_ssims = [], [], []
    for input_batch, target_batch in dataset_to_eval: # Assumes dataset_to_eval yields batches
        predicted_batch = generator_to_eval.predict_on_batch(input_batch)
        for i in range(target_batch.shape[0]): # Iterate through images in the batch
            target_img_norm, predicted_img_norm = target_batch[i].numpy(), predicted_batch[i]
            target_img_01, predicted_img_01 = (target_img_norm*0.5)+0.5, (predicted_img_norm*0.5)+0.5
            target_img_01, predicted_img_01 = np.clip(target_img_01,0,1), np.clip(predicted_img_01,0,1)
            all_maes.append(np.mean(np.abs(target_img_01-predicted_img_01))*255.0)
            try: psnr_val = skimage_psnr(target_img_01, predicted_img_01, data_range=1.0)
            except ZeroDivisionError: psnr_val = float('inf') # Handle potential all-black image case
            all_psnrs.append(psnr_val)
            current_win_size = min(SSIM_WIN_SIZE, target_img_01.shape[0]-1, target_img_01.shape[1]-1)
            if current_win_size % 2 == 0: current_win_size -=1
            if current_win_size < 3 : ssim_val = 0.0 # Avoid error for very small images if win_size becomes too small
            else: ssim_val = skimage_ssim(target_img_01, predicted_img_01, data_range=1.0, multichannel=True, channel_axis=-1, win_size=current_win_size)
            all_ssims.append(ssim_val)
    if not all_maes: print("No metrics calculated."); return
    avg_mae, std_mae = np.mean(all_maes), np.std(all_maes)
    finite_psnrs = [p for p in all_psnrs if np.isfinite(p)]; avg_psnr = np.mean(finite_psnrs) if finite_psnrs else 0.0
    avg_ssim = np.mean(all_ssims)
    print(f"\n--- Evaluation Results ({FILTER_NAME} - Generator) ---")
    print(f"Average MAE (0-255 scale): {avg_mae:.2f} (Target: <= 20)")
    print(f"Average PSNR: {avg_psnr:.2f} dB (Target: >= 25)")
    print(f"Average SSIM: {avg_ssim:.3f} (Target: >= 0.85)")
    print(f"MAE StdDev (Consistency): {std_mae:.2f} (Target: <= 5)")
    print("-------------------------------------------------------\n")

# --- 7. Prediction and Display ---
# ... 그대로 (display_pix2pix_predictions function from before, it should show 3 by default) ...
def display_pix2pix_predictions(generator_model, dataset, num_predictions=3):
    plt.figure(figsize=(15, num_predictions * 5))
    displayed_count = 0
    # dataset here is val_dataset, which is batched by BATCH_SIZE (currently 1)
    for input_batch, target_batch in dataset.take(num_predictions): # Takes num_predictions batches
        if displayed_count >= num_predictions: break
        if input_batch.shape[0] == 0 : continue # Skip empty batches if any edge case

        original_img_tensor = input_batch[0] # Since BATCH_SIZE is 1
        target_img_tensor = target_batch[0]  # Since BATCH_SIZE is 1

        prediction_batch = generator_model.predict_on_batch(input_batch)
        predicted_img_tensor = prediction_batch[0]

        original_img,target_img,predicted_img = original_img_tensor.numpy(), target_img_tensor.numpy(), predicted_img_tensor # predict_on_batch returns numpy
        original_img,target_img,predicted_img = (original_img*0.5)+0.5,(target_img*0.5)+0.5,(predicted_img*0.5)+0.5
        original_img,target_img,predicted_img = np.clip(original_img,0,1),np.clip(target_img,0,1),np.clip(predicted_img,0,1)

        plt.subplot(num_predictions, 3, displayed_count * 3 + 1); plt.title(f"Original Input ({displayed_count+1})"); plt.imshow(original_img); plt.axis('off')
        plt.subplot(num_predictions, 3, displayed_count * 3 + 2); plt.title(f"Target ({displayed_count+1})"); plt.imshow(target_img); plt.axis('off')
        plt.subplot(num_predictions, 3, displayed_count * 3 + 3); plt.title(f"Pix2Pix Predicted ({displayed_count+1})"); plt.imshow(predicted_img); plt.axis('off')
        displayed_count += 1
    plt.tight_layout(); plt.show()

# --- Main Execution ---
if __name__ == '__main__':
    assert NORMALIZE_TO_PLUS_MINUS_ONE, "Pix2Pix typically uses [-1,1] normalization."

    # The FILTER_NAME is now unique based on NUM_IMAGES_TO_USE.
    # This means logs, model saves, and checkpoints will go into new directories
    # specific to this data size, effectively starting "fresh" for this configuration.
    print(f"Starting run for configuration: {FILTER_NAME}")
     # This block should be updated to use the new constants and function name
    if not os.path.exists(MODEL_SAVE_DIR):
        os.makedirs(MODEL_SAVE_DIR)
        print(f"Created model save directory: {MODEL_SAVE_DIR}")

    # Use the new constants and generalized function here
    original_image_paths, target_image_paths = prepare_dataset_paths(
        INPUT_IMAGES_PATH, GROUND_TRUTH_IMAGES_PATH, NUM_IMAGES_TO_USE
    )

    if not original_image_paths or not target_image_paths:
        print("Exiting: Could not prepare dataset paths.")
    else:
        # The rest of your main execution logic remains the same
        trained_generator, history, val_dataset_for_eval = train_pix2pix_model(
            original_image_paths, target_image_paths
        )

        if trained_generator and val_dataset_for_eval:
            best_generator_path = os.path.join(MODEL_SAVE_DIR, GENERATOR_BEST_MODEL_NAME)
            generator_to_evaluate = None
            if os.path.exists(best_generator_path):
                print(f"\nLoading best saved generator for {FILTER_NAME} from: {best_generator_path}")
                try:
                    generator_to_evaluate = tf.keras.models.load_model(best_generator_path, compile=False)
                    print("Best generator loaded successfully.")
                except Exception as e:
                    print(f"Error loading best generator from {best_generator_path}: {e}. Using final epoch generator (if available).")
                    generator_to_evaluate = trained_generator
            else:
                print(f"Best generator not found at {best_generator_path}. Using generator from final epoch (if available).")
                if trained_generator is None : print("... and final generator is not available.")
                else: generator_to_evaluate = trained_generator

            if generator_to_evaluate:
                evaluate_pix2pix_generator(generator_to_evaluate, val_dataset_for_eval)
                display_pix2pix_predictions(generator_to_evaluate, val_dataset_for_eval, num_predictions=3)
            else:
                print("No generator model available for evaluation.")

        elif not trained_generator : # Attempt to load from checkpoint if training failed early
             checkpoint_load_dir = os.path.join(MODEL_SAVE_DIR, CHECKPOINT_MODEL_DIR_BASE) # Uses new FILTER_NAME
             if os.path.exists(checkpoint_load_dir):
                print("Training did not complete fully. Attempting to load generator from latest Pix2Pix checkpoint.")
                latest_pix2pix_checkpoint_file = tf.train.latest_checkpoint(checkpoint_load_dir)
                if latest_pix2pix_checkpoint_file:
                    print(f"Found Pix2Pix Keras checkpoint file: {latest_pix2pix_checkpoint_file}")
                    try:
                        temp_g = build_generator(); temp_d = build_discriminator()
                        pix2pix_model_for_load = Pix2Pix(temp_g, temp_d)
                        temp_g_opt = Adam(); temp_d_opt = Adam() # Dummy optimizers for compile
                        pix2pix_model_for_load.compile(temp_g_opt, temp_d_opt, generator_loss_fn, discriminator_loss_fn, dummy_loss)
                        pix2pix_model_for_load.load_weights(latest_pix2pix_checkpoint_file)
                        print("Successfully loaded weights into Pix2Pix model structure from checkpoint.")
                        generator_to_evaluate = pix2pix_model_for_load.generator

                        # Re-create val_dataset for evaluation if not available
                        if 'val_dataset_for_eval' not in locals() or val_dataset_for_eval is None:
                            if original_image_paths and target_image_paths: #Ensure paths were loaded
                                _, orig_val_paths, _, target_val_paths = train_test_split(original_image_paths, target_image_paths, test_size=0.1, random_state=42)
                                val_dataset_for_eval = create_tf_dataset(orig_val_paths, target_val_paths, BATCH_SIZE, shuffle=False) if orig_val_paths else None
                            else: val_dataset_for_eval = None

                        if val_dataset_for_eval and generator_to_evaluate:
                            evaluate_pix2pix_generator(generator_to_evaluate, val_dataset_for_eval)
                            display_pix2pix_predictions(generator_to_evaluate, val_dataset_for_eval, num_predictions=3)
                        else: print("Val dataset not available or generator extraction failed from checkpoint for eval.")
                    except Exception as e: print(f"Error loading Pix2Pix model or generator from checkpoint file: {e}")
                else: print("No suitable .keras checkpoint file found in directory for this configuration.")
             else: print("No generator trained and no checkpoint directory found for this configuration.")
    print("Script finished.")




