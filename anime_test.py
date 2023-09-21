# Define the loss functions
bce = BinaryCrossentropy() # The binary cross-entropy loss
sce = SparseCategoricalCrossentropy() # The sparse categorical cross-entropy loss

# Define the generator loss function
def generator_loss(fake_output):
  # The generator tries to make the discriminator output 1 for the fake images
  return bce(tf.ones_like(fake_output), fake_output)

# Define the discriminator loss function
def discriminator_loss(real_output, fake_output):
  # The discriminator tries to make the real output 1 and the fake output 0
  real_loss = bce(tf.ones_like(real_output), real_output)
  fake_loss = bce(tf.zeros_like(fake_output), fake_output)
  return real_loss + fake_loss

# Define the perceptual loss function
def perceptual_loss(real_image, fake_image):
  # The perceptual loss measures the similarity between the features of the real and fake images
  # Use a pre-trained VGG19 model as the feature extractor
  vgg19 = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
  # Freeze the weights of the VGG19 model
  vgg19.trainable = False
  # Extract the features from the intermediate layers
  features = [vgg19.get_layer(name).output for name in ['block1_conv2', 'block2_conv2', 'block3_conv4', 'block4_conv4', 'block5_conv4']]
  # Create a model that takes an image as input and outputs the features
  feature_extractor = Model(vgg19.input, features)
  # Get the features of the real and fake images
  real_features = feature_extractor(real_image)
  fake_features = feature_extractor(fake_image)
  # Calculate the mean squared error between the features
  mse = tf.keras.losses.MeanSquaredError()
  # Initialize a list to store the perceptual loss for each layer
  perceptual_loss = []
  # Loop through the features
  for real_feature, fake_feature in zip(real_features, fake_features):
    # Calculate the perceptual loss for the layer
    layer_loss = mse(real_feature, fake_feature)
    # Add the layer loss to the list
    perceptual_loss.append(layer_loss)
  # Return the sum of the perceptual loss
  return tf.reduce_sum(perceptual_loss)

# Define the cycle-consistency loss function
def cycle_consistency_loss(real_image, cycled_image):
  # The cycle-consistency loss measures the difference between the real image and the cycled image
  # Use the L1 norm as the difference metric
  l1 = tf.keras.losses.MeanAbsoluteError()
  # Return the cycle-consistency loss
  return LAMBDA * l1(real_image, cycled_image)

# Define the encoder-decoder loss function
def encoder_decoder_loss(real_caption, pred_caption):
  # The encoder-decoder loss measures the difference between the real caption and the predicted caption
  # Use the sparse categorical cross-entropy as the difference metric
  return sce(real_caption, pred_caption)

# Define the optimizers
generator_optimizer = Adam(learning_rate=GEN_LR, beta_1=BETA_1, beta_2=BETA_2)
discriminator_optimizer = Adam(learning_rate=DIS_LR, beta_1=BETA_1, beta_2=BETA_2)
encoder_decoder_optimizer = Adam(learning_rate=GEN_LR, beta_1=BETA_1, beta_2=BETA_2)

# Define the checkpoints
checkpoint_dir = './checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 encoder_decoder_optimizer=encoder_decoder_optimizer,
                                 generator=generator,
                                 discriminator=discriminator,
                                 encoder_decoder=encoder_decoder)

# Define the metrics
gen_loss_metric = tf.keras.metrics.Mean(name='gen_loss')
dis_loss_metric = tf.keras.metrics.Mean(name='dis_loss')
enc_dec_loss_metric = tf.keras.metrics.Mean(name='enc_dec_loss')
bleu_metric = Bleu(name='bleu')
rouge_metric = Rouge(name='rouge')
meteor_metric = Meteor(name='meteor')

# Define a function to generate an image from a story
def generate_image(story, noise):
  # Encode the story into a sequence of integers
  story_seq = encode(story, vocab)
  # Expand the dimensions of the story and the noise
  story_seq = tf.expand_dims(story_seq, 0)
  noise = tf.expand_dims(noise, 0)
  # Generate an image from the story and the noise using the generator
  image = generator([story_seq, noise], training=False)
  # Return the image
  return image

# Define a function to generate a caption from a story
def generate_caption(story):
  # Encode the story into a sequence of integers
  story_seq = encode(story, vocab)
  # Expand the dimensions of the story
  story_seq = tf.expand_dims(story_seq, 0)
  # Initialize an empty caption
  caption = []
  # Initialize the decoder input with the start token
  decoder_input = tf.expand_dims([vocab.index('<start>')], 0)
  # Loop until the end token is generated or the maximum length is reached
  for i in range(MAX_LEN):
    # Generate a word from the story and the decoder input using the encoder-decoder
    word = encoder_decoder([story_seq, decoder_input], training=False)
    # Get the index of the word with the highest probability
    word = tf.argmax(word, axis=-1)
    # Append the word to the caption
    caption.append(word)
    # Check if the end token is generated
    if word == vocab.index('<end>'):
      # Break the loop
      break
    # Update the decoder input with the word
    decoder_input = word
  # Decode the caption into a text
  caption = decode(caption, vocab)
  # Return the caption
  return caption

# Define a function to train the model for one epoch
@tf.function
def train_step(real_image, real_caption, story, noise):
  # Use a gradient tape to record the operations
  with tf.GradientTape(persistent=True) as tape:
    # Generate a fake image from the story and the noise using the generator
    fake_image = generator([story, noise], training=True)
    # Generate a cycled image from the fake image and the story using the generator
    cycled_image = generator([story, fake_image], training=True)
    # Generate a fake caption from the story using the encoder-decoder
    fake_caption = encoder_decoder([story, fake_image], training=True)
    # Discriminate the real image and the real caption using the discriminator
    real_output = discriminator([real_image, real_caption], training=True)
    # Discriminate the fake image and the fake caption using the discriminator
    fake_output = discriminator([fake_image, fake_caption], training=True)
    # Calculate the generator loss
    gen_loss = generator_loss(fake_output)
    # Calculate the discriminator loss
    dis_loss = discriminator_loss(real_output, fake_output)
    # Calculate the perceptual loss
    per_loss = perceptual_loss(real_image, fake_image)
    # Calculate the cycle-consistency loss
    cyc_loss = cycle_consistency_loss(real_image, cycled_image)
    # Calculate the encoder-decoder loss
    enc_dec_loss = encoder_decoder_loss(real_caption, fake_caption)
    # Calculate the total generator loss
    total_gen_loss = gen_loss + per_loss + cyc_loss + enc_dec_loss
  # Calculate the gradients of the generator loss with respect to the generator variables
  generator_gradients = tape.gradient(total_gen_loss, generator.trainable_variables)
  # Calculate the gradients of the discriminator loss with respect to the discriminator variables
  discriminator_gradients = tape.gradient(dis_loss, discriminator.trainable_variables)
  # Calculate the gradients of the encoder-decoder loss with respect to the encoder-decoder variables
  encoder_decoder_gradients = tape.gradient(enc_dec_loss, encoder_decoder.trainable_variables)
  # Apply the gradients to the generator optimizer
  generator_optimizer.apply_gradients(zip(generator_gradients, generator.trainable_variables))
  # Apply the gradients to the discriminator optimizer
  discriminator_optimizer.apply_gradients(zip(discriminator_gradients, discriminator.trainable_variables))
  # Apply the gradients to the encoder-decoder optimizer
  encoder_decoder_optimizer.apply_gradients(zip(encoder_decoder_gradients, encoder_decoder.trainable_variables))
  # Update the metrics
  gen_loss_metric.update_state(total_gen_loss)
  dis_loss_metric.update_state(dis_loss)
  enc_dec_loss_metric.update_state(enc_dec_loss)
  bleu_metric.update_state(real_caption, fake_caption)
  rouge_metric.update_state(real_caption, fake_caption)
  meteor_metric.update_state(real_caption, fake_caption)

# Define a function to train the model for a given number of epochs
def train_model(data, epochs):
  # Loop through the epochs
  for epoch in range(epochs):
    # Print the epoch number
    print('Epoch', epoch + 1)
    # Reset the metrics
    gen_loss_metric.reset_states()
    dis_loss_metric.reset_states()
    enc_dec_loss_metric.reset_states()
    bleu_metric.reset_states()
    rouge_metric.reset_states()
    meteor_metric.reset_states()
    # Loop through the data in batches
    for batch in tf.data.Dataset.from_tensor_slices(data).batch(BATCH_SIZE):
      # Get the real image, the real caption, and the story from the batch
      real_image, real_caption, story = batch
      # Generate a random noise vector
      noise = tf.random.normal([BATCH_SIZE, LATENT_DIM])
      # Train the model for one step
      train_step(real_image, real_caption, story, noise)
    # Print the metrics
    print('Generator loss:', gen_loss_metric.result().numpy())
    print('Discriminator loss:', dis_loss_metric.result().numpy())
    print('Encoder-decoder loss:', enc_dec_loss_metric.result().numpy())
    print('BLEU score:', bleu_metric.result().numpy())
    print('ROUGE score:', rouge_metric.result().numpy())
    print('METEOR score:', meteor_metric.result().numpy())
    # Save the model checkpoint
    checkpoint.save(file_prefix=checkpoint_prefix)
    # Generate some samples for evaluation
    print('Generating samples...')
    # Select a random story from the data
    story = np.random.choice(data[:, 2])
    # Generate a random noise vector
    noise = tf.random.normal([1, LATENT_DIM])
    # Generate an image from the story and the noise
    image = generate_image(story, noise)
    # Generate a caption from the story
    caption = generate_caption(story)
    # Print the story, the caption, and the image
    print('Story:', story)
    print('Caption:', caption)
    plt.imshow(image[0])
    plt.show()
