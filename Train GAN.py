import numpy as np
import tensorflow as tf

# Define GAN Parameters
latent_dim = 8  # Random noise vector size

# Generator Model
generator = tf.keras.models.Sequential([
    tf.keras.layers.Dense(16, input_dim=latent_dim, activation=tf.keras.layers.LeakyReLU(alpha=0.2)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(32, activation=tf.keras.layers.LeakyReLU(alpha=0.2)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(4, activation='tanh')  # 4 output features (X, Y, Doppler Velocity, Reflectivity)
])

# Discriminator Model
discriminator = tf.keras.models.Sequential([
    tf.keras.layers.Dense(32, input_dim=4, activation=tf.keras.layers.LeakyReLU(alpha=0.2)),
    tf.keras.layers.Dense(16, activation=tf.keras.layers.LeakyReLU(alpha=0.2)),
    tf.keras.layers.Dense(1, activation='sigmoid')  # Binary classification (Real vs. Fake)
])

# Compile Discriminator
discriminator.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(0.0002, 0.5), metrics=['accuracy'])

# Combine Generator & Discriminator into GAN Model
discriminator.trainable = False  # Freeze discriminator while training generator
gan = tf.keras.models.Sequential([generator, discriminator])
gan.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(0.0002, 0.5))

def train_gan(real_samples, epochs=5000, batch_size=64):
    for epoch in range(epochs):
        # Generate Fake RADAR Data
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        generated_data = generator.predict(noise)

        # Select Real Samples
        real_batch = real_samples[np.random.randint(0, real_samples.shape[0], batch_size)]

        # Train Discriminator
        d_loss_real = discriminator.train_on_batch(real_batch, np.ones((batch_size, 1)))  # Real = 1
        d_loss_fake = discriminator.train_on_batch(generated_data, np.zeros((batch_size, 1)))  # Fake = 0
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # Train Generator (via GAN model)
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        g_loss = gan.train_on_batch(noise, np.ones((batch_size, 1)))  # Trick Discriminator

        # Print progress every 500 epochs
        if epoch % 500 == 0:
            print(f"Epoch {epoch}: Discriminator Loss: {d_loss[0]}, Generator Loss: {g_loss}")

    # Save the trained generator
    generator.save("trained_radar_gan.h5")
    print("✅ GAN trained and saved as 'trained_radar_gan.h5'")

    return generator, discriminator
